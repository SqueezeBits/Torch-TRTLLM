# Copyright 2025 SqueezeBits, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code=misc
# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
import os
import time
from typing import Annotated, Literal

import click
import modelopt.torch.opt as mto
import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typer import Argument, Option, Typer

from .api import trtllm_build
from .constants import DEFAULT_DEVICE
from .contexts import disable_modelopt_peft_patches, disable_torch_jit_state
from .literals import DTypeLiteral
from .peft import load_peft_adapters
from .types import trt_to_torch_dtype_mapping

app = Typer(context_settings={"help_option_names": ["-h", "--help"]})

FLOATING_POINT_DTYPES = ("float32", "float16", "bfloat16")


@app.command()
def generate(
    model_id: Annotated[
        str, Argument(help="A pretrained model name or path. e.g., 'meta-llama/Meta-Llama-3-8B-Instruct'.")
    ],
    prompts: Annotated[list[str], Option(default_factory=list, help="List of input prompts.")],
    device: Annotated[str, Option(help="Device to run generation on.")] = DEFAULT_DEVICE,
    dtype: Annotated[str, Option(help="Data type to use for the model.")] = "auto",
    trust_remote_code: Annotated[bool, Option(help="Trust remote code.")] = False,
    max_output_len: Annotated[int, Option(help="Maximum number of new tokens to generate.")] = 100,
) -> None:
    """Generate text completions for given prompts using a language model."""
    if not prompts:
        logger.info("Using default prompts")
        prompts = ["Hey, are you conscious?"]
    _prompts = "\n".join(f"* {p}" for p in prompts)
    logger.info(f"prompts:\n{_prompts}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_model_dtype(dtype),
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"device: {device} | dtype: {model.config.torch_dtype}")
    if dtype == "auto" and model.config.torch_dtype == torch.float32:
        logger.warning(
            "Using FP32 model might consume significant amount of memory. "
            "Specify `--dtype float16` or `--dtype bfloat16` to reduce memory usage."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = model.config.pad_token_id or 0
    tokenizer.padding_side = "left"

    responses = run_generation(prompts, model, tokenizer, max_new_tokens=max_output_len, device=device)
    logger.info("============================Responses=============================")
    for response in responses:
        logger.info(response)
        logger.info("==================================================================")


def run_generation(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    max_new_tokens: int | None = None,
    device: str = DEFAULT_DEVICE,
) -> list[str]:
    """Run text generation for a list of prompts.

    Args:
        prompts (list[str]): List of input prompts to generate completions for
        model (PreTrainedModel): The language model to use for generation
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer for the model
        max_new_tokens (int | None, optional): Maximum number of new tokens to generate. Defaults to None.
        device (str, optional): Device to run generation on. Defaults to DEFAULT_DEVICE.

    Returns:
        list[str]: List of generated text completions with special tokens removed
    """
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device=device)
    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=max_new_tokens,
    )
    responses = tokenizer.batch_decode(outputs)
    return [
        response.replace(tokenizer.pad_token or "<pad>", "")
        .replace(tokenizer.eos_token or "<eos>", "")
        .replace(tokenizer.bos_token or "<bos>", "")
        for response in responses
    ]


@app.command()
@torch.no_grad()
def build(
    model_id: Annotated[str, Argument(help="A pretrained model name or path.")],
    add_output: Annotated[
        list[str],
        Option(default_factory=list, help="List of node names to add as output. See docs/DEBUG.md for details."),
    ],
    peft_ids: Annotated[
        list[str],
        Option(
            "-p",
            "--peft-ids",
            default_factory=list,
            help="List of LoRA adapter IDs to apply to the model.",
        ),
    ],
    output_dir: Annotated[
        str,
        Option(
            "-o",
            "--output-dir",
            help="Path to the output directory. If not specified, `./engines/<model_id>` will be used.",
        ),
    ] = "",
    dtype: Annotated[
        str,
        Option(
            click_type=click.Choice(("auto", *FLOATING_POINT_DTYPES)),
            help="Data type to use for the model. Defaults to `auto`.",
        ),
    ] = "auto",
    verbose_failure: Annotated[bool, Option(help="Show local variable values on failure.")] = False,
    trust_remote_code: Annotated[bool, Option(help="Trust remote code from Hugging Face Hub.")] = False,
    run_matmuls_in_fp32: Annotated[bool, Option(help="Run matmuls in fp32.")] = False,
    run_activations_in_model_dtype: Annotated[bool, Option(help="Run activations in model dtype.")] = True,
    max_batch_size: Annotated[int, Option(help="Maximum number of requests that the engine can schedule.")] = 2048,
    max_seq_len: Annotated[
        int | None, Option(help="Maximum total length of one request, including prompt and generated output.")
    ] = None,
    max_num_tokens: Annotated[
        int, Option(help="Maximum number of batched input tokens after padding is removed in each batch.")
    ] = 8192,
    opt_num_tokens: Annotated[
        int | None, Option(help="Optimal number of batched input tokens after padding is removed in each batch.")
    ] = None,
    max_beam_width: Annotated[int, Option(help="Maximum number of beams for beam search decoding.")] = 1,
    pp_size: Annotated[int, Option(help="N-way pipeline parallelism size.", min=1)] = 1,
    tp_size: Annotated[int, Option(help="N-way tensor parallelism size.", min=1)] = 1,
    logits_dtype: Annotated[
        DTypeLiteral,
        Option(
            click_type=click.Choice(FLOATING_POINT_DTYPES),
            help="Data type of logits. Defaults to `float32`.",
        ),
    ] = "float32",
    gather_context_logits: Annotated[bool, Option(help="Enable gathering context logits.")] = False,
    gather_generation_logits: Annotated[bool, Option(help="Enable gathering generation logits.")] = False,
    gather_all_logits: Annotated[
        bool, Option(help="Equivalent to `--gather-context-logits --gather-generation-logits`.")
    ] = False,
    run_routers_in_model_dtype: Annotated[
        bool, Option(help="Run linear layers for routers in MoE models in model dtype instead of FP32.")
    ] = False,
) -> None:
    """Build a TensorRT-LLM engine from a pretrained model."""
    assert not (
        (tp_size > 1 or pp_size > 1) and peft_ids
    ), "Tensor Parallelism or Pipeline Parallelism with LoRA is currently not supported"
    if gather_all_logits:
        gather_context_logits = gather_generation_logits = True
    output_dir = resolve_output_dir(output_dir, model_id)
    app.pretty_exceptions_show_locals = verbose_failure

    with disable_torch_jit_state(), disable_modelopt_peft_patches():
        logger.info(f"Loading model {model_id}")

        mto.enable_huggingface_checkpointing()        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=get_model_dtype(dtype),
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        if peft_ids:
            model = load_peft_adapters(model, peft_ids)
    logger.info(f"device: {model.device} | dtype: {model.config.torch_dtype}")
    if dtype == "auto" and model.config.torch_dtype == torch.float32:
        logger.warning(
            "Using FP32 model might consume significant amount of memory. "
            "Specify `--dtype float16` or `--dtype bfloat16` to reduce memory usage."
        )

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.perf_counter()
    trtllm_build(
        model,
        output_dir,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        debug_node_names=add_output,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        opt_num_tokens=opt_num_tokens,
        max_beam_width=max_beam_width,
        pp_size=pp_size,
        tp_size=tp_size,
        logits_dtype=logits_dtype,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        run_routers_in_model_dtype=run_routers_in_model_dtype,
    )
    minutes, seconds = divmod(int(time.perf_counter() - start_time), 60)
    logger.info(f"Build completed in {minutes:02d}:{seconds:02d}")


def get_model_dtype(dtype: str) -> torch.dtype | Literal["auto"]:
    """Get PyTorch dtype from string representation.

    Args:
        dtype (str): String representation of dtype or "auto"

    Returns:
        torch.dtype | Literal["auto"]: PyTorch dtype or "auto"

    Raises:
        ValueError: If dtype string is not recognized
    """
    if dtype == "auto":
        return "auto"
    try:
        return {
            str(torch_dtype).removeprefix("torch."): torch_dtype
            for torch_dtype in trt_to_torch_dtype_mapping().values()
        }[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported torch data type: {dtype}") from e


def resolve_output_dir(output_dir: str | None, model_id: str) -> str:
    """Resolve the output directory path.

    Args:
        output_dir (str | None): User-specified output directory path
        model_id (str): Model identifier used to generate default path

    Returns:
        str: Resolved output directory path

    Raises:
        AssertionError: If output_dir exists but is not a directory
    """
    if not output_dir:
        output_dir = get_default_output_dir(model_id)
        logger.info(f"Using default output directory: {output_dir}")

    if os.path.exists(output_dir):
        assert os.path.isdir(
            output_dir
        ), f"Invalid output directory: {output_dir} already exists, but it is not a directory."
        logger.warning("The contents in the output directory will be overwritten")

    return output_dir


def get_default_output_dir(model_id: str) -> str:
    """Get default output directory path based on model ID.

    Args:
        model_id (str): Model identifier or path

    Returns:
        str: Default output directory path
    """
    if os.path.isdir(model_id):
        return os.path.join("./engines", os.path.basename(model_id))
    return os.path.join("./engines", model_id)


@app.command()
def compare(
    x: Annotated[str, Argument(help="Path to the first output dictionary.")],
    y: Annotated[str, Argument(help="Path to the second output dictionary.")],
    transpose: Annotated[list[str], Option(default_factory=list, help="List of output names to be transposed")],
    show_only: Annotated[list[str], Option(default_factory=list, help="List of output names to be shown.")],
    floating_point_only: Annotated[bool, Option(help="Compare only floating point tensors.")] = False,
    device: Annotated[str, Option(help="Device to run comparison on.")] = DEFAULT_DEVICE,
) -> None:
    """Compare tensors between two PyTorch state dictionaries."""
    logger.info(f"Map location: {device}")
    tensors_x: dict[str, torch.Tensor] = torch.load(x, weights_only=True, map_location=device)
    tensors_y: dict[str, torch.Tensor] = torch.load(y, weights_only=True, map_location=device)
    keys_x = set(tensors_x)
    keys_y = set(tensors_y)
    if show_only:
        keys_x = keys_x.intersection(show_only)
        keys_y = keys_y.intersection(show_only)

    keys_only_x = keys_x - keys_y
    keys_only_y = keys_y - keys_x

    if keys_x != keys_y:
        logger.warning(f"Keys are different!\nOnly in {x}: {keys_only_x}\nOnly in {y}: {keys_only_y}\n")
    keys = keys_x & keys_y

    for name in keys:
        tx = tensors_x[name]
        ty = tensors_y[name]
        if floating_point_only and not (torch.is_floating_point(tx) and torch.is_floating_point(ty)):
            continue
        logger.info(f"======================Comparing {name}======================")
        if transpose_tx := name in transpose:
            logger.info(f"Transposing {name} in {x}")
        compare_tensors(x, tx, y, ty, transpose_tx=transpose_tx)

    if keys_only_x:
        logger.info(f"Tensors only found in {x}")
        for name in keys_only_x:
            t = tensors_x[name]
            logger.info(f"{name}\n{summary(t)}")

    if keys_only_y:
        logger.info(f"Tensors only found in {y}")
        for name in keys_only_y:
            t = tensors_y[name]
            logger.info(f"{name}\n{summary(t)}")


def compare_tensors(x: str, tx: torch.Tensor, y: str, ty: torch.Tensor, transpose_tx: bool = False) -> None:
    """Compare two tensors and log their differences.

    Args:
        x (str): Name/identifier for first tensor
        tx (torch.Tensor): First tensor
        y (str): Name/identifier for second tensor
        ty (torch.Tensor): Second tensor
        transpose_tx (bool, optional): Whether to transpose first tensor before comparison. Defaults to False.
    """
    if transpose_tx:
        tx = tx.t()
    has_same_shape = tx.shape == ty.shape
    has_same_dtype = tx.dtype == ty.dtype
    if has_same_shape:
        if not has_same_dtype:
            dtype = torch.promote_types(tx.dtype, ty.dtype)
            logger.warning(f"The tensors have different dtypes: {tx.dtype}, {ty.dtype}. Will promote dtypes to {dtype}")
            tx = tx.to(dtype)
            ty = ty.to(dtype)
        rtol = 1e-03 if tx.dtype == torch.float16 else 1e-05
        atol = 1e-03 if tx.dtype == torch.float16 else 1e-08
        allclose = torch.allclose(tx, ty, rtol=rtol, atol=atol)
        allsame = torch.all(tx == ty).item()
        (logger.info if allclose else logger.warning)(
            f"torch.allclose - {'pass' if allclose else 'fail'} (rtol: {rtol:.1e}, atol: {atol:.1e})"
        )
        (logger.info if allsame else logger.warning)(f"torch.all - {'pass' if allsame else 'fail'}")
        logger.info(f"absolute_difference\n{summary((tx - ty).abs())}")
        logger.info(f"percentage_error\n{summary((tx - ty).abs() / (ty.abs() + 1e-6), precision='.3%')}")
        logger.info(f"{x}\n{summary(tx)}")
        logger.info(f"{y}\n{summary(ty)}")
    else:
        logger.warning("Cannot compare tensors with different shapes")


def summary(t: torch.Tensor, *, precision: str = ".16e") -> str:
    """Generate a summary of tensor statistics.

    Args:
        t (torch.Tensor): Input tensor
        precision (str, optional): Format string for floating point values. Defaults to ".16e".

    Returns:
        str: Multi-line string containing tensor statistics
    """
    float_value = t.float()
    return "\n".join(
        [
            f"  * shape: {(*t.shape,)}",
            f"  * dtype: {t.dtype}",
            f"  * min: {format(float_value.min().item(), precision)}",
            f"  * mean: {format(float_value.mean().item(), precision)}",
            f"  * median: {format(float_value.median().item(), precision)}",
            f"  * max: {format(float_value.max().item(), precision)}",
            f"  * std: {float_value.std().item():.16e}",
            f"  * #nans: {float_value.isnan().sum().item()}",
            f"  * #infs: {float_value.isinf().sum().item()}",
        ]
    )


if __name__ == "__main__":
    app()
