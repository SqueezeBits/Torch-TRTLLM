import os
from typing import Annotated, Literal

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typer import Option, Typer

from .api import trtllm_build
from .constants import DEFAULT_DEVICE, DISABLE_TRANSFORMER_PATCHES
from .types import trt_to_torch_dtype_mapping

if not DISABLE_TRANSFORMER_PATCHES:
    # pylint: disable-next=unused-import
    from .patches import transformers  # noqa: F401

    logger.info("To prevent custom patches for transformers set the environment variable DISABLE_TRANSFORMER_PATCHES=1")

app = Typer()


@app.command()
def generate(
    model_id: str,
    prompts: Annotated[list[str], Option(default_factory=list)],
    device: str = DEFAULT_DEVICE,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    max_output_len: int = 100,
) -> None:
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
        response.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").replace(tokenizer.bos_token, "")
        for response in responses
    ]


@app.command()
@torch.no_grad()
def build(
    model_id: str,
    add_output: Annotated[list[str], Option(default_factory=list)],
    output_dir: str = "",
    device: str = DEFAULT_DEVICE,
    dtype: str = "auto",
    verbose: bool = False,
    trust_remote_code: bool = False,
    run_matmuls_in_fp32: bool = True,
    run_activations_in_model_dtype: bool = True,
) -> None:
    output_dir = resolve_output_dir(output_dir, model_id)
    app.pretty_exceptions_show_locals = verbose

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=get_model_dtype(dtype),
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"device: {device} | dtype: {model.config.torch_dtype}")

    engine, config = trtllm_build(
        model,
        run_matmuls_in_fp32=run_matmuls_in_fp32,
        run_activations_in_model_dtype=run_activations_in_model_dtype,
        debug_node_names=add_output,
    )

    os.makedirs(output_dir, exist_ok=True)

    def get_output_path(filename: str) -> str:
        output_path = os.path.join(output_dir, filename)
        assert not os.path.exists(output_path) or os.path.isfile(output_path)
        if os.path.exists(output_path):
            logger.warning(f"The file at {output_path} will be overwritten")
        return output_path

    engine_path = get_output_path("rank0.engine")
    logger.info(f"Writing serialized engine at {engine_path}")
    with open(engine_path, "wb") as engine_file:
        engine_file.write(engine)

    config_path = get_output_path("config.json")
    logger.info(f"Writing engine config at {config_path}")
    with open(config_path, "w") as config_file:
        config_file.write(config.model_dump_json(indent=2))


def get_model_dtype(dtype: str) -> torch.dtype | Literal["auto"]:
    if dtype == "auto":
        return "auto"
    try:
        return {
            str(torch_dtype).removeprefix("torch."): torch_dtype
            for torch_dtype in trt_to_torch_dtype_mapping().values()
        }[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported torch data type: {dtype}") from e


def resolve_output_dir(output_dir: str, model_id: str) -> str:
    if not output_dir:
        output_dir = get_default_output_dir(model_id)
        logger.info(f"Using default output directory: {output_dir}")

    if os.path.exists(output_dir):
        assert os.path.isdir(
            output_dir
        ), f"Invalid output directory: {output_dir} already exists, but it is not a directory."
        logger.warning("The contents in output directory will be overwritten")

    return output_dir


def get_default_output_dir(model_id: str) -> str:
    if os.path.isdir(model_id):
        return os.path.join("./engines", os.path.basename(model_id))
    return os.path.join("./engines", model_id)


@app.command()
def compare(
    x: str,
    y: str,
    transpose: Annotated[list[str], Option(default_factory=list)],
    show_only: Annotated[list[str], Option(default_factory=list)],
    floating_point_only: bool = False,
    device: str = DEFAULT_DEVICE,
) -> None:
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
