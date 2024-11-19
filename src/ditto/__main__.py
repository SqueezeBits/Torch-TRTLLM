import os
from collections.abc import Callable
from typing import Annotated, Any

import torch
from loguru import logger
from torch.fx import GraphModule
from torch.fx.graph import CodeGen
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StaticCache,
)
from typer import Option, Typer

from . import (
    DynamicCacheHandler,
    StaticCacheHandler,
    brief_tensor_repr,
    trtllm_build,
)
from .config import DEFAULT_DEVICE, TRTLLM_LLAMA2_7B_CONFIG

app = Typer()


@app.command()
def generate(
    model_id: str,
    prompts: Annotated[list[str], Option(default_factory=list)],
    device: str = DEFAULT_DEVICE,
    use_static_cache: bool = False,
    debug: bool = False,
    max_batch_size: int = 1,
    max_seq_len: int = 256,
) -> None:
    if not prompts:
        logger.info("Using default prompts")
        prompts = ["Tell me about pikachu.", "What's the difference between python2 and python3?"]
    _prompts = "\n".join(f"* {p}" for p in prompts)
    logger.info(f"prompts:\n{_prompts}")

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = model.config.pad_token_id or 0
    tokenizer.padding_side = "left"

    if debug:
        cache_handler = (
            StaticCacheHandler(
                config=model.config,
                batch_size=1,
                max_seq_len=max_seq_len,
            )
            if use_static_cache
            else DynamicCacheHandler(config=model.config)
        )

        def debug_hook(self: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], results: Any) -> None:
            _args, _kwargs = cache_handler.map_to_tensor(args, kwargs)
            with brief_tensor_repr():
                logger.info(f"args: {_args}")
                logger.info(f"kwargs: {_kwargs}")
                logger.info(f"results: {results}")

        model.register_forward_hook(debug_hook, with_kwargs=True)

    if use_static_cache:
        cache = StaticCache(
            model.config,
            max_batch_size=max_batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=torch.float16,
        )
    else:
        cache = DynamicCache()

    responses = run_generation(prompts, model, tokenizer, initial_cache=cache, device=device)
    logger.info("============================Responses=============================")
    for response in responses:
        logger.info(response)
        logger.info("==================================================================")


def run_generation(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    initial_cache: Cache,
    device: str = DEFAULT_DEVICE,
) -> list[str]:
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device=device)
    prompt_size = inputs.input_ids.shape[-1]
    outputs = model.generate(
        **inputs,
        past_key_values=initial_cache,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=(
            initial_cache.max_cache_len - 1 - prompt_size if isinstance(initial_cache, StaticCache) else 1024
        ),
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
    output_dir: str,
    add_output: Annotated[list[str], Option(default_factory=list)],
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    verbose: bool = False,
    trust_remote_code: bool = False,
    transpose_weights: bool = False,
    mm_in_fp32: bool = False,
) -> None:
    assert not os.path.exists(output_dir) or os.path.isdir(output_dir), f"Invalid output directory: {output_dir}"
    app.pretty_exceptions_show_locals = verbose
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    logger.info(f"device: {device} | dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    engine = trtllm_build(
        model,
        transpose_weights=transpose_weights,
        mm_in_fp32=mm_in_fp32,
        verbose=verbose,
        extra_passes=[add_outputs(add_output)] if add_output else [],
        output_names=["logits", *add_output],
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
        engine_file.write(engine.serialize())

    # TODO: implement config compilation [FOC-422](https://squeezebits.atlassian.net/browse/FOC-422?atlOrigin=eyJpIjoiMTM5MDEyOWRjMzVlNDJiZDlhNzU3YjlkYjIwNTNkNjQiLCJwIjoiaiJ9)
    config_path = get_output_path("config.json")
    logger.info(f"Writing engine config at {config_path}")
    with open(config_path, "w") as config_file:
        config_file.write(TRTLLM_LLAMA2_7B_CONFIG)


def add_outputs(names: list[str]) -> Callable[[GraphModule], GraphModule]:
    def reset_output(gm: GraphModule) -> GraphModule:
        nodes = {n.name: n for n in gm.graph.nodes}
        for node in reversed(gm.graph.nodes):
            if node.op == "output":
                break
        else:
            logger.exception("No output node found in the graph")
        try:
            outputs = node.args[0] + tuple(nodes[name] for name in names)
        except KeyError as e:
            gm.print_readable()
            logger.exception(f"No such node: {e}")
        logger.info(f"Adding new outputs to the graph: {', '.join(names)}")
        node.args = (outputs,)
        gm.graph._codegen = CodeGen()
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    return reset_output


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

    keys_x = set(tensors_x)
    keys_y = set(tensors_y)
    if show_only:
        keys_x = keys_x.intersection(show_only)
        keys_y = keys_y.intersection(show_only)
    if keys_x != keys_y:
        logger.warning(f"Keys are different!\nOnly in {x}: {keys_x - keys_y}\nOnly in {y}: {keys_y - keys_x}\n")
    keys = keys_x & keys_y
    for name in keys:
        tx = tensors_x[name]
        ty = tensors_y[name]
        if floating_point_only and not (torch.is_floating_point(tx) and torch.is_floating_point(ty)):
            continue
        logger.info(f"======================Comparing {name}======================")
        if name in transpose:
            logger.info(f"Transposing {name} in {x}")
            tx = tx.t()
        has_same_shape = tx.shape == ty.shape
        has_same_dtype = tx.dtype == ty.dtype
        if has_same_shape:
            if not has_same_dtype:
                dtype = torch.promote_types(tx.dtype, ty.dtype)
                logger.warning(
                    f"The tensors have different dtypes: {tx.dtype}, {ty.dtype}. Will promote dtypes to {dtype}"
                )
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

    if keys_only_x := keys_x - keys_y:
        logger.info(f"Tensors only found in {x}")
        for name in keys_only_x:
            t = tensors_x[name]
            logger.info(f"{name}\n{summary(t)}")

    if keys_only_y := keys_y - keys_x:
        logger.info(f"Tensors only found in {y}")
        for name in keys_only_y:
            t = tensors_y[name]
            logger.info(f"{name}\n{summary(t)}")


if __name__ == "__main__":
    app()
