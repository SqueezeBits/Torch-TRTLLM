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
    ArgumentsForExport,
    DynamicCacheHandler,
    StaticCacheHandler,
    TensorRTInferenceSession,
    brief_tensor_repr,
    build_engine,
    trtllm_build,
    trtllm_export,
)
from .config import DEFAULT_DEVICE, INPUT_IDS_UNSQUEEZE_DIM

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


@app.command()
@torch.no_grad()
def run(
    model_id: str,
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    engine_path: str = "",
    verbose: bool = False,
    trust_remote_code: bool = False,
    transpose_weights: bool = False,
    mm_in_fp32: bool = False,
) -> None:
    app.pretty_exceptions_show_locals = verbose
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    logger.info(f"device: {device} | dtype: {dtype}")
    suffix_items = ["fp16" if torch_dtype == torch.float16 else "fp32"]
    if transpose_weights:
        suffix_items.append("tw")
    if mm_in_fp32:
        suffix_items.append("fp32mm")
    engine_suffix = "-".join(suffix_items)

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
    )

    engine_path = engine_path or (
        os.path.join(f"{os.path.abspath(model_id)}-{engine_suffix}-ditto", "rank0.engine")
        if os.path.isdir(model_id)
        else f"{model_id}-{engine_suffix}.engine"
    )
    logger.info(f"Saving serialized engine at {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as engine_file:
        engine_file.write(engine.serialize())


@app.command()
@torch.no_grad()
def debug(
    model_id: str,
    skipped_optimizers: Annotated[list[str], Option(default_factory=list)],
    layer: str = "",
    target: str = "",
    prompt: str = "",
    transpose_weights: bool = False,
    mm_in_fp32: bool = False,
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    trust_remote_code: bool = False,
    verbose: bool = False,
) -> None:
    app.pretty_exceptions_show_locals = verbose
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    logger.info(f"device: {device} | dtype: {dtype}")

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = model.config.pad_token_id or 0
    tokenizer.padding_side = "left"

    if verbose:
        print(model)

    if not prompt:
        logger.info("Using default prompt")
        prompt = "Hey, are you conscious?"
    logger.info(f"prompt: {prompt}")
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int32, device=device)

    def graph_break(at: str) -> Callable[[GraphModule], GraphModule]:
        def reset_output(gm: GraphModule) -> GraphModule:
            if not at:
                return gm
            nodes = {n.name: n for n in gm.graph.nodes}
            for node in reversed(gm.graph.nodes):
                if node.op == "output":
                    break
            else:
                return gm
            try:
                node.args = (nodes[at],)
            except KeyError as e:
                logger.error(f"No such node: {at}")
                gm.print_readable()
                raise e
            gm.graph._codegen = CodeGen()
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            gm.recompile()
            return gm

        return reset_output

    def remove_unused_inputs(gm: GraphModule) -> GraphModule:
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for placeholder in placeholders:
            if placeholder.users:
                continue
            gm.graph.erase_node(placeholder)
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        return gm

    arguments_for_export = ArgumentsForExport.get_trtllm_inputs(
        device=device,
        use_cache=False,
    )

    graph_module = trtllm_export(
        model,
        arguments_for_export,
        skipped_optimizers=skipped_optimizers,
        transpose_weights=transpose_weights,
        mm_in_fp32=mm_in_fp32,
        extra_passes=[graph_break(target), remove_unused_inputs],
        verbose=verbose,
    )

    if verbose:
        graph_module.print_readable()

    logger.info("Building TensorRT engine ...")
    engine = build_engine(
        graph_module,
        (),
        arguments_for_export.torch_trt_inputs,
        name=type(model).__name__,
    )

    if layer:
        print(f"Hijacking the output from the layer {layer}")
        assert isinstance(model, torch.nn.Module)
        layer_outputs: tuple[torch.Tensor, ...] = ()

        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            nonlocal layer_outputs
            layer_outputs = output

        handle = model.get_submodule(layer).register_forward_hook(hook, with_kwargs=True)

        _ = model(input_ids.unsqueeze(INPUT_IDS_UNSQUEEZE_DIM))
        assert len(layer_outputs) > 0
        output = layer_outputs[0]
        handle.remove()
    else:
        print(f"Evaluating graph module output at node {target}")
        output = graph_module(input_ids)

    session = TensorRTInferenceSession(engine)
    output_name = session.output_tensor_names[0]
    session.allocate_outputs({output_name: output})
    trt_output = session.run({"input_ids": input_ids})[output_name]
    diff = (output - trt_output).abs()

    def summary(t: torch.Tensor) -> str:
        return "\n".join(
            [
                f"  * shape: {(*t.shape,)}",
                f"  * dtype: {t.dtype}",
                f"  * mean: {t.mean().item()}",
                f"  * std: {t.std().item()}",
            ]
        )

    for name, value in {
        "native output": output,
        "TRT output": trt_output,
        "Absolute diff": diff,
    }.items():
        logger.info(f"{name}: {value}\n{summary(value)}")


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


if __name__ == "__main__":
    app()
