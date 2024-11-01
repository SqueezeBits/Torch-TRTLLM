import os
from typing import Annotated, Any

import torch
import torch_tensorrt as torch_trt
from torch_tensorrt.dynamo.conversion import CompilationSettings
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
    brief_tensor_repr,
    build_engine,
    detailed_sym_node_str,
    export,
    get_inlined_graph_module,
)
from .constants import DEFAULT_DEVICE

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
        print("Using default ", end="")
        prompts = ["Tell me about pikachu.", "What's the difference between python2 and python3?"]
    _prompts = "\n".join(f"* {p}" for p in prompts)
    print(f"prompts:\n{_prompts}")

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
                print(f"args: {_args}")
                print(f"kwargs: {_kwargs}")
                print(f"results: {results}")

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
    print("============================Responses=============================")
    for response in responses:
        print(response)
        print("==================================================================")


@app.command()
@torch.no_grad()
def run(
    model_id: str,
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    engine_path: str = "",
    verbose: bool = False,
    trust_remote_code: bool = False,
    show_locals_on_exception: bool = False,
) -> None:
    app.pretty_exceptions_show_locals = show_locals_on_exception
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    print(f"device: {device} | dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    arguments_for_export = ArgumentsForExport.get_trtllm_inputs(
        device=device,
        use_cache=False,
    )

    if verbose:
        arguments_for_export.print_readable()

    print("torch.exporting module ...")
    exported_program = export(model, arguments_for_export)

    print("Lowering exported program into graph module ...")
    graph_module = get_inlined_graph_module(exported_program)

    model_name = type(model).__name__
    if verbose:
        with detailed_sym_node_str():
            with open(f"{model_name}_graph_module.txt", "w") as f:
                f.write(graph_module.print_readable(print_output=False))
            with open(f"{model_name}_graph.txt", "w") as f:
                f.write(f"{graph_module.graph}")

    print("Building TensorRT engine ...")
    settings = CompilationSettings(
        assume_dynamic_shape_support=True,
        enabled_precisions={torch_trt.dtype.f16, torch_trt.dtype.f32},
        debug=verbose,
        optimization_level=3,
        max_aux_streams=-1,
    )
    engine = build_engine(
        graph_module,
        (),
        arguments_for_export.torch_trt_inputs,
        settings=settings,
        name=type(model).__name__,
    )
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        print(f"({i}) {name}: {engine.get_tensor_shape(name)}")
    engine_path = engine_path or (
        os.path.join(
            f"{os.path.abspath(model_id)}-{'fp16' if torch_dtype == torch.float16 else 'fp32'}-ditto", "rank0.engine"
        )
        if os.path.isdir(model_id)
        else f"{model_id}.engine"
    )
    print(f"Saving serialized engine at {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as engine_file:
        engine_file.write(engine.serialize())


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
