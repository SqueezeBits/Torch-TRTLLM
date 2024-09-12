import pprint
from collections.abc import Callable
from typing import Annotated, Any

import torch
from torch.fx import GraphModule
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
from transformers.modeling_outputs import CausalLMOutputWithPast
from typer import Option, Typer

from . import (
    DynamicCacheHandler,
    DynamicDimension,
    ForwardArgumentCollector,
    PostExportWrapper,
    SDPBackend,
    StaticCacheHandler,
    brief_tensor_repr,
    convert,
    export,
)
from .constants import DEFAULT_DEVICE

app = Typer()


@app.command()
def generate(
    model_id: str,
    prompt: str = "Who are you?",
    device: str = DEFAULT_DEVICE,
    use_static_cache: bool = False,
    debug: bool = False,
    max_batch_size: int = 1,
    max_seq_len: int = 256,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
                pprint.pp(
                    {
                        "args": _args,
                        "kwargs": _kwargs,
                        "results": results,
                    },
                    indent=2,
                )

        model.register_forward_hook(debug_hook, with_kwargs=True)

    if use_static_cache:
        prompt_cache = StaticCache(
            model.config,
            max_batch_size=max_batch_size,
            max_cache_len=max_seq_len,
            device=device,
            dtype=torch.float16,
        )
    else:
        prompt_cache = DynamicCache()
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
    outputs = model.generate(
        **inputs,
        past_key_values=prompt_cache,
        do_sample=False,
        max_new_tokens=max_seq_len - inputs["input_ids"].shape[-1],
    )
    response = tokenizer.batch_decode(outputs)[0]
    pprint.pp(response)
    pprint.pp(type(prompt_cache))
    pprint.pp(len(prompt_cache.key_cache))
    pprint.pp(prompt_cache.key_cache[0].shape)
    pprint.pp(len(prompt_cache.value_cache))
    pprint.pp(prompt_cache.value_cache[0].shape)


@app.command()
@torch.no_grad()
def run(
    model_id: str,
    sdp_backends: Annotated[list[str], Option(default_factory=list)],
    prompts: Annotated[list[str], Option(default_factory=list)],
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    use_static_cache: bool = False,
    max_seq_len: int = 256,
    export_inner_model: bool = False,
    verbose: bool = False,
) -> None:
    backends = parse_sdp_backends(sdp_backends)
    print(f"Using SDP backends: {backends}")

    if not prompts:
        print("Using default ", end="")
        prompts = ["Tell me about pikachu.", "What's the difference between python2 and python3?"]
    _prompts = "\n".join(f"* {p}" for p in prompts)
    print(f"prompts:\n{_prompts}")

    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = model.config.pad_token_id or 0
    tokenizer.padding_side = "left"
    cache_handler = (
        StaticCacheHandler(
            config=model.config,
            batch_size=len(prompts),
            max_seq_len=max_seq_len,
        )
        if use_static_cache
        else DynamicCacheHandler(config=model.config)
    )

    model_to_export = model.model if export_inner_model else model

    def unpack_past_key_values(arguments: dict[str, Any]) -> dict[str, Any]:
        if not (
            isinstance((past_key_values := arguments.pop("past_key_values", None)), Cache)
            and isinstance((key_cache := getattr(past_key_values, "key_cache", None)), list)
            and isinstance((value_cache := getattr(past_key_values, "value_cache", None)), list)
        ):
            return arguments
        arguments.update({f"past_key_{i}": tensor for i, tensor in enumerate(key_cache)})
        arguments.update({f"past_value_{i}": tensor for i, tensor in enumerate(value_cache)})
        return arguments

    def pack_past_key_values(get_batch_size_from: str, dim: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def impl(arguments: dict[str, Any]) -> dict[str, Any]:
            cache = cache_handler.init_cache(batch_size=arguments[get_batch_size_from].shape[dim])
            i = 0
            while True:
                if not (
                    isinstance((key := arguments.pop(f"past_key_{i}", None)), torch.Tensor)
                    and isinstance((value := arguments.pop(f"past_value_{i}", None)), torch.Tensor)
                ):
                    break
                cache.update(key, value, i)
                i += 1
            if i > 0:
                arguments["past_key_values"] = cache
            return arguments

        return impl

    argument_collector = ForwardArgumentCollector(
        model_to_export,
        preprocess_inputs=unpack_past_key_values,
        verbose=verbose,
    )
    if not use_static_cache:
        # collect at most two arguments forwarded to the model with batch_size=1
        with argument_collector.collect(2):
            _ = run_generation(
                prompts[:1],
                model,
                tokenizer,
                initial_cache=cache_handler.init_cache(batch_size=1),
                device=device,
            )

    # collect at most two arguments forwarded to the model with batch_size=len(prompts)
    with argument_collector.collect(2):
        responses = run_generation(
            prompts,
            model,
            tokenizer,
            initial_cache=cache_handler.init_cache(batch_size=len(prompts), device=device),
            device=device,
        )

    print("==================Model Outputs Before Exporting==================")
    for response in responses:
        print(response)
        print("==================================================================")

    batch = DynamicDimension(name="N", min=1, opt=1, max=32)
    q_size = DynamicDimension(name="Sq", min=0, opt=1, max=16384)
    kv_size = DynamicDimension(name="Skv", min=0, opt=256, max=16384)
    qkv_size = q_size + kv_size

    arguments_for_export = argument_collector.get_arguments_for_export(
        dynamic_dims={
            "input_ids": {0: batch, 1: q_size},
            "attention_mask": {1: qkv_size},
            "past_key_0": {2: kv_size},
        },
    )

    if verbose:
        arguments_for_export.print_readable()

    exported_program = export(
        model_to_export,
        arguments_for_export,
        input_processors={"pack_past_key_values": pack_past_key_values("input_ids", 0)},
        output_processors={"unpack_past_key_values": unpack_past_key_values},
        sdp_backends=backends,
    )

    def cast_to_model_output(outputs: dict[str, Any]) -> CausalLMOutputWithPast:
        return CausalLMOutputWithPast(**outputs)

    graph_module = exported_program.module()
    graph_module._forward_pre_hooks.clear()
    assert isinstance(graph_module, GraphModule)
    exported_model = PostExportWrapper(
        graph_module,
        input_processors={"unpack_past_key_values": unpack_past_key_values},
        output_processors={
            "pack_past_key_values": pack_past_key_values("logits", 0),
            "cast_to_model_output": cast_to_model_output,
        },
    )

    if verbose:
        exported_model.model.print_readable()
    else:
        print("GraphModule exported!")

    if export_inner_model:
        model.model = exported_model
    else:

        def patched_top_model_forward(*args: Any, **kwargs: Any) -> Any:
            return exported_model(*args, **kwargs)

        model.forward = patched_top_model_forward

    print("======Model Outputs After Exporting (with two extra prompts)======")
    for response in run_generation(
        prompts + ["Hey, are you conscious?", "Why is pi irrational?"],
        model,
        tokenizer,
        initial_cache=cache_handler.init_cache(batch_size=len(prompts) + 2, device=device),
        device=device,
    ):
        print(response)
        print("==================================================================")

    print("Building TensorRT engine ...")
    convert(exported_program, arguments_for_export.torch_trt_inputs)


def parse_sdp_backends(sdp_backends: list[str]) -> list[SDPBackend] | SDPBackend:
    try:
        available_backends = {
            "CUDNN_ATTENTION": SDPBackend.CUDNN_ATTENTION,
            "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
            "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
            "MATH": SDPBackend.MATH,
        }
        backends = [available_backends[x.upper()] for x in sdp_backends]
        return backends or SDPBackend.MATH
    except KeyError as e:
        raise ValueError(f"--sdp-backends must be one of {','.join(available_backends.keys())}") from e


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
