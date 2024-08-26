import pprint
from typing import Any

import torch
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
from typer import Typer

from . import (
    DynamicCacheHandler,
    ForwardArgumentCollector,
    StaticCacheHandler,
    brief_tensor_repr,
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


@app.command("export")
@torch.no_grad()
def test_export(
    model_id: str,
    prompts: list[str] = ["Tell me about pikachu.", "What's the difference between python2 and python3?"],
    device: str = DEFAULT_DEVICE,
    dtype: str = "float16",
    use_static_cache: bool = False,
    max_seq_len: int = 256,
    export_inner_model: bool = False,
    verbose: bool = False,
) -> None:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
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

    def preprocess_inputs(arguments: dict[str, Any]) -> dict[str, Any]:
        arguments = {
            name: cache_handler.to_tensor(value) if isinstance(value, Cache) else value
            for name, value in arguments.items()
        }
        if not cache_handler.is_static and isinstance(
            (attention_mask := arguments.pop("attention_mask", None)), torch.Tensor
        ):
            if not isinstance((past_key_values := arguments.get("past_key_values")), torch.Tensor):
                raise ValueError(f"Expected past_key_values to be a tensor but got {past_key_values}")
            if not isinstance((input_ids := arguments.get("input_ids", None)), torch.Tensor):
                raise ValueError(f"Expected input_ids to be a tensor but got {input_ids}")
            input_ids_len = input_ids.shape[-1]
            cache_len = past_key_values.shape[-2]
            assert attention_mask.shape[-1] == input_ids_len + cache_len
            prefilled_attention_mask, generation_attention_mask = torch.split(
                attention_mask, [cache_len, input_ids_len], dim=-1
            )
            arguments["prefilled_attention_mask"] = prefilled_attention_mask
            arguments["generation_attention_mask"] = generation_attention_mask
        return arguments

    argument_collector = ForwardArgumentCollector(
        model_to_export,
        preprocess_inputs=preprocess_inputs,
        verbose=verbose,
    )
    if not use_static_cache:
        # collect two arguments forwarded to the model with batch_size=1
        with argument_collector.collect(2):
            _ = run_generation(
                prompts[:1],
                model,
                tokenizer,
                initial_cache=cache_handler.init_cache(batch_size=1),
                device=device,
            )

    # collect two arguments forwarded to the model with batch_size=len(prompts)
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

    example_inputs, dynamic_shapes = argument_collector.get_example_inputs_and_dynamic_shapes(
        dim_names={
            "input_ids": {0: "batch", 1: "seq_len"},
            "past_key_values": {4: "cache_size"},
        }
    )

    if verbose:
        with brief_tensor_repr():
            print("================Inputs for torch.export=================")
            for name, value in example_inputs.items():
                print(f"{name}: {value}")
            print("======================Constraints=======================")
            for name, constraint in dynamic_shapes.items():
                print(f"{name}: {constraint}")
            print("========================================================")

    exported_model = export(
        cache_handler,
        model_to_export,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
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

    print("==================Model Outputs After Exporting===================")
    for response in run_generation(
        prompts,
        model,
        tokenizer,
        initial_cache=cache_handler.init_cache(batch_size=len(prompts), device=device),
        device=device,
    ):
        print(response)
        print("==================================================================")


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
