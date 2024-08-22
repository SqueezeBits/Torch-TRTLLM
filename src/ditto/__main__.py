import pprint
from typing import Any

import torch
from torch.export import Dim
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
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
):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if debug:

        def debug_hook(self: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], results: Any) -> None:
            with brief_tensor_repr():
                pprint.pp(
                    {
                        "args": args,
                        "kwargs": kwargs,
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
    max_batch_size: int = 1 << 30,
    max_seq_len: int = 256,
    export_inner_model: bool = False,
    verbose: bool = False,
) -> None:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
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

    with ForwardArgumentCollector(
        model_to_export,
        cache_handler=cache_handler,
        max_num_arguments=4,
    ) as argument_collector:
        print("==================Model Outputs Before Exporting==================")
        for response in run_generation(
            prompts,
            model,
            tokenizer,
            initial_cache=cache_handler.init_cache(batch_size=len(prompts)),
            device=device,
        ):
            print(response)
            print("==================================================================")

    example_inputs, dynamic_shapes = argument_collector.get_example_inputs_and_dynamic_shapes(
        batch_dim=Dim("batch", min=1, max=max_batch_size)
    )

    if verbose:
        argument_collector.print_readable()
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

        def patched_top_model_forward(*args: Any, **kwargs: Any):
            return exported_model(*args, **kwargs)

        model.forward = patched_top_model_forward

    print("==================Model Outputs After Exporting===================")
    for response in run_generation(
        prompts,
        model,
        tokenizer,
        initial_cache=cache_handler.init_cache(batch_size=len(prompts)),
        device=device,
    ):
        print(response)
        print("==================================================================")


def run_generation(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
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
            initial_cache.max_cache_len - 1 - prompt_size if isinstance(initial_cache, StaticCache) else None
        ),
    )
    responses = tokenizer.batch_decode(outputs)
    return [response.strip(tokenizer.pad_token) for response in responses]


if __name__ == "__main__":
    app()
