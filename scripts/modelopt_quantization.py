
import torch
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

from transformers import AutoModelForCausalLM

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--export-tensorrt-llm-ckpt", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").cuda()

    def forward_loop(model):
        for _ in range(1):
            input_id = torch.ones((1, 128), dtype=torch.int64).to(model.device)
            attn_mask = torch.ones((1, 128), dtype=torch.int64).to(model.device)
            model(input_id, attn_mask)

    model = mtq.quantize(model, mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, forward_loop)

    if args.export_tensorrt_llm_ckpt:    
        with torch.inference_mode():
            export_tensorrt_llm_checkpoint(
                model,  # The quantized model.
                model.config.model_type,  # The type of the model as str, e.g gpt, gptj, llama.
                model.dtype,  # the weights data type to export the unquantized layers.
                args.out,  # The directory where the exported files will be stored.
                1,  # The number of GPUs used in the inference time for tensor parallelism.
                1,  # The number of GPUs used in the inference time for pipeline parallelism.
            )
    else:
        mto.enable_huggingface_checkpointing()
        model.save_pretrained(args.out)
