#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

main() {
    MODEL_SPECIFIC_ARGS=(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        "meta-llama/Llama-2-7b-chat-hf"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "mistralai/Mistral-7B-Instruct-v0.3 --model-type llama"
        "google/gemma-2-9b-it"
        "Qwen/Qwen2-7B-Instruct"
        "microsoft/Phi-3.5-mini-instruct"
        "microsoft/phi-4"
        "dnotitia/Llama-DNA-1.0-8B-Instruct"
        "CohereForAI/aya-expanse-8b --model-type commandr"
        "upstage/SOLAR-10.7B-Instruct-v1.0 --model-type llama"
        "42dot/42dot_LLM-SFT-1.3B --dtype float16 --model-type llama"
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --dtype float16 --model-type llama --trust-remote-code"
        # LoRA
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational \
            barissglc/tinyllama-tarot-v1 \
            snshrivas10/sft-tiny-chatbot"
        "meta-llama/Llama-2-7b-chat-hf \
            ArmaanSeth/Llama-2-7b-chat-hf-adapters-mental-health-counselling \
            ketchup123/llama-2-7b-chat-hf-safety-1000-HF \
            tricktreat/Llama-2-7b-chat-hf-guanaco-lora"
        # TP
        "meta-llama/Llama-3.1-8B-Instruct --tp-size 2"
        "mistralai/Mistral-7B-Instruct-v0.3 --model-type llama --tp-size 2"
        "google/gemma-2-9b-it --tp-size 2"
        "Qwen/Qwen2-7B-Instruct --tp-size 2"
        "microsoft/phi-4 --tp-size 2"
        "CohereForAI/aya-expanse-8b --model-type commandr --tp-size 2"
        # PP
        "meta-llama/Llama-3.1-8B-Instruct --pp-size 2"
        "mistralai/Mistral-7B-Instruct-v0.3 --model-type llama --pp-size 2"
        # "google/gemma-2-9b-it --pp-size 2" # unsupported in TensorRT-LLM
        "Qwen/Qwen2-7B-Instruct --pp-size 2"
        # "microsoft/phi-4 --pp-size 2" # unsupported in TensorRT-LLM
        # "CohereForAI/aya-expanse-8b --model-type commandr --pp-size 2" # unsupported in TensorRT-LLM
        # MoE
        "Qwen/Qwen1.5-MoE-A2.7B-Chat"
        "deepseek-ai/deepseek-moe-16b-chat --trust-remote-code --dtype bfloat16 --model-type deepseek_v1"
        "deepseek-ai/DeepSeek-V2-Lite-Chat  --trust-remote-code --dtype bfloat16"
        "mistralai/Mistral-7B-Instruct-v0.3 --model-type llama --tp-size 4"
        # W4A16 weight-only quantization (auto-gptq, autoawq)
        "saul95/Llama-3.2-1B-GPTQ --skip-native --print-output"
        "ciCic/llama-3.2-1B-Instruct-AWQ --skip-native --print-output"
        # W8A16 weight-only quantization (auto-gptq, autoawq)
        "fbaldassarri/TinyLlama_TinyLlama_v1.1-autogptq-int8-gs128-sym --skip-native --dtype float16 --print-output"
        "fbaldassarri/TinyLlama_TinyLlama_v1.1-autogptq-int8-gs128-asym --skip-native --dtype float16 --print-output"
        # FP8 per-tensor quantization (compressed-tensor)
        "neuralmagic/Llama-3.2-1B-Instruct-FP8 --skip-native --print-output"
        "neuralmagic/gemma-2-2b-it-FP8 --skip-native --print-output"
        "neuralmagic/starcoder2-3b-FP8 --skip-native --print-output"
        "nm-testing/TinyLlama-1.1B-Chat-v1.0-FP8-e2e --skip-native --print-output"
        "nm-testing/Phi-3-mini-128k-instruct-FP8 --skip-native --print-output"
    )

    for MODEL_SPECIFIC_ARG in "${MODEL_SPECIFIC_ARGS[@]}"; do
        echo "Testing $(echo ${MODEL_SPECIFIC_ARG} | sed 's/[[:space:]]\{2,\}/\n    /g')"
        ${SCRIPT_DIR}/compare.sh ${MODEL_SPECIFIC_ARG} --cleanup $@
    done
}

main $@
