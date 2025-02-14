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
    )

    for MODEL_SPECIFIC_ARG in "${MODEL_SPECIFIC_ARGS[@]}"; do
        echo "Testing $(echo ${MODEL_SPECIFIC_ARG} | sed 's/[[:space:]]\{2,\}/\n    /g')"
        ${SCRIPT_DIR}/compare.sh ${MODEL_SPECIFIC_ARG} $@
    done
}

main $@
