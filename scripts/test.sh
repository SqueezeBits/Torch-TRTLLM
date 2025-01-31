#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

main() {
    AUTO_DTYPE_MODELS=(
        "meta-llama/Llama-2-7b-chat-hf"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "meta-llama/Llama-3.1-8B-Instruct"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "google/gemma-2-9b-it"
        "Qwen/Qwen2-7B-Instruct"
        "microsoft/Phi-3.5-mini-instruct"
        "CohereForAI/aya-expanse-8b"
        "dnotitia/Llama-DNA-1.0-8B-Instruct"
        "upstage/SOLAR-10.7B-Instruct-v1.0"
    )

    FLOAT16_DTYPE_MODELS=(
        "42dot/42dot_LLM-SFT-1.3B"
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    )
    
    for MODEL_ID in "${AUTO_DTYPE_MODELS[@]}"; do
        echo "Testing ${MODEL_ID} with auto dtype"
        ${SCRIPT_DIR}/compare.sh ${MODEL_ID} $@
    done

    for MODEL_ID in "${FLOAT16_DTYPE_MODELS[@]}"; do
        echo "Testing ${MODEL_ID} with float16 dtype"
        ${SCRIPT_DIR}/compare.sh ${MODEL_ID} --dtype float16 $@
    done
}

main $@
