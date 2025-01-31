#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Command line arguments with default values
declare SKIP_DITTO_BUILD=false
declare SKIP_DITTO_RUN=false
declare SKIP_NATIVE_BUILD=false
declare SKIP_NATIVE_RUN=false
declare DEBUG_MODE=false
declare VERBOSE_MODE=false
declare PROMPT="Hey, are you conscious?"
declare TRTLLM_REPO=""
declare DTYPE="auto"
declare MODEL_TYPE=""

# Global variables for engine and artifact directories
declare DITTO_ENGINE_DIR=""
declare DITTO_ARTIFACTS_DIR=""
declare TRTLLM_ENGINE_DIR=""
declare TRTLLM_ARTIFACTS_DIR=""

# Print help message and exit
print_help() {
    echo "Usage: $0 MODEL_ID [options] [PEFT_ID1 PEFT_ID2 ...]"
    echo
    echo "Compares the outputs of TensorRT-LLM engines built using two different pipelines:"
    echo "1. Ditto pipeline: Uses Ditto's end-to-end build process"
    echo "2. Native pipeline: Uses TensorRT-LLM's native conversion and build tools"
    echo
    echo "The script validates that both pipelines produce identical outputs for:"
    echo "- Base models from Hugging Face Hub"
    echo "- PEFT/LoRA adapters from Hugging Face Hub"
    echo
    echo "Arguments:"
    echo "  MODEL_ID                 Base model identifier (e.g. meta-llama/Llama-2-7b-chat-hf)"
    echo "  [PEFT_ID1 PEFT_ID2 ...]  Optional PEFT model identifiers to apply"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message and exit"
    echo "  -v, --verbose           Show all terminal outputs"
    echo "  --skip-ditto           Skip building and running Ditto engine"
    echo "  --skip-ditto-build     Skip building Ditto engine"
    echo "  --skip-ditto-run       Skip running Ditto engine"
    echo "  --skip-native          Skip building and running TensorRT-LLM native engine"
    echo "  --skip-native-build    Skip building native TensorRT-LLM engine"
    echo "  --skip-native-run      Skip running native TensorRT-LLM engine"
    echo "  -d, --debug            Enable debug mode"
    echo "  --prompt TEXT          Input prompt (default: \"Hey, are you conscious?\")"
    echo "  --trtllm-repo PATH     Path to TensorRT-LLM repository"
    echo "                         (default: clone at user cache directory with version from pyproject.toml)"
    echo "  --dtype DTYPE          Data type for model conversion (default: auto)"
    echo "  --model-type TYPE      Model type for finding convert_checkpoint.py (default: auto-detected)"
    echo
    echo "[Example: Simply run meta-llama/Llama-2-7b-chat-hf]"
    echo "  $0 meta-llama/Llama-2-7b-chat-hf"
    echo
    echo "[Example: TinyLlama/TinyLlama-1.1B-Chat-v1.0 with 3 PEFT models]"
    echo "  $0 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\"
    echo "     givyboy/TinyLlama-1.1B-Chat-v1.0-mental-health-conversational \\"
    echo "     barissglc/tinyllama-tarot-v1 \\"
    echo "     snshrivas10/sft-tiny-chatbot"
    echo
    echo "[Example: meta-llama/Llama-2-7b-chat-hf with 3 PEFT models in debug mode skipping native TensorRT-LLM pipeline]"
    echo "  $0 --skip-native --debug meta-llama/Llama-2-7b-chat-hf \\"
    echo "     ArmaanSeth/Llama-2-7b-chat-hf-adapters-mental-health-counselling \\"
    echo "     ketchup123/llama-2-7b-chat-hf-safety-1000-HF \\"
    echo "     tricktreat/Llama-2-7b-chat-hf-guanaco-lora"
    exit 0
}

# Parse command line arguments
parse_args() {
    # Check for help flag
    for arg in "$@"; do
        if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
            print_help
        fi
    done

    # Check if at least 1 argument is provided
    if [ "$#" -lt 1 ]; then
        echo "Error: At least 1 argument required"
        echo "Try '$0 --help' for more information"
        exit 1
    fi

    # Parse arguments first to check for --trtllm-repo flag
    
    local ARGS=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-ditto)
                SKIP_DITTO_BUILD=true
                SKIP_DITTO_RUN=true
                shift
                ;;
            --skip-ditto-build)
                SKIP_DITTO_BUILD=true
                shift
                ;;
            --skip-ditto-run)
                SKIP_DITTO_RUN=true
                shift
                ;;
            --skip-native)
                SKIP_NATIVE_BUILD=true
                SKIP_NATIVE_RUN=true
                shift
                ;;
            --skip-native-build)
                SKIP_NATIVE_BUILD=true
                shift
                ;;
            --skip-native-run)
                SKIP_NATIVE_RUN=true
                shift
                ;;
            -d|--debug)
                DEBUG_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE_MODE=true
                shift
                ;;
            --prompt)
                PROMPT="$2"
                shift 2
                ;;
            --trtllm-repo)
                TRTLLM_REPO="$2"
                shift 2
                ;;
            --dtype)
                DTYPE="$2"
                shift 2
                ;;
            --model-type)
                MODEL_TYPE="$2"
                shift 2
                ;;
            -*)
                echo "Error: Unrecognized option: $1"
                echo "Try '$0 --help' for more information"
                exit 1
                ;;
            *)
                ARGS+=("$1")
                shift
                ;;
        esac
    done

    # Handle TensorRT-LLM repository path
    if [ -z "$TRTLLM_REPO" ]; then
        # Get default TensorRT-LLM directory only if --trtllm-repo was not specified
        DEFAULT_TRTLLM_DIR=$(python -c 'from platformdirs import user_cache_dir; print(f"{user_cache_dir()}/tensorrt-llm", end="")')
        if [ ! -d "$DEFAULT_TRTLLM_DIR" ]; then
            # Extract TensorRT-LLM version from pyproject.toml
            TRTLLM_VERSION=$(grep -Po 'tensorrt-llm = "\K[^"]*' "${SCRIPT_DIR}/../pyproject.toml")
            echo "Cloning TensorRT-LLM repository version ${TRTLLM_VERSION} to ${DEFAULT_TRTLLM_DIR} ..."
            git clone \
                -b v${TRTLLM_VERSION} \
                --single-branch \
                --depth 1 \
                --filter=blob:limit=300K \
                --no-tags \
                https://github.com/NVIDIA/tensorrt-llm \
                ${DEFAULT_TRTLLM_DIR}
        fi
        TRTLLM_REPO="$DEFAULT_TRTLLM_DIR"
    fi

    MODEL_ID=${ARGS[0]}
    unset ARGS[0]  # Remove first argument, leaving only PEFT_IDs
    PEFT_IDS=("${ARGS[@]}")  # Remaining arguments are PEFT_IDs
}

# Function to extract author names and create output dir suffix
get_output_suffix() {
    local suffix=""
    for peft_id in "$@"; do
        # Split by '/' and get the penultimate token
        local author=$(echo $peft_id | tr '/' '\n' | tail -n 2 | head -n 1)
        if [ -z "$suffix" ]; then
            suffix=$author
        else
            suffix="${suffix}+${author}"
        fi
    done
    echo $suffix
}

# Build PEFT_IDS arguments string for ditto
build_peft_args() {
    local args=""
    for peft_id in "$@"; do
        args="$args --peft-ids $peft_id"
    done
    echo $args
}

# Build lora_dir arguments string
build_lora_args() {
    local base_dir=$1
    local num_pefts=$2
    local args=""

    for i in $(seq 0 $(($num_pefts - 1))); do
        args="$args ${base_dir}/lora/$i"
    done
    echo $args
}

# Get HF cache dir
get_hf_cache_dir() {
    ${SCRIPT_DIR}/snapshot_download "$1"
}

# Build lora_dir arguments for TensorRT-LLM
build_trt_lora_args() {
    local args=""
    for peft_id in "$@"; do
        args="$args $(get_hf_cache_dir $peft_id)"
    done
    echo $args
}

# Append suffix to directory if not empty
append_suffix() {
    local dir=$1
    local suffix=$2
    if [ ! -z "$suffix" ]; then
        echo "${dir}+${suffix}"
    else
        echo "$dir"
    fi
}

# Extract output from run log
extract_output() {
    local run_log=$1
    local output_log=$2
    sed -e '1,/^Input \[Text 0\]:/d' \
        -e '/\[TensorRT-LLM\]\[INFO\] Refreshed the MPI local session/d' \
        "$run_log" > "$output_log"
}

# Set engine and artifact directories based on model ID and suffix
set_directories() {
    local model_id=$1
    local output_suffix=$(get_output_suffix "${PEFT_IDS[@]}")

    DITTO_ENGINE_DIR=$(append_suffix "engines/ditto/${model_id}" "$output_suffix")
    TRTLLM_ENGINE_DIR=$(append_suffix "engines/trtllm/${model_id}" "$output_suffix")

    if [ "$DEBUG_MODE" = true ]; then
        DITTO_ARTIFACTS_DIR=$(append_suffix "artifacts/ditto/${model_id}" "$output_suffix")
        TRTLLM_ARTIFACTS_DIR=$(append_suffix "artifacts/trtllm/${model_id}" "$output_suffix")
    fi

    if [ "$DTYPE" != "auto" ]; then
        DITTO_ENGINE_DIR="${DITTO_ENGINE_DIR}/${DTYPE}"
        TRTLLM_ENGINE_DIR="${TRTLLM_ENGINE_DIR}/${DTYPE}"
        if [ "$DEBUG_MODE" = true ]; then
            DITTO_ARTIFACTS_DIR="${DITTO_ARTIFACTS_DIR}/${DTYPE}"
            TRTLLM_ARTIFACTS_DIR="${TRTLLM_ARTIFACTS_DIR}/${DTYPE}"
        fi
    fi

    mkdir -p $DITTO_ENGINE_DIR $TRTLLM_ENGINE_DIR
}

# Build Ditto engine
ditto_build() {
    local model_id=$1
    local peft_ids=("${@:2}")

    echo "Building TensorRT-LLM engine using ditto ..."
    local cmd="DEBUG_ARTIFACTS_DIR=$DITTO_ARTIFACTS_DIR \
        ditto build ${model_id} \
        --output-dir $DITTO_ENGINE_DIR \
        --dtype $DTYPE \
        $(build_peft_args "${peft_ids[@]}")"

    if [ "$VERBOSE_MODE" = true ]; then
        eval "$cmd 2>&1 | tee $DITTO_ENGINE_DIR/build.log"
    else
        eval "$cmd &> $DITTO_ENGINE_DIR/build.log"
    fi
}

# Run inference with Ditto engine
ditto_run() {
    local model_id=$1
    local num_pefts=$2

    echo "Running TensorRT-LLM engine built by ditto ..."
    local base_cmd="python -u ${TRTLLM_REPO}/examples/run.py \
        --engine_dir $DITTO_ENGINE_DIR \
        --tokenizer_dir ${model_id} \
        --max_output_len 100 \
        --input_text \"$PROMPT\""

    if [ "$VERBOSE_MODE" = true ]; then
        eval "$base_cmd 2>&1 | tee $DITTO_ENGINE_DIR/run.log"
    else
        eval "$base_cmd &> $DITTO_ENGINE_DIR/run.log"
    fi
    extract_output "$DITTO_ENGINE_DIR/run.log" "$DITTO_ENGINE_DIR/output.log"

    for task_uid in $(seq 0 $(($num_pefts - 1))); do
        echo "Running TensorRT-LLM engine built by ditto with task_uid $task_uid ..."
        local cmd="$base_cmd \
            --lora_dir $(build_lora_args $DITTO_ENGINE_DIR $num_pefts) \
            --lora_task_uids $task_uid"

        if [ "$VERBOSE_MODE" = true ]; then
            eval "$cmd" 2>&1 | tee "$DITTO_ENGINE_DIR/run_task_uid_${task_uid}.log"
        else
            eval "$cmd &> $DITTO_ENGINE_DIR/run_task_uid_${task_uid}.log"
        fi
        extract_output "$DITTO_ENGINE_DIR/run_task_uid_${task_uid}.log" "$DITTO_ENGINE_DIR/output_task_uid_${task_uid}.log"
    done
}

# Build native TensorRT-LLM engine
native_build() {
    local model_id=$1
    local num_pefts=$2
    local peft_ids=("${@:3}")
    
    BASE_MODEL_DIR=$(get_hf_cache_dir ${model_id})
    TRTLLM_CKPT_DIR="${BASE_MODEL_DIR}/trtllm-ckpts"
    if [ "$DTYPE" != "auto" ]; then
        TRTLLM_CKPT_DIR="${TRTLLM_CKPT_DIR}/${DTYPE}"
    fi
    mkdir -p $TRTLLM_CKPT_DIR
    if [ ! -f "$TRTLLM_CKPT_DIR/config.json" ]; then
        echo "Converting checkpoint from $BASE_MODEL_DIR to $TRTLLM_CKPT_DIR ..."
        if [ -z "$MODEL_TYPE" ]; then
            MODEL_TYPE=$(echo "$model_id" | rev | cut -d'/' -f1 | rev | tr '[:upper:]' '[:lower:]' | tr '-' '_' | grep -o 'arctic\|baichuan\|bert\|blip2\|bloom\|chatglm\|cogvlm\|commandr\|dbrx\|deepseek_v1\|deepseek_v2\|dit\|eagle\|enc_dec\|exaone\|falcon\|gemma\|gpt\|gptj\|gptneox\|grok\|internlm\|internlm2\|jais\|llama\|lookahead\|mamba\|medusa\|mixtral\|mllama\|mpt\|nemotron\|nemotron_nas\|openai_triton\|opt\|phi\|prompt_lookup\|qwen\|qwenvl\|recurrentgemma\|sdxl\|skywork\|smaug\|whisper')
        fi
        local convert_script="${TRTLLM_REPO}/examples/${MODEL_TYPE}/convert_checkpoint.py"
        if [ ! -f "$convert_script" ]; then
            echo "Error: Failed to configure model type of $model_id. Please specify the --model-type flag manually."
            return 1
        fi
        echo "Using conversion script at $convert_script"
        if [ "$MODEL_TYPE" = "gemma" ]; then
            local convert_cmd="python $convert_script \
                --ckpt-type hf \
                --model-dir $BASE_MODEL_DIR \
                --output-model-dir $TRTLLM_CKPT_DIR \
                --dtype $DTYPE"
        else
            local convert_cmd="python $convert_script \
                --model_dir $BASE_MODEL_DIR \
                --output_dir $TRTLLM_CKPT_DIR \
                --dtype $DTYPE"
        fi

        if [ "$VERBOSE_MODE" = true ]; then
            eval "$convert_cmd 2>&1 | tee $TRTLLM_CKPT_DIR/convert.log"
        else
            eval "$convert_cmd &> $TRTLLM_CKPT_DIR/convert.log"
        fi
    else
        echo "Skipping conversion as TensorRT-LLM checkpoint directory already exists at $TRTLLM_CKPT_DIR"
    fi

    echo "Building native TensorRT-LLM engine ..."
    TRTLLM_BUILD_ARGS="--checkpoint_dir $TRTLLM_CKPT_DIR \
        --output_dir $TRTLLM_ENGINE_DIR \
        --gemm_plugin auto"

    if [ "$DEBUG_MODE" = true ]; then
        TRTLLM_BUILD_ARGS="$TRTLLM_BUILD_ARGS \
        --visualize_network \
        --profiling_verbosity detailed"
    fi

    if [ "$num_pefts" -gt 0 ]; then
        TRTLLM_BUILD_ARGS="$TRTLLM_BUILD_ARGS \
        --lora_plugin auto \
        --lora_dir $(build_trt_lora_args "${peft_ids[@]}")"
    fi

    if [ "$DEBUG_MODE" = true ]; then
        # Use the patched trtllm-build script in debug mode
        TRTLLM_BUILD_SCRIPT="${SCRIPT_DIR}/trtllm-build"
    else
        TRTLLM_BUILD_SCRIPT="trtllm-build"
    fi

    local build_cmd="DEBUG_ARTIFACTS_DIR=$TRTLLM_ARTIFACTS_DIR \
        $TRTLLM_BUILD_SCRIPT $TRTLLM_BUILD_ARGS"

    if [ "$VERBOSE_MODE" = true ]; then
        eval "$build_cmd 2>&1 | tee $TRTLLM_ENGINE_DIR/build.log"
    else
        eval "$build_cmd &> $TRTLLM_ENGINE_DIR/build.log"
    fi
}

# Run inference with native TensorRT-LLM engine
native_run() {
    local model_id=$1
    local num_pefts=$2
    
    echo "Running native TensorRT-LLM engine ..."
    local base_cmd="python -u ${TRTLLM_REPO}/examples/run.py \
        --engine_dir $TRTLLM_ENGINE_DIR \
        --tokenizer_dir ${model_id} \
        --max_output_len 100 \
        --input_text \"$PROMPT\""

    if [ "$VERBOSE_MODE" = true ]; then
        eval "$base_cmd 2>&1 | tee $TRTLLM_ENGINE_DIR/run.log"
    else
        eval "$base_cmd &> $TRTLLM_ENGINE_DIR/run.log"
    fi
    extract_output "$TRTLLM_ENGINE_DIR/run.log" "$TRTLLM_ENGINE_DIR/output.log"

    for task_uid in $(seq 0 $(($num_pefts - 1))); do
        echo "Running native TensorRT-LLM engine with task_uid $task_uid ..."
        local cmd="$base_cmd \
            --lora_dir $(build_lora_args $TRTLLM_ENGINE_DIR $num_pefts) \
            --lora_task_uids $task_uid"

        if [ "$VERBOSE_MODE" = true ]; then
            eval "$cmd 2>&1 | tee $TRTLLM_ENGINE_DIR/run_task_uid_${task_uid}.log"
        else
            eval "$cmd &> $TRTLLM_ENGINE_DIR/run_task_uid_${task_uid}.log"
        fi
        extract_output "$TRTLLM_ENGINE_DIR/run_task_uid_${task_uid}.log" "$TRTLLM_ENGINE_DIR/output_task_uid_${task_uid}.log"
    done
}

# Compare outputs between two files and print result
compare_output_files() {
    local file1=$1
    local file2=$2

    if [ ! -f "$file1" ] || [ ! -f "$file2" ] || [ ! -s "$file1" ] || [ ! -s "$file2" ]; then
        if [ ! -f "$file1" ]; then
            echo "[ERROR] Output file does not exist: $file1"
        elif [ ! -s "$file1" ]; then
            echo "[ERROR] Output file is empty: $file1"
        fi
        if [ ! -f "$file2" ]; then
            echo "[ERROR] Output file does not exist: $file2"
        elif [ ! -s "$file2" ]; then
            echo "[ERROR] Output file is empty: $file2"
        fi
        return 1
    fi

    # Run diff and capture both output and exit code
    local colorized_diff_output=$(diff --color --new-line-format='+%L' --old-line-format='-%L' --unchanged-line-format=' %L' "$file1" "$file2" | sed 's/^-/\x1b[31m-/;s/^+/\x1b[32m+/;s/^ /\x1b[34m /;s/$/\x1b[0m/')

    if [ -n "$(diff $file1 $file2)" ]; then
        echo -e "\033[31mFAILED\033[0m"
        if [ "$VERBOSE_MODE" = true ]; then
            echo "$colorized_diff_output"
        fi
        return 1
    fi
    echo -e "\033[32mSUCCESS\033[0m"
    if [ "$VERBOSE_MODE" = true ]; then
        echo "$colorized_diff_output"
    fi
    return 0
}

# Compare outputs between Ditto and native TensorRT-LLM engines
compare_outputs() {
    local num_pefts=$1

    if [ ! -d "$DITTO_ENGINE_DIR" ] || [ ! -d "$TRTLLM_ENGINE_DIR" ]; then
        echo "[ERROR] Either of the engine directories does not exist:"
        echo "  - ditto: $DITTO_ENGINE_DIR"
        echo "  - native: $TRTLLM_ENGINE_DIR"
        return 1
    fi

    echo "Comparing outputs between Ditto and native TensorRT-LLM engines..."
    
    # Compare base model outputs
    echo "Comparing base model outputs: "
    compare_output_files \
        "$TRTLLM_ENGINE_DIR/output.log" \
        "$DITTO_ENGINE_DIR/output.log"

    # Compare PEFT model outputs
    for task_uid in $(seq 0 $(($num_pefts - 1))); do
        echo "Comparing PEFT model outputs for task_uid ${task_uid}: "
        compare_output_files \
            "$DITTO_ENGINE_DIR/output_task_uid_${task_uid}.log" \
            "$TRTLLM_ENGINE_DIR/output_task_uid_${task_uid}.log"
    done

    return 0
}

main() {
    parse_args "$@"
    
    NUM_PEFTS=${#PEFT_IDS[@]}

    # Set global directory variables
    set_directories "$MODEL_ID"

    if [ "$SKIP_DITTO_BUILD" = false ]; then
        ditto_build "$MODEL_ID" "${PEFT_IDS[@]}"
    fi
    if [ "$SKIP_DITTO_RUN" = false ]; then
        ditto_run "$MODEL_ID" "$NUM_PEFTS"
    fi

    if [ "$SKIP_NATIVE_BUILD" = false ]; then
        native_build "$MODEL_ID" "$NUM_PEFTS" "${PEFT_IDS[@]}"
    fi
    if [ "$SKIP_NATIVE_RUN" = false ]; then
        native_run "$MODEL_ID" "$NUM_PEFTS"
    fi

    compare_outputs "$NUM_PEFTS"
    return $?
}

main "$@"
