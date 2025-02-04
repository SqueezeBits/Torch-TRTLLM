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
declare TRUST_REMOTE_CODE=false
declare REBUILD_DITTO=false
declare RERUN_DITTO=false
declare REBUILD_NATIVE=false
declare RERUN_NATIVE=false
declare TP_SIZE=1

# Global variables for engine and artifact directories
declare DITTO_ENGINE_DIR=""
declare DITTO_ARTIFACTS_DIR=""
declare TRTLLM_ENGINE_DIR=""
declare TRTLLM_ARTIFACTS_DIR=""

# Execute command and handle errors based on verbose mode
rich_execute() {
    local cmd="$1"
    local log_file="$2"
    local description="$3"

    echo -e "\033[1;32mExecuting the following command to $description redirecting output to $log_file\033[0m"
    echo -e "\033[1;38;5;213m$cmd\033[0m" | sed 's/[[:space:]]\{2,\}/\n    /g'
    if [ "$VERBOSE_MODE" = true ]; then
        set -o pipefail
        eval "$cmd" 2>&1 | tee "$log_file"
        local ret=$?
        set +o pipefail
        if [ $ret -ne 0 ]; then
            echo -e "\033[1;31mError: Failed to $description\033[0m"
            exit 1
        fi
    else
        eval "$cmd" &> "$log_file"
        if [ $? -ne 0 ]; then
            echo -e "\033[1;31mError: Failed to $description. Check $log_file for details.\033[0m"
            exit 1
        fi
    fi
}

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
    echo "  -d, --debug            Enable debug mode"
    echo "  --prompt TEXT          Input prompt (default: \"Hey, are you conscious?\")"
    echo "  --trtllm-repo PATH     Path to TensorRT-LLM repository"
    echo "                         (default: clone at user cache directory with version from pyproject.toml)"
    echo "  --dtype DTYPE          Data type for model conversion (default: auto)"
    echo "  --model-type TYPE      Model type for finding convert_checkpoint.py (default: auto-detected)"
    echo "  --tp-size SIZE         Tensor parallel size (default: 1)"
    echo "  --trust-remote-code    Trust remote code when loading models from Hugging Face"
    echo
    echo "  --skip                 Equivalent to --skip-build --skip-run"
    echo "  --skip-build           Equivalent to --skip-ditto-build --skip-native-build"
    echo "  --skip-run             Equivalent to --skip-ditto-run --skip-native-run"
    echo "  --skip-ditto           Equivalent to --skip-ditto-build --skip-ditto-run"
    echo "  --skip-ditto-build     Skip building Ditto engine"
    echo "  --skip-ditto-run       Skip running Ditto engine"
    echo "  --skip-native          Equivalent to --skip-native-build --skip-native-run"
    echo "  --skip-native-build    Skip building native TensorRT-LLM engine"
    echo "  --skip-native-run      Skip running native TensorRT-LLM engine"
    echo
    echo "  --redo                 Equivalent to --rebuild --rerun"
    echo "  --rebuild              Equivalent to --rebuild-ditto --rebuild-native"
    echo "  --rerun                Equivalent to --rerun-ditto --rerun-native"
    echo "  --redo-ditto           Equivalent to --rebuild-ditto --rerun-ditto"
    echo "  --rebuild-ditto        Rebuild Ditto engine even if it already exists"
    echo "  --rerun-ditto          Run Ditto engine even if output files exist"
    echo "  --redo-native          Equivalent to --rebuild-native --rerun-native"
    echo "  --rebuild-native       Rebuild native TensorRT-LLM engine even if it already exists"
    echo "  --rerun-native         Run native TensorRT-LLM engine even if output files exist"
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
            --skip)
                SKIP_DITTO_BUILD=true
                SKIP_NATIVE_BUILD=true
                SKIP_DITTO_RUN=true
                SKIP_NATIVE_RUN=true
                shift
                ;;
            --skip-build)
                SKIP_DITTO_BUILD=true
                SKIP_NATIVE_BUILD=true
                shift
                ;;
            --skip-run)
                SKIP_DITTO_RUN=true
                SKIP_NATIVE_RUN=true
                shift
                ;;
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
            --trust-remote-code)
                TRUST_REMOTE_CODE=true
                shift
                ;;
            --redo)
                REBUILD_DITTO=true
                REBUILD_NATIVE=true
                RERUN_DITTO=true
                RERUN_NATIVE=true
                shift
                ;;
            --rebuild)
                REBUILD_DITTO=true
                REBUILD_NATIVE=true
                shift
                ;;
            --rerun)
                RERUN_DITTO=true
                RERUN_NATIVE=true
                shift
                ;;
            --redo-ditto)
                REBUILD_DITTO=true
                RERUN_DITTO=true
                shift
                ;;
            --rebuild-ditto)
                REBUILD_DITTO=true
                shift
                ;;
            --rerun-ditto)
                RERUN_DITTO=true
                shift
                ;;
            --redo-native)
                REBUILD_NATIVE=true
                RERUN_NATIVE=true
                shift
                ;;
            --rebuild-native)
                REBUILD_NATIVE=true
                shift
                ;;
            --rerun-native)
                RERUN_NATIVE=true
                shift
                ;;
            --tp-size)
                TP_SIZE="$2"
                if ! [[ "$TP_SIZE" =~ ^0*[1-9][0-9]*$ ]]; then
                    echo "Error: TP_SIZE must be a positive integer."
                    exit 1
                fi
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
get_argument_suffix() {
    local dir=$MODEL_ID
    local suffix=""
    # Build suffix from PEFT_IDS concatenating author names
    for peft_id in "${PEFT_IDS[@]}"; do
        # Split by '/' and get the penultimate token
        local author=$(echo $peft_id | tr '/' '\n' | tail -n 2 | head -n 1)
        if [ -z "$suffix" ]; then
            suffix=$author
        else
            suffix="${suffix}+${author}"
        fi
    done
    if [ ! -z "$suffix" ]; then
        echo "${dir}+${suffix}"
    else
        echo "$dir"
    fi
}

# Append dtype and tp_size suffixes to directory
append_option_suffix() {
    local dir=$1

    if [ "$DTYPE" != "auto" ]; then
        dir="${dir}_${DTYPE}"
    fi

    if [ "$TP_SIZE" -gt 1 ]; then
        dir="${dir}_tp${TP_SIZE}"
    fi

    echo "$dir"
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
    local model_id=$MODEL_ID

    DITTO_ENGINE_DIR="engines/ditto/$(get_argument_suffix)"
    TRTLLM_ENGINE_DIR="engines/trtllm/$(get_argument_suffix)"

    DITTO_ENGINE_DIR=$(append_option_suffix "$DITTO_ENGINE_DIR")
    TRTLLM_ENGINE_DIR=$(append_option_suffix "$TRTLLM_ENGINE_DIR")

    if [ "$DEBUG_MODE" = true ]; then
        DITTO_ARTIFACTS_DIR="artifacts/ditto/$(get_argument_suffix)"
        TRTLLM_ARTIFACTS_DIR="artifacts/trtllm/$(get_argument_suffix)"

        DITTO_ARTIFACTS_DIR=$(append_option_suffix "$DITTO_ARTIFACTS_DIR")
        TRTLLM_ARTIFACTS_DIR=$(append_option_suffix "$TRTLLM_ARTIFACTS_DIR")
    fi

    mkdir -p $DITTO_ENGINE_DIR $TRTLLM_ENGINE_DIR
}

# Build Ditto engine
ditto_build() {
    local model_id=$1
    local peft_ids=("${@:2}")

    # Early return if engine already exists and --rebuild-ditto not specified
    if [ "$REBUILD_DITTO" = false ] && [ -f "$DITTO_ENGINE_DIR/config.json" ] && [ -f "$DITTO_ENGINE_DIR/rank0.engine" ]; then
        echo "Skip building Ditto engine as it already exists at $DITTO_ENGINE_DIR"
        return 0
    fi

    local cmd="ditto build ${model_id} \
        --output-dir $DITTO_ENGINE_DIR \
        --dtype $DTYPE \
        --tp-size $TP_SIZE \
        $(build_peft_args "${peft_ids[@]}")"

    if [ "$TRUST_REMOTE_CODE" = true ]; then
        cmd="$cmd --trust-remote-code"
    fi

    DEBUG_ARTIFACTS_DIR=$DITTO_ARTIFACTS_DIR rich_execute "$cmd" "$DITTO_ENGINE_DIR/build.log" "build Ditto engine"
}

# Run inference with engine
run_engine() {
    local model_id=$1
    local num_pefts=$2
    local engine_type=$3
    local engine_dir=$4
    local rerun=$5

    local base_cmd="python -u ${TRTLLM_REPO}/examples/run.py \
        --engine_dir $engine_dir \
        --tokenizer_dir ${model_id} \
        --input_text \"$PROMPT\" \
        --max_output_len 100"
    
    if [ "$TP_SIZE" -gt 1 ]; then
        base_cmd="mpirun -n $TP_SIZE $base_cmd"
    fi

    # Only run if output file doesn't exist or is empty or rerun flag is true
    OUTPUT_FILE="$engine_dir/output.log"
    RUN_FILE="$engine_dir/run.log"
    if [ "$rerun" = true ] || [ ! -f "$OUTPUT_FILE" ] || [ ! -s "$OUTPUT_FILE" ]; then
        rich_execute "$base_cmd" "$RUN_FILE" "run $engine_type engine"
        extract_output "$RUN_FILE" "$OUTPUT_FILE"
    else
        echo "Skip running $engine_type engine as output file already exists at $OUTPUT_FILE"
    fi

    for task_uid in $(seq 0 $(($num_pefts - 1))); do
        local cmd="$base_cmd \
            --lora_dir $(build_lora_args $engine_dir $num_pefts) \
            --lora_task_uids $task_uid"

        # Only run if output file doesn't exist or is empty or rerun flag is true
        RUN_FILE="$engine_dir/run_task_uid_${task_uid}.log"
        OUTPUT_FILE="$engine_dir/output_task_uid_${task_uid}.log"
        if [ "$rerun" = true ] || [ ! -f "$OUTPUT_FILE" ] || [ ! -s "$OUTPUT_FILE" ]; then
            rich_execute "$cmd" "$RUN_FILE" "run $engine_type engine with task_uid $task_uid"
            extract_output "$RUN_FILE" "$OUTPUT_FILE"
        else
            echo "Skip running $engine_type engine with task_uid $task_uid as output file already exists at $OUTPUT_FILE"
        fi
    done
}

# Build native TensorRT-LLM engine
native_build() {
    local model_id=$1
    local num_pefts=$2
    local peft_ids=("${@:3}")

    # Early return if engine already exists and --rebuild-native not specified
    if [ "$REBUILD_NATIVE" = false ] && [ -f "$TRTLLM_ENGINE_DIR/config.json" ] && [ -f "$TRTLLM_ENGINE_DIR/rank0.engine" ]; then
        echo "Skip building native TensorRT-LLM engine as it already exists at $TRTLLM_ENGINE_DIR"
        return 0
    fi

    local BASE_MODEL_DIR=$(get_hf_cache_dir ${model_id})
    local TRTLLM_CKPT_DIR=$(append_option_suffix "${BASE_MODEL_DIR}/trtllm-ckpts")
    mkdir -p $TRTLLM_CKPT_DIR
    if [ ! -f "$TRTLLM_CKPT_DIR/config.json" ]; then

        # EXAONE model directory needs to contain lower-cased "exaone" in its name
        if [[ "$BASE_MODEL_DIR" == *EXAONE* ]]; then
            BASE_MODEL_ROOT_DIR=`dirname $(dirname $BASE_MODEL_DIR)`
            EXAONE_DIR="$(dirname $BASE_MODEL_ROOT_DIR)/exaone"
            SYMLINK_PATH="./$(basename $BASE_MODEL_ROOT_DIR)"
            if [ ! -e "$EXAONE_DIR" ]; then
                ln -s "./$(basename $BASE_MODEL_ROOT_DIR)" $EXAONE_DIR
            fi
            BASE_MODEL_DIR="${BASE_MODEL_DIR/$BASE_MODEL_ROOT_DIR/$EXAONE_DIR}"
        fi

        echo "Converting checkpoint at $BASE_MODEL_DIR ..."
        if [ -z "$MODEL_TYPE" ]; then
            # Find all convert_checkpoint.py files and extract their parent directory names
            # Get list of available model types from TensorRT-LLM examples directory
            AVAILABLE_MODEL_TYPES=$(find "${TRTLLM_REPO}/examples" -type f -name convert_checkpoint.py | xargs -n1 dirname | xargs -n1 basename)

            # Convert model_id to lowercase for case-insensitive matching
            model_id_lower=$(echo "$model_id" | tr '[:upper:]' '[:lower:]')

            # Try to match each available model type
            for type in $AVAILABLE_MODEL_TYPES; do
                if [[ $model_id_lower == *"$type"* ]]; then
                    MODEL_TYPE=$type
                    break
                fi
            done
        fi

        local convert_script="${TRTLLM_REPO}/examples/${MODEL_TYPE}/convert_checkpoint.py"
        if [ ! -f "$convert_script" ]; then
            echo "Error: Failed to configure model type of $model_id. Please specify the --model-type flag manually."
            exit 1
        fi

        echo "Using conversion script at $convert_script"
        if [ "$MODEL_TYPE" = "gemma" ]; then
            local convert_cmd="python $convert_script \
                --ckpt-type hf \
                --model-dir $BASE_MODEL_DIR \
                --output-model-dir $TRTLLM_CKPT_DIR \
                --dtype $DTYPE \
                --world-size $TP_SIZE"
        else
            local convert_cmd="python $convert_script \
                --model_dir $BASE_MODEL_DIR \
                --output_dir $TRTLLM_CKPT_DIR \
                --dtype $DTYPE \
                --tp_size $TP_SIZE"
        fi

        rich_execute "$convert_cmd" "$TRTLLM_CKPT_DIR/convert.log" "convert checkpoint"
    else
        echo "Skipping checkpoint conversion as TensorRT-LLM it already exists at $TRTLLM_CKPT_DIR"
    fi

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

    local build_cmd="$TRTLLM_BUILD_SCRIPT $TRTLLM_BUILD_ARGS"

    DEBUG_ARTIFACTS_DIR=$TRTLLM_ARTIFACTS_DIR rich_execute "$build_cmd" "$TRTLLM_ENGINE_DIR/build.log" "build native TensorRT-LLM engine"
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
        echo -e "\033[31;1mFAILED\033[0m"
        if [ "$VERBOSE_MODE" = true ]; then
            echo "$colorized_diff_output"
        fi
        return 1
    fi
    echo -e "\033[32;1mSUCCESS\033[0m"
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
    local all_succeeded=0

    # Compare base model outputs
    compare_output_files \
        "$TRTLLM_ENGINE_DIR/output.log" \
        "$DITTO_ENGINE_DIR/output.log"
    if [ $? -ne 0 ]; then
        all_succeeded=1
    fi

    # Compare PEFT model outputs
    for task_uid in $(seq 0 $(($num_pefts - 1))); do
        echo "Comparing PEFT model outputs for task_uid ${task_uid}: "
        compare_output_files \
            "$DITTO_ENGINE_DIR/output_task_uid_${task_uid}.log" \
            "$TRTLLM_ENGINE_DIR/output_task_uid_${task_uid}.log"
        if [ $? -ne 0 ]; then
            all_succeeded=1
        fi
    done

    return $all_succeeded
}

main() {
    parse_args "$@"
    
    NUM_PEFTS=${#PEFT_IDS[@]}

    # Set global directory variables
    set_directories

    if [ "$SKIP_DITTO_BUILD" = false ]; then
        ditto_build "$MODEL_ID" "${PEFT_IDS[@]}"
    fi
    if [ "$SKIP_DITTO_RUN" = false ]; then
        run_engine "$MODEL_ID" "$NUM_PEFTS" "Ditto" "$DITTO_ENGINE_DIR" "$RERUN_DITTO"
    fi

    if [ "$SKIP_NATIVE_BUILD" = false ]; then
        native_build "$MODEL_ID" "$NUM_PEFTS" "${PEFT_IDS[@]}"
    fi
    if [ "$SKIP_NATIVE_RUN" = false ]; then
        run_engine "$MODEL_ID" "$NUM_PEFTS" "native TensorRT-LLM" "$TRTLLM_ENGINE_DIR" "$RERUN_NATIVE"
    fi

    compare_outputs "$NUM_PEFTS"
    return $?
}

main "$@"
