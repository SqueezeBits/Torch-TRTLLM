declare CONDA="conda"

get_script_directory() {
    local script_path
    local script_dir

    if [[ -n "$BASH_SOURCE" ]]; then
        script_path="$BASH_SOURCE"
    elif [[ -n "$ZSH_VERSION" ]]; then
        script_path="$0"
    else
        printf "Error: Unsupported shell. This script works only with bash or zsh.\n" >&2
        return 1
    fi

    while [ -L "$script_path" ]; do
        script_path=$(readlink "$script_path")
        [[ $script_path != /* ]] && script_path=$(dirname "$0")/"$script_path"
    done

    script_dir=$(cd "$(dirname "$script_path")" && pwd)
    printf $script_dir
}

install_ditto() {
    local install_path="$1"
    local editable="$2"
    if [ "$editable" = true ]; then
        echo "Installing Ditto from ${install_path} in editable mode ..."
        $CONDA run -n ${name} pip install -e "$install_path"
    else
        echo "Installing Ditto from ${install_path} ..."
        $CONDA run -n ${name} pip install "$install_path"
    fi
}

print_help() {
    echo "Usage: $0 [options]"
    echo "Create a conda environment with all dependencies installed. This script installs ditto from remote repository when either -r/--remote is specified or not attached to a terminal. Otherwise, it installs ditto from source."
    echo "Options:"
    echo "  -n, --name      Set the environment name (default: ditto)"
    echo "  -v, --version   Specify the version of Ditto to install (default: nightly, ignored when installing from source)"
    echo "  -c, --conda     Specify the conda executable to use (default: conda)"
    echo "  -r, --remote    Install Ditto from the remote repository (which is the default behavior when not attached to a terminal)"
    echo "  -e, --editable  Install Ditto in editable mode (ignored when installing from remote)"
    echo "  -h, --help      Display this help message and exit"
}

parse_args() {
    local -n name_ref=$1
    local -n version_ref=$2
    local -n remote_ref=$3
    local -n editable_ref=$4

    while [[ $# -gt 4 ]]; do
        case $5 in
            -n|--name)
                name_ref="$6"
                shift
                shift
                ;;
            -v|--version)
                version_ref="$6"
                shift
                shift
                ;;
            -c|--conda)
                CONDA="$6"
                shift
                shift
                ;;
            -e|--editable)
                editable_ref=true
                shift
                ;;
            -r|--remote)
                remote_ref=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo "Unknown option: $5"
                exit 1
                ;;
        esac
    done
}

resolve_args() {
    local -n remote_ref=$1
    local -n editable_ref=$2
    local -n ditto_path_ref=$3
    local version=$4
    local ditto_url_for_pip=$5

    script_dir=$(get_script_directory)
    if [ ! -f "${script_dir}/../pyproject.toml" ] && [ $remote_ref = false ]; then
        echo "[WARNING] No local ditto repository found. Installing Ditto from remote repository"
        remote_ref=true
    fi

    if $remote_ref; then
        if [ $editable_ref = true ]; then
            echo "[WARNING] -e/--editable will be ignored when installing from remote"
            editable_ref=false
        fi
        if [ -n "$version" ]; then
            ditto_path_ref="${ditto_url_for_pip}@v${version}"
        else
            ditto_path_ref="${ditto_url_for_pip}"
        fi
    else
        ditto_path_ref="$(realpath "$(get_script_directory)/..")"
        if [ -n "$version" ]; then
            echo "[WARNING] -v/--version ${version} will be ignored when installing from source"
        fi
    fi
}

main() {
    local name="ditto"
    local version=""
    local ditto_path=""
    local remote=false
    local editable=false
    local ditto_url_for_pip="git+https://github.com/SqueezeBits/Torch-TRTLLM.git"

    parse_args name version remote editable "$@"

    echo "Using `realpath ${CONDA}` to create environment named ${name}"
    $CONDA env create \
        -n ${name} \
        -c pytorch \
        -c nvidia \
        -c anaconda \
        -c conda-forge \
        python=3.10 \
        "pytorch>=2.5,<2.6" \
        "pytorch-cuda=12.4" \
        "mpi4py>=3,<4" \
        "openmpi>=4,<5" \
        pip
    
    if ! $CONDA env list | grep -q "${name}"; then
        echo "[ERROR] Failed to create environment named ${name}"
        exit 1
    fi

    resolve_args remote editable ditto_path "$version" "$ditto_url_for_pip"

    install_ditto "${ditto_path}" $editable
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Ditto"
        exit 1
    fi

    echo "Installing Torch-TensorRT ..."
    $CONDA run -n ${name} pip install torch-tensorrt==2.5.0 --no-deps
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install Torch-TensorRT"
        exit 1
    fi
}

main "$@"
