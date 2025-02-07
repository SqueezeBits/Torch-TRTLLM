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

main() {
    local name=${1:-ditto}
    local script_dir
    shift
    script_dir=`get_script_directory`
    CONDA=${CONDA:-conda}
    echo "Using `realpath ${CONDA}` to create environment named ${name}"
    $CONDA env create -f ${script_dir}/environment.yml -n ${name} $@ || (echo "Failed to create environment" && exit 1)
    echo "Installing ditto ..."
    $CONDA run -n ${name} pip install ${script_dir}/.. || (echo "Failed to install ditto" && exit 1)
    echo "Installing Torch-TensorRT ..."
    $CONDA run -n ${name} pip install torch-tensorrt==2.5.0 --no-deps || (echo "Failed to install Torch-TensorRT" && exit 1)
}

main $@
