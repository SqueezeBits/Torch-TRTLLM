ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.6.3-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}:${BASE_TAG} AS base

# https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html
# The default values come from `nvcr.io/nvidia/pytorch`
ENV BASH_ENV=${BASH_ENV:-/etc/bash.bashrc}
ENV ENV=${ENV:-/etc/shinit_v2}
SHELL ["/bin/bash", "-c"]

COPY docker/scripts/setup_env.sh setup_env.sh
RUN bash ./setup_env.sh && rm setup_env.sh

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install --no-install-recommends -y \
    build-essential manpages-dev wget zlib1g software-properties-common \
    git libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 \
    locales ninja-build openmpi-bin libopenmpi-dev git-lfs pigz \
    ccache python3-dev python-is-python3 python3-pip nano tree less && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

RUN echo 'export PYTHON_MAJOR_VERSION=$(python3 -c "import sys; print(sys.version_info.major)")' >> "${ENV}"
RUN echo 'export PYTHON_MINOR_VERSION=$(python3 -c "import sys; print(sys.version_info.minor)")' >> "${ENV}"
RUN rm /usr/lib/python${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION}/EXTERNALLY-MANAGED

RUN wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 -O /usr/bin/bazel && \
    chmod a+x /usr/bin/bazel

ARG PYTORCH_VERSION="2.5.1"
RUN mkdir -p /root/.cache/pip /root/.cache/ccache /root/.cache/bazel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==${PYTORCH_VERSION} torchvision 'numpy<2' \
    "accelerate>=0.25.0" "diffusers>=0.27.0" h5py==3.12.1 "nvidia-modelopt[torch]~=0.21.0" "onnx>=1.12.0" \
    "onnx_graphsurgeon>=0.5.2" "openai==1.54.3" "pynvml>=11.5.0" "sentencepiece>=0.1.99" \
    "transformers<=4.45.1,>=4.38.2" peft>=0.14.0 pillow==10.3.0 pybind11 \
    aenum build cuda-python click click_option_group colored evaluate lark optimum pandas packaging psutil pulp \
    pyyaml StrEnum pyzmq fastapi==0.115.4 uvicorn httpx setuptools ordered-set

ARG TRT_VER=10.7.0.23
ENV TRT_VER=${TRT_VER}
COPY docker/scripts/install_tensorrt.sh install_tensorrt.sh
RUN --mount=type=cache,target=/root/.cache bash ./install_tensorrt.sh && rm install_tensorrt.sh

ARG TRTLLM_BRANCH=v0.16.0
COPY docker/scripts/tensorrt_llm.patch /workspace/tensorrt_llm.patch
ENV CCACHE_DIR=/root/.cache/ccache
RUN git clone \
    --depth 1 \
    --single-branch \
    -b ${TRTLLM_BRANCH} \
    --shallow-submodules \
    --recursive https://github.com/NVIDIA/TensorRT-LLM \
    /workspace/tensorrt_llm
RUN bash /workspace/tensorrt_llm/docker/common/install_cmake.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    bash /workspace/tensorrt_llm/docker/common/install_polygraphy.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    bash /workspace/tensorrt_llm/docker/common/install_mpi4py.sh
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/nvcc_cache \
    cd /workspace/tensorrt_llm && \
    git lfs install && \
    git lfs pull && \
    git apply /workspace/tensorrt_llm.patch && \
    find $(python -c "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='')") \
        -type f -name "*.cmake" -exec sed -i 's/python_add_library/python3_add_library/g' {} + && \
    CUDA_CACHE_PATH=/root/.cache/nvcc_cache \
    python scripts/build_wheel.py \
        --clean \
        --trt_root /usr/local/tensorrt \
        --python_bindings \
        --benchmarks \
        --use_ccache \
        --skip_install_requirements \
        --install && \
    cd /workspace && \
    mv /workspace/tensorrt_llm/examples /workspace/tensorrt_llm_examples && \
    rm -rf /workspace/tensorrt_llm && \
    mkdir /workspace/tensorrt_llm && \
    mv /workspace/tensorrt_llm_examples /workspace/tensorrt_llm/examples

ARG TORCH_TRT_BRANCH=v2.5.0
COPY docker/scripts/torch_tensorrt.patch /workspace/torch_tensorrt.patch
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/bazel \
    git clone --depth 1 --single-branch -b ${TORCH_TRT_BRANCH} https://github.com/pytorch/TensorRT /workspace/torch_tensorrt && \
    cd /workspace/torch_tensorrt && \
    git apply /workspace/torch_tensorrt.patch && \
    MAX_JOBS=4 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 PYTHON_ONLY=1 \
    pip wheel . --no-deps --no-build-isolation -w dist && \
    pip install ./dist/*.whl --no-deps && \
    cd /workspace && \
    rm -rf /workspace/torch_tensorrt*

WORKDIR /workspace
