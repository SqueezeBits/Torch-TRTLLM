ARG BASE_IMAGE=nvcr.io/nvidia/cuda
ARG BASE_TAG=12.6.1-cudnn-devel-ubuntu24.04
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

ARG PYTORCH_VERSION=2.6.0.dev20241003
RUN mkdir -p /root/.cache/pip /root/.cache/ccache /root/.cache/bazel
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre torch==${PYTORCH_VERSION} torchvision 'numpy<2' \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu124
RUN pip install \
    "accelerate>=0.25.0" "diffusers>=0.27.0" h5py==3.10.0 "nvidia-modelopt~=0.15.0" "onnx>=1.12.0" "openai==1.39.0" \
    "pynvml>=11.5.0" "sentencepiece>=0.1.99" "transformers<=4.42.4,>=4.38.2" pillow==10.3.0 pybind11==2.6.2 \
    aenum build cuda-python click click_option_group colored evaluate lark optimum pandas packaging psutil pulp \
    pyyaml StrEnum

ARG TRT_VER=10.4.0.26
ENV TRT_VER=${TRT_VER}
COPY docker/scripts/install_tensorrt.sh install_tensorrt.sh
RUN --mount=type=cache,target=/root/.cache bash ./install_tensorrt.sh && rm install_tensorrt.sh

ARG TRTLLM_BRANCH=v0.13.0
RUN git clone -b ${TRTLLM_BRANCH} --recursive https://github.com/NVIDIA/TensorRT-LLM /workspace/tensorrt_llm
RUN git clone https://github.com/pytorch/TensorRT /workspace/torch_tensorrt

WORKDIR /workspace/tensorrt_llm
RUN bash ./docker/common/install_cmake.sh
RUN bash ./docker/common/install_polygraphy.sh
RUN bash ./docker/common/install_mpi4py.sh

WORKDIR /workspace/torch_tensorrt
ARG TORCH_TENSORRT_COMMIT=52df589bb3f6e1028abd6f1d1b64606c2e7a0459
COPY docker/scripts/torch_tensorrt.patch /workspace/torch_tensorrt.patch
RUN git checkout $TORCH_TENSORRT_COMMIT
RUN git apply /workspace/torch_tensorrt.patch
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/bazel \
    MAX_JOBS=4 LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 \
    pip wheel . --no-deps --no-build-isolation -w dist && \
    pip install ./dist/*.whl --no-deps && \
    rm -rf ./dist ./build

WORKDIR /workspace/tensorrt_llm
ENV CCACHE_DIR=/root/.cache/ccache
RUN git lfs install && git lfs pull
COPY docker/scripts/tensorrt_llm.patch /workspace/tensorrt_llm.patch
RUN git apply /workspace/tensorrt_llm.patch
RUN find $(python -c "import pybind11 as pb11; print(pb11.get_cmake_dir(),end='')") \
    -type f -name "*.cmake" -exec sed -i 's/python_add_library/python3_add_library/g' {} +
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/nvcc_cache \
    CUDA_CACHE_PATH=/root/.cache/nvcc_cache \
    python scripts/build_wheel.py \
    --clean \
    --trt_root /usr/local/tensorrt \
    --python_bindings \
    --benchmarks \
    --use_ccache \
    --skip_install_requirements \
    --install

WORKDIR /workspace
