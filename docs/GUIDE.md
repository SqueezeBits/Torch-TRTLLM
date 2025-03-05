# Getting Started

## A. Installation

Note that `pip` may complain about dependency mismatch due to `torch-tensorrt`, even if you follow the instructions below without any mistakes.
However, you are ***completely OK*** to ignore these messages as we've tested out `torch-tensorrt==2.5.0` to be compatible with `tensorrt-cu12*==10.7.0`.
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch-tensorrt 2.5.0 requires tensorrt-cu12==10.3.0, but you have tensorrt-cu12 10.7.0 which is incompatible.
torch-tensorrt 2.5.0 requires tensorrt-cu12-bindings==10.3.0, but you have tensorrt-cu12-bindings 10.7.0 which is incompatible.
torch-tensorrt 2.5.0 requires tensorrt-cu12-libs==10.3.0, but you have tensorrt-cu12-libs 10.7.0 which is incompatible.
```

### I. Using conda (recommended)
First, clone the repository.
```
git clone https://github.com/SqueezeBits/Torch-TRTLLM.git
cd Torch-TRTLLM
```
Then, run the following command to create an environment with dependencies installed.
```
conda create -f conda/environment.yml
```
Finally, install Ditto.
```
pip install . --extra-index-url https://pypi.nvidia.com
```

### II. Manual Installation

It is recommended to use conda to create a virtual environment with dependencies as above. However, if you prefer to manually set up your environment, you must manually ensure prerequisites below.

#### Prerequisites
* CUDA (recommended version is 12.4)
* `openmpi` and `mpi4py`: for example, on Ubuntu systems, you can install them by running `sudo apt install openmpi-bin libopenmpi-dev python3-mpi4py`.

#### Installation

With the prerequisites installed, install Ditto with following command. 
```
pip install git+https://github.com/SqueezeBits/Torch-TRTLLM.git@latest --extra-index-url https://pypi.nvidia.com
```

### III. Using Triton Inference Server
**Tested with nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3**

As NGC triton server docker image comes with broken dependencies, make sure to install and use Ditto in virtual envrionment.
However, when using the Ditto built engine built with triton server, make sure to deactivate the virtual environment so that the system can refer to it's preinstalled dependencies.

#### 1. Install python venv
```
apt update && apt install python3-venv -y
```

#### 2. Create and activate the virtual envrionment
```
python3 -m venv ditto
source ditto/bin/activate
```

#### 3. Install Ditto in the virtual environment
```
pip install git+https://github.com/SqueezeBits/Torch-TRTLLM.git@latest --extra-index-url https://pypi.nvidia.com
```

#### 4. Build engine with Ditto
```
ditto build  <model-id-or-hf-model-directory>
```

#### 5. Deactivate the venv when using the engine
```
deactivate
do sth with the engine
```

### IIII. Using Ditto Docker Image
#### 1. Build a docker image
**WARNING: This might take a few hours.**
Run from the root directory of the repository:
```
docker build -f docker/Dockerfile -t ditto:latest .
```

#### 2. Run the container
Run from the root directory of the repository:
```
docker run --rm -it --gpus all -v `pwd`:/workspace/ditto ditto:latest bash
```

#### 3. Install ditto
Now, inside the docker container, run
```
pip install /workspace/ditto
```
or, for editable installation,
```
pip install -e /workspace/ditto
```


## B. Quick Start Guide

#### Building a TRT-LLM engine
You can build a TRT-LLM engine from a pretrained model by running the following command:
```
ditto build <model-id-or-hf-model-directory> --output-dir <engine-output-directory (optional)>
```
For example, the following commands are equivalent:
* `ditto build meta-llama/Llama-2-7b-chat-hf`
* `ditto build meta-llama/Llama-2-7b-chat-hf --output-dir ./engines/meta-llama/Llama-2-7b-chat-hf`

#### Tensor Parallelism
To build a tensor parallelized engine, add the `--tp-size` option with a value greater than 1.
```
ditto build <model-id-or-hf-model-directory> --tp-size <tp-size> --output-dir <engine-output-directory (optional)>
```
For example, the following command builds a 2-way-tensor-parallelized engine:
```
ditto build meta-llama/Llama-2-7b-chat-hf --tp-size 2 --output-dir ./engines/meta-llama/Llama-2-7b-chat-hf-tp2
```

#### LoRA
Add the `-p/--peft-ids` option, possibly multiple times, to apply LoRA adapters to the model.
```
ditto build <model-id-or-hf-model-directory> --peft-ids <peft-id> [--peft-ids <peft-id> ...] --output-dir <engine-output-directory (optional)>
```
For example, the following command builds a LoRA-enabled engine with three LoRA adapters:
```
ditto build TinyLlama/TinyLlama-1.1B-Chat-v1.0 -p ArmaanSeth/Llama-2-7b-chat-hf-adapters-mental-health-counselling -p barissglc/tinyllama-tarot-v1 -p snshrivas10/sft-tiny-chatbot
```

#### Tensor Parallelism + LoRA
We do not support building a tensor-parallelized LoRA-enabled engine yet, but we will support it in the future.

#### Full Usage
```
 Usage: ditto build [OPTIONS] MODEL_ID                                                                                                                                     
                                                                                                                                                                           
 Build a TensorRT-LLM engine from a pretrained model.                                                                                                                      
                                                                                                                                                                           
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_id      TEXT  A pretrained model name or path. [default: None] [required]                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --add-output                                                                 TEXT                             List of node names to add as output. See docs/DEBUG.md    │
│                                                                                                               for details.                                              │
│                                                                                                               [default: <class 'list'>]                                 │
│ --peft-ids                        -p                                         TEXT                             List of LoRA adapter IDs to apply to the model.           │
│                                                                                                               [default: <class 'list'>]                                 │
│ --output-dir                      -o                                         TEXT                             Path to the output directory. If not specified,           │
│                                                                                                               `./engines/<model_id>` will be used.                      │
│ --dtype                                                                      [auto|float32|float16|bfloat16]  Data type to use for the model. Defaults to `auto`.       │
│                                                                                                               [default: auto]                                           │
│ --verbose-failure                     --no-verbose-failure                                                    Show local variable values on failure.                    │
│                                                                                                               [default: no-verbose-failure]                             │
│ --trust-remote-code                   --no-trust-remote-code                                                  Trust remote code from Hugging Face Hub.                  │
│                                                                                                               [default: no-trust-remote-code]                           │
│ --run-matmuls-in-fp32                 --no-run-matmuls-in-fp32                                                Run matmuls in fp32. [default: no-run-matmuls-in-fp32]    │
│ --run-activations-in-model-dtype      --no-run-activations-in-model-dtype                                     Run activations in model dtype.                           │
│                                                                                                               [default: run-activations-in-model-dtype]                 │
│ --max-batch-size                                                             INTEGER                          Maximum number of requests that the engine can schedule.  │
│                                                                                                               [default: 2048]                                           │
│ --max-seq-len                                                                INTEGER                          Maximum total length of one request, including prompt and │
│                                                                                                               generated output.                                         │
│                                                                                                               [default: None]                                           │
│ --max-num-tokens                                                             INTEGER                          Maximum number of batched input tokens after padding is   │
│                                                                                                               removed in each batch.                                    │
│                                                                                                               [default: 8192]                                           │
│ --opt-num-tokens                                                             INTEGER                          Optimal number of batched input tokens after padding is   │
│                                                                                                               removed in each batch.                                    │
│                                                                                                               [default: None]                                           │
│ --max-beam-width                                                             INTEGER                          Maximum number of beams for beam search decoding.         │
│                                                                                                               [default: 1]                                              │
│ --tp-size                                                                    INTEGER RANGE [x>=1]             N-way tensor parallelism size. [default: 1]               │
│ --logits-dtype                                                               [float32|float16|bfloat16]       Data type of logits. Defaults to `float32`.               │
│                                                                                                               [default: float32]                                        │
│ --gather-context-logits               --no-gather-context-logits                                              Enable gathering context logits.                          │
│                                                                                                               [default: no-gather-context-logits]                       │
│ --gather-generation-logits            --no-gather-generation-logits                                           Enable gathering generation logits.                       │
│                                                                                                               [default: no-gather-generation-logits]                    │
│ --gather-all-logits                   --no-gather-all-logits                                                  Equivalent to `--gather-context-logits                    │
│                                                                                                               --gather-generation-logits`.                              │
│                                                                                                               [default: no-gather-all-logits]                           │
│ --help                            -h                                                                          Show this message and exit.                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


### 3. Running inference with TensorRT-LLM
What Ditto gives you is just a standard TensorRT-LLM engine and its configuration file. You can directly use it for your existing TensorRT-LLM workflow.

Just as a simple example, you can use [TensorRT-LLM/examples/run.py](https://github.com/NVIDIA/TensorRT-LLM/blob/v0.16.0/examples/run.py) to check the outputs generated by your engine.

1. Clone the TensorRT-LLM repository. (It is already cloned in the container at `/workspace/tensorrt_llm`)
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git /path/to/tensorrt-llm
```

2. Then, you can run the inference as follows.
```
python /path/to/tensorrt-llm/examples/run.py --engine_dir ./engines/meta-llama/Llama-2-7b-chat-hf --tokenizer_dir meta-llama/Llama-2-7b-chat-hf --max_output_len 100 --input_text "Hey, are you conscious?"
```

## C. Visualizing and Debugging Engines
See [DEBUG.md](./DEBUG.md).
