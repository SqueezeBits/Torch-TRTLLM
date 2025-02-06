# Getting Started

## A. Environment Setup
### I. Using Docker
#### 1. Build a docker image
```
docker build -f docker/Dockerfile -t ditto:ubuntu24.04 .
```

#### 2. Run a container
```
docker run --rm -it --gpus all -v `pwd`:/workspace/ditto ditto:ubuntu24.04 bash
```

### II. Using Conda
Run `./conda/create_env.sh` to create an anaconda environment with all required packages for ditto.
See [conda/README.md](../conda/README.md) for more details.


## B. Quick Start Guide
### 1. Install ditto
Inside the docker container, run
```
pip install /workspace/ditto
```
or, for editable install,
```
pip install -e /workspace/ditto
```

Similarly, if you're using a conda environment, just replace the path `/workspace/ditto` by your local repository path.

### 2. Build a TRT-LLM engine
```
ditto build <model-id-or-hf-model-directory> --output-dir <engine-output-directory (optional)>
```
For example, the following commands are equivalent:
* `ditto build meta-llama/Llama-2-7b-chat-hf`
* `ditto build meta-llama/Llama-2-7b-chat-hf --output-dir ./engines/meta-llama/Llama-2-7b-chat-hf`

#### Full Usage
```
 Usage: ditto build [OPTIONS] MODEL_ID                                                                                                               
                                                                                                                                                     
 Build a TensorRT-LLM engine from a pretrained model.                                                                                                
                                                                                                                                                     
╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_id      TEXT  A pretrained model name or path. [default: None] [required]                                                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --add-output                                                                 TEXT               List of node names to add as output. See docs/DEBUG.md for details.                  │
│                                                                                                 [default: <class 'list'>]                                                            │
│ --peft-ids                                                                   TEXT               List of LoRA adapter IDs to apply to the model. [default: <class 'list'>]            │
│ --output-dir                                                                 TEXT               Path to the output directory. If not specified, `./engines/<model_id>` will be used. │
│ --dtype                                                                      TEXT               Data type to use for the model. [default: auto]                                      │
│ --verbose-failure                     --no-verbose-failure                                      Enable showing the value in those local variables. [default: no-verbose-failure]     │
│ --trust-remote-code                   --no-trust-remote-code                                    Trust remote code. [default: no-trust-remote-code]                                   │
│ --run-matmuls-in-fp32                 --no-run-matmuls-in-fp32                                  Run matmuls in fp32. [default: no-run-matmuls-in-fp32]                               │
│ --run-activations-in-model-dtype      --no-run-activations-in-model-dtype                       Run activations in model dtype. [default: run-activations-in-model-dtype]            │
│ --max-batch-size                                                             INTEGER            Maximum number of requests that the engine can schedule. [default: 256]              │
│ --max-seq-len                                                                INTEGER            Maximum total length of one request, including prompt and generated output.          │
│                                                                                                 [default: None]                                                                      │
│ --max-num-tokens                                                             INTEGER            Maximum number of batched input tokens after padding is removed in each batch.       │
│                                                                                                 [default: 8192]                                                                      │
│ --opt-num-tokens                                                             INTEGER            Optimal number of batched input tokens after padding is removed in each batch.       │
│                                                                                                 [default: None]                                                                      │
│ --max-beam-width                                                             INTEGER            Maximum number of beams for beam search decoding. [default: 1]                       │
│ --tp-size                                                                    INTEGER            N-way tensor parallelism size. [default: 1]                                          │
│ --logits-dtype                                                               [float16|float32]  Data type of logits. [default: float32]                                              │
│ --gather-context-logits               --no-gather-context-logits                                Enable gathering context logits. [default: no-gather-context-logits]                 │
│ --gather-generation-logits            --no-gather-generation-logits                             Enable gathering generation logits. [default: no-gather-generation-logits]           │
│ --gather-all-logits                   --no-gather-all-logits                                    Enable both `gather_context_logits` and `gather_generation_logits`.                  │
│                                                                                                 [default: no-gather-all-logits]                                                      │
│ --help                            -h                                                            Show this message and exit.                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


### 3. Run the inference with TensorRT-LLM
For example, you can use [TensorRT-LLM/examples/run.py](https://github.com/NVIDIA/TensorRT-LLM/blob/42a7b0922fc9e095f173eab9a7efa0bcdceadd0d/examples/run.py) to check the outputs generated by your engine.

```
python /workspace/tensorrt_llm/examples/run.py --engine_dir ./engines/meta-llama/Llama-2-7b-chat-hf --tokenizer_dir meta-llama/Llama-2-7b-chat-hf --max_output_len 100 --input_text "Hey, are you conscious?"
```


## C. Additional Features
### Tensor Parallelism
```bash
ditto build <model-id-or-hf-model-directory> --tp-size <tp-size> --output-dir <engine-output-directory (optional)>
```

### LoRA
TBD


## D. Visualizing and Debugging Engines
See [DEBUG.md](./DEBUG.md).
