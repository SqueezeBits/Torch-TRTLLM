# Ditto

## Setting up Docker container
### 1. Build a docker image
```
docker build -f docker/Dockerfile -t ditto:ubuntu24.04 .
```

### 2. Run a container
```
docker run --rm -it --gpus all -v `pwd`:/workspace/ditto -v /home/hdd/huggingface_models:/data ditto:ubuntu24.04 bash
```


## Quick start guide
### 1. Install ditto
```
pip install /workspace/ditto
```
or, for editable install,
```
pip install -e /workspace/ditto
```

### 2. Build a TRT-LLM engine
```
ditto build <model-id-or-hf-model-directory> --output-dir <engine-output-directory>
```
For example,
```
ditto build /data/Llama-2-7b-chat-hf --output-dir ./engines/Llama-2-7b-chat-hf-ditto
```

### 3. Run the inference with TRT-LLM example code
```
python /workspace/tensorrt_llm/examples/run.py --engine_dir ./engines/Llama-2-7b-chat-hf-ditto --tokenizer_dir /data/Llama-2-7b-chat-hf --max_output_len 100 --input_text "Hey, are you conscious?"
```


## Debugging engine files
### 1. Dumping Debug Artifacts while Building Engine with Ditto
Set the environment variable `DEBUG_ARTIFACTS_DIR` to dump intermediate build artifacts, such as graph module and TensorRT network definition, as files.
```
DEBUG_ARTIFACTS_DIR=./artifacts/ditto-build ditto build /data/Llama-2-7b-chat-hf --output-dir ./engines/Llama-2-7b-chat-hf-ditto
```

### 2. Dumping Debug Artifacts while Building Engine with trtllm-build
Use the [scripts/trtllm-build](scripts/trtllm-build) file instead.
* The only difference is the line `import ditto.patches` added in the original script.

Likewise, set the environment variable `DEBUG_ARTIFACTS_DIR` and **add the flag `--visualize_network --profiling_verbosity detailed`** to dump intermediate build artifacts as files.
```
DEBUG_ARTIFACTS_DIR=./artifacts/trtllm-build /workspace/ditto/scripts/trtllm-build --checkpoint_dir /data/Llama-2-7b-chat-hf-ckpt --output_dir ./engines/Llama-2-7b-chat-hf-trtllm --visualize_network --profiling_verbosity detailed
```
where the checkpoints in the directory `/data/Llama-2-7b-chat-hf-ckpt` is generated by one of the native TRT-LLM's `convert_checkpoint.py` scripts, for example,
```
python /workspace/tensorrt_llm/examples/llama/convert_checkpoint.py --model_dir /data/Llama-2-7b-chat-hf --dtype float16 --output_dir /data/Llama-2-7b-chat-hf-ckpt
```

#### Note
If you want to create `plugin.txt` file, you need to manually patch the function `gpt_attention_plugin` in the file `/workspace/tensorrt_llm/tensorrt_llm/functional.py` as written in [ditto/patches.py](src/ditto/patches.py).

Just copy and paste the sections wrapped by the following comments at the right place in the code:
```
# ============================ patch start ============================
<the contents to copy&paste>
# ============================ patch end ============================
```


### 3. Adding Extra Output Tensors into Engine with Ditto
* Use the flag `--add-output`, possibly multiple times.
* See the debug artifacts `graph_module.py` and `graph.txt` to see the node names.

For example, the following command will append the TRT tensors corresponding to the outputs of the FX nodes `mm_default` and `fake_gpt_attention_plugin` to the TensorRT network outputs.
```
ditto build /data/Llama-2-7b-chat-hf --output-dir ./engines/Llama-2-7b-chat-hf-ditto-more-outputs --add-output mm_default --add-output fake_gpt_attention_plugin
```


### 4. Adding Extra Output Tensors into Engine with trtllm-build
* Specify the environment variable `TRTLLM_ADD_OUTPUT` in JSON format
* The JSON contents must be a dictionary mapping the TRT layer names to an alias of your choice. It'll be helpful for the comparison later to choose the alias as the corresponding name of the FX node producing it.
* See the debug artifacts `trt_network_def.py` for names of the TRT tensors. The name of the TRT layer that produces each tensor is written in the comment above it.

For example, the following command will append the TRT tensors corresponding to the outputs of the FX nodes `mm_default` and `fake_gpt_attention_plugin` to the TensorRT network outputs in Llama2-7B.
```
TRTLLM_ADD_OUTPUT='{"LLaMAForCausalLM/transformer/layers/0/attention/qkv/multiply_collect_L272/multiply_and_lora_L246/matmul_L1048/cast_L855/CAST_0": "mm_default", "LLaMAForCausalLM/transformer/layers/0/attention/wrapper_L561/gpt_attention_L5154/PLUGIN_V2_GPTAttention_0": "fake_gpt_attention_plugin"}' /workspace/ditto/scripts/trtllm-build --checkpoint_dir /data/Llama-2-7b-chat-hf-ckpt --output_dir ./engines/Llama-2-7b-chat-hf-trtllm-more-outputs
```

### 5. Dumping Stepwise Inputs and Outputs while Running Inference
Use the [scripts/run.py](scripts/run.py) file instead.
* The only difference is the line `import ditto.patches` added in the original script.

Likewise, set the environment variable `DEBUG_ARTIFACTS_DIR` and **add the flags `--use_py_session --debug_mode`** to dump stepwise input and output tensors as files.

For example:
```
DEBUG_ARTIFACTS_DIR=./artifacts/ditto-run \
python /workspace/ditto/scripts/run.py \
    --engine_dir ./engines/Llama-2-7b-chat-hf-ditto-more-outputs \
    --tokenizer_dir /data/Llama-2-7b-chat-hf \
    --max_output_len 10 \
    --input_text "Hey, are you conscious?" \
    --use_py_session \
    --debug_mode

DEBUG_ARTIFACTS_DIR=./artifacts/trtllm-run \
python /workspace/ditto/scripts/run.py \
    --engine_dir ./engines/Llama-2-7b-chat-hf-trtllm-more-outputs \
    --tokenizer_dir /data/Llama-2-7b-chat-hf \
    --max_output_len 10 \
    --input_text "Hey, are you conscious?" \
    --use_py_session \
    --debug_mode
```

The stepwise input and output tensors will be written in each or the file named `./artifacts/ditto-run/step{i}.txt` or `./artifacts/trtllm-run/step{i}.txt` and  where `i` is the step index.

If you have added extra output tensors, the extra tensors will be saved as well.


### 5. Comparing Tensors
```
ditto compare ./artifacts/ditto-run/step0.pt ./artifacts/trtllm-run/step0.pt
```
