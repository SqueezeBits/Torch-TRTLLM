# Notice on Using Ditto-Built VLM Engines with TensorRT-LLM's LLM API

TensorRT-LLM is actively transitioning toward PyTorch-based, non-engine workflows, and as a result, feature support for engine-based workflows is being deprecated.

For example, the new LLM API in TensorRT-LLM currently does not support hybrid encoder-decoder architectures, such as vision encoder engine + LLM decoder engine. [This functionality was previously supported through the ENCODER_DECODER trtllm.Executor](https://github.com/NVIDIA/TensorRT-LLM/blob/dd3c736c7eeb7932bcc5b52a97ee1e7c64f6596d/tensorrt_llm/runtime/model_runner_cpp.py#L422-L425), but is no longer available in the latest LLM API, and it’s unlikely to be reintroduced.


To use Ditto-built engines for vision-language models (VLMs) with the latest LLM API, you will need to modify parts of the TensorRT-LLM codebase.

We provide an example of such a modification for qwen2.5-vl in [this commit](https://github.com/SqueezeBits/TensorRT-LLM/commit/f973dc24c5332a6045399df6676f9262b374de5c), demonstrating how to integrate a vision encoder engine with a decoder engine using the current LLM API. Please note that this example is still experimental and may not yield optimal performance.