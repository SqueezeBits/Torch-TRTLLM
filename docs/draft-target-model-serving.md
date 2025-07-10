# Running Draft-Target Models for Speculative Decoding with TensorRT-LLM

To run a draft-target model, we can use two methods: the native TensorRT-LLM workflow or the Triton Inference Server. The [official guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/draft_target_model/README.md) provides instructions for both approaches, which you can easily follow to get started.

This document focuses on the Triton Inference Server-based approach and highlights the limitations of the officially supported implementation.

Currently, Triton Inference Server supports serving draft-target models via `tensorrt_llm_bls`. However, the provided implementation has two major limitations:

1. It only supports a single request per batch (i.e., batch size = 1)
2. It does not support streaming responses.

The second limitation can be addressed with minor code modifications to enable streaming. However, the first issue requires a deeper investigation—specifically, whether the runtime or executor itself supports batching for draft-target models.

From various tests, we observed that the draft model does not appear to have any special logic—it simply processes requests with `min_tokens` and `num_output_tokens` set to the number of draft tokens. On the other hand, the target model seems to implement distinct logic for handling draft tokens. Based on this, we believe there is no fundamental reason batching should not be supported.

To test this hypothesis, we modified the `tensorrt_llm_bls` implementation to support asynchronous multi-request batching. During testing, the modified BLS was able to launch inference in batches, but we eventually encountered a CUDA error after a few internal executor iterations.

In conclusion, the current draft-target serving implementation is not practical for production use due to its inefficiencies and issues. To make speculative decoding with draft-target models viable in real-world environments, both the BLS and parts of the internal TensorRT-LLM will likely need to be extended or revised.


