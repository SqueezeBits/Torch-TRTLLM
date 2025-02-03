<div align="center">

<img src="./docs/assets/ditto_logo.png" alt="Ditto logo" width="200" />

[![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.5%2C%3C2.6-blue)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-%3C%3D4.45.1%2C%3E%3D4.38.2-yellow)](https://huggingface.co/transformers/)
[![tensorrt-llm](https://img.shields.io/badge/tensorrt--llm-0.16.0-green)](https://developer.nvidia.com/blog/tag/tensorrt-llm/)
[![torch-tensorrt](https://img.shields.io/badge/torch--tensorrt-2.5.0-lightgreen)](https://pytorch.org/TensorRT)
[![version](https://img.shields.io/badge/version-0.0.0-purple)](#)
[![license](https://img.shields.io/badge/license-Apache%202-red)](./LICENSE)

<div align="left">

# Ditto - Direct Torch to TensorRT-LLM Optimizer

Ditto is an open-source framework that enables **building TensorRT engines directly from PyTorch HuggingFace models**. Traditionally, building a TensorRT engine for a new transformer model requires implementing the model definition with TensorRT networks and converting the checkpoint into the corresponding format. Ditto eliminates the intermediate steps, allowing transformer models to be directly converted into TensorRT engines without additional effort. By simplifying this process, **Ditto aims to maximize efficiency in deploying transformer models on TensorRT**.

<div align="center">
<img src="./docs/assets/ditto_flow.png" alt="Ditto logo" width="800"/>
<div align="left">

## Key Advantages
- Directly converting new HF models into TensorRT engines that are not supported by TensorRT-LLM.
    - As of the publication date of this document (February 10, 2025), [Helium](https://huggingface.co/kyutai/helium-1-preview-2b) is supported in Ditto, while it is not in TensorRT-LLM.
- Directly converting quantized HF models.
- Enhancing the usability of custom plugins.
- Allowing for more flexible development of model architectures.

## Benchmarks

### Accuracy
TBD

### Latency
TBD

## Support Matrix

### Models
- Llama2-7B
- Llama3-8B
- LLama3.1-8B
- Llama3.2
- Llama3.3-70B
- Mistral-7B
- Gemma2-9B
- Phi4
- Phi3.5-mini
- Qwen2-7B
- Codellama
- Codestral
- ExaOne3.5-8B
- aya-expanse-8B
- Llama-DNA-1.0-8B
- SOLAR-10.7B
- Falcon
- Nemotron
- 42dot_LLM-SFT-1.3B
- Helium1-2B
- Sky-T1-32B
- SmolLM2-1.7B
- and many others that we haven't tested yet

### Plugins
- GPTAttentionPlugin
- GemmPlugin
- LoRAPlugin
- AllGatherPlugin, AllReducePlugin(for Tensor Parallelism)

## What's Next?
- Quantization
- MoE
- Multimodal
- Speculative Decoding
- Prefix Caching
- Pipeline Parallelism
- State Space Model
- Encode-Decoder Model

## Getting Started
Refer to [GUIDE.md](docs/GUIDE.md) for detailed usage.
