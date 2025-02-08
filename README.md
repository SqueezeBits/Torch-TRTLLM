<div align="center">

<img src="./docs/assets/ditto_logo.png" alt="Ditto logo" width="200" />

[![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.5%2C%3C2.6-blue)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-%3C%3D4.45.1%2C%3E%3D4.38.2-yellow)](https://huggingface.co/transformers/)
[![tensorrt-llm](https://img.shields.io/badge/tensorrt--llm-0.16.0-green)](https://developer.nvidia.com/blog/tag/tensorrt-llm/)
[![torch-tensorrt](https://img.shields.io/badge/torch--tensorrt-2.5.0-lightgreen)](https://pytorch.org/TensorRT)
[![version](https://img.shields.io/badge/version-0.1.0-purple)](#)
[![license](https://img.shields.io/badge/license-Apache%202-red)](./LICENSE)

<div align="left">

# Ditto - Direct Torch to TensorRT-LLM Optimizer

Ditto is an open-source framework that enables **direct conversion of HuggingFace `PreTrainedModel`s into TensorRT-LLM engines**. Normally, building a TensorRT-LLM engine consists of two steps - checkpoint conversion and `trtllm-build` - both of which rely on pre-defined model architectures. As a result, converting a novel model requires porting the model with [TensorRT-LLM's Python API](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#tensorrt-llm-overview) and writing a custom checkpoint conversion script. **By automating these dull procedures, Ditto aims to make TensorRT-LLM more accessible to the broader AI community**.

<div align="center">
<img src="./docs/assets/ditto_flow.png" alt="Ditto logo" width="800"/>
<div align="left">

## Getting Started
* [Installation](docs/GUIDE.md#a-installation)
* [Quick Start Guide](docs/GUIDE.md#b-quick-start-guide)
* [Debugging](docs/DEBUG.md)


## Key Advantages
- Ease-of-use: Ditto enables users to convert models with a single command.
```
ditto build <huggingface-model-name>
```
- Allowing novel model architectures to be converted into TensorRT engines.
    - As of the publication date of this document (February 10, 2025), [Helium](https://huggingface.co/kyutai/helium-1-preview-2b) is supported in Ditto, while it is not in TensorRT-LLM. (Note that you need to re-install transformers nightly-build after installing Ditto as `pip install git+https://github.com/huggingface/transformers.git`)

## Benchmarks

Note that the quality evaluation results are benchmarked using  [TensorRT-LLM llmapi](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/llmapi) with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main), while the performance results are benchmarked using [TensorRT-LLM gptManagerBenchmark](https://github.com/NVIDIA/TensorRT-LLM/tree/main/benchmarks/cpp). Both the GEMM plugin and the GPT attention plugin are enabled during all benchmarks.


### [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
| | MMLU<br/>(0-shot) | wikitext2 | gpqa_main_zeroshot | arc_challenge<br/>(0-shot) |ifeval<br>(0-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ditto | 0.819 | 3.9551 | 0.5067 | 0.9283 | 0.915025 |
| TensorRT-LLM | 0.819 | 3.9551 | 0.5067 | 0.9283 | 0.915025 |

| token throughput | 4 * A100-SXM4-80GB (4 way TP) |
| :--- | :---: |
| TensorRT-LLM | 1751.6 (token/sec) |
| Ditto | 1759.18 (token/sec) |

### [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
| | MMLU<br/>(0-shot) | wikitext2 | gpqa_main_zeroshot | arc_challenge<br/>(0-shot) |ifeval<br>(0-shot) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ditto | 0.6799 | 8.6402 | 0.3504 | 0.8234 | 0.8153 |
| TensorRT-LLM | 0.6799 | 8.6402 | 0.3504 | 0.8234 | 0.8153 |

| token throughput | A100-SXM4-80GB | A6000 | L40 |
| :--- | :---: | :---: | :---: |
| Ditto | 3357.89 (token/sec) | 1479.75 (token/sec) | 1085.23 (token/sec) |
| TensorRT-LLM | 3317.96 (token/sec) | 1508.59 (token/sec) | 1086.53 (token/sec) |

### [kyutai/helium-1-preview-2b](https://huggingface.co/kyutai/helium-1-preview-2b)
| | MMLU<br/>(0-shot) | wikitext2 |
| :--- | :---: | :---: |
| Ditto | 0.486 | 11.3724 |
| TensorRT-LLM | - | - |

| token throughput | A6000 | L40 |
| :--- | :---: | :---: |
| Ditto | 1439.5 (token/sec) | 1340.49 (token/sec) | 
| TensorRT-LLM | - | - | 


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
- AllGatherPlugin, AllReducePlugin (for Tensor Parallelism)

## What's Next?
- Quantization
- MoE
- Multimodal
- Speculative Decoding
- Prefix Caching
- Pipeline Parallelism
- State Space Model
- Encode-Decoder Model
