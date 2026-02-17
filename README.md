<div align="center">

# DeepSeek-V3: A Strong Mixture-of-Experts Language Model

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2412.19437-B31B1B.svg)](https://arxiv.org/abs/2412.19437)
[![GitHub](https://img.shields.io/badge/GitHub-DeepSeek--V3-blue?logo=github)](https://github.com/deepseek-ai/DeepSeek-V3)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DeepSeek-V3** is a state-of-the-art Mixture-of-Experts (MoE) language model with 671B total parameters, of which 37B are activated for each token. It introduces several architectural innovations to achieve superior performance with high efficiency.

[Introduction](#introduction) | [Key Features](#key-features) | [Architecture](#architecture) | [Quick Start](#quick-start) | [License](#license)

---

</div>

## Introduction

DeepSeek-V3 represents a significant leap in open-source AI. By leveraging Multi-head Latent Attention (MLA) and DeepSeekMoE, it provides training and inference efficiency that allows for scaling to hundreds of billions of parameters while maintaining a responsive activation footprint. This repository provides a clean PyTorch implementation of the core architectural components.

## Key Features

- 🚀 **Multi-head Latent Attention (MLA)**: Joint compression of KV and Queries to dramatically reduce KV cache overhead during generation.
- 🧩 **DeepSeekMoE**: High-granularity experts with an **Auxiliary-loss-free** load balancing strategy for stable training without performance trade-offs.
- 📈 **Multi-Token Prediction (MTP)**: A sequential prediction objective that densifies training signals and enables speculative decoding.
- 🔥 **FP8 Native Training**: Implementation of tile-wise and block-wise quantization for high-precision low-bitwidth training.

## Architecture Overview

| Component | Description |
|-----------|-------------|
| **Total Parameters** | 671B |
| **Activated Parameters** | 37B |
| **MLA Rank** | 512 (KV), 1536 (Q) |
| **Experts** | 256 routed + 1 shared |
| **Activated Experts** | 8 per token |

### Model Structure

```text
├── config.py          # Architectural hyper-parameters
├── attention.py       # MLA implementation with decoupled RoPE
├── moe.py             # DeepSeekMoE with bias-based balancing
├── model.py           # Core Transformer and CausalLM assembly
├── mtp.py             # Multi-token Prediction modules
├── quant.py           # FP8 tile/block-wise quantization
└── inference.py       # Demonstration and verification script
```

## Quick Start

### Installation

```bash
git clone git@github.com:Shahip2016/Deepseek-v3.git
cd Deepseek-v3
# Requires PyTorch 2.0+
pip install torch
```

### Running the Architecture Demo

To verify the implementation and inspect the layer structures:

```bash
python inference.py
```

## Implementation Details

### Multi-head Latent Attention (MLA)
DeepSeek-V3 uses low-rank joint compression to reduce the memory footprint of the KV cache. We implement the transformation from hidden states to latent vectors and the subsequent up-projection to decoupled keys and queries for Rotary Positional Embeddings (RoPE).

### DeepSeekMoE and Load Balancing
We implement the pioneering auxiliary-loss-free strategy. Instead of a static loss penalty, we use a dynamic bias term $b_i$ for each expert, updated based on the batch-wise load to ensure uniform expert utilization without impacting the model's convergence quality.

### Multi-token Prediction (MTP)
The MTP module consists of sequential blocks that predict the next series of tokens. It shares the embedding layer and LM head with the main model, providing a strong regularization signal and potential for high-speed speculative inference.

## Citation

If you use this code or the DeepSeek-V3 architecture in your research, please cite:

```bibtex
@misc{deepseekv3report,
      title={DeepSeek-V3 Technical Report}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
