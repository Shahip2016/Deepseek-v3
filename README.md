# DeepSeek-V3 Implementation

This repository contains a PyTorch implementation of the DeepSeek-V3 architecture as described in the paper [DeepSeek-V3 Technical Report].

## Key Features
- **Multi-head Latent Attention (MLA)**: Efficient attention mechanism with low KV cache overhead.
- **DeepSeekMoE**: Mixture-of-Experts with auxiliary-loss-free load balancing.
- **FP8 Training**: Support for mixed-precision training using FP8.
- **Multi-token Prediction (MTP)**: Enhanced training objective for better performance.

## Repository Structure
- `config.py`: Architectural hyper-parameters.
- `attention.py`: Multi-head Latent Attention (MLA) implementation.
- `moe.py`: DeepSeekMoE with Shared Experts and Aux-loss-free balancing.
- `model.py`: Core Transformer blocks and CausalLM assembly.
- `mtp.py`: Multi-token Prediction (MTP) modules.
- `quant.py`: Fine-grained FP8 quantization utilities.
- `inference.py`: Demonstration script for model initialization and forward pass.

## Usage
To run the demonstration script:
```bash
python inference.py
```

## Implementation Details
- **MLA**: Uses low-rank compression for KV and Queries, significantly reducing cache overhead.
- **DeepSeekMoE**: Implements the sigmoid-based gating and the bias-update strategy for load balancing without auxiliary loss degradation.
- **MTP**: Sequentially predicts future tokens, sharing the embedding and output head with the main model for efficiency.
- **FP8**: Includes tile-wise (1x128) and block-wise (128x128) quantization logic as described in the paper.
