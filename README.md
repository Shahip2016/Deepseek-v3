# DeepSeek-V3 Implementation

This repository contains a PyTorch implementation of the DeepSeek-V3 architecture as described in the paper [DeepSeek-V3 Technical Report].

## Key Features
- **Multi-head Latent Attention (MLA)**: Efficient attention mechanism with low KV cache overhead.
- **DeepSeekMoE**: Mixture-of-Experts with auxiliary-loss-free load balancing.
- **FP8 Training**: Support for mixed-precision training using FP8.
- **Multi-token Prediction (MTP)**: Enhanced training objective for better performance.

## Architecture Overview
DeepSeek-V3 is a strong Mixture-of-Experts (MoE) language model with 671B total parameters, of which 37B are activated for each token. It utilizes innovative architectural designs to achieve state-of-the-art performance.
