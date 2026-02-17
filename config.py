from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DeepSeekV3Config:
    # Model dimensions
    hidden_size: int = 7168
    num_layers: int = 61
    num_heads: int = 128
    head_dim: int = 128
    vocab_size: int = 129280  # Default BPE vocab size
    max_position_embeddings: int = 4096
    
    # MLA settings
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    
    # MoE settings
    moe_intermediate_size: int = 2048
    num_experts: int = 256
    num_shared_experts: int = 1
    num_activated_experts: int = 8
    moe_layer_freq: int = 1  # All layers except first three are MoE
    first_k_dense_layers: int = 3
    
    # Balancing settings
    aux_loss_alpha: float = 0.0001
    bias_update_speed: float = 0.001
    
    # MTP settings
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    
    # Training
    norm_eps: float = 1e-6
    initializer_range: float = 0.006
    
    # Scaling for scaled configuration
    @classmethod
    def small_config(cls):
        return cls(
            hidden_size=512,
            num_layers=12,
            num_heads=8,
            head_dim=64,
            q_lora_rank=256,
            kv_lora_rank=128,
            num_experts=16,
            num_activated_experts=4,
            moe_intermediate_size=512
        )
