import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import DeepSeekV3Config

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [max_seq_len, head_dim]
    # position_ids: [batch_size, seq_len]
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepSeekV3Attention(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # MLA Low-rank compression ranks
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        
        # Query projection
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), bias=False)
            
        # KV projection
        self.kv_a_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.config.v_head_dim), bias=False)
        
        # Decoupled RoPE Key projection
        self.k_rope_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_rope_head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.config.v_head_dim, self.hidden_size, bias=False)
        
        # Rotary embedding
        self.register_buffer("cos", torch.zeros(config.max_position_embeddings, self.qk_rope_head_dim))
        self.register_buffer("sin", torch.zeros(config.max_position_embeddings, self.qk_rope_head_dim))
        self._init_rope()

    def _init_rope(self):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.qk_rope_head_dim, 2).float() / self.qk_rope_head_dim))
        t = torch.arange(self.config.max_position_embeddings).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Step 1: Query projection (with LoRA if configured)
        if self.config.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)
        else:
            q = self.q_proj(hidden_states)
            
        q = q.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Step 2: KV projection (Latent compression)
        kv_latent = self.kv_a_proj(hidden_states)
        kv_latent = self.kv_a_layernorm(kv_latent)
        
        kv_up = self.kv_b_proj(kv_latent)
        kv_up = kv_up.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.config.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv_up, [self.qk_nope_head_dim, self.config.v_head_dim], dim=-1)
        
        # Step 3: Decoupled Key for RoPE
        k_rope = self.k_rope_proj(hidden_states)
        k_rope = k_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        
        # Step 4: Apply RoPE to decoupled parts
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, self.cos, self.sin, position_ids)
        
        # Step 5: Construct final Q and K
        # Query: [bsz, num_heads, q_len, qk_nope_head_dim + qk_rope_head_dim]
        # Key:   [bsz, num_heads, q_len, qk_nope_head_dim + qk_rope_head_dim]
        full_q = torch.cat([q_nope, q_rope], dim=-1)
        full_k = torch.cat([k_nope, k_rope], dim=-1)
        
        # Step 6: Attention calculation
        attn_weights = torch.matmul(full_q, full_k.transpose(-1, -2)) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.config.v_head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None  # Simplification: not returning kv_cache for now
