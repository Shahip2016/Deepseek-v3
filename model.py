import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import DeepSeekV3Config
from attention import DeepSeekV3Attention
from moe import DeepSeekV3MoE

class DeepSeekV3MLP(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekV3Block(nn.Module):
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = DeepSeekV3Attention(config)
        
        # In DeepSeek-V3, all FFNs except the first few are MoE
        if layer_idx >= config.first_k_dense_layers:
            self.ffn = DeepSeekV3MoE(config)
        else:
            self.ffn = DeepSeekV3MLP(config)
            
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.attn(hidden_states, position_ids, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        # FFN (MoE or Dense)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekV3Block(config, i) for i in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class DeepSeekV3ForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(input_ids, position_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss
