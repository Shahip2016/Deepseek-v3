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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        # Self-Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.attn(
            hidden_states, 
            position_ids, 
            past_key_value=past_key_value, 
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        
        # FFN (MoE or Dense)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value

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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        
        if position_ids is None:
            if past_key_values is not None:
                # Get the sequence length of the first KV cache entry
                past_len = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            
        next_decoder_cache = []
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(
                hidden_states, 
                position_ids, 
                past_key_value=past_key_value,
                attention_mask=attention_mask
            )
            next_decoder_cache.append(present_key_value)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        hidden_states, next_kv_cache = self.model(input_ids, position_ids, past_key_values, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        past_key_values = None
        curr_input_ids = input_ids
        
        for _ in range(max_new_tokens):
            logits, _, past_key_values = self.forward(curr_input_ids, past_key_values=past_key_values)
            
            # Focus on last token
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            curr_input_ids = next_token
            
        return input_ids
