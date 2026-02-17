import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from config import DeepSeekV3Config
from model import DeepSeekV3Block, DeepSeekV3ForCausalLM

class MTPModule(nn.Module):
    def __init__(self, config: DeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.block = DeepSeekV3Block(config, layer_idx)
        self.layernorm_h = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layernorm_emb = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        h_prev: torch.Tensor,
        emb_target: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # h_prev: [bsz, seq_len, hidden_size]
        # emb_target: [bsz, seq_len, hidden_size]
        
        # Combined representation
        combined = torch.cat([self.layernorm_h(h_prev), self.layernorm_emb(emb_target)], dim=-1)
        h_mtp = self.proj(combined)
        
        # Transformer block
        h_mtp = self.block(h_mtp, position_ids, attention_mask)
        return h_mtp

class DeepSeekV3WithMTP(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        # Main model
        self.causal_lm = DeepSeekV3ForCausalLM(config)
        
        # MTP Modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(config, config.num_layers + k) for k in range(config.mtp_depth)
        ])
        
        # Shared components (conceptually)
        # DeepSeek-V3 shares embedding and output head across all MTP modules and the main model.
        # In this implementation, we'll use the ones from self.causal_lm.

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        bsz, seq_len = input_ids.shape
        
        # 1. Main model forward
        hidden_states = self.causal_lm.model(input_ids, position_ids, attention_mask)
        main_logits = self.causal_lm.lm_head(hidden_states)
        
        losses = []
        if labels is not None:
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            losses.append(main_loss)
            
        # 2. MTP forward
        h_prev = hidden_states
        all_mtp_logits = []
        
        for k in range(self.config.mtp_depth):
            # Target token for depth k is token at i + 1 + k
            # For depth 1: h_i combined with Emb(t_{i+1}) predicts t_{i+2}
            # We need the embeddings of the target tokens
            # For simplicity, we slice the input_ids/embeddings
            if labels is not None:
                # Training mode: we use the ground truth tokens for the next depth
                # Emb(t_{i+k})
                target_token_ids = labels[:, k+1:]
                # Pad to match sequence length
                target_token_ids = F.pad(target_token_ids, (0, k+1), value=0) 
            else:
                # Inference mode: usually MTP is discarded or used for speculation
                # Here we just show the structure
                target_token_ids = torch.zeros_like(input_ids)

            target_embs = self.causal_lm.model.embed_tokens(target_token_ids)
            
            # Predict
            h_mtp = self.mtp_modules[k](h_prev, target_embs, position_ids, attention_mask)
            mtp_logits = self.causal_lm.lm_head(h_mtp)
            all_mtp_logits.append(mtp_logits)
            
            if labels is not None:
                # Prediction for depth k+1 (predicting t_{i+2+k})
                # Labels for this loss are tokens at 2+k onwards
                mtp_labels = labels[:, k+2:]
                mtp_loss_logits = mtp_logits[:, :-(k+2), :]
                if mtp_labels.numel() > 0:
                    mtp_loss = F.cross_entropy(mtp_loss_logits.reshape(-1, self.config.vocab_size), mtp_labels.reshape(-1))
                    losses.append(mtp_loss)
            
            h_prev = h_mtp

        total_loss = None
        if losses:
            # DeepSeek-V3 average MTP losses and weight by lambda
            main_loss = losses[0]
            mtp_loss_avg = torch.stack(losses[1:]).mean() if len(losses) > 1 else torch.tensor(0.0).to(main_loss.device)
            total_loss = main_loss + self.config.mtp_loss_weight * mtp_loss_avg

        return {
            "logits": main_logits,
            "mtp_logits": all_mtp_logits,
            "loss": total_loss
        }
