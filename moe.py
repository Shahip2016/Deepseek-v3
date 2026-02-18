import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import DeepSeekV3Config

class DeepSeekV3Expert(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekV3MoE(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_activated_experts = config.num_activated_experts
        self.num_shared_experts = config.num_shared_experts
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            DeepSeekV3Expert(config) for _ in range(self.num_shared_experts)
        ])
        
        # Routed experts
        self.experts = nn.ModuleList([
            DeepSeekV3Expert(config) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Bias for auxiliary-loss-free load balancing
        self.register_buffer("bias", torch.zeros(self.num_experts))
        self.bias_update_speed = config.bias_update_speed

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = hidden_states.view(-1, orig_shape[-1])
        
        # 1. Compute shared experts output
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(x)
            
        # 2. Gating logic
        # Propose: s_it = Sigmoid(u_t^T * e_i)
        # Here we use a linear layer as the e_i centroids
        logits = self.gate(x)
        scores = torch.sigmoid(logits)
        
        # Apply bias for routing (aux-loss-free)
        routing_scores = scores + self.bias
        
        # Select top-k experts
        topk_scores, topk_indices = torch.topk(routing_scores, self.num_activated_experts, dim=-1)
        
        # Gating values are derived from the original scores (not the biased ones)
        # Normalize among selected scores
        selected_scores = torch.gather(scores, 1, topk_indices)
        gating_values = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        
        # 3. Compute routed experts output using token grouping
        routed_output = torch.zeros_like(x)
        
        # In a real implementation like DeepSeek-V3, this might use specialized kernels
        # or scatter/gather operations. Here we optimize by grouping tokens.
        
        # expert_idx: [total_tokens, num_activated_experts]
        # gating_values: [total_tokens, num_activated_experts]
        
        # Flatten for grouping
        flat_topk_indices = topk_indices.view(-1)
        flat_gating_values = gating_values.view(-1)
        # Duplicate tokens for each activated expert they are assigned to
        flat_x = x.repeat_interleave(self.num_activated_experts, dim=0)
        
        # Sort tokens by expert index to process them in groups
        sorted_indices = torch.argsort(flat_topk_indices)
        sorted_experts = flat_topk_indices[sorted_indices]
        sorted_x = flat_x[sorted_indices]
        sorted_weights = flat_gating_values[sorted_indices]
        
        # Find unique experts and their counts
        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        
        # Dispatch to experts in groups
        start_idx = 0
        for expert_id, count in zip(unique_experts.tolist(), counts.tolist()):
            expert = self.experts[expert_id]
            end_idx = start_idx + count
            
            # Process group
            group_x = sorted_x[start_idx:end_idx]
            group_weights = sorted_weights[start_idx:end_idx].unsqueeze(-1)
            
            # Compute expert output and scatter back
            expert_out = expert(group_x) * group_weights
            
            # We need to scatter these back to their original token positions
            # The original positions are in sorted_indices
            target_indices = sorted_indices[start_idx:end_idx] // self.num_activated_experts
            routed_output.index_add_(0, target_indices, expert_out)
            
            start_idx = end_idx
            
        # Update bias for load balancing (aux-loss-free)
        if self.training and self.bias_update_speed > 0:
            load = torch.bincount(topk_indices.view(-1), minlength=self.num_experts).float()
            load = load / load.mean()  # Normalize load
            # If load > 1 (overloaded), decrease bias. If load < 1 (underloaded), increase bias.
            self.bias -= self.bias_update_speed * (load - 1.0)

        final_output = (shared_output + routed_output).view(orig_shape)
        return final_output
