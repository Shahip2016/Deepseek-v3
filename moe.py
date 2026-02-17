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
        
        # 3. Compute routed experts output
        routed_output = torch.zeros_like(x)
        for i in range(self.num_activated_experts):
            expert_idx = topk_indices[:, i]
            expert_weight = gating_values[:, i].unsqueeze(-1)
            
            # This is a naive implementation: in reality, we'd group tokens by expert
            for idx in range(self.num_experts):
                mask = (expert_idx == idx)
                if mask.any():
                    routed_output[mask] += expert_weight[mask] * self.experts[idx](x[mask])
                    
        # Update bias for load balancing (simplified version for demo/impl)
        if self.training and self.bias_update_speed > 0:
            load = torch.bincount(topk_indices.view(-1), minlength=self.num_experts).float()
            load = load / load.mean()  # Normalize load
            # If load > 1 (overloaded), decrease bias. If load < 1 (underloaded), increase bias.
            self.bias -= self.bias_update_speed * (load - 1.0)

        final_output = (shared_output + routed_output).view(orig_shape)
        return final_output
