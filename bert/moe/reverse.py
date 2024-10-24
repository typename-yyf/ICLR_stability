import torch
import torch.nn as nn
from .experts import FeedForwardNetwork
from einops import rearrange

class ReverseMoE(nn.Module):
    def __init__(self, config, 
        num_experts = None,
        is_scale_prob = True,
        expert = FeedForwardNetwork,
        loss_coef = 1e-2,
    ):
        super().__init__()
        self.n_experts = num_experts if num_experts else config.num_experts
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = is_scale_prob

        self.experts = nn.ModuleList([expert(config) for _ in range(self.n_experts)])
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)

        self.loss_coef = loss_coef
        self.loss = None

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)
        final_output = x.new_zeros((self.n_experts, batch_size * seq_len, d_model))
        counts = x.new_zeros((self.n_experts, len(x)))

        route_prob = self.softmax(self.switch(x))
        route_prob = rearrange(route_prob, 'N e -> e N')

        k = int(self.capacity_factor * len(x) / self.n_experts)
        route_prob_values, routes = torch.topk(route_prob, k=k, dim=-1, largest=True, sorted=False)

        for i in range(self.n_experts):
            counts[i][routes[i]] = 1
            # final_output[i] = x[:, :]
            expert_input = x[routes[i, :], :]
            expert_output = self.experts[i](expert_input)
            expert_output = expert_output * route_prob_values[i].view(-1, 1) if self.is_scale_prob else expert_output * (route_prob_values[i] / route_prob_values[i].detach()).view(-1, 1)
            final_output[i, routes[i, :], :] = expert_output

        final_output = torch.sum(final_output, dim=0)
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output