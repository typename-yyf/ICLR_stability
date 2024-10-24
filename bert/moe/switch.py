# https://nn.labml.ai/transformers/switch/index.html

import torch
import torch.nn as nn
from .experts import FeedForwardNetwork

from torch.utils.tensorboard import SummaryWriter

class SwitchFeedForward(nn.Module):
    def __init__(self, config, 
        is_scale_prob = True,
        drop_tokens = False,
        expert = FeedForwardNetwork,
        loss_coef = 1e-2
    ):
        super().__init__()
        self.n_experts = config.num_experts
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = is_scale_prob
        self.drop_tokens = drop_tokens

        self.experts = nn.ModuleList([expert(config) for _ in range(self.n_experts)])
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)

        self.loss = None
        self.loss_coef = loss_coef

    def load_balance_loss(self, counts, route_prob):
        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_prob / total
        load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

        return load_balancing_loss

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.contiguous().view(-1, d_model)
        final_output = x.new_zeros(x.shape)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        # self.loss = self.loss_coef * self.load_balance_loss(counts, route_prob)
        
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) > capacity:
                    indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                    dropped.append(indexes_list[i][capacity:])
                    indexes_list[i] = indexes_list[i][:capacity]

        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
        
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        final_output = final_output * route_prob_max.view(-1, 1) if self.is_scale_prob else final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output