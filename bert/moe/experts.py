from einops import rearrange
import torch
import torch.nn as nn
import math
import util

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class DividedExpert(nn.Module):
    def __init__(self, config):
        super(DividedExpert, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size // config.num_experts)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size // config.num_experts, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class DividedFFN(nn.Module):
    def __init__(self, config,
        expert = DividedExpert,
    ) -> None:
        super(DividedFFN, self).__init__()
        self.config = config
        self.n_experts = config.num_experts

        self.experts = nn.ModuleList([expert(config) for _ in range(self.n_experts)])

    def forward(self, x: torch.Tensor):
        expert_output = [self.experts[i](x) for i in range(self.n_experts)]
        final_output = torch.mean(torch.stack(expert_output), dim=0)
        return final_output

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)
    
# expert class
class Experts(nn.Module):
    def __init__(self, config, activation = nn.GELU):
        super().__init__()

        w1 = torch.zeros(config.num_experts, config.hidden_size, config.intermediate_size)
        w2 = torch.zeros(config.num_experts, config.intermediate_size, config.hidden_size)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out