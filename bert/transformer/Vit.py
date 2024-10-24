import torch
import torch.nn as nn
from einops import rearrange
from moe.switch import SwitchFeedForward
from transformer.Transformer import TransformerLayer

# https://nn.labml.ai/transformers/vit/index.html
class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.hidden_size
        patch_size = config.patch_size
        in_channels = config.in_channels

        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c') # batches, patches, d_model
        return x

class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, config, max_len: int = 5_000):
        super().__init__()
        d_model = config.hidden_size
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:len(x)]
        return x + pe

class VisionTransformerMoE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = config.hidden_size
        n_classes = config.n_classes

        # self.cls_token_emb = nn.Parameter(torch.randn(1, 1, config.hidden_size), requires_grad=True)

        self.patch_emb = PatchEmbeddings(config)
        self.pos_emb = LearnedPositionalEmbeddings(config)
        self.switch = SwitchFeedForward(config)
        self.head = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        x = self.pos_emb(x)

        # cls_token_emb = self.cls_token_emb.expand(len(x), -1, -1)
        # x = torch.cat([cls_token_emb, x], dim=1)

        x = self.switch(x)
        x = torch.sum(x, dim=1)
        x = self.head(x)

        return x

class VisionTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = config.hidden_size
        n_classes = config.n_classes

        self.patch_emb = PatchEmbeddings(config)
        self.pos_emb = LearnedPositionalEmbeddings(config)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

        self.classification = nn.Linear(d_model, n_classes)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.LayerNorm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        x = self.pos_emb(x)

        batch_size, patch_size, _ = x.shape

        cls_token_emb = self.cls_token_emb.expand(batch_size, -1, -1)
        hidden_states = torch.cat([cls_token_emb, x], dim=1)
        attention_mask = hidden_states.new_ones((batch_size, patch_size + 1))
        
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        
        logit = hidden_states[:, 0]
        logit = self.LayerNorm(logit)
        logit = self.classification(logit)
        return logit
        