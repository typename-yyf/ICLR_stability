import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from .Transformer import TransformerLayers
from transformers import BertConfig

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 1, repeat(indexes, 'b t -> b t c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        B, T, _ = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes]), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes]), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:, :remain_T]

        return patches, forward_indexes, backward_indexes

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
        max_len = (config.image_size // config.patch_size) ** 2
        self.positional_encodings = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings
        return x + pe

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_emb = PatchEmbeddings(config)
        self.pos_emb = LearnedPositionalEmbeddings(config)
        self.shuffle = PatchShuffle(config.mask_ratio)
        self.transformer = TransformerLayers(config, config.encoder_layers)

    def forward(self, x: torch.Tensor):
        x = self.patch_emb(x)
        x = self.pos_emb(x)

        patches, forward_indexes, backward_indexes = self.shuffle(x)
        features = self.transformer(patches)

        return features, backward_indexes

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_emb = LearnedPositionalEmbeddings(config, max_len=(config.image_size // config.patch_size) ** 2)

        self.transformer = TransformerLayers(config, config.decoder_layers)
        self.head = torch.nn.Linear(config.hidden_size, config.in_channels * config.patch_size ** 2)

    def forward(self, features: torch.Tensor, backward_indexes):
        B, T, _ = features.shape
        features = torch.cat([features, self.mask_token.expand(B, backward_indexes.shape[1] - T, -1)], dim=1)
        features = take_indexes(features, backward_indexes)
        features = self.pos_emb(features)
        features = self.transformer(features)

        patches = self.head(features)
        image = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=self.image_size//self.patch_size)

        mask = torch.zeros_like(patches)
        mask[:, T:] = 1
        mask = take_indexes(mask, backward_indexes)
        mask = rearrange(mask, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=self.image_size//self.patch_size)

        return image, mask

class MaskedAutoEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, x: torch.Tensor):
        features, backward_indexes = self.encoder(x)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask

if __name__ == "__main__":
    config = BertConfig.from_json_file('config/mae.json')
    mae = MaskedAutoEncoder(config)

    img = torch.rand(7, 3, 64, 64)
    mae(img)