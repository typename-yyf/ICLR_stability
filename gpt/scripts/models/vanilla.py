import torch
import torch.nn as nn
import numpy as np

from .none import MultiheadAttention, FeedForward, Decoder

def get_vanilla_model(rank, epoch, batch, vocab_size, layer_num, embed_dim, heads_num, window_size, expert_num, moe_at, ckpt_dir):
    model = VanillaMoEGPT(vocab_size, .1, layer_num, embed_dim, heads_num, window_size, .1, .1, expert_num, moe_at).to(rank)

    if batch > 0 or epoch > 0:
        weights = torch.load(f'{ckpt_dir}/{epoch}_{batch}.pth', map_location='cpu')
        model.load_state_dict(weights)

    model.to(rank)
    return model


class VanillaMOEDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num,) -> None:
        super().__init__()
        embed_dim = embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)

        self.experts = nn.ModuleList([FeedForward(embed_dim, ff_drop) for _ in range(expert_num)])
        self.gate = nn.Linear(embed_dim, expert_num)
        self.expert_num = expert_num


    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        
        bs, seq_len = x.size(0), x.size(1)
        x = x.reshape(bs * seq_len, -1)
        final_output = x.new_zeros(x.shape)
        router_res = self.gate(x) # [bs * seq_len, expert_num], [bs * seq_len]
        router_res = torch.softmax(router_res, dim=-1)
        indices = router_res.argmax(dim=-1) # [bs * seq_len]
        idx_list = [torch.eq(indices, i).nonzero(as_tuple=True)[0] for i in range(len(self.experts))]
        expert_output = [expert(x[idx_list[i], :]) for i, expert in enumerate(self.experts)]
        for i, idx in enumerate(idx_list):
            final_output[idx, :] = expert_output[i]

        router_res = router_res.max(dim=-1)[0] # [bs, seq_len]
        final_output = final_output * router_res.unsqueeze(-1) # [bs, seq_len, embed_dim]

        final_output = final_output.reshape(bs, seq_len, -1)

        return final_output


class VanillaMoEGPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num, moe_at) -> None:
        super().__init__()
        self.max_len = window_size
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, embed_dim))
        self.dropout = nn.Dropout(emb_drop)
        self.decoders = nn.ModuleList([VanillaMOEDecoder(embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num) if i >= 6 else Decoder(embed_dim, heads_num, window_size, attn_drop, ff_drop) for i in range(layer_num)])

        self.decoders = nn.ModuleList()
        for l in range(layer_num):
            if l in moe_at:
                self.decoders.append(VanillaMOEDecoder(embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num))
            else:
                self.decoders.append(Decoder(embed_dim, heads_num, window_size, attn_drop, ff_drop))
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.fc(self.ln(x))

        return x

    def get_tok_emb(self, x):
        return self.tok_emb(x)

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            seq_len = x.size(1) # x = [bs, seq_len, vocab_size]
            tok_x = self.tok_emb(x) # tok_emb = [bs, seq_len, embed_dim]
            pos_emb = self.pos_emb[:, :seq_len, :]
            x = self.dropout(tok_x) + pos_emb
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

