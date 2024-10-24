import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F



def get_gpt(rank, epoch, batch, vocab_size, layer_num, embed_dim, heads_num, window_size, ckpt_dir, bit = False):
    if bit == False:
        model = GPT(vocab_size, .1, layer_num, embed_dim, heads_num, window_size, .1, .1)
    else:
        model = BitGPT(vocab_size, .1, layer_num, embed_dim, heads_num, window_size, .1, .1)
    
    if epoch > 0 or batch > 0:
        weights = torch.load(f'{ckpt_dir}/{epoch}_{batch}.pth', map_location='cpu', weights_only=True)
        # print(get_ckpt_path(epoch, batch, domain_count, embed_dim, 'none'))
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        model.load_state_dict(weights)
    
    model.to(rank)
    return model



class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        self.window_size = window_size
        assert embed_dim % heads_num == 0, 'Embedding dimension must be divisible by number of heads.'

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, self.window_size, self.window_size), diagonal=0)
        )

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)

        # x = [bs, seq_len, embed_dim]
        k = self.key(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        q = self.query(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        v = self.value(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.embed_dim // self.heads_num)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn = attn.masked_fill(mask == 0, float('-inf'))  # 不能填 inf，不然第一行全是 inf 就出 nan 了

        # [b, h, n, n] 代表了每一个 token 对其他 token 的 attention
        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class GeneralFeedForward(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)



class Decoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)
        self.feed_fwd = FeedForward(embed_dim, ff_drop)
        
        self.get_attn_output_hook = lambda x, y, z: None
        self.get_ffn_output_hook = lambda x, y, z: None

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.get_attn_output(x)
        x = self.get_ffn_output(x)

        return x
    
    def get_attn_output(self, x):
        if isinstance(x, tuple):
            x, _ = x
        attn_out = self.attn(x)
        out = attn_out + x
        ###
        out = self.ln1(out)
        self.get_attn_output_hook(attn_out, x, out)
        return out
    
    def get_ffn_output(self, x):
        ffn_out = self.get_ffn_output_wo_ln(x)
        out = ffn_out + x
        self.get_ffn_output_hook(ffn_out, x, out)
        out = self.ln2(out)
        return out
    
    def get_ffn_output_wo_ln(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.feed_fwd(x)
        return x
    
    def ffn_ln(self, x):
        return self.ln2(x)


class StaticMOEDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop, expert_num, pretrained_module = None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, heads_num, window_size, attn_drop)

        self.feed_fwd = nn.ModuleList([FeedForward(embed_dim, ff_drop) for _ in range(expert_num)])
        self.expert_num = expert_num

        if pretrained_module is not None:
            self.ln1.load_state_dict(pretrained_module.ln1.state_dict())
            self.ln2.load_state_dict(pretrained_module.ln2.state_dict())
            self.attn.load_state_dict(pretrained_module.attn.state_dict())
            for i in range(expert_num):
                if isinstance(pretrained_module.feed_fwd, nn.ModuleList):
                    self.feed_fwd[i].load_state_dict(pretrained_module.feed_fwd[i].state_dict())
                elif isinstance(pretrained_module.feed_fwd, FeedForward):
                    self.feed_fwd[i].load_state_dict(pretrained_module.feed_fwd.state_dict())

    def forward(self, x):
        x, domain = x
        x = x + self.attn(x)
        x = self.ln1(x)

        res = []
        for d, sent in zip(domain, x):
            res.append(self.feed_fwd[d](sent))
        x = x + torch.stack(res, dim=0)
        x = self.ln2(x)

        return x, domain


class GPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.dropout = nn.Dropout(emb_drop)
        self.decoders = nn.Sequential(*[Decoder(embed_dim, heads_num, window_size, attn_drop, ff_drop) for _ in range(layer_num)])
        self.ln_post_emb = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.decode(x)

        return x

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            x = self.embed(x)
            ###
            x = self.ln_post_emb(x)
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1)
        # self.embed(x)
        # for j in range(layer):
        #     x = self.decoders[j](x)
        x = self.decoders[layer].get_attn_output(x)
        return x

    def decode(self, x):
        x = self.fc(self.ln(x))
        return x

    def embed(self, x):
        seq_len = x.size(1)
        # x = [bs, seq_len, vocab_size]
        tok_x = self.tok_emb(x)
        # tok_emb = [bs, seq_len, embed_dim]
        pos_emb = self.pos_emb[:, :seq_len, :]
        x = self.dropout(tok_x) + pos_emb
        return x
    
class SubLN(nn.Module):
    def __init__(self, eps=1e-5):
        super(SubLN, self).__init__()
        self.eps = eps

    def forward(self, x):
        # 计算均值 E(x)
        mean = x.mean(dim=-1, keepdim=True)
        # 计算方差 Var(x)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # 标准化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm

def activation_quant(x: Tensor):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: Tensor, eps: float = 1e-5):
    """
    Quantizes the weight tensor to {-1, 0, +1} using absmean quantization.

    Args:
        w (Tensor): The weight tensor.
        eps (float): Small epsilon value for numerical stability.
    
    Returns:
        Tensor: Quantized weight tensor.
    """
    # Compute the absolute mean of the weight matrix
    abs_mean = w.abs().mean()

    # Scale the weight matrix by the absolute mean
    scaled_w = w / (abs_mean + eps)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Quantize the weights to the nearest value in {-1, 0, +1}
    ###！！！ 这里已经提前做了反量化了，实际上并没有做1 bit的矩阵乘法
    ### 而且反量化的方式也有待商榷
    w_quant = scaled_w.round().clamp_(-1, 1) * abs_mean
    
    return w_quant


class BitLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer using quantized weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        w = self.weight
        # Apply RMSNorm to normalize the input activations
        # 使用 SubLN 进行归一化
        ### 原代码使用的是Zeta库的SimpleRMSNorm，存在依赖问题，目前的SubLN是和论文中公式是一致的
        subln = SubLN()
        x_norm = subln(x)
        
        # Quantize the activations and weights
        # 激活量化，使用SJE技术
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # 权重量化，使用SJE技术
        w_quant = w + (weight_quant(w) - w).detach()

        # Perform the linear transformation with quantized weights and activations
        y = F.linear(x_quant, w_quant)

        return y

class BitMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        self.window_size = window_size
        assert embed_dim % heads_num == 0, 'Embedding dimension must be divisible by number of heads.'

        self.key = BitLinear(embed_dim, embed_dim)
        self.query = BitLinear(embed_dim, embed_dim)
        self.value = BitLinear(embed_dim, embed_dim)
        self.proj = BitLinear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(attn_drop)
        self.register_buffer('mask',
            torch.tril(torch.ones(1, 1, self.window_size, self.window_size), diagonal=0)
        )

    def forward(self, x):
        bs = x.size(0)
        seq_len = x.size(1)

        # x = [bs, seq_len, embed_dim]
        k = self.key(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        q = self.query(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        v = self.value(x).view(bs, seq_len, self.heads_num, self.embed_dim // self.heads_num).transpose(1, 2)
        # k, q, v = [bs, heads_num, seq_len, embed_dim // heads_num]

        # [b, h, n, d] * [b, h, d, n] = [b, h, n, n]
        attn = (torch.matmul(q, k.transpose(-2, -1))) / math.sqrt(self.embed_dim // self.heads_num)
        mask = self.mask[:, :, :seq_len, :seq_len] #[1, 1, n, n]
        attn = attn.masked_fill(mask == 0, float('-inf'))  # 不能填 inf，不然第一行全是 inf 就出 nan 了

        # [b, h, n, n] 代表了每一个 token 对其他 token 的 attention
        # attn[b, 0, n] = q[b, 0, d] * k[b, d, n] * mask_fill
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # [b, h, n, n] * [b, h, n, d] = [b, h, n, d]     x[b, 0, d] = attn[b, 0, n] * v[b, n, d]
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class BitFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            BitLinear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            BitLinear(4 * embed_dim, embed_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)


class BitGeneralFeedForward(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, ff_drop) -> None:
        super().__init__()
        self.feed_fwd = nn.Sequential(
            BitLinear(in_dim, hid_dim),
            nn.GELU(),
            BitLinear(hid_dim, out_dim),
            nn.Dropout(ff_drop)
        )

    def forward(self, x):
        return self.feed_fwd(x)

class BitDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = BitMultiheadAttention(embed_dim, heads_num, window_size, attn_drop)
        self.feed_fwd = BitFeedForward(embed_dim, ff_drop)
        
        self.get_attn_output_hook = lambda x, y, z: None
        self.get_ffn_output_hook = lambda x, y, z: None

    def forward(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.get_attn_output(x)
        x = self.get_ffn_output(x)

        return x
    
    def get_attn_output(self, x):
        if isinstance(x, tuple):
            x, _ = x
        attn_out = self.attn(x)
        out = attn_out + x
        ### 下面的原本GPT代码没有，但是BitNet和原始的Transformer都有
        out = self.ln1(out)
        self.get_attn_output_hook(attn_out, x, out)
        return out
    
    def get_ffn_output(self, x):
        ffn_out = self.get_ffn_output_wo_ln(x)
        out = ffn_out + x
        self.get_ffn_output_hook(ffn_out, x, out)
        out = self.ln2(out)
        return out
    
    def get_ffn_output_wo_ln(self, x):
        if isinstance(x, tuple):
            x, _ = x
        x = self.feed_fwd(x)
        return x
    
    def ffn_ln(self, x):
        return self.ln2(x)

class BitGPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, window_size, attn_drop, ff_drop) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size-1)
        self.pos_emb = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        self.dropout = nn.Dropout(emb_drop)
        self.decoders = nn.Sequential(*[BitDecoder(embed_dim, heads_num, window_size, attn_drop, ff_drop) for _ in range(layer_num)])
        self.ln_post_emb = nn.LayerNorm(embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = BitLinear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.get_decoder_output(x, len(self.decoders) - 1)
        x = self.decode(x)

        return x

    def get_decoder_output(self, x, i, prev = None):
        if prev is None:
            x = self.embed(x)
            ###！！！ BitNet这里会做一个Post emb norm
            x = self.ln_post_emb(x)
            for j in range(i + 1):
                x = self.decoders[j](x)
            return x
        else:
            return self.decoders[i](prev)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1)
        # self.embed(x)
        # for j in range(layer):
        #     x = self.decoders[j](x)
        x = self.decoders[layer].get_attn_output(x)
        return x

    def decode(self, x):
        x = self.fc(self.ln(x))
        return x

    def embed(self, x):
        seq_len = x.size(1)
        # x = [bs, seq_len, vocab_size]
        tok_x = self.tok_emb(x)
        # tok_emb = [bs, seq_len, embed_dim]
        pos_emb = self.pos_emb[:, :seq_len, :]
        x = self.dropout(tok_x) + pos_emb
        return x


