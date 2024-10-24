import torch
import torch.nn as nn
from einops import rearrange
from moe.switch import SwitchFeedForward
from moe.reverse import ReverseMoE
from moe.experts import DividedExpert, DividedFFN

# refer to:
# https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/transformer_vanilla/mhsa.py
class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.attention_head_size = int(config.hidden_size / config.naum_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        self.to_qkv = nn.Linear(self.input_dim, self.input_dim * 3, bias=False)
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, 3 * num_head * head_dim
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states)
        q, k, v = tuple(rearrange(qkv, 'b n (k h d) -> k b h n d', k=3, h=self.num_attention_heads))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, :, None] == 0, -10000)
        attention = self.softmax(scaled_dot_prod)

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result
    
    def freeze(self, heads):
        num_head = self.num_attention_heads
        head_size = self.attention_head_size

        head_mask = torch.ones_like(self.to_qkv.weight.data)
        for head in heads:
            for k in range(3):
                start = k * self.input_dim
                head_mask[start + head_size * head : start + head_size * (head + 1)] = 0
        
        def hook_factory(keep_mask):
            def hook(grads):
                return grads * keep_mask
            return hook
        self.hook = self.to_qkv.weight.register_hook(hook_factory(head_mask))

    def reset(self):
        if hasattr(self, 'hook'):
            self.hook.remove()

class MHSA(nn.Module):
    def __init__(self, config):
        super(MHSA, self).__init__()
        self.config = config
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
        
        self.attention_hook_after_qk = lambda m, input, output: None
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        qkv = torch.stack([self.heads[h](hidden_states) for h in range(self.num_attention_heads)])
        # qkv = torch.stack([self.heads[h](hidden_states) * self.head_mask[h] for h in range(self.num_attention_heads)])
        # batch_size, seq_len, _ = hidden_states.shape
        # qkv = torch.stack([
        #     self.heads[h](hidden_states) if self.head_mask[h] else hidden_states.new_zeros((batch_size, seq_len, self.attention_head_size * 3))
        #     for h in range(self.num_attention_heads)
        # ])
        q, k, v = tuple(rearrange(qkv, 'h b n (k d) -> k b h n d', k=3))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
           
        self.attention_hook_after_qk(self, (q, k, v), scaled_dot_prod)
              
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
        attention = self.softmax(scaled_dot_prod)
        self.attention = attention

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = MHSA(config) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):
        hidden_states = self.self(input_tensor, attention_mask)
        atten_out = hidden_states.detach().norm().item()
        
        hidden_states = self.dense(hidden_states)
        mlp_out = hidden_states.detach().norm().item()
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        lnorm_out = hidden_states.detach().norm().item()
        
        
        return hidden_states

class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = Attention(config)

        if hasattr(config, 'moe') and config.moe:
            self.ffn = SwitchFeedForward(config, expert=DividedExpert)
            # self.ffn = ReverseMoE(config, expert=DividedExpert)
        else:
            self.ffn = FeedForward(config)
            # self.ffn = DividedFFN(config)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class TransformerLayers(nn.Module):
    def __init__(self, config, n_layers):
        super(TransformerLayers, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(n_layers)])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if not attention_mask:
            batch_size, seq_len, _ = hidden_states.shape
            attention_mask = hidden_states.new_ones((batch_size, seq_len))
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.embeddings = Embeddings(config)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        #print("====")
        embeddings = self.embeddings(input_ids)
        output = self.encoder(embeddings, attention_mask)
        return output