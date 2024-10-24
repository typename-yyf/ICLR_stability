class GPTConfig:
    attn_drop = .1
    ff_drop = .1
    emb_drop = .1
    moe = "vanilla" # domain, vanilla, none, cluster
    expert_num = 2
    moe_layers = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

    def __init__(self, window_size, vocab_size, embed_dim, layer_num, heads_num) -> None:
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.heads_num = heads_num


