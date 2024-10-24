import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import numpy as np
import random

from .none import GeneralFeedForward
from utils import cluster_embedding_space


def get_continue_gpt(rank, epoch, batch, vocab_size, layer_num, embed_dim, heads_num, window_size, dataset, base_model, moe_at, ffn_expert_num, common_pct_threashold, ckpt_dir):
    # check if centers all exists
    if dataset is not None:
        indices = list(range(100000))
        selected_indices = random.sample(indices, 2048)
        sampled_data = torch.stack([dataset[i][0] for i in selected_indices])

    centers, radii, longtail_indices = {}, {}, {}
    for layer in moe_at:
        if os.path.exists(f'{ckpt_dir}/centers_layer{layer}.npy') and os.path.exists(f'{ckpt_dir}/radius_layer{layer}.npy'):
            print(f'load layer{layer} centers & radii from checkpoints...', end='\r')
            centers[layer] = np.load(f'{ckpt_dir}/centers_layer{layer}.npy')
            radii[layer] = np.load(f'{ckpt_dir}/radius_layer{layer}.npy')
            longtail_indices[layer] = np.load(f'{ckpt_dir}/longtail_idx_layer{layer}.npy')
            print(f'layer {layer}: {len(longtail_indices[layer])} longtail clusters.')
        else:
            if dataset is None or base_model is None:
                raise ValueError('dataset or base model is None and centers checkpoint not exists.')
            print(f'cluster layer {layer} embedding space...', end='\r')
            center, radius, count_dict = cluster_embedding_space(rank, base_model, sampled_data, layer, ffn_expert_num)
            print(f'layer {layer}: {count_dict}')
            torch.cuda.empty_cache()
            common_idx, longtail_idx = [], []
            presum = 0
            for idx, count in count_dict.items():
                if presum < common_pct_threashold * len(sampled_data):
                    common_idx.append(idx)
                else:
                    longtail_idx.append(idx)
                presum += count

            longtail_idx = np.array(longtail_idx)
            np.save(f'{ckpt_dir}/centers_layer{layer}.npy', center)
            np.save(f'{ckpt_dir}/radius_layer{layer}.npy', radius)
            np.save(f'{ckpt_dir}/longtail_idx_layer{layer}.npy', longtail_idx)
            centers[layer] = center
            radii[layer] = radius
            longtail_indices[layer] = longtail_idx
            print(f'layer {layer}: {len(longtail_idx)} longtail clusters. ')

    model = ContinueGPT(vocab_size, .1, layer_num, embed_dim, heads_num, window_size, .1, .1, base_model, moe_at, ffn_expert_num, (centers, radii, longtail_indices)).to(rank)

    if epoch >= 0 and batch > 0:
        weights = torch.load(f'{ckpt_dir}/{epoch}_{batch}.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(weights)
    else:
        print('no model checkpoint loaded.')

    model.to(rank)
    return model



class ContinueDecoder(nn.Module):
    def __init__(self, embed_dim, heads_num, window_size, attn_drop, ff_drop, base_module, ffn_intermediate, ffn_expert_num, centers, radius, longtail_idx) -> None:
        super().__init__()
        self.base_module = base_module
        # self.base_module.requires_grad_(False)

        # assert len(centers) == len(longtail_idx)
        # assert len(radius) == len(longtail_idx)

        self.longtail_idx = longtail_idx
        self.longtail_idx2feed_fwd_idx = {idx: i for i, idx in enumerate(longtail_idx)}
        self.feed_fwd = nn.ModuleList([])
        for idx in longtail_idx:
            expert = GeneralFeedForward(embed_dim, ffn_intermediate, embed_dim, ff_drop)
            expert.feed_fwd[2].weight.data.zero_()
            expert.feed_fwd[2].bias.data.zero_()
            self.feed_fwd.append(expert)

        self.centers, self.radius = centers, radius

    def forward(self, x, need_label = False):
        labels = None
        if isinstance(x, tuple):
            x, labels = x

        x = self.get_attn_output(x)
        x, longtail_label = self.get_ffn_output(x, labels)

        if need_label:
            return x, longtail_label
        return x, longtail_label

    def get_attn_output(self, x):
        return self.base_module.get_attn_output(x)

    def get_ffn_output(self, x, labels):
        ffn_out, longtail_lable = self.get_ffn_output_wo_ln(x, labels)
        x = self.ffn_ln(x + ffn_out)

        return x, longtail_lable

    def get_ffn_output_wo_ln(self, x, labels):
        x_baseffn_output= self.base_module.get_ffn_output_wo_ln(x)
        if isinstance(x_baseffn_output, tuple):
            x_baseffn_output, _ = x_baseffn_output

        if labels is None:
            labels = self.cal_ffn_indices(x)
        else:
            pass
            # print(f'use given label: {labels}')
        x_final_output = []
        longtail_lable = []
        # print('used idx: ', end= '')
        for i, idx in enumerate(labels):
            if idx in self.longtail_idx:
                feed_fwd_idx = self.longtail_idx2feed_fwd_idx[idx]
                # print(f'{feed_fwd_idx} ', end='')
                x_final_output.append(x_baseffn_output[i] + self.feed_fwd[feed_fwd_idx](x[i]))
                longtail_lable.append(idx)
            else:
                x_final_output.append(x_baseffn_output[i])
                # print(f'-1 ', end='')
                longtail_lable.append(idx)
        # print('')

        x_final_output = torch.stack(x_final_output)
        
        return x_final_output, longtail_lable

    def cal_ffn_indices(self, x):
        # 这里可以优化，在 GPU 上做更快
        x = x.mean(1).detach().cpu().numpy() # [N, embed_dim]
        dists = []
        # x in on cuda. move it to numpy and cal dists and index
        for center, radius in zip(self.centers, self.radius):
            dist = np.linalg.norm(x - center, axis=1) / radius
            dists.append(dist)
        dists = np.stack(dists, axis=1)
        indices = np.argmin(dists, axis=1)

        return indices

    def ffn_ln(self, x):
        return self.base_module.ffn_ln(x)


class ContinueGPT(nn.Module):
    def __init__(self, vocab_size, emb_drop, layer_num, embed_dim, heads_num, window_size, attn_drop, ff_drop, 
                 base_model, moe_at, ffn_expert_num, ffn_center_radius) -> None:
        super().__init__()
        self.max_len = window_size
        self.base_model = base_model
        self.base_model.tok_emb.requires_grad_(False)
        self.base_model.pos_emb.requires_grad_(False)

        centers, radius, longtail_idx = ffn_center_radius
        self.decoders = nn.ModuleList()
        for l in range(layer_num):
            if l in moe_at:
                self.decoders.append(
                    ContinueDecoder(embed_dim, heads_num, window_size, attn_drop, ff_drop, 
                                    self.base_model.decoders[l], embed_dim * 4 // max(len(longtail_idx[l]), 1), 
                                    ffn_expert_num, centers[l], radius[l], longtail_idx[l]))
            else:
                self.base_model.decoders[l].requires_grad_(False)
                self.decoders.append(self.base_model.decoders[l])

    def forward(self, x, need_label = False):
        x = self.get_decoder_output(x, len(self.decoders) - 1, need_label)
        if need_label or isinstance(x, tuple):
            x, label = x
        x = self.decode(x)
        if need_label:
            return x, label
        return x, label

    def get_decoder_output(self, x, i, need_label, prev = None):
        labels = []
        if prev is None:
            x = self.embed(x)
            for j in range(i + 1):
                if isinstance(self.decoders[j], ContinueDecoder):
                    x = self.decoders[j](x, need_label)
                    if need_label:
                        _, label = x
                        labels.append(label)
                else:
                    x = self.decoders[j](x)
            if need_label:
                return x, labels
            return x
        else:
            return self.decoders[i](prev, need_label)

    def get_attn_output(self, x, layer):
        x = self.get_decoder_output(x, layer - 1, need_label = False)
        return self.decoders[layer].get_attn_output(x)

    def decode(self, x):
        return self.base_model.decode(x)

    def embed(self, x):
        return self.base_model.embed(x)

