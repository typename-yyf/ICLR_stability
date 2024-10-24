import torch
from models import get_gpt
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

def train_gpt(rank, start_epoch, start_batch, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, task_name, save_every_batch = 128):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Run training on rank {rank}.")

    ckpt_dir = f'../checkpoint/3-mixed-768d/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model = get_gpt(rank, start_epoch, start_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, ckpt_dir)
    model.train()
    print(f'rank {rank} model ok. params: {sum(p.numel() for p in model.parameters())}')
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print(f'rank {rank} data ok.')
    breakpoint()

    loss_fn = nn.CrossEntropyLoss(ignore_index = vocab_size - 1).to(rank)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)  # 3e-4 才好，别的 1e-4 也行，2e-4一般。其他全都 train 不动。
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 150000)

    for epoch in range(start_epoch, 1):
        for batch, (source, target, _) in enumerate(dataloader):
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
                with record_function("model_training"):
                    source = source.to(rank)
                    target = target.to(rank)
                    # domain_label = domain_label.to(rank)
                    optimizer.zero_grad()

                    output = model(source) # [bs, n, 768]
                    loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
            print(prof.key_averages())#.table(sort_by="self_cuda_flops", row_limit=10))
            breakpoint()

            batch_idx = batch + 1 + start_batch
            print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}')
            if (batch_idx) % save_every_batch == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{batch_idx}.pth')
            
            # cal rank for FFN matrix
        torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{batch_idx}.pth')


def test_gpt(rank, start_epoch, start_batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, task_name):
    print(f"Running test on rank {rank}.")

    ckpt_dir = f'../checkpoint/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    dataloader = DataLoader(dataset, batch_size=40)
    for test_batch in range(start_batch, 20000 + 1, 250):
        model = get_gpt(rank, start_epoch, test_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, ckpt_dir)
        model.eval()

        print(f'rank {rank} data ok.')

        metric_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=vocab_size-1).to(rank)
        perp = []
        data_idx = 0
        with torch.no_grad():
            for batch, (source, target, _) in enumerate(dataloader):
                source = source.to(rank)
                target = target.to(rank)

                output = model(source)

                batch_perp = []
                for i in range(output.size(0)):
                    loss = metric_fn(output[i].reshape([-1, vocab_size]), target[i].reshape([-1]))
                    loss = loss.exp()
                    perp.append(loss.item())
                    batch_perp.append(loss.item())
                    data_idx += 1

        print(f'test_batch: {test_batch}, test data: {len(perp)} all perplexity: {sum(perp) / len(perp):.2f}')

    return sum(perp)/len(perp),


def draw_grad_consis(rank, dataset):
    dataloader = DataLoader(dataset, batch_size=24)
    metric_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=50256)

    label2 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 0, 1, 2, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 2, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 0, 1, 2, 2, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 1, 2, 2, 2, 2, 0, 1, 0, 2, 0, 2, 1, 1, 2, 0, 2, 1, 1, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 0, 0, 0, 1, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 0, 1, 0, 1])

    common_consist, long_consist, longlong_consist = [], [], []

    for test_batch in range(0, 150000 + 1, 1500):
        model = get_gpt(rank, 0, test_batch, 50257, 12, 768, 64, 256, '../checkpoint/3-mixed-768d/baseline')
        model.requires_grad_(False)
        model.decoders[10].feed_fwd.feed_fwd[2].weight.requires_grad_(True)

        data_idx = 0
        common_grad, long_grad, longlong_grad = [], [], []

        for batch, (source, target, _) in enumerate(dataloader):
            if batch > 512 // 24 - 1:
                break
            source = source.to(rank)
            target = target.to(rank)
            output = model(source)

            for i in range(24):
                loss = metric_fn(output[i].view(-1, 50257), target[i].view(-1))
                loss.backward(retain_graph=True)
                grad = model.decoders[10].feed_fwd.feed_fwd[2].weight.grad.detach().cpu().numpy()
                if label2[data_idx] == 0:
                    common_grad.append(grad)
                elif label2[data_idx] == 1:
                    long_grad.append(grad)
                else:
                    longlong_grad.append(grad)
                data_idx += 1
                model.zero_grad()

        cg, lg, llg, bg = sum(common_grad) / len(common_grad), sum(long_grad) / len(long_grad), sum(longlong_grad) / len(longlong_grad), sum(common_grad + long_grad + longlong_grad) / (len(common_grad) + len(long_grad) + len(longlong_grad))
        np.save(f'../figure_data/common_grad_{test_batch}.npy', cg)
        np.save(f'../figure_data/long_grad_{test_batch}.npy', lg)
        np.save(f'../figure_data/longlong_grad_{test_batch}.npy', llg)
        np.save(f'../figure_data/batch_grad_{test_batch}.npy', bg)

        common_consist.append(np.dot(cg.flatten(), bg.flatten()) / (np.linalg.norm(cg) * np.linalg.norm(bg)))
        long_consist.append(np.dot(lg.flatten(), bg.flatten()) / (np.linalg.norm(lg) * np.linalg.norm(bg)))
        longlong_consist.append(np.dot(llg.flatten(), bg.flatten()) / (np.linalg.norm(llg) * np.linalg.norm(bg)))

        print(f'batch: {test_batch}, common: {common_consist[-1]}, long: {long_consist[-1]}, longlong: {longlong_consist[-1]}')

    plt.plot(common_consist, label='common')
    plt.plot(long_consist, label='long')
    plt.plot(longlong_consist, label='longlong')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('grad similarity')
    plt.title('grad similarity of different data')
    plt.savefig('../figure_data/grad_consist.png')
    plt.close()


def cal_model_rank(model, epoch, batch, th = 1.5e-1):

    for layer, decoder in enumerate(model.decoders):
        _,s,_ = torch.svd(decoder.feed_fwd.feed_fwd[0].weight)
        s = s / s.max()
        fw1_rank = (s > th).sum()
        _,s,_ = torch.svd(decoder.feed_fwd.feed_fwd[2].weight)
        s = s / s.max()
        fw2_rank = (s > th).sum()
        print(f'epoch: {epoch}, batch: {batch}, layer: {layer}, fw1_rank: {fw1_rank}, fw2_rank: {fw2_rank}')

