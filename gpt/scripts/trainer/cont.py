import torch
from models import get_continue_gpt, get_gpt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup

def train_continue_gpt(rank, start_epoch, start_batch, end_epoch, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, task_name, base_model, moe_at, ffn_expert_num, common_pct_threashold, save_every_batch = 128):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Run training on rank {rank}.")

    ckpt_dir = f'../checkpoint/3-mixed-768d/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model = get_continue_gpt(rank, -1, start_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, dataset, base_model, moe_at, ffn_expert_num, common_pct_threashold, ckpt_dir)
    model.train()

    print(f'rank {rank} model ok. params: {sum(p.numel() for p in model.parameters())}, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'rank {rank} data ok.')

    loss_fn = nn.CrossEntropyLoss(ignore_index = vocab_size - 1).to(rank)

    # include only trainable params to optimizer
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)  # 3e-4 才好，别的 1e-4 也行，2e-4一般。其他全都 train 不动。
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 3000, 150000)

    for epoch in range(start_epoch, end_epoch):
        for batch, (source, target, _) in enumerate(dataloader):
            # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("model_training"):
            source = source.to(rank)
            target = target.to(rank)
            # domain_label = domain_label.to(rank)
            optimizer.zero_grad()

            output = model(source) # [bs, n, 768]
            loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # print(prof.key_averages())
            # breakpoint()

            batch_idx =  batch + 1 + start_batch
            print(f'epoch: {epoch}, batch: {batch_idx}, loss: {loss.item()}')
            if (batch_idx) % save_every_batch == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{batch_idx}.pth')
                
        torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{batch_idx}.pth')


def test_continue_gpt(rank, start_epoch, start_batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, task_name, base_model, moe_at, ffn_expert_num):
    # print(f"Running test on rank {rank}.")

    ckpt_dir = f'../checkpoint/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    dataloader = DataLoader(dataset, batch_size=40, shuffle=False)

    for test_batch in range(start_batch, 20000, 250):
        model = get_continue_gpt(rank, start_epoch, test_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, None, base_model, moe_at, ffn_expert_num, .5, ckpt_dir)
        model.eval()

        print(f'rank {rank} model ok.')

        metric_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=vocab_size-1).to(rank)
        data_idx = 0
        perp = []
        with torch.no_grad():
            for batch, (source, target, _) in enumerate(dataloader):
                source = source.to(rank)
                target = target.to(rank)

                output, label = model(source, need_label = False)

                # label = np.array(label).sum(axis=0) #[batch_size]
                for output_, target_, label_ in zip(output, target, label):
                    output_ = output_.reshape([-1, vocab_size])
                    target_ = target_.reshape([-1])
                    loss = metric_fn(output_, target_)
                    loss = loss.exp()
                    perp.append(loss.item())
                    data_idx += 1


        print(f'test_batch: {test_batch}, test data: {len(perp)}, perplexity: {sum(perp)/len(perp):.2f}')

    return 


