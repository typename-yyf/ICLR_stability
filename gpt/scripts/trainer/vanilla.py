import torch
from models import get_vanilla_model
import torch.nn as nn
import os
import torch.optim as optim

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

def train_vanilla(rank, start_epoch, start_batch, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, expert_num, moe_at, task_name, save_every_batch = 128):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Run training on rank {rank}.")

    ckpt_dir = f'../checkpoint/3-mixed-768d/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    print(f'rank {rank} data ok.')
    model = get_vanilla_model(rank, start_epoch, start_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, expert_num, moe_at, ckpt_dir)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index = vocab_size - 1).to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 150000)

    print(f'rank {rank} model ok. params: {sum(p.numel() for p in model.parameters())}')

    for epoch in range(start_epoch, 1):
        for batch, (source, target, domain_label) in enumerate(dataloader):
            source = source.to(rank)
            target = target.to(rank)
            domain_label = domain_label.to(rank)
            optimizer.zero_grad()

            output = model(source)
            loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_batch = batch + 1 + start_batch
            print(f'epoch: {epoch}, batch: {train_batch}, loss: {loss.item()}')
            if train_batch % save_every_batch == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{train_batch}.pth')


def test_vanilla(rank, start_epoch, start_batch, dataloader, vocab_size, layer_num, embed_dim, heads_num, window_size, domain_count):
    # 会自动传入一个参数比如 rank 表示第几个进程
    print(f"Running test on rank {rank}.")

    model = get_vanilla_model(rank, start_epoch, start_batch, vocab_size, layer_num, embed_dim, heads_num, window_size, domain_count)
    model.eval()

    print(f'rank {rank} data ok.')

    metric_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=vocab_size-1).to(rank)
    perp = []
    with torch.no_grad():
        for batch, (source, target, domain) in enumerate(dataloader):
            source = source.to(rank)
            target = target.to(rank)
            domain = domain.to(rank)

            output = model(source)

            loss = metric_fn(output.reshape([-1, vocab_size]), target.reshape([-1]))
            loss = loss.exp()
            perp.append(loss.item())

            print(f'batch: {batch}, perplexity: {loss.item()}')

    print(f'rank: {rank}, perplexity: {sum(perp)/len(perp)}')

    return sum(perp)/len(perp)



