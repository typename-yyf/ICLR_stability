import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from utils import get_dataset, Tokenized_data
from models import get_vanilla_model
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup


DEVICE = [0, 1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 40
LR = 4e-4

WINDOW_SIZE = 256
VOCAB_SIZE = 50257
LAYER_NUM = 12
EMBED_DIM = 768
HEADS_NUM = 12
EXPERT_NUM = 2
MOE_AT = [1, 3, 5, 7, 9, 11]
TASK_NAME = 'switch140M'


def main(rank, world_size):
    print(f'rank: {rank}, world_size: {world_size}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 设定设备
    device = f'cuda:{rank}'
    dataset = Tokenized_data(WINDOW_SIZE, is_test=False) # get_dataset('openweb', WINDOW_SIZE, is_test=False) # 21849600 = 68280 batch  273120
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=sampler)
    print(f'Data ok on device {rank}.')

    model = get_vanilla_model(device, 0, 0, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, EXPERT_NUM, MOE_AT, ckpt_dir)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()
    print(f'Model ok on device {rank}. params: {sum(p.numel() for p in model.parameters())}')

    loss_fn = nn.CrossEntropyLoss(ignore_index = VOCAB_SIZE - 1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=.1)  # 3e-4 才好，别的 1e-4 也行，2e-4一般。其他全都 train 不动。
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 500, 20000) # 261120 total steps

    for epoch in range(0, 1):
        sampler.set_epoch(epoch)
        for batch, (source, target, _) in enumerate(dataloader, 1):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(source) # [bs, n, 768]
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item():.3f}')
                if batch % 250 == 0:
                    torch.save(model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/opt_{epoch}_{batch}.pth')
                    torch.save(lr_scheduler.state_dict(), f'{ckpt_dir}/lr_{epoch}_{batch}.pth')

        if rank == 0:
            torch.save(model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')

    dist.destroy_process_group()


ckpt_dir = f'../checkpoint/{TASK_NAME}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# 通常情况下，这个main函数会被一个启动脚本调用，该脚本负责启动多个进程
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

