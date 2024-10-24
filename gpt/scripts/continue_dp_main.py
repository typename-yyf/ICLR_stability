import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from utils import get_dataset, Tokenized_data
from models import get_gpt, get_continue_gpt
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup


# 训练参数
DEVICE = [0, 1, 2, 3, 4, 5, 6, 7]
BATCH_SIZE = 40
LR = 8e-4
SAVE_EVERY = 250

# 模型通用参数
WINDOW_SIZE = 256
VOCAB_SIZE = 50257
LAYER_NUM = 12
EMBED_DIM = 768
HEADS_NUM = 12

# baseline model 参数，我们的方法是在这个模型上引入 moe、freeze param 并继续训练
LM_EPOCH = 0  # baseline checkpoint 的 epoch
LM_BATCH = 2750  # baseline checkpoint 的 batch
LM_NAME = '140Mbaseline' # baseline checkpoint 的 task name，与 baseline training 脚本里的 TASK_NAME 一致

# ours model 额外参数
MOE_LAYERS = [6, 7, 8, 9, 10, 11] # 在哪些层引入 moe，目前决定在后一半
FFN_EXPERT_NUM = 4 # 每个 moe layer 的 expert 数量
COMMON_PERCENT = 0 # 多少数据会触发分发，现在是所有数据都会分了所以就设为 0 好了
TASK_NAME = f'continue140M_from{LM_BATCH}_{FFN_EXPERT_NUM}experts_{COMMON_PERCENT}_lr{LR}' # 保存 ours 模型的 task name


def main(rank, world_size):
    print(f'rank: {rank}, world_size: {world_size}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    device = f'cuda:{rank}'
    dataset = Tokenized_data(WINDOW_SIZE, is_test=False, start_from = LM_BATCH * BATCH_SIZE + 1)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=sampler)
    # test_dataset = Tokenized_data(WINDOW_SIZE, is_test=True)
    # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=test_sampler)
    print(f'Data ok on device {rank}.')

    base_model = get_gpt(rank, LM_EPOCH, LM_BATCH, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, f'../checkpoint/{LM_NAME}')
    print(f'Baseline ok on device {rank}.')
    continue_model = get_continue_gpt(rank, -1, 0, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, None, base_model, 
                                      moe_at = MOE_LAYERS, ffn_expert_num = FFN_EXPERT_NUM, common_pct_threashold = COMMON_PERCENT, 
                                      ckpt_dir=f'../checkpoint/{TASK_NAME}')
    continue_model = DDP(continue_model, device_ids=[rank], find_unused_parameters=True)
    continue_model.train()
    print(f'Model ok on device {rank}. Trainable params: {sum(p.numel() for p in continue_model.parameters() if p.requires_grad)}')

    loss_fn = nn.CrossEntropyLoss(ignore_index = VOCAB_SIZE - 1)
    optimizer = optim.AdamW([p for p in continue_model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=.1)  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 100, 18000) # 261120 total steps

    for epoch in range(0, 1):
        sampler.set_epoch(epoch)
        for batch, (source, target, _) in enumerate(dataloader, LM_BATCH + 1):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            output, labels = continue_model(source) # [bs, n, 768]
            if rank == 0:
                pass
                # print(labels)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item():.3f}')
                if batch % SAVE_EVERY == 0:
                    torch.save(continue_model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/opt_{epoch}_{batch}.pth')
                    torch.save(lr_scheduler.state_dict(), f'{ckpt_dir}/lr_{epoch}_{batch}.pth')
                    # perp = []
                    # with torch.no_grad():
                    #     for test_batch, (source, target, _) in enumerate(test_dataloader):
                    #         source = source.to(rank)
                    #         target = target.to(rank)

                    #         output, label = continue_model(source, need_label = False)

                    #         for output_, target_, label_ in zip(output, target, label):
                    #             output_ = output_.reshape([-1, VOCAB_SIZE])
                    #             target_ = target_.reshape([-1])
                    #             loss = loss_fn(output_, target_)
                    #             loss = loss.exp()
                    #             perp.append(loss.item())

                    # print(f'test_batch: {batch}, perplexity: {sum(perp)/len(perp):.2f}')

        if rank == 0:
            torch.save(continue_model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')

    dist.destroy_process_group()



def single_main(rank, world_size):
    print(f'rank: {rank}, world_size: {world_size}')

    device = f'cuda:{rank}'
    dataset = Tokenized_data(WINDOW_SIZE, is_test=False, start_from = LM_BATCH * BATCH_SIZE + 1)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataset = Tokenized_data(WINDOW_SIZE, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f'Data ok on device {rank}.')

    base_model = get_gpt(rank, LM_EPOCH, LM_BATCH, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, f'../checkpoint/{LM_NAME}')
    print(f'Baseline ok on device {rank}.')
    continue_model = get_continue_gpt(rank, -1, 0, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, None, base_model, 
                                      moe_at = MOE_LAYERS, ffn_expert_num = FFN_EXPERT_NUM, common_pct_threashold = COMMON_PERCENT, 
                                      ckpt_dir=f'../checkpoint/{TASK_NAME}')
    continue_model.train()
    print(f'Model ok on device {rank}. Trainable params: {sum(p.numel() for p in continue_model.parameters() if p.requires_grad)}')

    loss_fn = nn.CrossEntropyLoss(ignore_index = VOCAB_SIZE - 1)
    optimizer = optim.AdamW([p for p in continue_model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=.1)  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 2, 18000) # 261120 total steps

    for epoch in range(0, 1):
        for batch, (source, target, _) in enumerate(dataloader, LM_BATCH + 1):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            output, labels = continue_model(source) # [bs, n, 768]
            if rank == 0:
                pass
                # print(labels)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item():.3f}')
                if batch % SAVE_EVERY == 0:
                    torch.save(continue_model.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/opt_{epoch}_{batch}.pth')
                    torch.save(lr_scheduler.state_dict(), f'{ckpt_dir}/lr_{epoch}_{batch}.pth')
                    perp = []
                    with torch.no_grad():
                        for test_batch, (source, target, _) in enumerate(test_dataloader):
                            if test_batch > 1024 // 24:
                                break
                            source = source.to(rank)
                            target = target.to(rank)

                            output, label = continue_model(source, need_label = False)

                            for output_, target_, label_ in zip(output, target, label):
                                output_ = output_.reshape([-1, VOCAB_SIZE])
                                target_ = target_.reshape([-1])
                                loss = loss_fn(output_, target_)
                                loss = loss.exp()
                                perp.append(loss.item())

                    print(f'test_batch: {batch}, perplexity: {sum(perp)/len(perp):.2f}')

        if rank == 0:
            torch.save(continue_model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')



ckpt_dir = f'../checkpoint/{TASK_NAME}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


def cal_cluster_centers():
    # get continue gpt 这个函数在第一次运行的时候，如果对应 task name 的 checkpoint 文件夹下没有 cluster center，就会自动计算并保存
    # 下次再调用的时候就会直接 load 上次的 center 而不是重新计算。所以在多卡训练之前，需要先在单卡上运行这个函数，防止训练的时候多卡上
    # 的进程竞争计算 center 导致错误
    dataset = Tokenized_data(WINDOW_SIZE, is_test=False)
    base_model = get_gpt(1, 0, LM_BATCH, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, f'../checkpoint/{LM_NAME}')
    continue_model = get_continue_gpt(1, -1, 0, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, dataset, base_model, 
                                      moe_at = MOE_LAYERS, ffn_expert_num = FFN_EXPERT_NUM, common_pct_threashold = COMMON_PERCENT, 
                                      ckpt_dir=f'../checkpoint/{TASK_NAME}')
    breakpoint()
    return


# 通常情况下，这个main函数会被一个启动脚本调用，该脚本负责启动多个进程
if __name__ == "__main__":
    # cal_cluster_centers()
    # single_main(0, 1)
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

