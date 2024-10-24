import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from utils import get_dataset, Tokenized_data
from models import get_gpt
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


class hook:
    def __init__(self, writer: SummaryWriter, name: str, mname: str) -> None:
        self.step = 0
        self.writer = writer
        self.name = name
        self.mname = mname
    
    @torch.no_grad()
    def __call__(self, out, res, out_add_res):
        if self.step % 10 == 0:
            # print("====", out.shape)
            s_out = torch.linalg.svdvals(out[0])
            s_res = torch.linalg.svdvals(res[0])
            s_out_add_res = torch.linalg.svdvals(out_add_res[0])
            
            self.writer.add_scalar(f"{self.mname}_out_erank/{self.name}", (s_out ** 2).sum() / s_out[0] ** 2)
            self.writer.add_scalar(f"{self.mname}_res_erank/{self.name}", (s_res ** 2).sum() / s_res[0] ** 2)
            self.writer.add_scalar(f"{self.mname}_out+res_erank/{self.name}", (s_out_add_res ** 2).sum() / s_out_add_res[0] ** 2)

@torch.no_grad()
def reset_params(model: nn.Module, temp=1.0, method="softmax"):
    def smooth(m, s, scale_factor=0.25):
        if m == "maxeig":
            top_index = s.size()[0] // 20
            for i in range(top_index):
                s[i] = s[top_index]
        # scale_factor = 1 / s[0]
        return s * scale_factor
    for name, p in model.named_parameters():
        if "weight" in name and not ("ln" in name) and not ("emb" in name) and not ("fc" in name):
            
            u, s, v = torch.linalg.svd(p.data, full_matrices=False)
            # s *= 0.2
            # ss = torch.sum(s) / 5
            s = smooth(method, s)

            p.data = u @ torch.diag(s) @ v

DEVICE = [3]
BATCH_SIZE = 32
LR = 2e-3
PORT = "12348"

TAG="bit_wo_warmup_baseline"
WINDOW_SIZE = 256
VOCAB_SIZE = 50257
LAYER_NUM = 12 # 12 36
EMBED_DIM = 768 # 768 1280
HEADS_NUM = 12 # 12 20
TASK_NAME = f'lr={LR}_{TAG}' # 会在 ../checkpoint 下面产生一个同名文件夹保存 checkpoint
RESET = False

def main(rank, world_size):
    if rank == 0:
        writer = SummaryWriter(f"/home/mychen/Stability/gpt/log/lr={LR}_tag_{TAG}")
        
    print(f'rank: {DEVICE[rank]}, world_size: {world_size}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 设定设备
    device = f'cuda:{DEVICE[rank]}'
    dataset = Tokenized_data(WINDOW_SIZE, is_test=False) # get_dataset('openweb', WINDOW_SIZE, is_test=False) # 21849600 = 68280 batch  273120
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) #sampler=sampler)
    print(f'Data ok on device {DEVICE[rank]}.')

    model = get_gpt(device, 0, 0, VOCAB_SIZE, LAYER_NUM, EMBED_DIM, HEADS_NUM, WINDOW_SIZE, ckpt_dir, bit = True)
    # model = DDP(model, device_ids=[DEVICE[rank]])
    model.train()
    
    if rank == 0:
        for name, m in model.named_modules():
            if hasattr(m, "get_attn_output_hook"):
                m.get_attn_output_hook = hook(writer, name, "attn")
                m.get_ffn_output_hook = hook(writer, name, "ffn")
    
    print(f'Model ok on device {DEVICE[rank]}. params: {sum(p.numel() for p in model.parameters())}')

    loss_fn = nn.CrossEntropyLoss(ignore_index = VOCAB_SIZE - 1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=.1)  # 3e-4 才好，别的 1e-4 也行，2e-4一般。其他全都 train 不动。
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 200000) # 261120 total steps 1000
    


    his_gradnorm = 0
    reset_token = 0

    for epoch in range(0, 1):
        # sampler.set_epoch(epoch)
        for batch, (source, target, _) in enumerate(dataloader, 1):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(source) # [bs, n, 768]
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            loss.backward()
            
            # g = 0
            # for p in model.parameters():
            #     if not (p.grad is None):
            #         g += p.grad.norm().item()
            # if rank == 0:
            #     writer.add_scalar("grad_norm", g, batch)
            
            # if batch > 50 and (g > his_gradnorm * 2.5) and reset_token <= 0 and RESET:
            #     reset_params(model, 0.25, "maxeig")
            #     reset_token = 30
            #     print("reset params")
                
            #     if rank == 0:
            #         writer.add_text("reset_log", f"g:{g}, his_g:{his_gradnorm}", batch)
            #     continue
            # else:
            #     alpha = 0.1
            #     his_gradnorm = (1 - alpha) * his_gradnorm + alpha * g
            
            if rank == 0:
                writer.add_scalar("train_loss", loss, batch)

            optimizer.step()
            lr_scheduler.step()

            if rank == 0:
                print(f'epoch: {epoch}, batch: {batch}, loss: {loss.item():.3f}')
                if batch % 2500 == 0:
                    torch.save(model.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')
                    torch.save(optimizer.state_dict(), f'{ckpt_dir}/opt_{epoch}_{batch}.pth')
                    torch.save(lr_scheduler.state_dict(), f'{ckpt_dir}/lr_{epoch}_{batch}.pth')
            reset_token -= 1
            
        if rank == 0:
            torch.save(model.module.state_dict(), f'{ckpt_dir}/{epoch}_{batch}.pth')

    dist.destroy_process_group()


ckpt_dir = f'../checkpoint/{TASK_NAME}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# 通常情况下，这个main函数会被一个启动脚本调用，该脚本负责启动多个进程
if __name__ == "__main__":
    
    world_size = DEVICE.__len__()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = PORT
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

