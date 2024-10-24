import torch
import torch.nn as nn
import torch.optim as optim
import os


from models import get_mot_model, TransformerConfig, MoTConfig, DecoderConfig, MoT, ClusterDispatcher, get_cls_model
from utils import get_ckpt_path
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# 跑实验 checklist
# 1. save dir
# 2. magic_arg
# 3. log file
# 4. device

CKPT_INTERVAL = 500

def train_cls(device, lm_model, lr, task_datasets, cls_head_nums, batch_size, embed_dim, task_name, save_every_batch = 128):
    print(f"Running train on GPU {device}.")

    ckpt_dir = f'../checkpoint/{task_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # load cls
    cls_model = get_cls_model(device, embed_dim, cls_head_nums, lm_model, model_name=task_name)
    lm_model.requires_grad_(False)

    # train
    print(f'GPU {device} model ok. params: {sum(p.numel() if p.requires_grad else 0 for p in cls_model.parameters())}')

    cls_loss_fn = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW(cls_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = None # get_linear_schedule_with_warmup(optimizer, 3072, 18717 * 3) # 2700

    for epoch in range(0, 20):
        batch = 1
        for dataset in task_datasets:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            batch = train_one_epoch(epoch, batch, device, cls_model, optimizer, lr_scheduler, dataloader, cls_loss_fn)
        torch.save(cls_model.state_dict(), f'{ckpt_dir}/epoch{epoch}.pth')


def train_one_epoch(epoch, start_batch, device, model, optimizer, lr_scheduler, dataloader, cls_loss_fn):
    for batch, (text, labels, task_idx) in enumerate(dataloader, start=start_batch):
        text = text.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        logits = model((text, task_idx))
        if logits.size(1) < max(labels)+1:
            exit()
        cls_loss = cls_loss_fn(logits, labels)
        
        cls_loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f'epoch: {epoch}, batch: {batch}, cls_loss: {cls_loss.item()}')
    return batch + 1



def test_cls(device, lm_model, task_datasets, cls_head_nums, batch_size, embed_dim, task_name):
    print(f"Running testing on GPU {device}.")

    accs = [0 for _ in range(len(task_datasets))]
    for test_epoch in range(1, 20):
        cls_model = get_cls_model(device, embed_dim, cls_head_nums, lm_model, test_epoch, 0, model_name=task_name)
        cls_model.eval()
        for task_id, dataset in enumerate(task_datasets, 1):
            dataloader = DataLoader(dataset, batch_size=batch_size)
            acc, total = 0, 0
            for b, (text, labels, task_idx) in enumerate(dataloader):
                text = text.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    logits = cls_model((text, task_idx))

                acc += sum(logits.argmax(dim=1) == labels)
                total += len(labels)

            accs[task_id-1] = max(accs[task_id-1], acc / total)
            print(f'epoch {test_epoch} task{task_id}_acc: {acc / total:.4f}')

    print(f'final accs')
    for task_id, acc in enumerate(accs, 1):
        print(f'task{task_id}: {acc:.4f}')
    print(f'avg acc: {sum(accs) / len(accs):.4f}')

