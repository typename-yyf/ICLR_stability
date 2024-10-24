import torch.nn as nn
import exp_models
import base_models
from transformers import BertConfig
from Dataset import Wikitext
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
import transformer.Transformer
import math
import torch
import numpy as np
import random
import pickle
import os
from torch.cuda.amp import autocast, GradScaler

from log_util import *


import argparse

import matplotlib.pyplot as plt

from collections import OrderedDict
#from file_writer import file_writer

@torch.no_grad
def reset_params(model: nn.Module, temp=1.0, method="softmax"):
    def smooth(m, s, scale_factor=1.0):
        if m == "softmax":
            s = softmax(s * 0.25) * s.sum()
        elif m == "random":
            s = s[torch.randperm(s.size(0))]
        elif m == "conv":
            s = torch.cat([
                s[1].unsqueeze(-1),
                s,
                s[-2].unsqueeze(-1),
            ])
            
            s = torch.nn.functional.conv1d(
                s.unsqueeze(0),
                torch.ones((1, 1, 3), device=s.device, dtype=s.dtype) / 3,
                padding="valid"
            )
            s = s.flatten()
        elif m == "avg":
            s = torch.ones_like(s, device=s.device)
        # scale_factor = 1 / s[0]
        return s * scale_factor
    
    
    softmax = torch.nn.Softmax(dim=0)
    
    for name, p in model.named_parameters():
        if "weight" in name and "encoder" in name and (not "LayerNorm" in name):
            if "heads" in name:
                
                wa = p.data.view(3, -1, 768)
                for i in range(3):
                    w = wa[i]
                    u, s, v = torch.linalg.svd(w, full_matrices=False)

                    s = smooth(method, s)
                    # s *= 0.2
                    wa[i] = u @ torch.diag(s) @ v
                p.data = wa.view(-1, 768)
            
            else:
                u, s, v = torch.linalg.svd(p.data, full_matrices=False)
                # s *= 0.2
                # ss = torch.sum(s) / 5
                s = smooth(method, s)

                p.data = u @ torch.diag(s) @ v
    print("reset params")

class bert_test(exp_models.exp_models):
    def _save(self, file_path: str):
        bias = self._base_model.head.predictions.decoder.bias
        self._base_model.head.predictions.decoder.bias = torch.nn.Parameter(bias.detach().clone())
        self._accelerator.save_state(file_path)
        
        print(f"checkpoint saved at {file_path}")
        
    def _load(self, file_path: str):
        torch.cuda.empty_cache()
        
        self._accelerator.load_state(file_path)
        
        self._optimizer = optim.AdamW(
            self._base_model.parameters(), 
            lr=self._args.lr, 
            weight_decay=0.01, 
            betas=[0.9, 0.999], 
            eps=1e-6
        )

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=self._args.warmup_steps,
            num_training_steps=self._args.train_steps
        )
        
        print(f"load from checkpoint {file_path}")

    def __init__(self, args):
        config = BertConfig.from_json_file(args.config)
        
        self._base_model   = base_models.BertForMLM(config=config)
        self._dataset      = Wikitext(config=config)
        
        self._writer       = SummaryWriter(f"{args.log_dir}/{args.tag}")
        
        self._train_loader = self._dataset.train_loader
        self._test_loader  = self._dataset.test_loader
        self._grad_scaler = GradScaler()
        self._args = args
        self._accelerator  = Accelerator()
    
        if not os.path.exists(f"{args.log_dir}/{args.tag}_t"):
            os.makedirs(f"{args.log_dir}/{args.tag}_t")
    
    def init_model(self, pth_path: str="") -> None:    
        self._train_steps = self._args.train_steps

        self._optimizer = optim.AdamW(
            self._base_model.parameters(), 
            lr=self._args.lr, 
            weight_decay=0.01, 
            betas=[0.9, 0.999], 
            eps=1e-6
        )

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=self._args.warmup_steps,
            num_training_steps=self._args.train_steps
        )

        
            
        self._base_model, self._optimizer, self._lr_scheduler, \
            self._train_loader, self._val_loader, self._test_loader, self._grad_scaler= \
        self._accelerator.prepare(
            self._base_model, 
            self._optimizer, 
            self._lr_scheduler, 
            self._train_loader, 
            self._val_loader, 
            self._test_loader,
            self._grad_scaler
            
        )
        
        if pth_path != "":
            self._load(pth_path)
        
    
    def train(self) -> None:

        steps = 0
        
        loglist = set()
        for name, p in self._base_model.named_parameters():
            if ("layers.0" in name) or ("layers.11" in name) or ("layers.3" in name) or ("layers.7" in name):
                if ("heads.0" in name) or ("dense_1.weight" in name) or ("attention.dense.weight" in name):
                    loglist.add(name)
        for i in (0, 4, 7, 11):
            self._base_model.bert.encoder.layers[i].attention.self.attention_hook_after_qk = \
                attention_after_qk_hook(self._writer, f"layer_{i}")
            
        
        
        while steps < self._train_steps:
                
            self._base_model.train()
                
            for i, batch in enumerate(self._train_loader):
                
                
                with autocast(enabled=self._args.amp, dtype=torch.bfloat16):
                    loss, _ = self._base_model(**batch)
                    self._optimizer.zero_grad()                    
                    
                if self._args.amp:
                    self._grad_scaler.scale(loss).backward()
                    self._grad_scaler.unscale_(self._optimizer)
                    self._grad_scaler.step(self._optimizer)
                    self._grad_scaler.update()
                
                else:
                    loss.backward()
                    self._optimizer.step()                 
                
                if steps % 20 == 0:
                    if steps % 500 == 0:
                        log_condition(steps, self._base_model, self._writer, loglist)
                    else:
                        log_condition(steps, self._base_model, self._writer, loglist, loghist=False)
                
                self._lr_scheduler.step()
                
                
                if self._args.reset_steps > 0 and steps % self._args.reset_steps == 0 and steps > 0:
                    reset_params(self._base_model, 0.25, "conv")
                    
                    if self._args.reset_lr > 0:
                        self._optimizer = optim.AdamW(
                            self._base_model.parameters(), 
                            lr=self._args.reset_lr, 
                            weight_decay=0.01, 
                            betas=[0.9, 0.999], 
                            eps=1e-6
                        )

                        self._lr_scheduler = get_cosine_schedule_with_warmup(
                            optimizer=self._optimizer,
                            num_warmup_steps=25,
                            num_training_steps=8000
                        )
                        
                        self._optimizer, self._lr_scheduler = self._accelerator.prepare(
                            self._optimizer, self._lr_scheduler
                        )
                    
                    print("reset params")
                
                # if steps == self._args.reset_steps:
                #     reset_params(self._base_model, 0.25, "conv")
                    
                # if steps >= 0:
                # if self._args.reset_steps > 0 and steps >= self._args.reset_steps:
                #     if steps < self._args.reset_steps + 30:
                #         for param_group in self._optimizer.param_groups:
                #             param_group['lr'] *= 0.01
                            
                #     else:
                #         for param_group in self._optimizer.param_groups:
                #             param_group['lr'] *= 0.1
                    
                    
                
                self._writer.add_scalar("lr", self._optimizer.param_groups[0]["lr"], steps)
                self._writer.add_scalar("train_loss", loss.item(), steps)
                if steps % 500 == 0:
                    log_test_loss(steps, self._base_model, self._writer, self._test_loader)
                steps += 1
                
                # if steps == 900:
                #     self._save(f"/home/mychen/Stability/bert/checkpoints/step_lr={self._args.lr}_{steps}")
                
                if steps >= self._train_steps:
                    break 

def get_available_cuda_device() -> int:
    max_devs = torch.cuda.device_count()
    for i in range(max_devs):
        try:
            mem = torch.cuda.mem_get_info(i)
        except:
            continue
        if mem[0] / mem[1] > 0.85:
            return i
    return -1

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str)
    parser.add_argument("--log-dir", type=str, default="log")

    parser.add_argument("--train-steps", type=int, default=50000)
    
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=4e-5)
    
    parser.add_argument("--gpu", type=int, default=None)
    
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--reset-steps", type=int, default=-1)
    
    parser.add_argument("--reset-lr", type=float, default=-1)
    
    parser.add_argument("--load-from", type=str, default="")
    
    args = parser.parse_args()

    if args.gpu is None:
        args.gpu = get_available_cuda_device()
    
    if args.tag is None:
        setattr(args, "tag", f"seed_{args.seed}_amp_{args.amp}")
    
    print("################# training args ###################")
    d = dir(args)
    for p in d:
        if not p.startswith("__") and not callable(getattr(args, p)):
            print(f"{p}: {getattr(args, p)}")
    print("###################################################")
    return args

if __name__ == "__main__":
    
    args = parse()
    set_seed(args.seed)
    model = bert_test(args)
    
    torch.cuda.set_device(args.gpu)
    
    model.init_model(args.load_from)
    model.train()