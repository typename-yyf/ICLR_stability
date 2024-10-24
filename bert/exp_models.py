import torch.nn as nn
import torch.optim as optim

r'''
该文件定义了模型相关，每个试验模型需要实现两个方法：train与init_model，
train用于训练模型，init_model用于初始化一些超参
具体实例查看bert_re.py文件
'''

class exp_models:
    def _unfinished_method_train(self, *args, **largs):
        raise Exception("exp_models: Unfinished method 'train'")
    def _unfinished_method_init_model(self, *args, **largs):
        raise Exception("exp_models: Unfinished method 'init_model'")
    
    _base_model: nn.Module = None
    _dataset = None
    _name: str = None
    
    _optimizer: optim
    _lr_scheduler = None
    _train_loader = None
    _val_loader   = None
    _test_loader  = None
    
    train: callable = _unfinished_method_train
    init_model: callable = _unfinished_method_init_model
    
    def __init__(self, base_model: nn.Module):
        self._base_model = base_model
    
    
    