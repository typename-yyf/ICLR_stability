from contextlib import redirect_stdout
from utils import get_dataloader, get_bert_dataloader, get_dataset, FakeData, CLS_Single_Tokenized_data, Tokenized_data
from trainer import *
from models import get_gpt, get_continue_gpt
import torch

DOMAINS = ['openweb', 'legal', 'med', 'acl', 'review']
CLS_TASKS = [
    'casehold', 'citation_intent', 'climate_sentiment',
    'env_claims', 'FPB', 'mag',
    'netzero_reduction', 'QQP', 'RTE',
    'sciie-re', 'chemprot', 'climate_detect',
    'COLA', 'euadr', 'GAD',
    'GNLI', 'MRPC', 'overruling',
    'rct-20k', 'sci-cite', 'SST2']

CLS_HEAD = [
    2, 6, 3,
    2, 3, 7,
    3, 2, 2,
    7, 13, 2,
    2, 2, 2,
    2, 25, 2,
    5, 3, 2]

def run_training(device, epoch, batch, lr, batch_size, moe, vocab_size, layer_num, embed_dim, heads_num, window_size, domain_count, save_every, args):
    if device == -1:
        return

    if moe == 'cls':
        # dataset = get_cls_dataset(CLS_TASKS, is_equal=True)
        task_datasets = [CLS_Single_Tokenized_data(task, i, window_size, False, 50000) for i, task in enumerate(CLS_TASKS)]
        lmmodel = get_gpt(device, 0, 50000, vocab_size, layer_num, embed_dim, heads_num, window_size, '../checkpoint/big_model/')
        train_cls(device, lmmodel, lr, task_datasets, CLS_HEAD, batch_size, embed_dim, 'big50000_tasks', save_every)
    elif moe == 'none':
        dataset = get_dataset(DOMAINS[0], window_size, is_test=False, from_idx = 0, max_total= 60000)
        train_gpt(device, 0, 0, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, 'baseline', save_every)
    elif moe == 'conti1':
        dataset = get_dataset(DOMAINS[:1], window_size, is_test=False, from_idx = 1200000, max_total= 4800000)
        base_model = get_gpt(device, 0, 30000, vocab_size, layer_num, embed_dim, heads_num, window_size, '../checkpoint/baseline/')
        train_continue_gpt(device, epoch, 30000, epoch + 1, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, 'continue_cluster4_layer6_from20_nofreeze', base_model, moe_at = [6,7,8,9, 10,11], ffn_expert_num = 4, common_pct_threashold = .7, save_every_batch = save_every)
    elif moe == 'conti2':
        dataset = get_dataset(DOMAINS[:1], window_size, is_test=False, from_idx = 4000000, max_total= 2000000)
        basebase_model = get_gpt(device, 0, 62499, vocab_size, layer_num, embed_dim, heads_num, window_size, f'../checkpoint/baseline/')
        base_model = get_continue_gpt(device, 0, 125000, vocab_size, layer_num, embed_dim, heads_num, window_size, None, basebase_model, moe_at = 10, ffn_expert_num = 4, common_pct_threashold = .5,  ckpt_dir = f'../checkpoint/continue_2-of-3/')
        train_continue_gpt(device, epoch, 125000, epoch + 1, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, 'continue_3-of-3', base_model, moe_at = 10, ffn_expert_num = 8, common_pct_threashold = .7, save_every_batch = save_every)
    elif moe == 'vanilla':
        dataset = get_dataset(DOMAINS[:1], window_size, is_test=False, from_idx = 0, max_total= 6000000)
        train_vanilla(device, epoch, batch, lr, dataset, batch_size, vocab_size, layer_num, embed_dim, heads_num, window_size, expert_num=2, moe_at=[1,3,5,7,9,11], task_name='switch_1e-5', save_every_batch = save_every)


def run_test(device, epoch, batch, batch_size, moe, vocab_size, layer_num, embed_dim, heads_num, window_size, domain_count):
    if device == -1:
        return
    if moe == 'cls':
        task_datasets = [CLS_Single_Tokenized_data(task, i, window_size, True, 1000) for i, task in enumerate(CLS_TASKS)]
        lmmodel = get_gpt(device, 0, 50000, vocab_size, layer_num, embed_dim, heads_num, window_size, '../checkpoint/big_model/')
        test_cls(device, lmmodel, task_datasets, CLS_HEAD, batch_size, embed_dim, 'big50000_tasks')
    else:
        dataset = Tokenized_data(window_size, is_test=True)
        if moe == 'none':
            test_gpt(device, epoch, batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, '140Mbaseline')
        elif moe == 'conti1':
            base_model = get_gpt(device, 0, 1250, vocab_size, layer_num, embed_dim, heads_num, window_size, '../checkpoint/140Mbaseline/')
            test_continue_gpt(device, 0, batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, 'continue140M_from1250_4experts_0_lr0.0008', base_model, moe_at=[6,7,8,9,10,11], ffn_expert_num=4)
        elif moe == 'conti2':
            basebase_model = get_gpt(device, 1, 0, vocab_size, layer_num, embed_dim, heads_num, window_size, f'../checkpoint/baseline/')
            base_model = get_continue_gpt(device, 1, 0, vocab_size, layer_num, embed_dim, heads_num, window_size, None, basebase_model, moe_at = 10, ffn_expert_num = 4, common_pct_threashold = .5,  ckpt_dir = f'../checkpoint/continue_2-of-3/')
            test_continue_gpt(device, epoch, batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, 'continue_3-of-3', base_model, moe_at=10, ffn_expert_num=8)
        elif moe == 'vanilla':
            test_vanilla(device, epoch, batch, dataset, vocab_size, layer_num, embed_dim, heads_num, window_size, expert_num=2, moe_at=[1,3,5,7,9,11], task_name='switch_1e-5')


def test_cls_with_load(device, test_epoch, test_batch, task_datasets, task_name):
    lmmodel = get_gpt(device, test_epoch, test_batch, 50257, 12, 768, 12, 256, '../checkpoint/big_model/')
    test_cls(device, lmmodel, task_datasets, CLS_HEAD, 40, 768, task_name)


def myfunc(args):
    print('start myfunc.')
    from multiprocessing import Process
    task_datasets = [CLS_Single_Tokenized_data(task, i, 256, True, 1000) for i, task in enumerate(CLS_TASKS)]
    task_names = [
        'big5000_tasks',
        'big15000_tasks',
        'big30000_tasks',
        'big50000_tasks',
        'big_tasks',
        'big1_tasks',
        'big2_tasks',
        'big3_tasks',
    ]
    epoch_batch    = [
        (0, 5000),
        (0, 15000),
        (0, 30000),
        (0, 50000),
        (0, 65000),
        (1, 65000),
        (2, 65000),
        (3, 65000),
    ]
    devices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    processes = []
    for device, task_name, (test_epoch, test_batch) in zip(devices, task_names, epoch_batch):
        # create a process running train_sparse with arguments above
        out_file = open(f'../logs/test1000_{task_name}.txt', 'w')
        with redirect_stdout(out_file):
            p = Process(target=test_cls_with_load, args=(device, test_epoch, test_batch, task_datasets, task_name))
            p.start()
        processes.append(p)
        print(f'process {p.pid} started with task name {task_name}')

    # wait for all processes to finish
    for p in processes:
        p.join()

    print('all done')



def print_all_args(args):
    print('experiments start with args:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')


import argparse
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # task params
    argparser.add_argument('--task', type=str, default='train')
    argparser.add_argument('--moe', type=str, default='none')  # none, cluster, vanilla, domain
    argparser.add_argument('--device', type=int, default = 0)
    argparser.add_argument('--epoch', type=int, default=0)
    argparser.add_argument('--batch', type=int, default=0)
    argparser.add_argument('--domain', type=int, default=3)
    argparser.add_argument('--batch_size', type=int, default=40)
    argparser.add_argument('--lr', type=float, default=1e-4)

    # model params
    argparser.add_argument('--size', type=str, default='custom') # small, medium, large
    argparser.add_argument('--embed_dim', type=int, default=768)
    argparser.add_argument('--heads_num', type=int, default=12)
    argparser.add_argument('--layer_num', type=int, default=12)

    # const params
    argparser.add_argument('--window_size', type=int, default=256)
    argparser.add_argument('--vocab_size', type=int, default=50257)
    argparser.add_argument('--save_every', type=int, default=10000)

    argparser.add_argument('--local_rank', type=int, default=0)
    argparser.add_argument('--deepspeed_config', '-d', type=str, default='../configs/deepspeed.json')
    # parser = deepspeed.add_config_arguments(argparser)

    args = argparser.parse_args()


    if args.size == 'medium':
        args.embed_dim = 1024
        args.heads_num = 16
        args.layer_num = 24
        # args.batch_size = 16
    elif args.size == 'large':
        args.embed_dim = 1280
        args.heads_num = 20
        args.layer_num = 36
        args.batch_size = 4
    elif args.size == 'small':
        args.embed_dim = 768
        args.heads_num = 12
        args.layer_num = 12
        args.batch_size = 32
    elif args.size == 'xlarge':
        args.embed_dim = 1600
        args.heads_num = 25
        args.layer_num = 48
        args.window_size = 1024
    elif args.size == 'custom':
        pass


    if args.device == -1:
        args.device = 'cpu'

    if args.task == 'train':
        run_training(args.device, args.epoch, args.batch, args.lr, args.batch_size, args.moe, args.vocab_size, args.layer_num, args.embed_dim, args.heads_num, args.window_size, args.domain, args.save_every, args)
    elif args.task == 'test':
        run_test(args.device, args.epoch, args.batch, args.batch_size, args.moe, args.vocab_size, args.layer_num, args.embed_dim, args.heads_num, args.window_size, args.domain)
    elif args.task == 'myfunc':
        myfunc(args)



# 从别的地方copy过来
# rsync -r 10.176.34.112:/home/gjzhao/workspace2023/soundstream-zgj/source_separation/experiment/origin_vq_vae /home/gjzhao/workspace2023/soundstream-zgj/source_separation/experiment/

# matplotlib scikit-learn tiktoken transformers

# https://drive.usercontent.google.com/download?id=1OGHyJrkaVpbrdbmxsDotG-tI3LiKyxuC&export=download&authuser=0&confirm=t&uuid=fb78d8db-1850-4df2-8adb-d9491811d46a&at=APZUnTXCszW0HYgkSTIJ2jJRjr0k%3A1708662677992

#467456 tensorboard

