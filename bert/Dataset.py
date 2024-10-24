import os
import multiprocessing

from datasets import load_dataset, load_from_disk, DatasetDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class Wikitext():
    def group_texts(self, examples):
        block_size = self.block_size

        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    def preprocess(self, config, path):
        num_proc = multiprocessing.cpu_count() // 2

        raw_datasets = load_dataset('wikitext', config.dataset_name)
        tokenized_datasets = raw_datasets.map(lambda dataset: self.tokenizer(dataset['text']), batched=True, num_proc=num_proc, remove_columns=["text"])
        lm_dataset = tokenized_datasets.map(self.group_texts, batched=True)
        lm_dataset.save_to_disk(path)
        return lm_dataset

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        #self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        path = os.path.join(config.dataset_cache[config.dataset_name], str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
        lm_datasets = load_from_disk(path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.train_loader = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.train_loader_full = DataLoader(lm_datasets['train'], batch_size=1, shuffle=False, collate_fn=data_collator)
        self.train_loader_hessian = DataLoader(lm_datasets['train'], batch_size=20, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(lm_datasets['validation'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.test_loader = DataLoader(lm_datasets['test'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        
class SST2:
    def preprocess(self, config, path):
        # Load dataset from TSV file
        train_file = '/home/mychen/ER_TextSpeech/BERT/data/datasets/SST-2/train.tsv'
        val_file = '/home/mychen/ER_TextSpeech/BERT/data/datasets/SST-2/dev.tsv'
                
        datasets = DatasetDict({
            'train': load_dataset('csv', data_files=train_file, delimiter='\t', column_names=['sentence', 'label'])['train'],
            'validation': load_dataset('csv', data_files=val_file, delimiter='\t', column_names=['sentence', 'label'])['train'],
        })
                
        column_names = ['sentence']
        
        def tokenize_function(examples):
            # Tokenize the texts
            return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.block_size)

        tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=column_names)
        tokenized_datasets.save_to_disk(path)

        return tokenized_datasets

    def __init__(self, config):
        self.block_size = config.seq_len
        self.batch_size = config.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('/home/mychen/ER_TextSpeech/BERT/pretrained/tokenizer/roberta-base')


        path = os.path.join("/home/mychen/ER_TextSpeech/BERT/data/datasets/tokenized/SST-2", str(self.block_size))
        if not config.preprocessed:
            self.preprocess(config, path)
            
        lm_datasets = load_from_disk(path)        
        train_data = lm_datasets['train']
        val_data = lm_datasets['validation']
        
        # Create PyTorch DataLoader
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        def collate_fn(batch):            
            labels = torch.tensor([int(item['label']) for item in batch])
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])            
            batch_new = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
            return batch_new        
        
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        # self.train_loader_unshuffle = DataLoader(lm_datasets['train'], batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)