import torch.nn as nn
import torchvision
from transformer.Transformer import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

r'''
这个文件定义了试验所用的基础模型，目前仅使用到了bert。
如果需要其他模型，请在此定义。
'''

class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1))

        return mlm_loss, scores

class ResnetForCifar(nn.Module):
    def __init__(self):
        super(ResnetForCifar, self).__init__()
        self.model = torchvision.models.resnet50(pretrained = False)
        inchannel = self.model.fc.in_features 
        self.model.fc = nn.Linear(inchannel, 10)
        
    
