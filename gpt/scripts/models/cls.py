import torch
import numpy as np
import torch.nn as nn


class LMClassifier(nn.Module):
    def __init__(self, model, embed_dim, class_nums):
        super(LMClassifier, self).__init__()
        self.model = model
        if isinstance(class_nums, int):
            class_nums = [class_nums]
        self.classifier = nn.ModuleList([nn.Linear(embed_dim, class_num) for class_num in class_nums])

    def forward(self, x):
        x, task_indices = x

        logits = []
        for seq, task_index in zip(x, task_indices):
            for first_eot in range(0, len(seq)):
                if seq[first_eot] == 50256:
                    break
            seq = seq[:first_eot]
            seq = seq.unsqueeze(0)
            with torch.no_grad():
                emb = self.model.get_decoder_output(seq, 11)

            emb = emb.mean(dim=1)
            logit = self.classifier[task_index](emb)
            logits.append(logit)

        logits = torch.cat(logits, dim=0)
        return logits


def get_cls_model(device, embed_dim, class_nums, LMmodel, epoch = 0, batch = 0, model_name = '') -> LMClassifier:
    # print(f'unique_dims {unique_dims} with magic_arg {magic_arg}')

    LMclassifier = LMClassifier(LMmodel, embed_dim, class_nums)

    if epoch > 0:
        LMclassifier.load_state_dict(torch.load(f'../checkpoint/{model_name}/epoch{epoch}.pth', map_location='cpu'))

    LMclassifier.to(device)
    return LMclassifier

