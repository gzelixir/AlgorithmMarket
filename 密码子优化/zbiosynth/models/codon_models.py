from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers
from zbiosynth.layers import functional
from zbiosynth.core import Registry as R
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel



@R.register("models.CodonOptModel")
class CodonOptModel(nn.Module, core.Configurable):
    def __init__(self, model_name, model_path, dropout=0.5, freeze_backbone=False):
        super().__init__()
        model_names = ["codonlm_8m", "codonlm_35m", "codonlm_650m"]
        if model_name not in model_names:
            raise ValueError(f"model_name {model_name} not supported")
        else:
            self.codon_lm = EsmModel.from_pretrained(f'{model_path}/codon_LM/{model_name}/')
            
            
        self.output_dim = self.codon_lm.config.hidden_size
            
        # config = self.codon_lm.config
        # self.dropout = nn.Dropout(dropout)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, batch, all_loss=None, metric=None):
        outputs = {}
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output = self.codon_lm(input_ids, attention_mask=attention_mask)
        features = output.last_hidden_state
        outputs['feature'] = features
        
        return outputs
        # features = self.dropout(features)
        # x = self.dense(features)
        # x = self.dropout(x)
        # x = self.decoder(x)
        # return x