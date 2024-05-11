from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers
from zbiosynth.layers import functional
from zbiosynth.core import Registry as R
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel, EsmForProteinFolding

@R.register("models.Esm2Model")
class Esm2Model(nn.Module, core.Configurable):
    def __init__(self, model_name, model_path, freeze_backbone=False):
        super(Esm2Model, self).__init__()
        
        model_names = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3B", "esm2_15B"]
        if model_name not in model_names:
            raise ValueError(f"model_name {model_name} not supported")
        
        self.esm = EsmModel.from_pretrained(f"{model_path}/prot_LM/{model_name}/")
        self.output_dim = self.esm.config.hidden_size
        
        # freeze
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
        
    
    def forward(self, batch, all_loss=None, metric=None):
        outputs = {}
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output = self.esm(input_ids, attention_mask=attention_mask)
        pool_feat = output.pooler_output
        outputs['feature'] = pool_feat
        
        return outputs
        # output = self.dropout(pool_feat)
        # output = self.fc(output)
        # return output
        
        
@R.register("models.Esm2DDG_Model")
class Esm2DDG_Model(nn.Module, core.Configurable):
    def __init__(self, model_name, model_path, freeze_backbone=None):
        super(Esm2DDG_Model, self).__init__()
        
        self.freeze_backbone = freeze_backbone
        model_names = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3B", "esm2_15B", "esmfold_v1"]
        if model_name not in model_names:
            raise ValueError(f"model_name {model_name} not supported")
        self.model_name = model_name
        if self.model_name != 'esmfold_v1':
            self.esm = EsmModel.from_pretrained(f"{model_path}/prot_LM/{model_name}/")
            self.output_dim = self.esm.config.hidden_size * 3
        else:
            self.esm = EsmForProteinFolding.from_pretrained(f"{model_path}/prot_LM/{model_name}/")
            self.esm.esm = self.esm.esm.half()
            self.esm.trunk.set_chunk_size(64)
            self.output_dim = 1024 * 3
        # freeze
        if self.freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
            self.esm = self.esm.eval()
        
    def forward(self, batch, all_loss=None, metric=None):
        outputs = {}
        
        if self.freeze_backbone:
            with torch.no_grad():
                wt_input_ids, wt_attention_mask = batch['wt_input_ids'], batch['wt_attention_mask']
                wt_output = self.esm(wt_input_ids, attention_mask=wt_attention_mask)
                mut_input_ids, mut_attention_mask = batch['mut_input_ids'], batch['mut_attention_mask']
                mut_output = self.esm(mut_input_ids, attention_mask=mut_attention_mask)
        else:
            wt_input_ids, wt_attention_mask = batch['wt_input_ids'], batch['wt_attention_mask']
            wt_output = self.esm(wt_input_ids, attention_mask=wt_attention_mask)
            mut_input_ids, mut_attention_mask = batch['mut_input_ids'], batch['mut_attention_mask']
            mut_output = self.esm(mut_input_ids, attention_mask=mut_attention_mask)
            
            
        if self.model_name != 'esmfold_v1':
            wt_feat = wt_output.last_hidden_state[:, 1:-1,:]
            mut_feat = mut_output.last_hidden_state[:, 1:-1,:]
        else:
            wt_feat = wt_output['s_s']
            mut_feat = mut_output['s_s']
        
        ## mutation position
        wt_feat = torch.gather(wt_feat, 1, batch['position'].view(-1, 1, 1).expand(-1, -1, wt_feat.size(2)))
        mut_feat = torch.gather(mut_feat, 1, batch['position'].view(-1, 1, 1).expand(-1, -1, mut_feat.size(2)))
        wt_feat = wt_feat.squeeze(1)
        mut_feat = mut_feat.squeeze(1)
        
        feat = torch.cat([wt_feat, mut_feat, wt_feat - mut_feat], dim=-1)
        outputs['feature'] = feat
        return outputs