from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers
from zbiosynth.layers import functional
from zbiosynth.core import Registry as R
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel, EsmForProteinFolding
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformers import BertModel



@R.register("models.ProtMolModel")
class ProtMolModel(nn.Module, core.Configurable):
    def __init__(self, prot_model_name, mol_model_name, prot_model_path, mol_model_path, freeze_backbone=False):
        super(ProtMolModel, self).__init__()
        
        prot_model_names = ["esm2_8m", "esm2_35m", "esm2_150m", "esm2_650m", "esm2_3B", "esm2_15B"]
        if prot_model_name not in prot_model_names:
            raise ValueError(f"model_name {prot_model_name} not supported")
        
        mol_model_names = ["molt5-base", "molt5-small", "smole-bert"]
        if mol_model_name not in mol_model_names:
            raise ValueError(f"model_name {mol_model_name} not supported")
        
        self.prot_model_name = prot_model_name
        self.mol_model_name = mol_model_name
        
        
        self.prot_model = EsmModel.from_pretrained(f"{prot_model_path}/prot_LM/{prot_model_name}/")
        self.prot_output_dim = self.prot_model.config.hidden_size
                
        if 't5' in mol_model_name:
            self.mol_model = T5EncoderModel.from_pretrained(f"{mol_model_path}/mol_LM/{mol_model_name}/")
            self.mol_output_dim = self.mol_model.config.d_model
            self.mol_pooler = nn.AdaptiveAvgPool1d(1)
        elif 'bert' in mol_model_name:
            self.mol_model = BertModel.from_pretrained(f"{mol_model_path}/mol_LM/{mol_model_name}/")
            self.mol_output_dim = self.mol_model.config.hidden_size
            
            
        self.output_dim = self.prot_output_dim + self.mol_output_dim

        
        if freeze_backbone:
            for param in self.prot_model.parameters():
                param.requires_grad = False
            # for param in self.prot_model.embed_tokens.parameters():
            #     param.requires_grad = True
            
            for param in self.mol_model.parameters():
                param.requires_grad = False
        
    
    def forward(self, batch, all_loss=None, metric=None):
        outputs = {}
        prot_input_ids, prot_attention_mask = batch['prot_input_ids'], batch['prot_attention_mask']
        mol_input_ids, mol_attention_mask = batch['mol_input_ids'], batch['mol_attention_mask']
        prot_output = self.prot_model(prot_input_ids, attention_mask=prot_attention_mask)
        prot_feat = prot_output.pooler_output
        mol_output = self.mol_model(mol_input_ids, attention_mask=mol_attention_mask)
        
        if 't5' in self.mol_model_name:
            mol_feat = mol_output.last_hidden_state[:,:-1,:]
            mol_feat = self.mol_pooler(mol_feat.permute(0, 2, 1)).squeeze(-1)
        elif 'bert' in self.mol_model_name:
            mol_feat = mol_output.pooler_output
            
            
        pool_feat = torch.cat([prot_feat, mol_feat], dim=-1)
        outputs['feature'] = pool_feat
        return outputs
        # output = self.dropout(pool_feat)
        # output = self.fc(output)
        # return output