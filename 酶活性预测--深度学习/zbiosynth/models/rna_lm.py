from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers
from zbiosynth.layers import functional
from zbiosynth.core import Registry as R
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel

# @R.register("models.RNALM")
# class RNALM(nn.Module, core.Configurable):
#     def __init__(self, model_name='rnalm_8m'):
#         super(RNALM, self).__init__()
        
#         tokenizer_8m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_8m/")  
#         tokenizer_35m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_35m/")
#         tokenizer_150m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_150m/")
#         tokenizer_650m = EsmTokenizer.from_pretrained(f"/share/liufuxu/zBioSynth/resources/pretrained_weights/RNA_LM/rnalm_650m/")
    
#         if model_name == 'rnalm_8m':
#             model_config = EsmConfig.from_pretrained(f'/share/liufuxu/zBioSynth/resources/pretrained_weights/esm8m_mars50parts/config.json')  
#             state_dict = torch.load(f'/share/liufuxu/zBioSynth/resources/pretrained_weights/esm8m_mars50parts/pytorch_model.bin', map_location='cpu')
#         elif model_name == 'rna_lm_35m':
#             model_config = EsmConfig.from_pretrained(f'/share/liufuxu/zBioSynth/resources/pretrained_weights/esm35m_mars25parts/config.json')
#             state_dict = torch.load(f'/share/liufuxu/zBioSynth/resources/pretrained_weights/esm35m_mars25parts/pytorch_model.bin', map_location='cpu')
#         else:
#             raise ValueError(f"model_name {model_name} not supported")
                
#         model = EsmModel(model_config)
#         self.output_dim = model_config.hidden_size
        
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key.replace('esm.', '')
#             new_state_dict[new_key] = state_dict[key]
            
#         same_key_nums = 0
#         raw_keys = model.state_dict().keys()
#         for k in new_state_dict.keys():
#             if k in raw_keys:
#                 same_key_nums += 1
#         print(f"same key nums: {same_key_nums}")   
    
#         model.load_state_dict(new_state_dict, strict=False)
        
#         self.rna_lm = model
    
#     def forward(self, batch, all_loss=None, metric=None):
#         outputs = {}
#         input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
#         output = self.rna_lm(input_ids, attention_mask=attention_mask)
#         pool_feat = output.pooler_output
#         outputs['feature'] = pool_feat
        
        
#         return outputs
#         # output = self.dropout(pool_feat)
#         # output = self.fc(output)
#         # return output

@R.register("models.RNALM")
class RNALM(nn.Module, core.Configurable):
    def __init__(self, model_name, model_path, freeze_backbone=False):
        super(RNALM, self).__init__()
        
        model_names = ["rnalm_8m", "rnalm_35m", "rnalm_150m", "rnalm_650m"]
        if model_name not in model_names:
            raise ValueError(f"model_name {model_name} not supported")
        
        self.rna_lm = EsmModel.from_pretrained(f'{model_path}/RNA_LM/{model_name}/')  
                
        self.output_dim = self.rna_lm.config.hidden_size

        # freeze
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.rna_lm.parameters():
                param.requires_grad = False
                
    def forward(self, batch, all_loss=None, metric=None):
        
        outputs = {}
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        output = self.rna_lm(input_ids, attention_mask=attention_mask)
        pool_feat = output.pooler_output
        outputs['feature'] = pool_feat
        
        return outputs
        # output = self.dropout(pool_feat)
        # output = self.fc(output)
        # return output