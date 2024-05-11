import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R


@R.register("datasets.kcat")
class KcatDataset(torch_data.Dataset, core.Configurable):
    def __init__(self, path, mode, fold, data_name, label_name, 
                 feat_type='onehot', prot_model_name=None, mol_model_name=None, prot_model_path=None, mol_model_path=None, 
                 max_length=None, padding_mode='max_length'):
        self.path = path    
        self.mode = mode
        self.fold = fold
        self.feat_type = feat_type
        self.max_length = max_length
        self.data_name = data_name
        self.label_name = label_name
        self.padding_mode = padding_mode
        self.prot_model_name = prot_model_name
        self.mol_model_name = mol_model_name
        self.ps_featurize = data.ProteinSequence(lm_model_name=prot_model_name, model_path=prot_model_path)
        self.mol_featurize = data.MoleculeSequence(lm_model_name=mol_model_name, model_path=mol_model_path)

        if self.feat_type == 'language_model':
            if self.prot_model_name is None or self.mol_model_name == '':
                raise ValueError('model_name must be provided for language_model feature type')   
            self.ps_pad_token_id = self.ps_featurize.pad_token_id
            self.mol_pad_token_id = self.mol_featurize.pad_token_id
        
        df = pd.read_csv(self.path)
        
        if self.mode == 'train':
            self.df = df[df['split'] == 'train'].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['split'] == 'valid'].reset_index(drop=True)
        elif self.mode == 'test':
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
        elif self.mode == 'infer':
            self.df = df
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
                
        print(f'{mode} dataset size: {len(self.df)}')

    def __getitem__(self, index):    
        row = self.df.iloc[index]
        prot_seq = row['seq']
        mol_seq = row['smile']
        y = row[self.label_name]
        
        data_dcit = {}
        if self.feat_type == 'onehot':
            data = self.ps_featurize.featurize(prot_seq)
            data = torch.FloatTensor(data)
        else:
            # if self.max_length:
            #     seq = seq[:self.max_length]
            prot_data = self.ps_featurize.featurize(prot_seq)
            data_dcit['prot_input_ids'] = prot_data['input_ids']
            data_dcit['prot_attention_mask'] = prot_data['attention_mask']
            mol_data = self.mol_featurize.featurize(mol_seq)
            data_dcit['mol_input_ids'] = mol_data['input_ids']
            data_dcit['mol_attention_mask'] = mol_data['attention_mask']
        
            data_dcit['prot_pad_token_id'] = self.ps_pad_token_id
            data_dcit['mol_pad_token_id'] = self.mol_pad_token_id 
            
                
        return data_dcit, y

    def __len__(self):
        return len(self.df)
