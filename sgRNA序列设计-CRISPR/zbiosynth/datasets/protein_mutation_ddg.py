import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R


@R.register("datasets.protein_mutaiton_ddg")
class ProteinMutationDdgDataset(torch_data.Dataset, core.Configurable):
    def __init__(self, path, mode, fold, data_name, label_name, 
                 feat_type='onehot', lm_model_name=None, model_path=None,
                 max_length=None, padding_mode='max_length'):
        self.path = path    
        self.mode = mode
        self.fold = fold
        self.feat_type = feat_type
        self.max_length = max_length
        self.data_name = data_name
        self.label_name = label_name
        self.padding_mode = padding_mode
        self.lm_model_name = lm_model_name
        self.featurize = data.ProteinSequence(lm_model_name=lm_model_name, model_path=model_path)

        if self.feat_type == 'language_model':
            if self.lm_model_name is None:
                raise ValueError('model_name must be provided for language_model feature type')   
            self.pad_token_id = self.featurize.pad_token_id
        
        df = pd.read_csv(self.path)
        
        if self.mode == 'train':
            self.df = df[df['split'] == 'train'].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['split'] == 'valid'].reset_index(drop=True)
            # self.df = df[df['split'] == 'test'].reset_index(drop=True)
        elif self.mode == 'test':
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
            # self.df = df[df['split'] == 'valid'].reset_index(drop=True)
        elif self.mode == 'infer':
            self.df = df
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
                
        print(f'{mode} dataset size: {len(self.df)}')

    def __getitem__(self, index):  
            
        row = self.df.iloc[index]
        wt_seq = row['wt_seq']
        mut_seq = row['mut_seq']
        ddg = float(row[self.label_name])
        position = int(row['position']) - 1
        
        wt_data = self.featurize.featurize(wt_seq)
        mut_data = self.featurize.featurize(mut_seq)
        
        wt_data['pad_token_id'] = self.pad_token_id 
        mut_data['pad_token_id'] = self.pad_token_id
                
        return wt_data, mut_data, position, ddg

    def __len__(self):
        return len(self.df)
    

