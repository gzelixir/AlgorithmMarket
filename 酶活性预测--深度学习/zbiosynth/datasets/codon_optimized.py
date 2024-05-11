import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R


@R.register("datasets.codon_optimized")
class CodonOptimizedDataset(torch_data.Dataset, core.Configurable):
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
        
        self.featurize = data.CodonSequence(lm_model_name=lm_model_name, model_path=model_path)

        if self.feat_type == 'language_model':
            if self.lm_model_name is None:
                raise ValueError('model_name must be provided for language_model feature type')   
            self.pad_token_id = self.featurize.pad_token_id
        
        
        df = pd.read_csv(self.path)
        
        if self.mode == 'train':
            self.df = df[df['split'] == 'train'].reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['split'] == 'valid'].reset_index(drop=True)
        elif self.mode == 'test':
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
               
        
        # self.X = self.df[self.data_name].values
        # if self.mode not in ['train', 'valid', 'test']:
        #     self.y = self.df[self.label_name].values
                
        print(f'{mode} dataset size: {len(self.df)}')
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        aa_seq = row[f'{self.data_name}'] + '*'    
            
        label_codon_seq = None
        if self.mode in ['train', 'valid', 'test']:
            label_codon_seq = row[f'{self.label_name}']
                    
        ## random map
        input_text = self.featurize.random_map_codon_sequence(aa_seq)

        if label_codon_seq:
            label_codons = [label_codon_seq[i:i+3] for i in range(0, len(label_codon_seq), 3)]
            label_text = ' '.join(label_codons)
        else:
            label_text = input_text    
        
        feature = self.featurize.featurize(input_text)
        label_feature = self.featurize.featurize(label_text)
        
        feature['pad_token_id'] = self.featurize.pad_token_id  
        label_feature['input_ids'][:, 0] = -100
        label_feature['input_ids'][:, -1] = -100
        
        return feature, label_feature

    def __len__(self):
        return len(self.df)

