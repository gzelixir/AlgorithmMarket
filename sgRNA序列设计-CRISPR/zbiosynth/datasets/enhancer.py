import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R


# @R.register("datasets.Enhancer")
# class EnhancerDataset(torch_data.Dataset, core.Configurable):
#     def __init__(self, X, y, mode='train', onehot=False, tokenizer=None):
#         self.X, self.y = X, y
#         self.mode = mode
#         self.onehot = onehot
#         self.tokenizer = tokenizer
                
#         print(f'{mode} dataset size: {len(self.X)}')
    
#     def __len__(self):
#         return len(self.X)

#     def get_single(self, index):
        
#         if self.onehot:
#             data = self.X[index]
#             data = torch.FloatTensor(data)
#             data_dict = {"input": data}
#         else:
#             seq = self.X[index]
#             input_text = ' '.join(seq)
#             features = self.tokenizer(input_text, return_tensors='pt')
#             data_dict = {"input_ids": features['input_ids'].squeeze(0),  "attention_mask": features['attention_mask'].squeeze(0)}
            
#         activity = self.y[index]
#         activity = torch.FloatTensor([float(activity)])
#         data_dict["label"] = activity

#         return data_dict

#     def __getitem__(self, index):

#         data_dict = self.get_single(index=index)

#         return data_dict



@R.register("datasets.Enhancer_Activity")
class EnhancerActivityDataset(torch_data.Dataset, core.Configurable):
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
        
        self.featurize = data.NucleotideSequence(lm_model_name=lm_model_name, model_path=model_path)

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
               
        self.X = self.df[self.data_name].values
        self.y = self.df[self.label_name].values
                
        print(f'{mode} dataset size: {len(self.df)}')
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = row[self.data_name]
        y = row[self.label_name]
        
        if self.feat_type == 'onehot':
            data = self.featurize.featurize(seq)
            data = torch.FloatTensor(data)
        else:
            if self.max_length:
                seq = seq[:self.max_length]
            data = self.featurize.featurize(seq)
            data['pad_token_id'] = self.pad_token_id 
        return data, y

    def __len__(self):
        return len(self.df)

