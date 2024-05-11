import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R


@R.register("datasets.enzyme_ecnumber")
class EnzymeECNumberDataset(torch_data.Dataset, core.Configurable):
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
        
        if self.mode in ['train', 'valid', 'test']:
            self.df = df[df['split'] == self.mode].reset_index(drop=True)
        elif self.mode == 'infer':
            self.df = df
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
                               
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
    
def data_collate(batch):
    x, y = zip(*batch)
    max_length = max([len(onehot) for onehot in x])
    ## padding
    x = [np.pad(onehot, ((0, max_length - len(onehot)), (0, 0)), 'constant', constant_values=0) for onehot in x]
    x = np.array(x)
    
    data_dict = {}
    data_dict['input'] = torch.FloatTensor(x)
    data_dict['label'] = torch.FloatTensor([y])
    return data_dict


# class ProteinSolubilityDataset(torch_data.Dataset, core.Configurable):
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

