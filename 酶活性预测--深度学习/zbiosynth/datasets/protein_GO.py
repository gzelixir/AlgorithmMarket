import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data as torch_data
from zbiosynth import core, data, utils
from zbiosynth.core import Registry as R




@R.register("datasets.protein_go")
class ProteinGODataset(torch_data.Dataset, core.Configurable):
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
            # self.df = self.df.sample(1000).reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['split'] == 'valid'].reset_index(drop=True)
        elif self.mode == 'test':
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
        elif self.mode == 'infer':
            self.df = df
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
                
        self.path = os.path.split(__file__)[0]
        self.bp_go2id = np.load(f'{self.path}/data/BP_class_dict.npy', allow_pickle=True).item()
        self.mf_go2id = np.load(f'{self.path}/data/MF_class_dict.npy', allow_pickle=True).item()
        self.cc_go2id = np.load(f'{self.path}/data/CC_class_dict.npy', allow_pickle=True).item()
        
        self.num_bp_classes = len(self.bp_go2id)
        self.num_mf_classes = len(self.mf_go2id)
        self.num_cc_classes = len(self.cc_go2id)
        
        
        print(f'{mode} dataset size: {len(self.df)}')

    def __getitem__(self, index):    
        row = self.df.iloc[index]
        seq = row[self.data_name]
        
        bp = row['BP']
        mf = row['MF']
        cc = row['CC']
        bp_y = torch.zeros(self.num_bp_classes)
        mf_y = torch.zeros(self.num_mf_classes)
        cc_y = torch.zeros(self.num_cc_classes)
        if str(bp) != 'nan':
            bp = np.array([self.bp_go2id[go] for go in bp.split(',') if go in self.bp_go2id])
            bp_y[bp] = 1
        if str(mf) != 'nan':
            mf = np.array([self.mf_go2id[go] for go in mf.split(',') if go in self.mf_go2id])
            mf_y[mf] = 1
        if str(cc) != 'nan':
            cc = np.array([self.cc_go2id[go] for go in cc.split(',') if go in self.cc_go2id])
            cc_y[cc] = 1
            
        if self.feat_type == 'onehot':
            data = self.featurize.featurize(seq)
            data = torch.FloatTensor(data)
        else:
            if self.max_length:
                seq = seq[:self.max_length]
            data = self.featurize.featurize(seq)
            data['pad_token_id'] = self.pad_token_id  
                           
        return data, bp_y, mf_y, cc_y

    def __len__(self):
        return len(self.df)
    
@R.register("datasets.protein_go_single")
class ProteinGOSingleDataset(torch_data.Dataset, core.Configurable):
    def __init__(self, path, mode, fold, data_name, label_name, task='BP',
                 feat_type='onehot', lm_model_name=None, model_path=None,
                 max_length=None, padding_mode='max_length'):
        self.path = path    
        self.mode = mode
        self.fold = fold
        self.feat_type = feat_type
        self.max_length = max_length
        self.data_name = data_name
        self.label_name = label_name
        self.task = task
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
            # self.df = self.df.sample(1000).reset_index(drop=True)
        elif self.mode == 'valid':
            self.df = df[df['split'] == 'valid'].reset_index(drop=True)
        elif self.mode == 'test':
            self.df = df[df['split'] == 'test'].reset_index(drop=True)
        elif self.mode == 'infer':
            self.df = df
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
                
        self.path = os.path.split(__file__)[0]
        if self.task == 'BP':
            self.go2id = np.load(f'{self.path}/data/BP_class_dict.npy', allow_pickle=True).item()
        elif self.task == 'MF':
            self.go2id = np.load(f'{self.path}/data/MF_class_dict.npy', allow_pickle=True).item()
        elif self.task == 'CC':
            self.go2id = np.load(f'{self.path}/data/CC_class_dict.npy', allow_pickle=True).item()
        else:
            raise ValueError(f'Invalid task: {self.task}')
        
        self.num_classes = len(self.go2id)
        
        print(f'{mode} dataset size: {len(self.df)}')

    def __getitem__(self, index):    
        row = self.df.iloc[index]
        seq = row[self.data_name]
        
        go_label = row[self.label_name]
        y = torch.zeros(self.num_classes)
        go_label = np.array([self.go2id[go] for go in go_label.split(',') if go in self.go2id])
        y[go_label] = 1
            
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


