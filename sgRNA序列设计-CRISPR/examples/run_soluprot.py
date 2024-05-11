import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"    
import pandas as pd
import numpy as np
import json
import torch
import zbiosynth
from zbiosynth import datasets
from zbiosynth import core, models, tasks
import transformers
from transformers import get_scheduler
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from scipy.stats import pearsonr
import argparse


def load_data(fold=0):
    df_train = pd.read_csv('/share/liufuxu/zBioSynth/dataset/soluprot/train.csv')
    df_test = pd.read_csv('/share/liufuxu/zBioSynth/dataset/soluprot/test.csv')
    
    train_df = df_train[df_train['fold'] != fold].reset_index(drop=True)
    valid_df = df_train[df_train['fold'] == fold].reset_index(drop=True)
    
    
    train_X = train_df['seq'].values
    train_y = train_df['solubility'].values
    valid_X = valid_df['seq'].values
    valid_y = valid_df['solubility'].values
    test_X = df_test['seq'].values
    test_y = df_test['solubility'].values

    return train_X, train_y, train_df, valid_X, valid_y, valid_df, test_X, test_y, df_test


def onehot_data_collate(batch):
    x, y = zip(*batch)
    max_length = max([len(onehot) for onehot in x])
    ## padding
    x = [np.pad(onehot, ((0, max_length - len(onehot)), (0, 0)), 'constant', constant_values=0) for onehot in x]
    x = np.array(x)
    
    data_dict = {}
    data_dict['input'] = torch.FloatTensor(x)
    data_dict['label'] = torch.FloatTensor(y).view(-1,1)
    return data_dict

class Esm_data_collate(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        x, y = zip(*batch)
        max_length = max([len(seq) for seq in x]) + 2
        ## padding
        x = self.tokenizer.batch_encode_plus(x, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        
        data_dict = {}
        data_dict['input_ids'] = x['input_ids']
        data_dict['attention_mask'] = x['attention_mask']
        data_dict['label'] = torch.FloatTensor(y).view(-1,1)
        return data_dict
    

def main(config):
    
    warmup_epochs = 5
    num_epochs = 40
    lr = 3e-4
    gpu_id = 0
    num_workers = 8
    
    train_X, train_y, train_df, valid_X, valid_y, valid_df, test_X, test_y, df_test = load_data(fold=0)
    

    if not config.lm:
        train_dataset = datasets.ProteinSolubilityDataset(X=train_X, y=train_y, mode='train', onehot=True)
        valid_dataset = datasets.ProteinSolubilityDataset(X=valid_X, y=valid_y, mode='valid', onehot=True)
        test_dataset = datasets.ProteinSolubilityDataset(X=test_X, y=test_y, mode='test', onehot=True)
        model = models.ProteinCNN1D(input_dim=21, hidden_dims=[64, 128, 256], kernel_size=3, stride=1, padding=1,
                activation='relu', short_cut=False, concat_hidden=False, pool="max")
        data_collate = onehot_data_collate
        
    else:
        train_dataset = datasets.ProteinSolubilityDataset(X=train_X, y=train_y, mode='train', onehot=False, max_lenght=config.max_length)
        valid_dataset = datasets.ProteinSolubilityDataset(X=valid_X, y=valid_y, mode='valid', onehot=False, max_lenght=config.max_length)
        test_dataset = datasets.ProteinSolubilityDataset(X=test_X, y=test_y, mode='test', onehot=False, max_lenght=config.max_length)

        model = models.Esm2Model(model_name=config.model_name)
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{config.model_name}")  
        data_collate = Esm_data_collate(tokenizer)
        
    
    
    task = tasks.ProteinSolubilityPrediction(model, task=['protein_solubility_prediction'], criterion="bce", 
                                            metric=("bce", "acc", "auroc","mcc", "precision"), 
                                            num_mlp_layer=2,normalization=False, num_class=1, mlp_batch_norm=True, mlp_dropout=0.4, verbose=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
    )
    
    if len(config.gpu_ids) == 1:
        gpus = [int(config.gpu_ids)]
        solver = core.Engine(task, train_dataset, valid_dataset, test_dataset,  optimizer, data_collate=data_collate, scheduler=lr_scheduler, batch_size=config.batch_size, gpus=gpus)  ## single GPU
    else:
        gpus = [int(gpu_id) for gpu_id in config.gpu_ids.split(',')]
        solver = core.Engine(task, train_dataset, valid_dataset, test_dataset,  optimizer, data_collate=data_collate, scheduler=lr_scheduler, batch_size=config.batch_size, gpus=gpus)  ## multi GPU
    solver.train(num_epoch=num_epochs)
    solver.evaluate("valid")
    solver.evaluate("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--lm", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="", help='model name for language model,[esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D')
    parser.add_argument("--local_rank", default=-1, type=int)
    ## gpu_ids
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0,1,2,3 for multiple gpus, one number for single gpu")
    config = parser.parse_args()
    
    if config.lm and config.model_name == "":
        raise ValueError("Please specify the model name for the language model")
    
    main(config)