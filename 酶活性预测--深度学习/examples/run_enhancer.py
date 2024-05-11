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
    random_df = pd.read_csv('/share/liufuxu/zBioSynth/dataset/enhancer_activity/E12.rice_Random.csv')
    en_df = pd.read_csv('/share/liufuxu/zBioSynth/dataset/enhancer_activity/E11.rice_Enhancer.csv')   
    random_one_hot = np.load('/share/liufuxu/zBioSynth/dataset/enhancer_activity/random_seq_one_hot.npy')
    en_one_hot = np.load('/share/liufuxu/zBioSynth/dataset/enhancer_activity/enhancer_seq_one_hot.npy') 
    
    raw_df = pd.concat([random_df, en_df], axis=0).reset_index(drop=True)
    raw_X = np.concatenate([random_one_hot, en_one_hot], axis=0)
    raw_X = raw_X.transpose(0,2,1)
    
    train_df = raw_df[raw_df['fold'] != fold].reset_index(drop=True)
    train_X = raw_X[raw_df['fold'] != fold]
    train_y = train_df.activity.values
    
    valid_df = raw_df[raw_df['fold'] == fold].reset_index(drop=True)
    valid_X = raw_X[raw_df['fold'] == fold]
    valid_y = valid_df.activity.values
    
    return train_X, train_y, train_df, valid_X, valid_y, valid_df


def main(config):
    
    warmup_epochs = 5
    num_epochs = 40
    lr = 3e-4
    batch_size = 128
    num_workers = 8
    
    train_X, train_y, train_df, valid_X, valid_y, valid_df= load_data(fold=0)

    if not config.lm:
        train_dataset = datasets.EnhancerDataset(X=train_X, y=train_y, mode='train', onehot=True)
        valid_dataset = datasets.EnhancerDataset(X=valid_X, y=valid_y, mode='valid', onehot=True)
        model = models.Deepstarr(input_dim=4, hidden_dims=[256, 60, 60, 120], kernel_sizes=[7,3,5,3], output_dim=1800,
                 activation='relu')
    else:
        tokenizer = EsmTokenizer.from_pretrained("/share/liufuxu/zBioSynth/resources/vocab_files/vocab_esm_mars_all.txt")  
        train_seqs, val_seqs = train_df.seq.values, valid_df.seq.values
        train_dataset = datasets.EnhancerDataset(X=train_seqs, y=train_y, mode='train', onehot=False, tokenizer=tokenizer)
        valid_dataset = datasets.EnhancerDataset(X=val_seqs, y=valid_y, mode='valid', onehot=False, tokenizer=tokenizer)
        model = models.RNALM(model_name=config.model_name)
        
    
    task = tasks.EnhancerActivityPrediction(model, task=['enhancer_avtivity'], criterion="mse", metric=("pearsonr", "mse"), num_mlp_layer=2,
                 normalization=False, num_class=1, mlp_batch_norm=True, mlp_dropout=0.4,
                 verbose=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=num_epochs,
    )
    
    if len(config.gpu_ids) == 1:
        gpus = [int(config.gpu_ids)]
        solver = core.Engine(task, train_dataset, valid_dataset, valid_dataset, optimizer,scheduler=lr_scheduler, batch_size=batch_size, gpus=gpus)  ## single GPU
    else:
        gpus = [int(gpu_id) for gpu_id in config.gpu_ids.split(',')]
        solver = core.Engine(task, train_dataset, valid_dataset, valid_dataset, optimizer,scheduler=lr_scheduler, batch_size=batch_size, gpus=gpus)  ## multi GPU
    solver.train(num_epoch=num_epochs)
    solver.evaluate("valid")
    solver.evaluate("test")
    
    

    ## save and load
    ## 这种方式会直接保存整个task，包括model，optimizer，scheduler, data等，不是很高效的样子。
    # with open("/share/liufuxu/zBioSynth/dataset/tmp/enhancer_cnn.json", "w") as fout:
    #     json.dump(solver.config_dict(), fout)
    # solver.save("/share/liufuxu/zBioSynth/dataset/tmp/enhancer_cnn.pth")
    
    # with open("/share/liufuxu/zBioSynth/dataset/tmp/enhancer_cnn.json", "r") as fin:
    #     solver = core.Configurable.load_config_dict(json.load(fin))
    # solver.load("/share/liufuxu/zBioSynth/dataset/tmp/enhancer_cnn.pth")

    # solver.evaluate("valid")
    # solver.evaluate("test")
    
    
    data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        preds = []
        for batch in tqdm(data_loader):
            for key in batch:
                batch[key] = batch[key].to(task.model.device)
            output = task.predict(batch)
            preds.extend(output.cpu().numpy().reshape(-1))
    print(pearsonr(preds, valid_y))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lm", type=bool, default=False)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--model_name", type=str, default="", help='model name for language model,[rna_lm_8m, rna_lm_35m]')
    ## gpu_ids
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0,1,2,3 for multiple gpus, one number for single gpu")
    config = parser.parse_args()
    
    main(config)