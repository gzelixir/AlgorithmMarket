import argparse
import pprint
import shutil
import torch
import mlflow
import mlflow.pytorch
import yaml
from zbiosynth.utils.tools import load_config, create_working_directory, get_root_logger
from zbiosynth import data, core, utils, datasets, tasks, models, layers
from zbiosynth.utils.data_collate import *
from zbiosynth.utils import comm
import pandas as pd
from tqdm import tqdm
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="/user/liufuxu/project/zBioSynth/config/soluprot/mlflow_eval.yaml")
    return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    cfg_model = load_config(f'{cfg.mlflow_running_path}/artifacts/running_config.yaml')
    cfg_model.dataset.path = cfg.file_path
    
    if cfg.with_label:
        cfg_model.dataset.mode='test'
    else:
        cfg_model.dataset.mode='infer'
        
    if cfg_model.engine.gpus is None:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        
        # if len(gpus) != self.world_size:
        #     error_msg = "World size is %d but found %d GPUs in the argument"
        #     if self.world_size == 1:
        #         error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        #     raise ValueError(error_msg % (self.world_size, len(gpus)))
        # self.device = torch.device(gpus[self.rank % len(gpus)])
        
    test_dataset = core.Configurable.load_config_dict(cfg_model.dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg_model.engine.batch_size, shuffle=False, num_workers=cfg_model.engine.num_worker, collate_fn=eval(cfg_model.collate_fn))
    mlflow_model = mlflow.pytorch.load_model(f'{cfg.mlflow_running_path}/artifacts/model')
    mlflow_model.eval()

    # all_preds,all_labels = [],[]
    # for batch in test_loader:
    #     for key in batch:
    #         batch[key] = batch[key].cuda()
    #     preds = mlflow_model.predict(batch, postprocess=True)
    #     all_preds.append(preds.detach().cpu().numpy())
    # all_preds = (np.concatenate(all_preds).reshape(-1) > 0.5).astype(int)

    preds = []
    targets = []
    for batch in test_loader:
        if device.type == "cuda":
            batch = utils.cuda(batch, device=device)
        pred, target = mlflow_model.predict_and_target(batch)
        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    if cfg.with_label:
        metric = mlflow_model.evaluate(pred, target)
        print(metric)
    



