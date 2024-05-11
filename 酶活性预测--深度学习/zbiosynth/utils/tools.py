import os
import sys
import time
import random
import logging

import yaml
import easydict
import jinja2
import json

from torch import distributed as dist
from zbiosynth.utils import comm



def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()
    cfg = yaml.load(raw_text, Loader=yaml.CLoader)
    cfg = easydict.EasyDict(cfg)

    return cfg


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    output_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                              cfg.task["class"], cfg.dataset["class"],
                              cfg.task.model["class"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(output_dir)
        os.makedirs(output_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            output_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(output_dir)
    return output_dir

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w') as file:
        yaml.dump(dict_value, file)

def json_to_yaml(jsonPath):
    with open(jsonPath, encoding="utf-8") as f:
        datas = json.load(f)
    yamlDatas = yaml.dump(datas, indent=5)
    return yamlDatas

def generate_yaml_file(cfg, yamlPath):
    jsonPath = yamlPath.replace('.yaml', '.json')
    with open(jsonPath, 'w') as f:
        json.dump(dict(cfg), f, indent=4)        
    yamlDatas = json_to_yaml(jsonPath)
    with open(yamlPath,'w') as f:
        f.write(yamlDatas)
    os.remove(jsonPath)
    
def create_mlflow_intermediate_dir(args, run_id, artifact_uri):
    cfg = load_config(args.config)
    
    ckpt_dir = os.path.join(artifact_uri, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    cfg.mlflow.RUN_ID = run_id
    cfg.mlflow.ARTIFACT_URI = artifact_uri
    # update config file
    generate_yaml_file(cfg, os.path.join(artifact_uri, "running_config.yaml"))
    
    return ckpt_dir


def get_gpus_used(gpus):
    if gpus == 0:
        return None
    elif gpus == 1:
        return [0]
    else:
        return [i for i in range(gpus)]
    
    
        
def config_setting(args, cfg, current_dir, parent_dir):
        
    if args.task == 'kcat':
        cfg.dataset.prot_model_name = args.embedding_model.split(',')[0]
        cfg.dataset.mol_model_name = args.embedding_model.split(',')[1]
        
        cfg.task.model.prot_model_name = args.embedding_model.split(',')[0]
        cfg.task.model.mol_model_name = args.embedding_model.split(',')[1]
    else:
        cfg.dataset.lm_model_name = args.embedding_model
        cfg.task.model.model_name = args.embedding_model
        

    ## platform
    if args.platform:
        print('running on platform')
        cfg.output_dir = f'{current_dir}/{cfg.output_dir}'
        
        if os.path.exists( f'{parent_dir}/data/{cfg.dataset.path}'):
            print("find csv in fearure set")
            cfg.dataset.path = f'{parent_dir}/data/{cfg.dataset.path}'
        else:
            print("find csv in git data dir")
            cfg.dataset.path = f'{current_dir}/data/{cfg.dataset.path}'


        if args.task == 'kcat':
            if 'prot_model_path' in cfg['task']['model'] and 'mol_model_path' in cfg['task']['model']:
                # cfg['task']['model']['prot_model_path'] = f'{parent_dir}/prot_mol_LM/' 
                # cfg['dataset']['model_path'] = cfg['task']['model']['prot_model_path']
                # cfg['task']['model']['mol_model_path'] = f'{parent_dir}/prot_mol_LM/'
                # cfg['dataset']['model_path'] = cfg['task']['model']['mol_model_path']
                cfg['task']['model']['prot_model_path'] = f'{parent_dir}' 
                cfg['dataset']['prot_model_path'] = cfg['task']['model']['prot_model_path']
                cfg['task']['model']['mol_model_path'] = f'{parent_dir}'
                cfg['dataset']['mol_model_path'] = cfg['task']['model']['mol_model_path']
        else:
            if 'model_path' in cfg['task']['model']:
                cfg['task']['model']['model_path'] = parent_dir
                cfg['dataset']['model_path'] = cfg['task']['model']['model_path']

    else:
        cfg.output_dir = f'{current_dir}/{cfg.output_dir}'
        if os.path.exists( f'{parent_dir}/data/{cfg.dataset.path}'):
            print("find csv in fearure set")
            cfg.dataset.path = f'{parent_dir}/data/{cfg.dataset.path}'
        else:
            print("find csv in git data dir")
            cfg.dataset.path = f'{current_dir}/data/{cfg.dataset.path}'
        cfg['mlflow']['TRACKING_URI'] = f'{current_dir}/mlruns'

        if args.task == 'kcat':  
            cfg['dataset']['prot_model_path'] = cfg['task']['model']['prot_model_path']
            cfg['dataset']['mol_model_path'] = cfg['task']['model']['mol_model_path']
        else:
            cfg['dataset']['model_path'] = cfg['task']['model']['model_path']

    cfg.engine.batch_size = args.batch_size
    cfg.engine.gpus = get_gpus_used(args.gpus)
    cfg.optimizer.lr = args.learning_rate
    cfg.lr_scheduler.num_epochs = args.epochs
    
    return cfg
        