import argparse
import pprint
import shutil
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import yaml
from zbiosynth.utils.tools import load_config, create_working_directory, get_root_logger
from zbiosynth import core, datasets, tasks, models, layers
from zbiosynth.utils.data_collate import *
from zbiosynth.utils import comm
import pandas as pd
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel, get_scheduler
from tqdm import tqdm
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))



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
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="/user/liufuxu/project/zBioSynth/config/soluprot/proteincnn1d.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    return parser.parse_known_args()[0]


def set_seed(seed):
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def build_solver(cfg, logger):
    
    # build dataset
    cfg.dataset.mode='train'
    train_dataset = core.Configurable.load_config_dict(cfg.dataset)
    cfg.dataset.mode='valid'
    valid_dataset = core.Configurable.load_config_dict(cfg.dataset)
    cfg.dataset.mode='test'
    test_dataset = core.Configurable.load_config_dict(cfg.dataset)

        
    if comm.get_rank() == 0:
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_dataset), len(valid_dataset), len(test_dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    
    # build solver
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    
    if not "lr_scheduler" in cfg:
        scheduler = None
    else:
        lr_scheduler = get_scheduler(
            cfg.lr_scheduler.name,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_scheduler.num_warmup_epochs,
            num_training_steps=cfg.lr_scheduler.num_epochs,
        )

    solver = core.Engine(task, train_dataset, valid_dataset, test_dataset,  optimizer, collate_fn=eval(cfg.collate_fn), scheduler=lr_scheduler, **cfg.engine)
    
    
    ## resume training
    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint, load_optimizer=False)

    return solver


def train_and_validate(cfg, solver, ckpt_dir):
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.lr_scheduler.num_epochs > 0:
        return solver, best_epoch
    
    ## topk model
    topk_epcoh = []
    
    for epoch in tqdm(range(0, cfg.lr_scheduler.num_epochs, cfg.eval.eval_epochs)):
        num_epochs_to_eval = min(cfg.eval.eval_epochs, cfg.lr_scheduler.num_epochs - epoch)
        train_records = solver.train(num_epochs_to_eval)
        
        if comm.get_rank() == 0:
            mlflow.log_metrics(train_records, step=epoch)
            
        score = []
        if cfg.eval.do_eval:
            metric = solver.evaluate("valid")
            ## mlflow logging
            if comm.get_rank() == 0:
                for k, v in metric.items():
                    mlflow.log_metric('valid_'+k, v, step=epoch)
                
            for k, v in metric.items():
                if k in cfg.eval.eval_metric:
                    score.append(v)
            score = np.mean(score)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                ## save the model to mlflow artifact ckpt_dir
                solver.save(os.path.join(ckpt_dir, f"model_epoch_{best_epoch}.pth"))
                topk_epcoh.append(best_epoch)
                if len(topk_epcoh) > cfg.eval.save_top_k:
                    os.remove(os.path.join(ckpt_dir, f"model_epoch_{topk_epcoh[0]}.pth"))
                    topk_epcoh.pop(0)
        
    solver.save(os.path.join(ckpt_dir, f"model_epoch_final.pth"))
    solver.load(os.path.join(ckpt_dir, f"model_epoch_{best_epoch}.pth"))
    
    return solver, best_epoch


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
        
    val_metric = solver.evaluate("valid")
    test_metric = solver.evaluate("test")    

    if comm.get_rank() == 0:
        ## mlflow logging
        for k, v in val_metric.items():
            mlflow.log_metric('best_valid_'+k, v)
        for k, v in test_metric.items():
            mlflow.log_metric('test_'+k, v)
    return


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = load_config(args.config)
    cfg_tmp = load_config(args.config)

    set_seed(args.seed)
    logger = get_root_logger()

    solver = build_solver(cfg, logger)
        
    if comm.get_rank() == 0:
        ### the location of mlflow artifact store ###
        # mlflow.set_tracking_uri(cfg.mlflow.TRACKING_URI)  ## when using local store
        mlflow.set_tracking_uri("http://127.0.0.1:5001")   ## when using database-backed store
        mlflow.set_experiment(cfg.mlflow.EXPERIMENT)
        # with mlflow.start_run(experiment_id=cfg.mlflow.EXPERIMENT_ID, run_name=cfg.mlflow.RUN_NAME) as run:
        with mlflow.start_run(run_name=cfg.mlflow.RUN_NAME) as run:
            artifact_uri = run.info.artifact_uri
            run_id = run.info.run_id
            
            ckpt_dir = create_mlflow_intermediate_dir(args, run_id, artifact_uri)
            
            solver, best_epoch = train_and_validate(cfg, solver, ckpt_dir)
            logger.warning("Best epoch on valid: %d" % best_epoch) 
            test(cfg, solver)

            mlflow.pytorch.log_model(solver.model, artifact_path="model")
            
            logger.warning("load model from mlflow artifact, and test it")
            
            ## load the model from mlflow artifact
            loaded_model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("model"))
            solver.model = loaded_model
            test(cfg, solver)
            
    else:
        solver, best_epoch = train_and_validate(cfg, solver)
        test(cfg, solver)

    if comm.get_rank() == 0:
        mlflow.end_run()  
        