import os
try:
    import pandas as pd
except:
    os.system('pip uninstall pandas -y')
    os.system('pip install pandas')
try:
    import matplotlib
except:
    os.system('pip uninstall matplotlib -y')
    os.system('pip install matplotlib')

import argparse
import pprint
import shutil
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature
from zbiosynth.utils.tools import load_config, create_working_directory, get_root_logger, get_gpus_used
from zbiosynth.utils.tools import create_mlflow_intermediate_dir, config_setting
from zbiosynth import core, data, datasets, tasks, models, layers, mlflow_pyfunc_models
from zbiosynth.mlflow_pyfunc_models import *
from zbiosynth.utils.data_collate import *
from zbiosynth.utils import comm
import pandas as pd
from transformers import get_scheduler
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from warmup_scheduler import GradualWarmupScheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
        self.name = 'GradualWarmupScheduler'
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="specify task name", default='')
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="/user/liufuxu/project/zBioSynth/config/soluprot/proteincnn1d.yaml")
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--platform", help="", type=bool, default=False)
    parser.add_argument("--gpus", help="numbers of gpu are used", type=int, default=1)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=3.0e-4)
    parser.add_argument("--epochs", help="epochs", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--data_name", help="the name of data file", default='data.csv')
    parser.add_argument("--embedding_model", help="the name of embedding model", default='esm2_8m')
    parser.add_argument("--freeze_backbone", type=bool, help="Whether to freeze embedding_model", default=False)


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
    
    if not "lr_scheduler" in cfg:
        scheduler = None
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    else:
        if cfg.lr_scheduler.name == "GradualWarmupScheduler":
            ## modified optimizer lr
            cfg.optimizer.lr = cfg.optimizer.lr * 0.1
            optimizer = core.Configurable.load_config_dict(cfg.optimizer)
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.lr_scheduler.num_epochs-cfg.lr_scheduler.num_warmup_epochs)
            lr_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=cfg.lr_scheduler.num_warmup_epochs, after_scheduler=scheduler_cosine)
        else:
            optimizer = core.Configurable.load_config_dict(cfg.optimizer)
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


def train_and_validate(cfg, solver):
    if cfg.eval.metric_mode == 'max':
        best_score = float("-inf")
    else:
        best_score = float("inf")
        
    best_epoch = -1

    if not cfg.lr_scheduler.num_epochs > 0:
        return solver, best_epoch

    for epoch in tqdm(range(0, cfg.lr_scheduler.num_epochs, cfg.eval.eval_epochs)):
        num_epochs_to_eval = min(cfg.eval.eval_epochs, cfg.lr_scheduler.num_epochs - epoch)
        train_metric = solver.train(num_epochs_to_eval)
        
        if comm.get_rank() == 0:
            mlflow.log_metrics(train_metric, step=epoch)
            
        score = []
        if cfg.eval.do_eval:
            metric = solver.evaluate("valid")
            if comm.get_rank() == 0:
                for k, v in metric.items():
                    if 'threshold' in k:
                        continue
                    mlflow.log_metric('valid_'+k, v, step=epoch)
                    
            for k, v in metric.items():
                if k in cfg.eval.eval_metric:
                    score.append(v)
                        
            score = np.mean(score)
            
            if cfg.eval.metric_mode == 'max':
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    # if 'best_threshold' in cfg.eval:
                    #     cfg.eval.best_threshold = threshold_tmp
                    solver.save(f"model_epoch_{best_epoch}.pth") 
            else:
                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    solver.save(f"model_epoch_{best_epoch}.pth") 
                

    solver.save(f"model_epoch_final.pth")
    solver.load(f"model_epoch_{best_epoch}.pth")
    ## delete the checkpoint files
    os.system(f'rm -rf *.pth')
    return solver, best_epoch


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
        
    val_metric = solver.evaluate("valid")
    test_metric = solver.evaluate("test")    

    if comm.get_rank() == 0:
        ## mlflow logging
        for k, v in val_metric.items():
            if 'threshold' in k:
                continue
            mlflow.log_metric('best_valid_'+k, v)
        for k, v in test_metric.items():
            if 'threshold' in k:
                continue
            mlflow.log_metric('test_'+k, v)
    
    if 'best_threshold' in cfg.eval:
        cfg.eval.best_threshold = test_metric[f'{cfg.eval.eval_metric[0]}_threshold']
        
    return cfg

if __name__ == "__main__":
    args = parse_args()
    
    task2config = {
        'enhancer_activity':'./config/enhancer_activity/rnalm_mlflow.yaml',
        'protein_solubility':'./config/soluprot/esm2_mlflow.yaml',
        'protein_mutaiton_ddg':'./config/ddg/esm2_mlflow.yaml',    ## batchsize always 1? can > 1
        'enzyme_ecnumber':'./config/enzyme_ecnumber/esm2_mlflow.yaml',   ## 8m maximum batch size 16
        'kcat':'./config/kcat/lm_mlflow.yaml',
        'codon_optimization':'./config/codon_optimized/codonlm_mlflow.yaml',
        'promoter':'./config/promoter/rnalm_mlflow.yaml',
        'terminator':'./config/terminator/rnalm_mlflow.yaml',
        'sgrna_offtarget':'./config/sgrna_offtarget/rnalm_mlflow.yaml',
        'transcription_factor_binding_sites':'./config/transcription_factor_binding_sites/rnalm_mlflow.yaml',
        'go': './config/go/esm2_mlflow.yaml',
        'go_bp': './config/go_single/BP_esm2_mlflow.yaml',
        'go_mf': './config/go_single/MF_esm2_mlflow.yaml',
        'go_cc': './config/go_single/CC_esm2_mlflow.yaml',
    }
    
    if args.task in task2config:
        args.config = task2config[args.task]
    else:
        raise ValueError(f'task {args.task} not supported, supported tasks are: {list(task2config.keys())}')
    
    
    args.config = os.path.realpath(args.config)
    cfg = load_config(args.config)
    cfg.dataset.path = args.data_name
    if args.freeze_backbone:
        cfg.task.model.freeze_backbone = True
    
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    conda_env = load_config(f'{current_dir}/zbiosynth.yml')
    
    # cfg.engine.gpus = get_gpus_used(args.gpus)
    
    
    # # print(cfg['task']['model'])
    # # print(cfg['task']['model']['model_path'])
    # # print(cfg['dataset']['model_path'])
    # # exit(0)
    
    
    cfg = config_setting(args, cfg, current_dir, parent_dir)
        
    set_seed(args.seed)
    output_dir = create_working_directory(cfg)
    logger = get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))
    os.chdir(output_dir)
    
    gpus = cfg.engine.gpus
    
    
    ## if need mlflow model with cpu, need a solver cpu instance
    # cfg.engine.gpus = None
    # solver_cpu = build_solver(cfg, logger)
    # cfg.engine.gpus = gpus
    
    solver = build_solver(cfg, logger)
    
    
    if comm.get_rank() == 0:
        
        RNA_NAME = None
        ## 本地训练的时候需要
        if not args.platform:
            mlflow.set_tracking_uri(cfg.mlflow.TRACKING_URI)  ## when using local store    
            # mlflow.set_tracking_uri("http://127.0.0.1:5001")   ## when using database-backed store
            mlflow.set_experiment(cfg.mlflow.EXPERIMENT)
            
            RNA_NAME = cfg.mlflow.RUN_NAME
        # with mlflow.start_run(run_name=cfg.mlflow.RUN_NAME) as run:
        ## 本地训练的时候需要
        with mlflow.start_run(run_name=RNA_NAME) as run:
            print(f'tracking_uri: {mlflow.get_tracking_uri()}')
            artifact_uri = run.info.artifact_uri
            run_id = run.info.run_id
            ckpt_dir = create_mlflow_intermediate_dir(args, run_id, artifact_uri)
            
            print(f'artifact_uri: {artifact_uri}')
            print(f'run_id: {run_id}')
            print(f'ckpt_dir: {ckpt_dir}')
            
            
            solver, best_epoch = train_and_validate(cfg, solver)
            logger.warning("Best epoch on valid: %d" % best_epoch)
        
            cfg = test(cfg, solver)
            
            # mlflow.pytorch.log_model(solver.model, artifact_path="torch_model", conda_env=conda_env)
            # # mlflow.pytorch.log_model(solver.model, artifact_path=os.path.join(f'{output_dir}/{artifact_uri}/model'))
                
            # logger.warning("load model from mlflow artifact, and test it")
            
            # ## load the model from mlflow artifact
            # loaded_model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("torch_model"))
            # solver.model = loaded_model
            # test(cfg, solver)
            
            
            ## change the model device from gpu to cpu
            # solver.model = solver.model.to('cpu')    
            # solver.model.device = torch.device('cpu')
            
            
            model_state_dict = solver.model.state_dict()
            ## device = cpu
            for key in model_state_dict:
                model_state_dict[key] = model_state_dict[key].cpu()
                
            ## if need mlflow model with cpu, solver_cpu load the model_state_dict
            # solver_cpu.model.load_state_dict(model_state_dict)
            
            ## save the pyfunc model            
            signature = get_signature(cfg['task']['class'])            
            if args.task == 'kcat':
                ps_featurize = solver.valid_set.ps_featurize
                mol_featurize = solver.valid_set.mol_featurize
                pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver.model, 
                # pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver_cpu.model, 
                                                                                   ps_featurize,
                                                                                   mol_featurize,
                                                                                   signature=signature)
            elif 'go' in args.task:
                featurize = solver.valid_set.featurize
                threshold = cfg.eval.best_threshold
                print(threshold)
                if 'go' == args.task:
                    bp_go2id = solver.valid_set.bp_go2id
                    mf_go2id = solver.valid_set.mf_go2id
                    cc_go2id = solver.valid_set.cc_go2id
                    pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver.model,
                                                                                featurize, 
                                                                                thresholds=threshold,
                                                                                bp_go2id=bp_go2id,
                                                                                mf_go2id=mf_go2id,
                                                                                cc_go2id=cc_go2id,
                                                                                signature=signature)
                else:
                    go2id = solver.valid_set.go2id
                    pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver.model,
                                                                                    featurize, 
                                                                                    threshold=threshold,
                                                                                    go2id=go2id,
                                                                                    signature=signature)
            else:
                featurize = solver.valid_set.featurize
                pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver.model,
                # pymodel = mlflow_pyfunc_models.model2pymodel[cfg['task']['class']](solver_cpu.model,
                                                                                   featurize, 
                                                                                   signature=signature)
            
            # mlflow.pyfunc.log_model(artifact_path="pyfunc_model", python_model=pymodel, signature=signature, conda_env=conda_env)
            # mlflow.pyfunc.log_model(artifact_path="model", python_model=pymodel, signature=signature, conda_env=conda_env, code_path=['/user/liufuxu/project/zBioSynth/zbiosynth'])
            
            ## mlflow model inference input example data
            if os.path.exists(os.path.join(parent_dir,"data",args.data_name)):
                df = pd.read_csv(os.path.join(parent_dir,"data",args.data_name))
            else:
                df = pd.read_csv(os.path.join(current_dir,"data",args.data_name))
            test_df = df[df['split'] == 'test']
            test_data = test_df.iloc[[-1]]
            test_data = test_data.dropna(axis=1, how='all')
            print("test_data",test_data)
            pred = pymodel.predict(model_input = test_data, context="")
            print("pred",pred)
            
            params = {"task": args.task, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "data_name": args.data_name, "embedding_model":args.embedding_model, "epochs": args.epochs}
            mlflow.log_params(params)

            params = {"learning_rate": args.learning_rate, "batch_size": args.batch_size, "data_name": args.data_name, "embedding_model":args.embedding_model, "epochs": args.epochs}
            mlflow.log_params(params)
            
            if args.platform:
                mlflow.pyfunc.log_model(artifact_path="model", python_model=pymodel, signature=signature, conda_env=conda_env, code_path=[f'{current_dir}/zbiosynth'], input_example=test_data)
            else:
                mlflow.pyfunc.log_model(artifact_path="model", python_model=pymodel, signature=signature, conda_env=conda_env, input_example=test_data)
            # mlflow.pyfunc.log_model(artifact_path="model", python_model=pymodel, signature=signature, conda_env=conda_env, code_path=[f'{current_dir}/zbiosynth-0.2.0-py3-none-any.whl'])
    
            # from mlflow.utils.file_utils import path_to_local_file_uri
            # print(path_to_local_file_uri('/user/liufuxu/project/zBioSynth/zbiosynth-0.2.0-py3-none-any.whl'))
            # mlflow.pyfunc.log_model(artifact_path=os.path.join(f'{output_dir}/{artifact_uri}/pyfunc_model'), python_model=pymodel, signature=signature)
            
            ## copy conda yaml file to pyfunc_model
            # shutil.copyfile(f'{current_dir}/zbiosynth.yml', os.path.join(f'{output_dir}/{artifact_uri}/pyfunc_model', "conda.yaml"))
            # shutil.copyfile(f'{current_dir}/zbiosynth.yml', os.path.join(f'{output_dir}/{artifact_uri}/pyfunc_model', "conda.yaml"))   
        
    else:
        solver, best_epoch = train_and_validate(cfg, solver)
        test(cfg, solver)
    
    if comm.get_rank() == 0:
        mlflow.end_run()  
