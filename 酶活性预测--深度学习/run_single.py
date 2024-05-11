import argparse
import pprint
import shutil
import torch
import mlflow
import mlflow.pytorch
from zbiosynth.utils.tools import load_config, create_working_directory, get_root_logger
from zbiosynth import core, datasets, tasks, models, layers
from zbiosynth.utils.data_collate import *
from zbiosynth.utils import comm
import pandas as pd
from transformers import EsmForMaskedLM, EsmConfig, EsmTokenizer, EsmModel, get_scheduler
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



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


def train_and_validate(cfg, solver):
    best_score = float("-inf")
    best_epoch = -1

    if not cfg.lr_scheduler.num_epochs > 0:
        return solver, best_epoch

    for epoch in tqdm(range(0, cfg.lr_scheduler.num_epochs, cfg.eval.eval_epochs)):
        num_epochs_to_eval = min(cfg.eval.eval_epochs, cfg.lr_scheduler.num_epochs - epoch)
        train_metric = solver.train(num_epochs_to_eval)
        score = []
        if cfg.eval.do_eval:
            metric = solver.evaluate("valid")
            for k, v in metric.items():
                if k in cfg.eval.eval_metric:
                    score.append(v)
            score = np.mean(score)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                solver.save(f"model_epoch_{best_epoch}.pth") 
                   

    solver.save(f"model_epoch_final.pth")
    solver.load(f"model_epoch_{best_epoch}.pth")
    return solver, best_epoch


def test(cfg, solver):
    if "test_batch_size" in cfg:
        solver.batch_size = cfg.test_batch_size
        
    solver.evaluate("valid")
    solver.evaluate("test")

    return


if __name__ == "__main__":
    args = parse_args()
    args.config = os.path.realpath(args.config)
    cfg = load_config(args.config)

    ## tmp code
    # os.system('pip uninstall pandas -y')
    # os.system('pip install pandas')
    # os.system('pip uninstall matplotlib -y')
    # os.system('pip install matplotlib')
    set_seed(args.seed)
    output_dir = create_working_directory(cfg)
    os.makedirs(output_dir, exist_ok=True)
    logger = get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
        logger.warning("Output dir: %s" % output_dir)
        shutil.copyfile(args.config, os.path.basename(args.config))
    os.chdir(output_dir)


    solver = build_solver(cfg, logger)
    solver, best_epoch = train_and_validate(cfg, solver)
    if comm.get_rank() == 0:
        logger.warning("Best epoch on valid: %d" % best_epoch)
    test(cfg, solver)