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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", help="specify task name", default='')
    parser.add_argument("--gpus", help="numbers of gpu are used", type=int, default=1)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=3.0e-4)
    parser.add_argument("--epochs", help="epochs", type=int, default=1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--data_name", help="the name of data file", default='data.csv')
    parser.add_argument("--embedding_model", help="the name of embedding model", default='esm2_8m')
    parser.add_argument("--freeze_backbone", type=str, help="Whether to freeze embedding_model", default="False")

    return parser.parse_known_args()[0]

if __name__ == "__main__":
    args = parse_args()
    
    if args.gpus == 0 or args.gpus == 1:
        cmd = f'python run_single_pltform_mlflow.py \
                --task {args.task} \
                --platform True \
                --gpus {args.gpus} \
                --learning_rate {args.learning_rate} \
                --epochs {args.epochs} \
                --batch_size {args.batch_size} \
                --data_name {args.data_name} \
                --embedding_model {args.embedding_model}'
        if args.freeze_backbone.upper() == 'TRUE':
            cmd += ' --freeze_backbone True'
    else:
        cmd = f'python -m torch.distributed.launch --nproc_per_node={args.gpus} run_single_pltform_mlflow.py \
            --task {args.task} \
            --platform True \
            --gpus {args.gpus} \
            --learning_rate {args.learning_rate} \
            --epochs {args.epochs} \
            --batch_size {args.batch_size} \
            --data_name {args.data_name} \
            --embedding_model {args.embedding_model}'
        if args.freeze_backbone.upper() == 'TRUE':
            cmd += ' --freeze_backbone True'
    
    
    # if args.gpus == 0 or args.gpus == 1:
    #     cmd = f'python run_single_pltform_mlflow.py \
    #             --task {args.task} \
    #             --gpus {args.gpus} \
    #             --learning_rate {args.learning_rate} \
    #             --epochs {args.epochs} \
    #             --batch_size {args.batch_size} \
    #             --data_name {args.data_name} \
    #             --embedding_model {args.embedding_model}'
    #     if args.freeze_backbone.upper() == 'TRUE':
    #         cmd += ' --freeze_backbone True'
    # else:
    #     cmd = f'python -m torch.distributed.launch --nproc_per_node={args.gpus} run_single_pltform_mlflow.py \
    #         --task {args.task} \
    #         --gpus {args.gpus} \
    #         --learning_rate {args.learning_rate} \
    #         --epochs {args.epochs} \
    #         --batch_size {args.batch_size} \
    #         --data_name {args.data_name} \
    #         --embedding_model {args.embedding_model}'
    #     if args.freeze_backbone.upper() == 'TRUE':
    #         cmd += ' --freeze_backbone True'
        
    os.system(cmd)
        