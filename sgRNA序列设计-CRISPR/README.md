create a new Task, e.g. DNA enhancer activity prediction
- 1. create a new task class in zbiosynth/tasks/enhancer_activity_prediction.py, and add it to __init__.py
- 2. create a new dataset class in zbiosynth/datasets/enhancer.py, and add it to __init__.py
- 3. create a new model class in zbiosynth/models/enhancer.py, and add it to __init__.py (create Deepstarr model in zbiosynth/models/cnn.py, create rna_lm in zbiosynth/models/rna_lm.py)


## training
```bash
python -m torch.distributed.launch --nproc_per_node=2 examples/run_single_mlflow.py --config config/soluprot/proteincnn1d_mlflow.yaml
```



### 制作whl包
```bash
python setup.py bdist_wheel
```


```bash
python run_single_pltform_mlflow.py \
--task codon_optimization \
--learning_rate 3e-4 \
--epochs 5 \
--batch_size 4 \
--gpus 1 \
--data_name codon_optimized.csv \
--embedding_model esm2_10m
```