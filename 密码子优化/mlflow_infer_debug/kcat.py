import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import mlflow
import mlflow.pytorch


pymodel_mlflow = mlflow.pyfunc.load_model('/user/liufuxu/project/zBioSynth/mlruns/5/ba6d12265be7486b9c189d33872dafb6/artifacts/model')

df = pd.read_csv('/user/liufuxu/project/zBioSynth/data/kcat.csv')
df = df[df['split'] == 'test'].reset_index(drop=True)

kcat = df.kcat.values
pred = pymodel_mlflow.predict(df)
preds = pred['kcat']
print(pearsonr(kcat, preds))