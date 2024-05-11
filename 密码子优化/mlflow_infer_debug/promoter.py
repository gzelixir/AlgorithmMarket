import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import mlflow
import mlflow.pytorch


pymodel_mlflow = mlflow.pyfunc.load_model('/user/liufuxu/project/zBioSynth/mlruns/8/063028df4ad74c5b99dcc15604a641cf/artifacts/model')

df = pd.read_csv('/user/liufuxu/project/zBioSynth/data/promoter.csv')
df = df[df['split'] == 'test'].reset_index(drop=True)
labels = df.label.values
pred = pymodel_mlflow.predict(df)
scores = pred['score']
print(roc_auc_score(labels, scores))
