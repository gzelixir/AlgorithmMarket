import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import mlflow
import mlflow.pytorch


pymodel_mlflow = mlflow.pyfunc.load_model('/user/liufuxu/project/zBioSynth/mlruns/9/95849af82a0943a08da50aa810840563/artifacts/model')

df = pd.read_csv('/user/liufuxu/project/zBioSynth/data/sgrna_offtarget_v2.csv')
df = df[df['split'] == 'test'].reset_index(drop=True)
labels = df.label.values
pred = pymodel_mlflow.predict(df)
scores = pred['score']
auc = roc_auc_score(labels, scores)
bacc = balanced_accuracy_score(labels, (scores>0.5).astype(int))
print(f'auc: {auc}, bacc: {bacc}')
