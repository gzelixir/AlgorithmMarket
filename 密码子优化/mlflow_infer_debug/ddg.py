import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import mlflow
import mlflow.pytorch


pymodel_mlflow = mlflow.pyfunc.load_model('/user/liufuxu/project/zBioSynth/mlruns/3/c2f50a2aa41740ce965649eb6a696b91/artifacts/model')

df = pd.read_csv('/share/liufuxu/zBioSynth/dataset/ddg/ddg.csv')
df = df[df['split'] == 'test'].reset_index(drop=True)
d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
df = df.astype(d)

labels = df.ddG.values
pred = pymodel_mlflow.predict(df)
scores = pred['ddg']
print(pearsonr(labels, scores))