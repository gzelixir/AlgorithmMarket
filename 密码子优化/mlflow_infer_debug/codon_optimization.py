import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
from scipy.stats import pearsonr
import mlflow
import mlflow.pytorch


pymodel_mlflow = mlflow.pyfunc.load_model('/user/liufuxu/project/zBioSynth/mlruns/7/0b11c99f95264a81b24ea25d44f06974/artifacts/model')

df = pd.read_csv('/user/liufuxu/project/zBioSynth/data/codon_optimized.csv')
df = df[df['split'] == 'test'].reset_index(drop=True)
# d = dict.fromkeys(df.select_dtypes(np.int64).columns, np.int32)
# df = df.astype(d)

dna_seqs = df.dna_seq.values
output = pymodel_mlflow.predict(df)
output_df = pd.DataFrame(output)
print(output_df)