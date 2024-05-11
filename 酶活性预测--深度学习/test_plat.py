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


import pandas as pd
import zbiosynth
print(zbiosynth)


# print(os.listdir('./'))
# df = pd.read_csv('../codon_optimized/train.csv')

# print(df.head())