import pandas as pd
import numpy as np


test_X = pd.read_csv('test_X_PrBOtR0_kr21GXn.csv',sep=',')
train_X = pd.read_csv('train_X_bi3kZtl_HpFoQzd.csv',sep=',')
train_Y = pd.read_csv('train_Y_YuFWD9r_L2f1EvL.csv',sep=',')
test_Y = pd.read_csv('test_Y_randomized_9SyBmQ0_jkWJtkA.csv',sep=',')

# Je supprime la prediction associée à T=10 ainsi que l'observation associée à Model = 0
train_Xbis = train_X.loc[train_X.TIME <10]
train_Xbis = train_Xbis[train_Xbis['MODEL'] != 0]

# A partir du script benchmark j'ai reshape le df pour être du même format que train_Y
dfs = []
for i in train_Xbis.DATASET.unique():
    for j in train_Xbis.MODEL.unique():
        for k in train_Xbis.TIME.unique() :
            df = train_Xbis[train_Xbis.DATASET == i]
            df = df[df.MODEL == j]
            df = df[df.TIME == k]
            means = np.mean(df.VALUE.to_numpy().reshape(192,16),1)
            vars = np.var(df.VALUE.to_numpy().reshape(192,16),1)
            df = pd.DataFrame()
            df['MEAN'] = means
            df['VARIANCE'] = vars
            df['DATASET'] = i
            df['POSITION'] = [i for i in range(len(df))]
            df['TIME'] = k
            df['MODEL'] = j
            df = df[['DATASET', 'POSITION', 'MODEL', 'TIME', 'MEAN', 'VARIANCE']]
            dfs.append(df)


df2 = pd.concat(dfs)
df3 = pd.pivot_table(df2, values = ['MEAN', 'VARIANCE'], index = ['DATASET', 'POSITION'], 
                     columns = ['TIME'], aggfunc=np.sum)

df_train = pd.pivot_table(train_Y, values = ['MEAN'], index = ['DATASET', 'POSITION'])
df3['pred_mean'] = df_train.MEAN
df3 = df3.reset_index()
df3.columns = ['DATASET','POSITION', 'MEAN_0', 'MEAN_1', 'MEAN_2', 'MEAN_3', 'MEAN_4', 
               'MEAN_5', 'MEAN_6', 'MEAN_7', 'MEAN_8', 'MEAN_9', 
               'VAR_0', 'VAR_1', 'VAR_2', 'VAR_3', 'VAR_4', 'VAR_5', 'VAR_6', 'VAR_7', 'VAR_8', 'VAR_9',
               'PRED_MEAN']

# Le df montre donc les moyennes et variances d'anomalies de températures pour les 192 positions pour chacun des 5 datasets
# Une ligne possède les 10 ans de données moyennées entre les 22 modèles le tout à basse résolution + la colonne de prediction.
# Donc à partir de ça on peut construire un modèle qui s'entraine sur les anomalies pour prédire la dernière colonne.
