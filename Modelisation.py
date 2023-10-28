#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:35:14 2022


"""
import pandas as pd
import numpy as np


test_X = pd.read_csv('test_X_PrBOtR0_kr21GXn.csv',sep=',')
train_X = pd.read_csv('train_X_bi3kZtl_HpFoQzd.csv',sep=',')
train_Y = pd.read_csv('train_Y_YuFWD9r_L2f1EvL.csv',sep=',')
test_Y = pd.read_csv('test_Y_randomized_9SyBmQ0_jkWJtkA.csv',sep=',')

train_X[(train_X['POSITION'] == 0) &
        (train_X['TIME'] == 10)].mean()


# On pourrait moyenner les données des modèles, avoir un df avec les données de modèles sur 10 ans 
# et les données obs sur 10 ans. 

# On entraine le y-train sur le X-train pour obtenir un modèle qui puisse prédire au mieux selon les
# observations et les modèles

train_current = train_X[train_X['TIME'] !=10]

train_X_obs = train_current[train_current['MODEL'] == 0]
train_X_mod = train_current[train_current['MODEL'] != 0]

####### TRAIN_X_OBS
dfs = []
for i in train_X_obs.DATASET.unique():
    for j in train_X_obs.MODEL.unique():
        for k in train_X_obs.TIME.unique() :
            df = train_X_obs[train_X_obs.DATASET == i]
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
train_X_obs = pd.pivot_table(df2, values = ['MEAN', 'VARIANCE'], index = ['DATASET', 'POSITION']
                             ,aggfunc=np.mean)

#df_train = pd.pivot_table(train_Y, values = ['MEAN'], index = ['DATASET', 'POSITION'], aggfunc = np.mean)
#df3['pred_mean'] = df_train.MEAN
train_X_obs = train_X_obs.reset_index()
train_X_obs.columns = ['DATASET', 'POSITION', 'MEAN_OBS', 'VARIANCE_OBS']



######## TRAIN_X_MOD
dfs = []
for i in train_X_mod.DATASET.unique():
    for j in train_X_mod.MODEL.unique():
        for k in train_X_mod.TIME.unique() :
            df = train_X_mod[train_X_mod.DATASET == i]
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
train_X_mod = pd.pivot_table(df2, values = ['MEAN', 'VARIANCE'], index = ['DATASET', 'POSITION']
                             ,aggfunc=np.mean)

#df_train = pd.pivot_table(train_Y, values = ['MEAN'], index = ['DATASET', 'POSITION'], aggfunc = np.mean)
#df3['pred_mean'] = df_train.MEAN
train_X_mod = train_X_mod.reset_index()
train_X_mod.columns = ['DATASET', 'POSITION', 'MEAN_MOD', 'VARIANCE_MOD']

train_X_merged = pd.concat([train_X_obs, train_X_mod], axis = 1)
train_X_merged = train_X_merged.iloc[:,[0,1,2,3,6,7]]


###### TEST X

test_current = test_X[test_X['TIME'] !=10]

test_X_obs = test_current[test_current['MODEL'] == 0]
test_X_mod = test_current[test_current['MODEL'] != 0]

####### TEST_X_OBS
dfs = []
for i in test_X_obs.DATASET.unique():
    for j in test_X_obs.MODEL.unique():
        for k in test_X_obs.TIME.unique() :
            df = test_X_obs[test_X_obs.DATASET == i]
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
test_X_obs = pd.pivot_table(df2, values = ['MEAN', 'VARIANCE'], index = ['DATASET', 'POSITION']
                             ,aggfunc=np.mean)

#df_train = pd.pivot_table(train_Y, values = ['MEAN'], index = ['DATASET', 'POSITION'], aggfunc = np.mean)
#df3['pred_mean'] = df_train.MEAN
test_X_obs = test_X_obs.reset_index()
test_X_obs.columns = ['DATASET', 'POSITION', 'MEAN_OBS', 'VARIANCE_OBS']


######## TEST_X_MOD
dfs = []
for i in test_X_mod.DATASET.unique():
    for j in test_X_mod.MODEL.unique():
        for k in test_X_mod.TIME.unique() :
            df = test_X_mod[test_X_mod.DATASET == i]
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
test_X_mod = pd.pivot_table(df2, values = ['MEAN', 'VARIANCE'], index = ['DATASET', 'POSITION']
                             ,aggfunc=np.mean)

#df_train = pd.pivot_table(train_Y, values = ['MEAN'], index = ['DATASET', 'POSITION'], aggfunc = np.mean)
#df3['pred_mean'] = df_train.MEAN
test_X_mod = test_X_mod.reset_index()
test_X_mod.columns = ['DATASET', 'POSITION', 'MEAN_MOD', 'VARIANCE_MOD']

test_X_merged = pd.concat([test_X_obs, test_X_mod], axis = 1)
test_X_merged = test_X_merged.iloc[:,[0,1,2,3,6,7]]



from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

clf = RandomForestRegressor(n_estimators = 200, random_state=1)
clf.fit(train_X_merged.iloc[:,[2,4]], train_Y.MEAN)

y_pred = clf.predict(test_X_merged.iloc[:, [2,4]])

# ON VARIANCE NOW

clf2 = RandomForestRegressor(n_estimators = 200, random_state=1)
clf2.fit(train_X_merged.iloc[:,[3,5]], train_Y.MEAN)
#y_pred = clf2.predict(test_X_merged.iloc[:, [3,5]])

y_pred_mean = pd.DataFrame(clf.predict(test_X_merged.iloc[:, [2,4]]), columns = ['MEAN'])
y_pred_var = pd.DataFrame(clf.predict(test_X_merged.iloc[:, [3,5]]), columns = ['VARIANCE'])
y_pred_merged = pd.concat([test_Y.ID,test_Y.DATASET,test_Y.POSITION,y_pred_mean, y_pred_var], axis = 1)

from climate_challenge_custom_metric import climate_metric_function

the_score=climate_metric_function(test_Y, y_pred_merged)
