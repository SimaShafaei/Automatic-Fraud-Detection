# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:11:46 2019

@author: sshaf
"""


import os
import pandas as pd
import numpy as np

os.getcwd()
pd.options.display.max_colwidth = 5000

#load data and remove unwanted variables
mypath='..\\data\\'
df = pd.read_csv (mypath+'fraud_balenced_30000.csv', index_col=[0])
#replace Nan with a negative number that doesn't exist in DB
"""new_num=np.min((np.min(df,axis=1)),axis=0)-100
df.fillna(new_num,inplace=True)"""
#removing quasi-constant dataset
from sklearn.feature_selection import VarianceThreshold
quasi_constant_filter = VarianceThreshold(threshold=0.01)
quasi_constant_filter.fit(df)
#len(train_features.columns[quasi_constant_filter.get_support()])
quasi_constant_columns = [column for column in df.columns
                    if column not in df.columns[quasi_constant_filter.get_support()]]
df.drop(labels=quasi_constant_columns, axis=1, inplace=True)

#removing duplicate

#normalization: scale all data between 0,1
"df[df==new_num]=np.nan"
df -= df.min()
df /= df.max()

#Using Pearson Correlation
correlated_features = set()
correlated_to=[]

correlation_matrix = df.corr()
import matplotlib.pyplot as plt
plt.subplots(figsize=(12,9))
#sns.heatmap(correlation_matrix, vmax=0.9, square=True)

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            colname_prime = correlation_matrix.columns[j]
            correlated_features.add(colname)
            correlated_to.append([colname_prime,colname,abs(correlation_matrix.iloc[i, j])])
           
            

df.drop(labels=correlated_features, axis=1, inplace=True)
df.to_csv(mypath+'fraud_featureSelection1.csv')

#chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
"selector=SelectKBest(chi2,100)"
selector=SelectPercentile(chi2,60)
newdt=selector.fit_transform(df.drop(labels=['isFraud'], axis=1), df['isFraud'])
mask=selector.get_support()

col_names=df.drop(labels=['isFraud'], axis=1).columns
df_chi2=df
for i in range(0,len(mask)):
    if mask[i]==False:
        df_chi2=df_chi2.drop([col_names[i]],axis='columns')
   
df_chi2.to_csv(mypath+'feature_selected_fraud_30000.csv')

#Information gain
"""
sklearn.feature_selection.mutual_info_classif


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(
    df.drop(labels=['isFraud'], axis=1),
    df['isFraud'],
    test_size=0.1,
    random_state=41)
"""

"""
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
"""
