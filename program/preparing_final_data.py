# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:43:08 2019

@author: sshaf
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:21:35 2019

@author: sshaf
"""



import os
import pandas as pd

os.getcwd()
pd.options.display.max_colwidth = 5000

#load data and remove unwanted variables
mypath='..\\data\\'
dataset = pd.read_csv (mypath+'fraud_test.csv')
dataset=dataset.drop(['P_emaildomain','R_emaildomain','DeviceInfo'],axis='columns')

#handling missing value:
categorical_features = dataset.select_dtypes(include=[object])
for c in categorical_features:
    if dataset[c].isna().any():
        dataset[c].fillna('undefined',inplace=True)


float_features = dataset.select_dtypes(include=['float64','float32','float16'])
for f in float_features:
    if dataset[f].isna().any():
        dataset[f].fillna(dataset[f].mean(),inplace=True)

int_features = dataset.select_dtypes(include=['int64','int32','int16','int8'])
for i in float_features:
    if dataset[i].isna().any():
        dataset[i].fillna(dataset[i].median(),inplace=True)    



#one hot encoding: convert categorical variable to binary features
for variable in categorical_features.columns:    
    dummies_col=pd.get_dummies(dataset[variable],prefix=variable,prefix_sep='_')
    dataset=pd.concat([dataset,dummies_col],axis='columns')
    dataset=dataset.drop([variable],axis='columns')
   

mypath='..\\data\\'
df = pd.read_csv (mypath+'feature_selected_fraud_30000.csv')

dataset2=dataset['TransactionID']
for i in df.columns:
        try:
            dataset2=pd.concat([dataset2,dataset[i]],axis='columns')
        except:
            df=df.drop([i],axis='columns')

                

df.to_csv(mypath+'feature_selected_fraud_30000.csv')
dataset2.to_csv(mypath+'fraud_test_prepared_30000.csv')
