# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:39:01 2019

@author: sima_shaf

this program create numerical and balenced dataset for fraud detection
it covers following steps:
    on hot encoding
    upsampling
    downsampling
"""


import os
import pandas as pd

os.getcwd()
pd.options.display.max_colwidth = 5000

#load data and remove unwanted variables
mypath='..\\data\\'
dataset = pd.read_csv (mypath+'fraud.csv')
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
    dummies_col=pd.get_dummies(dataset[variable],prefix=variable,prefix_sep='_',drop_first=True)
    dataset=pd.concat([dataset,dummies_col],axis='columns')
    dataset=dataset.drop([variable],axis='columns')


#balence classes
fraud_part = dataset[(dataset['isFraud'] ==1)]
non_fraud_part=dataset[(dataset['isFraud']==0)]
fraud_count=fraud_part.shape[0]
non_fraud_count=non_fraud_part.shape[0]
ideal_ds_size=20000

##upsampling for non_fraud_part
up_sampling_num=ideal_ds_size - fraud_count

### randomly sample from class fraud with replacement
from sklearn.utils import resample
fraud_upsampled = resample(fraud_part, replace=False,n_samples=up_sampling_num)
balenced_fraud=pd.concat([fraud_upsampled, fraud_part],axis='index')

##downsampling for fraud_part 
balenced_non_fraud = resample(non_fraud_part, replace=False,n_samples=ideal_ds_size)

final_fraud=pd.concat([balenced_non_fraud, balenced_fraud],axis='index')

final_fraud.to_csv(mypath+'fraud_balenced_30000.csv')
