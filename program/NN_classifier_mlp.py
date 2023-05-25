# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 10:02:14 2019

@author: sshaf
"""

import os
import pandas as pd
import numpy as np

os.getcwd()
pd.options.display.max_colwidth = 5000

#load data and remove unwanted variables
mypath='..\\data\\'
df = pd.read_csv (mypath+'feature_selected_fraud.csv')

X=np.array(df.drop(['isFraud'],axis=1))
y=np.array(df['isFraud'])

from sklearn.ensemble import RandomForestClassifier


rf_clf=RandomForestClassifier(n_estimators=100, criterion='gini')
from sklearn.model_selection import cross_validate
cv_results = cross_validate(rf_clf, X, y, cv=10)
print(cv_results['test_score'].mean())
#0.9210833333333334
