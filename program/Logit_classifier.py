# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:32:11 2019

@author: sshaf
using logistic classifier for fraud detection 
 

"""
import os
import pandas as pd
import numpy as np

os.getcwd()
pd.options.display.max_colwidth = 5000

#load data and remove unwanted variables
mypath='..\\data\\'
df = pd.read_csv (mypath+'feature_selected_fraud_30000.csv')
df=df.drop(['Unnamed: 0'],axis=1)
X=np.array(df.drop(['isFraud'],axis=1))
y=np.array(df['isFraud'])

from sklearn.linear_model import LogisticRegression


logit_clf=LogisticRegression(max_iter=100, penalty='l1')



from sklearn.model_selection import cross_validate
cv_results = cross_validate(logit_clf, X, y, cv=10)



from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(logit_clf, X, y, cv=10, scoring='f1')

print('f1 score mean:'+str(cv_results2.mean()))
print('accuracy mean:'+str(cv_results['test_score'].mean()))
print('fit time mean:'+str(cv_results['fit_time'].mean()))
print('score time mean:'+str(cv_results['score_time'].mean()))

fold = [1, 2, 3,4,5,6,7,8,9,10]

import matplotlib.pyplot as plt
plt.plot(fold, cv_results['fit_time'])
plt.xlabel('fold number')
plt.ylabel('fit time(s)')


plt.plot(fold, cv_results['score_time'])
plt.xlabel('fold number')
plt.ylabel('score time (s)')

plt.plot(fold, cv_results['test_score'])
plt.xlabel('fold number')
plt.ylabel('accuracy')

plt.plot(fold, cv_results2)
plt.xlabel('fold number')
plt.ylabel('f1_score')
#0.7574666666666666
"""
{'fit_time': array([29.08936119,  7.05363417, 14.34764552,  4.6270895 ,  8.67309785,
        15.92534065, 12.05754614,  9.94897866, 28.95665693,  8.79142976]),
 'score_time': array([0.01115561, 0.        , 0.00804615, 0.        , 0.        ,
        0.00400162, 0.        , 0.        , 0.        , 0.01008821]),
 'test_score': array([0.7745    , 0.76983333, 0.77666667, 0.756     , 0.7325    ,
        0.75966667, 0.74466667, 0.77883333, 0.74033333, 0.73733333])}
"""
