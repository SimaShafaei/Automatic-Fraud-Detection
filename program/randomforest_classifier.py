# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:54:52 2019

@author: sshaf

using randomforest classifier for fraud detection 
n_estimators : The number of trees in the forest. :50,100,200
criterion =gini or entropy
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

from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(n_estimators=100, criterion='gini')
from sklearn.model_selection import cross_validate
cv_results = cross_validate(rf_clf, X, y, cv=10)

from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(rf_clf, X, y, cv=10, scoring='f1')
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

rf_clf.fit(X,y)
df_test = pd.read_csv (mypath+'fraud_test_prepared_30000.csv')
df_test=df_test.drop(['Unnamed: 0'],axis=1)
X2=np.array(df_test.drop(['isFraud','TransactionID'],axis=1))
y2=np.array(df_test['isFraud'])
y_pred=rf_clf.predict(X2)




from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y2, y_pred))
print("f1_score:",metrics.f1_score(y2, y_pred))


#0.9210833333333334



#X3=np.array(df_test.drop(['TransactionID','isFraud'],axis=1))
#y_pred=rf_clf.predict(X3)
#data = {'isFraud': y_pred} 
#  
## Convert the dictionary into DataFrame 
#result = pd.DataFrame(data) 
#result=pd.concat([result,df_test['TransactionID']],axis='columns')
#result.to_csv(mypath+'result3000_randomforest.csv')