# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:15:50 2019

@author: sshaf
using SVM classifier for fraud detection 

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

#chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector=SelectKBest(chi2,20)
newdt=selector.fit_transform(df.drop(labels=['isFraud'], axis=1), df['isFraud'])
mask=selector.get_support()

col_names=df.drop(labels=['isFraud'], axis=1).columns
df_chi2=df
for i in range(0,len(mask)):
    if mask[i]==False:
        df_chi2=df_chi2.drop([col_names[i]],axis='columns')
        
X=np.array(df_chi2.drop(['isFraud'],axis=1))
y=np.array(df_chi2['isFraud'])


from sklearn import svm
svm_clf=svm.SVC(kernel='rbf' , degree=2)  #kernel='rbf' , gamma =auto,degree=3
from sklearn.model_selection import cross_validate
cv_results = cross_validate(svm_clf, X, y, cv=5)



from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(svm_clf, X, y, cv=5, scoring='f1')
print('f1 score mean:'+str(cv_results2.mean()))
print('accuracy mean:'+str(cv_results['test_score'].mean()))
print('fit time mean:'+str(cv_results['fit_time'].mean()))
print('score time mean:'+str(cv_results['score_time'].mean()))

fold = [1, 2, 3,4,5]

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


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=4)
#x_train_mod=x_train.reshape(-1,1)
#x_test_mod=x_test.reshape(-1,1)
#y_train_mod=y_train.reshape(-1,1)
#y_test_mod=y_test.reshape(-1,1)


#SVM Classifier
from sklearn import svm
model=svm.SVC(kernel='rbf' , degree=3)  #kernel='rbf' , gamma =auto,degree=3
model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
#accuracy=model.score(x_test_mod,y_test_mod)
#prediction=model.predict(x_test_mod,y_test_mod)

y_pred=model.predict(x_test)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("f1_score:",metrics.f1_score(y_test, y_pred))
