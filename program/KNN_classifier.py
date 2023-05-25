# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:19:56 2019

@author: sshaf
"""

"""
Created on Fri Oct 18 14:54:52 2019

@author: sshaf

using K nearset neighbors classifier for fraud detection 
n_neighbors 

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

from sklearn.neighbors import KNeighborsClassifier


knn_clf=KNeighborsClassifier(n_neighbors=2)
from sklearn.model_selection import cross_validate
cv_results = cross_validate(knn_clf, X, y, cv=10)



from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
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

knn_clf.fit(X,y)
df_test = pd.read_csv (mypath+'feature_selected_original_ds_30000.csv')
df_test=df_test.drop(['Unnamed: 0'],axis=1)
X2=np.array(df_test.drop(['isFraud','TransactionID'],axis=1))
y2=np.array(df_test['isFraud'])
y_pred=knn_clf.predict(X2)




from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y2, y_pred))
print("f1_score:",metrics.f1_score(y2, y_pred))


a=[0,0,0,0,0,0,0,0,0]
a[1]=cv_results2.mean()
knn_clf=KNeighborsClassifier(n_neighbors=2)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean (N=2):'+str(cv_results2.mean()))
a[2]=cv_results2.mean()

knn_clf=KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean (N=3):'+str(cv_results2.mean()))
a[3]=cv_results2.mean()


knn_clf=KNeighborsClassifier(n_neighbors=4)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean:(N=4)'+str(cv_results2.mean()))
a[4]=cv_results2.mean()

knn_clf=KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean(N=5):'+str(cv_results2.mean()))
a[5]=cv_results2.mean()


knn_clf=KNeighborsClassifier(n_neighbors=6)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean(N=6):'+str(cv_results2.mean()))
a[6]=cv_results2.mean()

knn_clf=KNeighborsClassifier(n_neighbors=10)
from sklearn.model_selection import cross_val_score
cv_results2 = cross_val_score(knn_clf, X, y, cv=5, scoring='f1')
print('f1 score mean(N=10):'+str(cv_results2.mean()))
a[7]=cv_results2.mean()


"""
n_neighbors=1  --> 0.74
n_neighbors=3 --> 0.60
n_neighbors=5 --> 0.57 
n_neighbors=10 --> 0.52
"""
