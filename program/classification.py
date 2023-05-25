# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:23:45 2019

@author: sshaf
"""





#split dataset for test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=4)
x_train_mod=x_train.reshape(-1,1)
x_test_mod=x_test.reshape(-1,1)
y_train_mod=y_train.reshape(-1,1)
y_test_mod=y_test.reshape(-1,1)

