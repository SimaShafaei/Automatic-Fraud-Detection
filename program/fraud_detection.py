
import os
import pandas as pd

os.getcwd()
pd.options.display.max_colwidth = 5000
#seperation of feature and class

X=np.array(fraud_data.drop(['isFraud'],axis=1))
Y=np.array(fraud_data['isFraud'])



#split dataset for test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=4)
x_train_mod=x_train.reshape(-1,1)
x_test_mod=x_test.reshape(-1,1)
y_train_mod=y_train.reshape(-1,1)
y_test_mod=y_test.reshape(-1,1)







