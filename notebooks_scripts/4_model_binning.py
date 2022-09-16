import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

import matplotlib.pyplot as plt

#ML baseline models

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge



#metrics
from sklearn.metrics import mean_squared_error


#reading data
training=pd.read_csv(r'data/2_Feature_Engineering/2_feature_training.csv')

# train test split 
X = training.drop('cnt', axis =1)
y = training.cnt


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#implementing the models

#Decision Tree

dt = tree.DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)


dt_y_train_pred = dt.predict(X_train)
dt_y_test_pred = dt.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, dt_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, dt_y_test_pred,squared=False)),2))
#These i'm a bit scared about the error here, maybe this is wrong?!


#Random Forest
rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train,y_train)

rf_y_train_pred = rf.predict(X_train)
rf_y_test_pred = rf.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, rf_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, rf_y_test_pred,squared=False)),2))
#SAME THOUGHTS that with Decision Tree

#Ridge Regression ### The error here is 0, but I suspect that something is wrong, so not going to use this
rid = Ridge(alpha=0.13)
rid.fit(X_train,y_train)

rid_y_train_pred = rid.predict(X_train)
rid_y_test_pred = rid.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, rid_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, rid_y_test_pred,squared=False)),2))


#predict the data test, we'll with random forest as is the best RMSE for the baseline model
test=pd.read_csv(r'data/2_Feature_Engineering/2_feature_test.csv')


#I will upload both preduction, with Decision tree and Random Forest

#cvillarragamo(2)
dt_y_test_pred = dt.predict(test)
pred = pd.DataFrame(dt_y_test_pred)
pred.columns=['pred']
pred.to_csv('cvillarragamo.csv',index=False)

#cvillarragamo(3)
rf_y_test_pred = rf.predict(test)
pred = pd.DataFrame(rf_y_test_pred)
pred.columns=['pred']
pred.to_csv('cvillarragamo.csv',index=False)