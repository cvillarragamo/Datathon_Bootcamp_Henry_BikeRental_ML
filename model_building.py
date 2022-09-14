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
training=pd.read_excel('bike_train.xlsx')

#choose relevant columns 
training.columns
model=training[['season','yr','mnth','hr','holiday','weekday','workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt']]

#the data came with dummies for all categorical features, and some normalization for continuous variables, so we can go to train the model

# train test split 
X = model.drop('cnt', axis =1)
y = model.cnt


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#implementing the models

#Decision Tree

dt = tree.DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)


dt_y_train_pred = dt.predict(X_train)
dt_y_test_pred = dt.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, dt_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, dt_y_test_pred,squared=False)),2))
#These model is not the best fit for the data due the overfiting in the results!


#Random Forest
rf = RandomForestRegressor(random_state = 42)
rf.fit(X_train,y_train)

rf_y_train_pred = rf.predict(X_train)
rf_y_test_pred = rf.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, rf_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, rf_y_test_pred,squared=False)),2))


#Ridge Regression
rid = Ridge(alpha=1.0)
rid.fit(X_train,y_train)

rid_y_train_pred = rf.predict(X_train)
rid_y_test_pred = rf.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, rid_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, rid_y_test_pred,squared=False)),2))

