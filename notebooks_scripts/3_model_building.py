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
training=pd.read_excel(r'data/bike_train.xlsx')

#choose relevant columns 
training.columns
model=training[['season','yr','mnth','hr','holiday','weekday','workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt']]

#the baseline model will be made with the raw data, since there are normalization for continuous variables and dummies for categorical
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

rid_y_train_pred = rid.predict(X_train)
rid_y_test_pred = rid.predict(X_test)

print('RMSE train data:', round((mean_squared_error(y_train, rid_y_train_pred,squared=False)),2))
print('RMSE test data:', round((mean_squared_error(y_test, rid_y_test_pred,squared=False)),2))


#predict the data test, we'll with random forest as is the best RMSE for the baseline model
test=pd.read_excel(r'data/bike_test.xlsx')
test=test[['season','yr','mnth','hr','holiday','weekday','workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed']]


rf_y_test_pred = rf.predict(test)
pred = pd.DataFrame(rf_y_test_pred)
pred.columns=['pred']
pred.to_csv('cvillarragamo.csv',index=False)