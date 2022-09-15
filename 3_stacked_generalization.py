import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np

import matplotlib.pyplot as plt

from discover_feature_relationships import discover

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


def rmse(est,X_train, X_test, y_train,y_test):
    est.fit(X_train,y_train)

    est_y_train_pred = est.predict(X_train)
    est_y_test_pred = est.predict(X_test)

    print('RMSE train data:', round((mean_squared_error(y_train, est_y_train_pred,squared=False)),2))
    print('RMSE test data:', round((mean_squared_error(y_test, est_y_test_pred,squared=False)),2))

    pred = pd.DataFrame(est_y_test_pred, columns = ['pred'])
    return pred


#Decision Tree
dt = tree.DecisionTreeRegressor(random_state=42)
decision_Tree=rmse(dt, X_train, X_test, y_train,y_test)


#Random Forest
rf = RandomForestRegressor(random_state = 42)
randomf=rmse(rf,X_train, X_test, y_train,y_test)

#Ridge Regression
rid = Ridge(alpha=0.13)
ridge=rmse(rid,X_train, X_test, y_train,y_test)





#predict the data test, we'll with random forest as is the best RMSE for the baseline model
test=pd.read_excel('bike_test.xlsx')
test=test[['season','yr','mnth','hr','holiday','weekday','workingday','weathersit', 'temp', 'atemp', 'hum', 'windspeed']]


rf_y_test_pred = rf.predict(test)
pred = pd.DataFrame(rf_y_test_pred, columns = ['pred'])
pred.to_csv('cvillarragamo.csv',index=False)