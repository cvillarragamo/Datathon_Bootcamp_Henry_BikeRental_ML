import pandas as pd
from sklearn.model_selection import train_test_split


training=pd.read_excel('bike_train.xlsx')

# choose relevant columns 
training.columns
def split_train():
    model=training[['season','mnth','hr','weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt']]

#the data came with dummies for all categorical features, and some normalization for continuous variables, so we can go to train the model

# train test split 
    X = model.drop('cnt', axis =1)
    y = model.cnt.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test