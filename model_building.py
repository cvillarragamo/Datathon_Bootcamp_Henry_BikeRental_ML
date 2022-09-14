import pandas as pd
from sklearn.model_selection import train_test_split


training=pd.read_excel('bike_train.xlsx')

# choose relevant columns 
def split_train():
    model=training[['season','mnth','hr','weathersit', 'temp', 'atemp', 'hum', 'windspeed','cnt']]
    y = model.cnt.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test