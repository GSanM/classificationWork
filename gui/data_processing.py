import pydotplus
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def getDataset(filepath):
    return pd.read_csv(filepath)

def getXAndY(dataset, target_column):
    X = dataset.drop([target_column], axis=1)
    y = dataset[[target_column]]

    return X, y

def getTrainingAndTesting(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def undersample(dataset, samples, target_column, values):
    parts = []
    for value in values:
        parts.append(dataset[dataset[target_column]==value].sample(samples))
    
    result = pd.concat(parts)
    return result

def getAllColumnValues(dataset, column):
    classes = []
    for index, row in dataset.iterrows():
        if not (row[column] in classes):
            classes.append(int(row[column]))
    return classes
            