import pydotplus
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.externals.six import StringIO 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from data_processing import *

class Trainer:
    def setConfigs(self, configs):
        pass
    
    def fit(self, X_train, X_test, y_train, y_test):
        pass


class KNNTrainer(Trainer):
    
    def setConfigs(self, configs):
        self.test_size = configs['test_size']
        self.n = configs['n']

    def fit(self, X_train, X_test, y_train, y_test):
        self.knn = KNeighborsClassifier(self.n)

        self.knn.fit(X_train.values, y_train.values.ravel())

        self.predictions = self.knn.predict(X_test.values)

        self.accuracy = accuracy_score(y_test.values, self.predictions)

        return self.accuracy
    
    def predict(self, X_test):
        prediction = self.knn.predict(X_test)

        return prediction
    