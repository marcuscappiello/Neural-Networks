# Simple neural network to determine flower type

from __future__ import print_function # for python 2 and 3 compatibility
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_iris = pd.read_csv('iris.csv', header=None,
                      names=['sepal_length', 'sepal_width', 'label_str'])

# add bias feature
df_iris['bias'] = 1

# make column with binary label
str_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df_iris['label_int'] = df_iris['label_str'].apply(lambda label_str: str_to_int[label_str])
# Lambda is used as a place holder for a function when you don't need to call something many times
df_iris.tail()

# select data, features, labels
X = df_iris[['sepal_length', 'sepal_width', 'bias']]
y = df_iris['label_int']
X, y = X[:100], y[:100] # use 2 labels only (binary classification)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Using a seed for the random generator (42) allows you to replicate yourself (same data)

class BinaryLogisticRegressorPurePython(object):
    
    def __init__(self, lr, n_iter): # Initialization function
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            p_y = self.sigmoid_fn(X)
            error = (y - p_y)
            self.w_[1:] += self.lr * X.T.dot(error) 
            self.w_[0] += self.lr * error.sum()
            
            cost = -y.dot(np.log(p_y)) - ((1 - y).dot(np.log(1 - p_y)))
            self.cost_.append(cost)
        return self

    def net_input(self, X): # Dot product of x and weight
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.sigmoid_fn(X) >= 0.5, 1, 0)
    
    def sigmoid_fn(self, X):
        z = self.net_input(X)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid

        
        # fit model
model_3 = BinaryLogisticRegressorPurePython(n_iter=100, lr=0.001) # lr - how much to change the weights
model_3.fit(X_train.values, y_train.values)
plt.plot(model_3.cost_)
plt.show()
