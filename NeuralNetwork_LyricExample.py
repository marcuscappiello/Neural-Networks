# Code to determine how lyrics can determine what genre the music is

from __future__ import print_function # for python 2 and 3 compatibility

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read csv
all_charts = pd.read_csv('BillboardLyricData.txt', sep='\t', encoding='utf-8')
all_charts = all_charts.dropna()

# countvecotrize data
num_features = 5000
vectorizer = CountVectorizer(max_df=0.5, min_df=0.0, max_features=num_features, stop_words='english')
X = np.asarray(vectorizer.fit_transform(all_charts.lyrics).todense()).astype(np.float32)

input_feature_size = [50, 100, 500, 1000, 5000]
hidden_nodes = [10, 20, 50, 100]

# y to ints
labels = np.unique(all_charts.chart).tolist()
num_labels = len(labels)
class_mapping = {label:idx for idx,label in enumerate(labels)}
y = all_charts.chart.map(class_mapping)

for i in range(input_feature_size):
    for j in range(hidden_nodes):


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X[1:input_feature_size(i)], y, test_size=0.3)

# scale
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

model = MLPClassifier(alpha=1e-5,
                      hidden_layer_sizes=(100, 50),
                      activation='logistic',
                      batch_size=10,
                      learning_rate_init=0.01,
                      learning_rate='constant')
model.fit(X_train_std, y_train)


# evaluate model
train_acc = model.score(X_train_std, y_train)
test_acc = model.score(X_test_std, y_test)
print('Train accuracy: {}'.format(train_acc))
print('Test accuracy: {}'.format(test_acc))