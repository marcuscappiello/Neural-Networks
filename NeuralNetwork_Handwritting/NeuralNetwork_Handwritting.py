# Script for simple neural network to determine what character is written

from __future__ import print_function # for python 2 and 3 compatibility
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ['label'] # Set up name for loading data
feature_names = map(str, range(784))  # Create a list of strings with number of items
names.extend(feature_names) # Add on the list of strings to names

df_mnist_train = pd.read_csv('mnist_train.csv', header=None, nrows=1000, names=names) # Load data for training, returns dataframe
df_mnist_test = pd.read_csv('mnist_test.csv', header=None, nrows=100, names=names) # Load data for test

max_pixel =  255 # Set max_pixel
# normalize data
df_mnist_train.iloc[:, 1:] /= max_pixel # dataframe.iloc - integer location of axis
df_mnist_test.iloc[:, 1:] /= max_pixel # df_mnist_train.iloc = df_mnist_train.iloc/255

# print(df_mnist_train.shape, df_mnist_test.shape) # Look at sizes of data

# plot the first 3 examples of the mnist train dataset

n_examples = 3
for i in range(n_examples):
    example = df_mnist_train.iloc[i,:].values # df_mnist_train.iloc[index of example,pixels of example]
    label = int(example[0]) # What the example is supposed to be

    pixels = example[1:] # Collect pixels
    pixels = np.array(pixels) # Convert to array (1D)
    pixels = pixels.reshape((28, 28)) # Convert to 2D matrix

    plt.title('Label is {}'.format(label)) # Set the title of the plot
    plt.imshow(pixels, cmap='gray') # Draw image to current figure
    plt.show() # Display figure on gui backend

    # select subset of features and labels

X_train, y_train = df_mnist_train.iloc[:, 1:], df_mnist_train['label'] # For train sets: X-input, Y-correct answer
X_test, y_test = df_mnist_test.iloc[:, 1:], df_mnist_test['label'] # For test sets: X-input, Y-correct answer

# print(X_train.shape, y_train.shape) # Make sure the sizes make sense


class MultiClassLogisticRegressorPurePython(object): # Set up class
    def __init__(self, lr, n_iter): # Initialize class variables
        self.lr = lr # Set learning rate for the current class
        self.n_iter = n_iter # Set the number of iterations to current class

    def fit(self, X, y): # Fit the current dataset (X and Y)
        self.n_classes_ = len(np.unique(y))  # All unique possibilities (1-10)
        self.m_ = X.shape[0] # Size of training set
        self.w_ = np.zeros((X.shape[1], self.n_classes_)) # Initialize weights as 0
        self.cost_ = [] # Initialize cost function (empty)
        for i in range(self.n_iter): # For each iteration
            # print(i)
            z = self.net_input(X) # Get dotproduct of input with weights, output 'nodes' [10,1000]
            assert not np.isnan(np.sum(z)) # Catch any NaNs
            p_y = self.softmax_fn(z) # Take result so far and turn it into probability distribution [10,1000]
            y_onehot = self.onehot_fn(y) # Create binary sequence for each possible target [10,1000]
            error = (y_onehot - p_y) # Compute error as difference between prob dist. and correct answer [10,1000]
            grad = (-1 / self.m_) * X.T.dot(error) # Calculate gradient
            self.w_ = self.w_ - (self.lr * grad) # Adjust weights in the direction of gradient by learning rate
            cost = (-1 / self.m_) * np.sum(y_onehot * np.log(p_y)) # Ccalulate logistic regression cost function
            self.cost_.append(cost) # Keep track of cost for each iteration

        return self # Return all variables in self

    def onehot_fn(self, y): # Set up onehot_fn function
        onehot = np.eye(self.n_classes_)[y] # Create identity matrix for each class, apply this to all values in y
        return onehot # Return the onehot variable

    def net_input(self, X): # Set up net_input function
        return np.dot(X, self.w_) # Find dot product between input and weights

    def predict(self, X): # Set up predict function
        z = self.net_input(X) # Get dot product between input and weights
        return np.argmax(self.softmax_fn(z), axis=1) # Return prediction (as probability matrix)

    def softmax_fn(self, z): # Set up softmax_fn function
        z -= np.max(z) # z needs to be negative (the more negative, the farther from the most probable)
        softmax = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T # Create prob distribution using softmax function
        return softmax # Return softmax variable

# # instantiate and fit model
n_iter = 10 # Set number of iterations
model_3 = MultiClassLogisticRegressorPurePython(n_iter=n_iter, lr=0.01) # Run n_iter iterations of logistic regression
model_3.fit(X_train, y_train) # Fit training set
plt.plot(model_3.cost_) # Create the cost function plot
plt.show() # Show plot

# # evaluate model
train_acc = np.sum(model_3.predict(X_train) == y_train) / float(len(X_train)) # For training: Average the number of times the prediction matches the target
test_acc = np.sum(model_3.predict(X_test) == y_test) / float(len(X_test)) # For test: Average the number of times the prediction matches
print('Train accuracy: {}'.format(train_acc)) # Print out train accuracy
print('Test accuracy: {}'.format(test_acc)) # Print out test accuracy

class confusion_matrix_class(object): # Set up HW class
    def __init__(self, n_iter): # Initialize class variables
        self.n_iter = n_iter # Set the number of iterations to current class
    def confusion_matrix(self, y, y_pred): # Set up confusion_matrix function
        self.CM_ = pd.crosstab(y, y_pred) # Find cross-tabulation of the data the sexy way

        # Find confusion matrix the non sexy way (I'm sure there's a better way to do this, but this works)
        self.CM_2 = np.zeros((10,10)) # Initialize the confustion matrix to zeros
        for i in np.unique(y.values): # For all target values
            for j in np.unique(y_pred): # For all predicted values

                idxy = np.where(y==i) # Find index where they target equals the current target
                idxy_pred = np.where(y_pred==j) # Find index where they target equals the current predicted value
                self.CM_2[i,j] = len(np.intersect1d(idxy, idxy_pred)) # Collect the number of times the target and predicted values are equal

        # print(self.CM_) # Print the sexy confusion matrix
        print(self.CM_2) # Print the not-so-sexy confusion matrix
        return self

    def plot_wrongs(self,wrong_list,num_plot):
        for i in range(num_plot): # Loop through the number of examples
            curEx = wrong_list[i] # Get index of example to plot
            example = df_mnist_train.iloc[curEx,:].values # Get pixel values of the current exampl
            label = int(example[0]) # Get the label for the current example
            pixels = example[1:] # Collect pixels
            pixels = np.array(pixels) # Convert to array (1D)
            pixels = pixels.reshape((28, 28)) # Convert to 2D matrix

            plt.title('Label is {}'.format(label)) # Set the title of the plot
            plt.imshow(pixels, cmap='gray') # Draw image to current figure
            plt.show() # Display figure on gui backend


confusion_matrix_class_1 = confusion_matrix_class(n_iter) # Set up the HW class
confusion_matrix_class_1.confusion_matrix(y_train,model_3.predict(X_train)) # Run the confusion matrid

wrong_list = np.where(model_3.predict(X_train) != y_train) # Find examples that were wrong
confusion_matrix_class_1.plot_wrongs(wrong_list[0],3) # Run the plotting function
