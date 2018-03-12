# Singular value decomposition script

import sys, csv
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# input parameters
data_file = 'data1.csv'
dim1 = 0  # what SV dimensions to plot
dim2 = 1

# read data into a numpy matrix
data_matrix = np.array(list(csv.reader(open(data_file, "r"), delimiter=","))).astype("float")

# create some labels for the data
feature_labels = []
item_labels = []
sv_labels = []
for i in range(data_matrix.shape[0]):
    item_labels.append("item" + str(i + 1))
for i in range(data_matrix.shape[1]):
    sv_labels.append("sv" + str(i + 1))
    feature_labels.append("f" + str(i + 1))

# heatmap of the raw data
fig0, ax0 = plt.subplots()
ax0.imshow(data_matrix, interpolation='none')
ax0.set_xticks(np.arange(data_matrix.shape[1]) + 0.5, minor=False)
ax0.set_yticks(np.arange(data_matrix.shape[0]) + 0.5, minor=False)
ax0.set_xticklabels(feature_labels, minor=False)
ax0.set_yticklabels(item_labels, minor=False)

# compute the SVD
U, s, V = np.linalg.svd(data_matrix)
print(s)

# plot U and V (can be thought of as row and column-wise principal components)
fig1, axarr1 = plt.subplots(1, 2)
for ax, mat, ytlabels in zip(axarr1, [U, V], [item_labels, feature_labels]):
    ax.imshow(mat, interpolation='none')
    ax.set_yticks(np.arange(V.shape[0]) + 0.5, minor=False)
    ax.set_yticklabels(ytlabels, minor=False)
    ax.set_xticks([])
    # plot singular values on top of imshow (equal to square roots of the eigenvalues of covariance matrix)
    ax.plot(s)

# dimensionality reduction using above specified dimensions
# eigenvectors (of covariance matrix) are given by columns in U
fig2, ax2 = plt.subplots()
ax2.plot(U[:, dim1], U[:,dim2], "o")
ax2.set_xlabel('dim1')
ax2.set_ylabel('dim2')
plt.suptitle('Dimensionality Reduction with First 2 PCs', fontweight='bold', fontsize=14)

# plot the cluster diagram
data_dist = pdist(data_matrix) # computing the distance
data_link = linkage(data_dist) # computing the linkage
fig3, ax3 = plt.subplots()
dendrogram(data_link, ax=ax3)
ax3.set_xlabel('Items')
ax3.set_ylabel('Distance')
plt.suptitle('Hierarchical clustering', fontweight='bold', fontsize=14)

# show all figures
plt.show()
