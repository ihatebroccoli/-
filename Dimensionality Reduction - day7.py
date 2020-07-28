# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:32:25 2020

@author: grago
"""

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np

"""
mnist = fetch_openml('mnist_784')
n_train = int(mnist.data.shape[0] * 0.8)
X_train = mnist.data[:n_train]
y_train = mnist.target[:n_train]
X_test = mnist.data[n_train:]
y_test = mnist.target[n_train:]


pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d= np.argmax(cumsum >= 0.95) + 1

print(d)

pca = PCA(n_components= 0.95)
X_reduced = pca.fit_transform(X_train)


plt.plot(cumsum)
plt.axis([0,500, 0.0, 1.0])
plt.show()

pca = PCA(n_components= 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)


incremental PCa
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)


Random PCA
rnd_pca = PCA(n_components = 154, svd_solver = 'randomized')
X_reduced = rnd_pca.fit_transform(X_train)

Kernel PCA

"""
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
X,color = make_swiss_roll(n_samples= 1000, noise = 0.2, random_state= 0)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
plt.axis('tight')
plt.show()


rbf_pca = KernelPCA (n_components = 2, kernel ="rbf", gamma = 0.04)
X_reduced  = rbf_pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X_reduced[:,0], X_reduced[:, 1],c = color)
plt.show()

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components= 2, n_neighbors = 10)
X_reduced  = lle.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(X_reduced[:,0], X_reduced[:, 1],c = color)
plt.show()

