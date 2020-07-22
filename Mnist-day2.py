# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:42:37 2020

@author: grago
"""
import tensorflow as tf
data_train, data_test = tf.keras.datasets.mnist.load_data()
(X_train, y_train) = data_train
(X_test, y_test) = data_test

import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter = 5, random_state = 42)
sgd_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled,y_train,cv = 3, scoring = "accuracy"))


from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)

from sklearn.metrics import f1_score

y_train_knn_pred = cross_val_score(knn_clf, X_train, y_multilabel, cv = 3, n_jobs = -1)
print(f1_score(y_multilabel, y_train_knn_pred, average = "marco"))