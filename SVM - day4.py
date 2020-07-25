# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:08:51 2020

@author: grago
"""
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris["data"][:, (2,3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C = 1, loss = "hinge")),
    ])
svm_clf.fit(X,y)
print(svm_clf.predict([[5.5,1.7]]))

from sklearn.preprocessing import PolynomialFeatures

X,y = datasets.make_moons(n_samples = 100, noise = 0.15, random_state = 42)

polynomial_svc_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree = 3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C = 10, loss = "hinge", max_iter= 2000))
    ])

polynomial_svc_clf.fit(X,y)

from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler())
        ("svm_clf", SVC(kernel = "poly", degeree = 3, coef =  1, C =5))
    ])
poly_kernel_svm_clf.fit(X,y)

rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel = "rbf", gamma = 5, C =0.001))
    ])
rbf_kernel_svm_clf.fit(X,y)
