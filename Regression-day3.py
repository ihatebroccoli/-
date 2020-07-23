# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:07:19 2020

@author: grago
"""
import numpy as np
m = 100
X = 6 * np.random.rand(m,1) - 3
y = X


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 10, include_bias = False)
X_Poly = poly_features.fit_transform(X)


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha = 1, solver = "cholesky")
ridge_reg.fit(X,y)
print(ridge_reg.predict([[1.5]]))

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 5, penalty = "l2")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))

from sklearn.linear_model import Lasso
lasso_reg =Lasso(alpha = 0.1)
lasso_reg.fit(X,y)
print(lasso_reg.predict([[1.5]]))

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elastic_net.fit(X,y)
print(elastic_net.predict([[1.5]]))

import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver = 'liblinear')
log_reg.fit(X,y)

X_new  = np.linspace(0, 3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-",label = "Iris-Virginica")
plt.plot(X_new, y_proba[:,0], "b--",label = "Non Iris-Virginica")
plt.show()

X = iris["data"][:,(2,3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver = "lbfgs", C=10)
softmax_reg.fit(X,y)
print(softmax_reg.predict_proba([[5,2]]))