# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:27:03 2020

@author: grago
"""

from sklearn.datasets import load_wine

wine = load_wine()
X = wine["data"]
y = wine["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(y_train.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

tre_clf = DecisionTreeClassifier(max_depth = 3)
svc_clf = SVC(kernel="poly", degree=3, coef0 = 1, C = 5)
rnd_clf = RandomForestClassifier(n_estimators = 100, max_depth = 3)

GBC = GradientBoostingClassifier()

tre_clf.fit(X_train,y_train)
svc_clf.fit(X_train,y_train)
rnd_clf.fit(X_train,y_train)

t_pred = tre_clf.predict(X_test)
s_pred = svc_clf.predict(X_test)
r_pred = rnd_clf.predict(X_test)
from sklearn.metrics import accuracy_score


print (accuracy_score(t_pred, y_test))
print (accuracy_score(s_pred, y_test))
print (accuracy_score(r_pred, y_test))

import numpy as np

new_arr = np.array([t_pred,s_pred,r_pred])

new_arr = np.transpose(new_arr)

GBC.fit(new_arr, y_test)
GBC_pred = GBC.predict(new_arr)
print(accuracy_score(y_test, GBC_pred))
