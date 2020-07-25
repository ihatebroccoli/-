# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:28:04 2020

@author: grago
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth= 2)
tree_clf.fit(X,y)
"""
from sklearn.tree import export_graphviz
export_graphviz(
       tree_clf,
       out_file= ("iris_tree.dot"),
       feature_names= iris.feature_names[2:],
       class_names= iris.target_names,
       rounded = True,
       filled = True
    )
"""

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth = 2)
tree_reg.fit(X,y)