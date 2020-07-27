# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:11:22 2020

@author: grago
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets

X,y = datasets.make_moons(n_samples = 500, noise = 0.15, random_state = 42)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2) 

log_clf = LogisticRegression(solver = 'liblinear')
rnd_clf = RandomForestClassifier(n_estimators = 10)
svn_clf = SVC(gamma = 'auto')

voting_clf = VotingClassifier(
        estimators = [('lr', log_clf), ('rf', rnd_clf), ('svc',svn_clf )],
        voting = 'hard')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svn_clf, voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators = 500,
        max_samples= 100, bootstrap=True, n_jobs = -1, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
print(bag_clf.oob_score_)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import  RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes= 16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)

iris  = datasets.load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name,score)
    
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth= 1), n_estimators= 200,
        algorithm= "SAMME.R", learning_rate = 0.5
    )
ada_clf.fit(X_train, y_train)

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth = 2)
tree_reg1.fit(X,y)

y2 = tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth = 2)
tree_reg2.fit(X,y2)

y3 = tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth = 2)
tree_reg3.fit(X,y3)

from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators= 3, learning_rate = 1.0)
gbrt.fit(X,y)