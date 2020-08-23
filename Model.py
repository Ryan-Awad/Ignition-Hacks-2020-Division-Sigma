# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 12:46:31 2020

@author: Robbot
"""

from sklearn import svm
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf = svm.SVC()
clf.fit(X, y)

print(clf.predict([[2, 2]]))
