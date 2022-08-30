#!/usr/bin/python
# -*- coding: utf-8 -*-
# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler and by Aldebaro Klautau
# License: BSD 3 clause

import numpy as np
import csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ml4comm.qam_analyzer import ser


def train_classifier(clf_name, X, y, num_classes):
    """
    Parameters
    ==========
    clf_name: str
        Classifier name to be selected
    X:
    y:
    num_classes: int
        Total number of classes

    """
    names = ["Naive Bayes",
             "Decision Tree", "Random Forest",
             "AdaBoost",
             "Linear SVM", "RBF SVM", "Gaussian Process",
             "Neural Net",
             "QDA", "Nearest Neighbors"]

    classifiers = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=100),
        RandomForestClassifier(max_depth=100, n_estimators=30),
        AdaBoostClassifier(),
        LinearSVC(), #linear SVM (maximum margin perceptron)
        SVC(gamma=1, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        MLPClassifier(alpha=0.1, max_iter=500),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3)]

    assert(clf_name in names)

    clf_ind = names.index(clf_name)
    clf = classifiers[clf_ind]
    clf.fit(X, y)

    return clf


def main():
    file_name = 'qam_awgn.csv'
    with open(file_name, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    data = np.array(data, dtype=float)

    X = data[:,:-1] # Data points
    y = data[:,-1]  # Labels

    train_size = int(0.5*len(y))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test  = X[train_size:]
    y_test  = y[train_size:]

    # iterate over classifiers
    names = ["Naive Bayes",
             "Decision Tree", "Random Forest",
             "AdaBoost",
             "Linear SVM", "RBF SVM", "Gaussian Process",
             "Neural Net",
             "QDA", "Nearest Neighbors"]

    for name in names:
        print("###### Training classifier: ", name)

        clf = train_classifier(name, X_train, y_train, np.max(y))
        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        print('\nPrediction accuracy for the train dataset:\t {:.2%}'.format(
            metrics.accuracy_score(y_train, pred_train)
        ))

        pred_test = clf.predict(X_test)
        print('\nPrediction accuracy for the test dataset:\t {:.2%}'.format(
            metrics.accuracy_score(y_test, pred_test)
        ))

        ser_test = ser(clf, X_test, y_test)
        print('\nSymbol error rate (SER) for the test dataset:\t {:.2%}'.format(
            ser_test
        ))

        print('\n\n')

if __name__ == '__main__':
    main()

