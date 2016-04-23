#!/usr/bin/env python

from __future__ import division
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import numpy

def run_clfs(train_data, train_lab, test_data, test_lab):

    # classification: Multinomial Naive Bayes
    print "\nMULTINOMIAL NAIVE BAYES\n"

    nb_clf = MultinomialNB()
    nb_clf = nb_clf.fit(train_data, train_lab)
    predicted = nb_clf.predict(test_data)
    nb_acc = accuracy_score(test_lab, predicted) * 100
    nb_f1 = f1_score(test_lab, predicted, average=None)

    print "accuracy:", nb_acc
    print "f1 score:", nb_f1

    # classification: Support Vector Machine
    print "\nSUPPORT VECTOR MACHINE"

    num_classes = len(numpy.unique(test_lab))
    num_sessions = 10
    svm_avg_acc = 0
    svm_avg_f1 = numpy.zeros(num_classes)

    for it in range(num_sessions):
        sgd_clf = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=0.4, n_iter=25,
                                                  random_state=it)), ])

        sgd_clf = sgd_clf.fit(train_data, train_lab)
        predicted = sgd_clf.predict(test_data)
        svm_acc = accuracy_score(test_lab, predicted) * 100
        svm_f1 = f1_score(test_lab, predicted, average=None)
        svm_avg_acc += svm_acc
        svm_avg_f1 += svm_f1

        print "\ntraining session", it
        print "accuracy:", svm_acc
        print "f1 score:", svm_f1

    svm_avg_acc /= num_sessions
    svm_avg_f1 /= num_sessions

    print "\naverages over sessions"
    print "avg. accuracy:", svm_avg_acc
    print "avg. f1 score:", svm_avg_f1
