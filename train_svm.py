#!/usr/bin/env python

from __future__ import division
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy

def run_clfs(train_data, train_lab, test_data, test_lab):

    # Multinomial Naive Bayes classification
    print "\nMULTINOMIAL NAIVE BAYES"
    nb_clf = MultinomialNB()
    nb_clf = nb_clf.fit(train_data, train_lab)

    # performance on training set
    predicted = nb_clf.predict(train_data)
    nb_acc = accuracy_score(train_lab, predicted) * 100
    nb_f1_avg = f1_score(train_lab, predicted, average='weighted')
    nb_f1_cat = f1_score(train_lab, predicted, average=None)

    print "\nperformance on training set:\n"
    print "accuracy:", nb_acc
    print "f1 average:", nb_f1_avg
    print "f1 by class:", nb_f1_cat

    # performance on test set
    predicted = nb_clf.predict(test_data)
    nb_acc = accuracy_score(test_lab, predicted) * 100
    nb_f1_avg = f1_score(test_lab, predicted, average='weighted')
    nb_f1_cat = f1_score(test_lab, predicted, average=None)

    print "\nperformance on test set:\n"
    print "accuracy:", nb_acc
    print "f1 average:", nb_f1_avg
    print "f1 by class:", nb_f1_cat

    # Support Vector Machine classification
    print "\nSUPPORT VECTOR MACHINE"
    svm_train_acc = 0
    svm_train_f1 = 0
    svm_test_acc = 0
    svm_test_f1 = 0
    num_sessions = 10

    for it in range(num_sessions):
        sgd_clf = Pipeline([('scale', RobustScaler()), 
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=0.2, n_iter=25,
                                                  random_state=it)), ])
        sgd_clf = sgd_clf.fit(train_data, train_lab)

        # performance on training set
        predicted = sgd_clf.predict(train_data)
        svm_acc = accuracy_score(train_lab, predicted) * 100
        svm_f1_avg = f1_score(train_lab, predicted, average='weighted')
        svm_f1_cat = f1_score(train_lab, predicted, average=None)
        svm_train_acc += svm_acc
        svm_train_f1 += svm_f1_avg

        print "\nsession", it, "training set:\n"
        print "accuracy:", svm_acc
        print "f1 average:", svm_f1_avg
        print "f1 by class:", svm_f1_cat

        # performance on test set
        predicted = sgd_clf.predict(test_data)
        svm_acc = accuracy_score(test_lab, predicted) * 100
        svm_f1_avg = f1_score(test_lab, predicted, average='weighted')
        svm_f1_cat = f1_score(test_lab, predicted, average=None)
        svm_test_acc += svm_acc
        svm_test_f1 += svm_f1_avg

        print "\nsession", it, "test set:\n"
        print "accuracy:", svm_acc
        print "f1 average:", svm_f1_avg
        print "f1 by class:", svm_f1_cat

        '''
        # optional: save confusion matrix
        svm_cm = confusion_matrix(test_lab, predicted)
        numpy.save("confusion_" + str(it) + ".npy", svm_cm)
        '''

    # average performance over sessions
    svm_train_acc /= num_sessions
    svm_train_f1 /= num_sessions
    svm_test_acc /= num_sessions
    svm_test_f1 /= num_sessions

    print "\naverages over sessions"
    print "train accuracy:", svm_train_acc
    print "train f1 score:", svm_train_f1
    print "test accuracy:", svm_test_acc
    print "test f1 score:", svm_test_f1