#!/usr/bin/env python

from __future__ import division
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy

# trains Naive Bayes and support vector machine classifiers

def run_clfs(train_data, train_lab, test_data, test_lab):

    # Multinomial Naive Bayes classification
    print "\nMULTINOMIAL NAIVE BAYES"
    nb_clf = MultinomialNB()
    nb_clf = nb_clf.fit(train_data, train_lab)

    # performance on training set
    predicted = nb_clf.predict(train_data)
    nb_acc = accuracy_score(train_lab, predicted) * 100
    nb_f1 = f1_score(train_lab, predicted, average=None)

    print "\nperformance on training set:\n"
    print "accuracy:", nb_acc
    print "f1 by class:", nb_f1

    # performance on test set
    predicted = nb_clf.predict(test_data)
    nb_acc = accuracy_score(test_lab, predicted) * 100
    nb_f1 = f1_score(test_lab, predicted, average=None)

    print "\nperformance on test set:\n"
    print "accuracy:", nb_acc
    print "f1 by class:", nb_f1

    # Support Vector Machine classification
    print "\nSUPPORT VECTOR MACHINE"
    num_sessions = 10
    num_classes = len(numpy.unique(test_lab))
    svm_avg_acc_train = 0
    svm_avg_acc_test = 0
    svm_avg_f1_train = numpy.zeros(num_classes)
    svm_avg_f1_test = numpy.zeros(num_classes)

    for it in range(num_sessions):
        sgd_clf = Pipeline([('scale', RobustScaler()), 
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=0.2, n_iter=25,
                                                  random_state=it)), ])
        sgd_clf = sgd_clf.fit(train_data, train_lab)

        # performance on training set
        predicted = sgd_clf.predict(train_data)
        svm_acc = accuracy_score(train_lab, predicted) * 100
        svm_f1 = f1_score(train_lab, predicted, average=None)
        svm_avg_acc_train += svm_acc
        svm_avg_f1_train += svm_f1

        print "\nsession", it, "training set:\n"
        print "accuracy:", svm_acc
        print "f1 by class:", svm_f1

        # performance on test set
        predicted = sgd_clf.predict(test_data)
        svm_acc = accuracy_score(test_lab, predicted) * 100
        svm_f1 = f1_score(test_lab, predicted, average=None)
        svm_avg_acc_test += svm_acc
        svm_avg_f1_test += svm_f1

        print "\nsession", it, "test set:\n"
        print "accuracy:", svm_acc
        print "f1 by class:", svm_f1

        '''
        # optional: save confusion matrix
        svm_cm = confusion_matrix(test_lab, predicted)
        numpy.save("confusion_" + str(it) + ".npy", svm_cm)
        '''

    # average performance over sessions
    svm_avg_acc_train /= num_sessions
    svm_avg_acc_test /= num_sessions
    svm_avg_f1_train /= num_sessions
    svm_avg_f1_test /= num_sessions

    print "\naverages over sessions"
    print "train accuracy:", svm_avg_acc_train
    print "train f1 score:", svm_avg_f1_train
    print "test accuracy:", svm_avg_acc_test
    print "test f1 score:", svm_avg_f1_test
