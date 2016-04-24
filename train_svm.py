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

    # classification: Multinomial Naive Bayes
    print "\nMULTINOMIAL NAIVE BAYES\n"

    nb_clf = MultinomialNB()
    nb_clf = nb_clf.fit(train_data, train_lab)
    predicted = nb_clf.predict(test_data)

    nb_acc = accuracy_score(test_lab, predicted) * 100
    nb_f1_avg = f1_score(test_lab, predicted, average='weighted')
    nb_f1_cat = f1_score(test_lab, predicted, average=None)

    print "accuracy:", nb_acc
    print "f1 average:", nb_f1_avg
    print "f1 by class:", nb_f1_cat

    # classification: Support Vector Machine
    print "\nSUPPORT VECTOR MACHINE"

    svm_avg_acc = 0
    svm_avg_f1 = 0
    num_sessions = 10

    for it in range(num_sessions):
        sgd_clf = Pipeline([('scale', RobustScaler()), 
                            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=0.2, n_iter=25,
                                                  random_state=it)), ])

        sgd_clf = sgd_clf.fit(train_data, train_lab)
        predicted = sgd_clf.predict(test_data)
        svm_acc = accuracy_score(test_lab, predicted) * 100
        svm_f1_avg = f1_score(test_lab, predicted, average='weighted')
        svm_f1_cat = f1_score(test_lab, predicted, average=None)
        svm_avg_acc += svm_acc
        svm_avg_f1 += svm_f1_avg

        '''
        # optional: save confusion matrix
        svm_cm = confusion_matrix(test_lab, predicted)
        numpy.save("cm_" + str(it) + ".npy", svm_cm)
        '''

        print "\ntraining session", it
        print "accuracy:", svm_acc
        print "f1 average:", svm_f1_avg
        print "f1 by class:", svm_f1_cat

    svm_avg_acc /= num_sessions
    svm_avg_f1 /= num_sessions

    print "\naverages over sessions"
    print "avg. accuracy:", svm_avg_acc
    print "avg. f1 score:", svm_avg_f1