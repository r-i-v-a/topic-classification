# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def run_clfs(train_data, train_lab, test_data, test_lab):
    """ Run multinomial naive bayes and SVM clfs """

    nb_clf = MultinomialNB()
    nb_clf = nb_clf.fit(train_data, train_lab)
    predicted = nb_clf.predict(test_data)

    nb_ac = np.mean(predicted == test_lab) * 100.0
    print('NB acc: %.2f' % nb_ac)
    # print('CR\n', classification_report(test_lab, predicted))

    # These parameters are set based on grid search on train data

    svm_scores = []
    for it in range(10):
        sgd_clf = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                  alpha=0.4, n_iter=25,
                                                  random_state=it+5)), ])

        sgd_clf = sgd_clf.fit(train_data, train_lab)
        predicted = sgd_clf.predict(test_data)

        svm_ac = np.mean(predicted == test_lab) * 100.0
        svm_scores.append(svm_ac)
        print('SVM acc: %.2f' % svm_ac)
        # print('CR\n', classification_report(test_lab, predicted))

    svm_avg = np.mean(svm_scores)
    print 'SVM avg:', svm_avg
    print 'SVM std:', np.std(svm_scores)
    return [nb_ac, svm_avg]
