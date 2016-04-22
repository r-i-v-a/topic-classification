#!/usr/bin/env python

import cPickle as pickle
import numpy
import sys
import train_svm

datadir = sys.argv[1]

vectors_mi = datadir + "/vectors_mi"
vectors_tf_idf = datadir + "/vectors_tf_idf"
vectors_x2 = datadir + "/vectors_x2"

train_data = numpy.load(vectors_mi + "/x_train_200.npy")
train_lab = numpy.load(vectors_mi + "/y_train_200.npy")
test_data = numpy.load(vectors_mi + "/x_test_200.npy")
test_lab = numpy.load(vectors_mi + "/y_test_200.npy")

print type(train_data)
print type(train_lab)
print type(test_data)
print type(test_lab)

# train_svm.run_clfs(train_data, train_lab, test_data, test_lab)