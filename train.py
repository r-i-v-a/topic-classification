#!/usr/bin/env python

import cPickle as pickle
import numpy
import sys
import train_svm

datadir = sys.argv[1]

vectors_mi = datadir + "/vectors_mi"
vectors_tf_idf = datadir + "/vectors_tf_idf"
vectors_x2 = datadir + "/vectors_x2"

vectors = vectors_mi
k = 600

train_x = numpy.load(vectors + "/x_train_" + str(k) + ".npy")
train_y = numpy.load(vectors + "/y_train_" + str(k) + ".npy")
test_x = numpy.load(vectors + "/x_test_" + str(k) + ".npy")
test_y = numpy.load(vectors + "/y_test_" + str(k) + ".npy")

print "data source:", vectors
print "vector size:", k

train_svm.run_clfs(train_x, train_y, test_x, test_y)