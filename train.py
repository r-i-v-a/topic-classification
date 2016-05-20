#!/usr/bin/env python

import cPickle as pickle
import numpy
import sys
import train_svm

# call training script on target files

datadir = sys.argv[1]

# directories containing document vectors for each method
vectors_mi = datadir + "/vectors_mi"
vectors_tf_idf = datadir + "/vectors_tf_idf"
vectors_x2 = datadir + "/vectors_x2"

# method + vector size to select
vectors = vectors_tf_idf
k = 200

train_x = numpy.load(vectors + "/x_train_" + str(k) + ".npy")
train_y = numpy.load(vectors + "/y_train_" + str(k) + ".npy")
test_x = numpy.load(vectors + "/x_test_" + str(k) + ".npy")
test_y = numpy.load(vectors + "/y_test_" + str(k) + ".npy")

print "data source:", vectors
print "vector size:", k

# run training script
train_svm.run_clfs(train_x, train_y, test_x, test_y)
