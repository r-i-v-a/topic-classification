#!/usr/bin/env python

from __future__ import division
import numpy
import sys
import topic

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tfidf = datadir + "/vectors_tfidf"
vectors_x2 = datadir + "/vectors_x2"

x_10 = numpy.load(vectors_mi + "/x_10.npy")
y_10 = numpy.load(vectors_mi + "/y_10.npy")

x_20 = numpy.load(vectors_mi + "/x_20.npy")
y_20 = numpy.load(vectors_mi + "/y_20.npy")

print "\n10:"
print len(x_10), "x", len(x_10[0])
print len(y_10)

print "\n20:"
print len(x_20), "x", len(x_20[0])
print len(y_20)