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

x_10 = numpy.load(open(vectors_mi + "/x_10.npy", 'rb'))
y_10 = numpy.load(open(vectors_mi + "/y_10.npy", 'rb'))

print x_10[1]
print y_10

x_20 = numpy.load(open(vectors_mi + "/x_20.npy", 'rb'))
y_20 = numpy.load(open(vectors_mi + "/y_20.npy", 'rb'))

print x_20
print y_20