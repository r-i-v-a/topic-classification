#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import numpy
import sys
import topic

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tfidf = datadir + "/vectors_tfidf"
vectors_x2 = datadir + "/vectors_x2"

x_10 = pickle.load(open(vectors_mi + "/x_10.p", 'rb'))
y_10 = pickle.load(open(vectors_mi + "/y_10.p", 'rb'))

print x_10.shape()
print y_10.shape()

x_20 = pickle.load(open(vectors_mi + "/x_20.p", 'rb'))
y_20 = pickle.load(open(vectors_mi + "/y_20.p", 'rb'))

print x_20.shape()
print y_20.shape()