#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import matplotlib.pyplot as pyplot
import numpy
import sys
import topic

def rescale(a):
	a -= numpy.mean(a)
	a /= numpy.std(a)
	return a

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tf_idf = datadir + "/vectors_tf_idf"
vectors_x2 = datadir + "/vectors_x2"

k = 20

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))
top_tf_idf = pickle.load(open(features + "/top_tf_idf.p", 'rb'))
top_freq = pickle.load(open(features + "/top_freq.p", 'rb'))

mi = [v for (k,v) in sorted(top_mi, key = lambda (k,v): k)]
x2 = [v for (k,v) in sorted(top_x2, key = lambda (k,v): k)]
tf_idf = [v for (k,v) in sorted(top_tf_idf, key = lambda (k,v): k)]
freq = [v for (k,v) in sorted(top_freq, key = lambda (k,v): k)]

ax = pyplot.subplot()
ax.scatter(numpy.log10(freq), rescale(mi), marker='+')
pyplot.show()

ax = pyplot.subplot()
ax.scatter(numpy.log10(freq), rescale(x2), marker='+')
pyplot.show()

ax = pyplot.subplot()
ax.scatter(numpy.log10(freq), rescale(tf_idf), marker='+')
pyplot.show()