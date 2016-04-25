#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import matplotlib.pyplot as pyplot
import sys
import topic

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

az_mi = sorted(top_mi, key = lambda (k,v): k)
az_x2 = sorted(top_x2, key = lambda (k,v): k)
az_tf_idf = sorted(top_tf_idf, key = lambda (k,v): k)
az_freq = sorted(top_freq, key = lambda (k,v): k)

print "\na-z: mi\n"
print az_mi[:k]
print "\na-z: x2\n"
print az_x2[:k]
print "\na-z: tf-idf\n"
print az_tf_idf[:k]
print "\na-z: freq\n"
print az_freq[:k]

ax = pyplot.subplot()
ax.hist([v for (k,v) in az_mi], 100, alpha=0.8)
pyplot.show()