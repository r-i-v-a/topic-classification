#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import sys
import topic

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tf_idf = datadir + "/vectors_tf_idf"
vectors_x2 = datadir + "/vectors_x2"

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_tf_idf = pickle.load(open(features + "/top_tf_idf.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))
top_freq = pickle.load(open(features + "/top_freq.p", 'rb'))

k = 10

print "\ntop_mi\n"
print top_mi[:k]
print "\ntop_tf_idf\n"
print top_tf_idf[:k]
print "\ntop_x2\n"
print top_x2[:k]
print "\ntop_freq\n"
print top_freq[:k]