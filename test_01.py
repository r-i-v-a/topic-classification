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

print len(top_mi)
print len(top_tf_idf)
print len(top_x2)
print len(top_freq)

print "top_mi\n", top_mi[:10]
print "top_tf_idf\n", top_tf_idf[:10]
print "top_x2\n", top_x2[:10]
print "top_freq\n", top_freq[:10]