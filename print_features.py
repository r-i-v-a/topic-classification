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

k = 20

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))
top_tf_idf = pickle.load(open(features + "/top_tf_idf.p", 'rb'))
top_freq = pickle.load(open(features + "/top_freq.p", 'rb'))

print "\nbest mi\n"
print top_mi[:k]
print "\nbest x2\n"
print top_x2[:k]
print "\nbest tf-idf\n"
print top_tf_idf[:k]
print "\nbest freq\n"
print top_freq[:k]

top_mi.reverse()
top_tf_idf.reverse()
top_x2.reverse()
top_freq.reverse()

print "\nworst mi\n"
print top_mi[:k]
print "\nworst x2\n"
print top_x2[:k]
print "\nworst tf-idf\n"
print top_tf_idf[:k]
print "\nworst freq\n"
print top_freq[:k]