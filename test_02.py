#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import sys
import topic

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tfidf = datadir + "/vectors_tfidf"
vectors_x2 = datadir + "/vectors_x2"

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_tfidf = pickle.load(open(features + "/top_tfidf.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))

print "MI"
print top_mi[:10]

print "X2"
print top_x2[:10]

print "TF-IDF"
print top_tfidf[:10]