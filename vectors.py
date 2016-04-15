#!/usr/bin/env python

from __future__ import division
import chi_squared
import cPickle as pickle
import mutual_information
import numpy
import random
import sys
import tfidf
import topic

datadir = sys.argv[1]

features = datadir + "/features"
vectors_mi = datadir + "/vectors_mi"
vectors_tfidf = datadir + "/vectors_tfidf"
vectors_x2 = datadir + "/vectors_x2"

# k = number of features to select = final vector size
k_vals = [10, 20]

# load document data
doc_cats = pickle.load(open(datadir + "/doc_cats.p", 'rb'))
doc_terms = pickle.load(open(datadir + "/doc_terms.p", 'rb'))
set_test = pickle.load(open(datadir + "/set_test.p", 'rb'))

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_tfidf = pickle.load(open(features + "/top_tfidf.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))

# generate term frequency vectors
print "saving MI: document vectors"
topic.make_vectors(top_mi, vectors_mi, doc_cats, doc_terms, set_test, k_vals)

# generate term frequency vectors
print "saving X2: document vectors"
topic.make_vectors(top_x2, vectors_x2, doc_cats, doc_terms, set_test, k_vals)

# generate term frequency vectors
print "saving TF-IDF: document vectors"
topic.make_vectors(top_tfidf, vectors_tfidf, doc_cats, doc_terms, set_test, k_vals)