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

# k = number of features to select = final vector size
k_vals = [200, 400, 600, 800]

# load document data
doc_cats = pickle.load(open(datadir + "/doc_cats.p", 'rb'))
doc_terms = pickle.load(open(datadir + "/doc_terms.p", 'rb'))
set_train = pickle.load(open(datadir + "/set_svm_train.p", 'rb'))
set_test = pickle.load(open(datadir + "/set_svm_test.p", 'rb'))

# load feature lists
top_mi = pickle.load(open(features + "/top_mi.p", 'rb'))
top_tf_idf = pickle.load(open(features + "/top_tf_idf.p", 'rb'))
top_x2 = pickle.load(open(features + "/top_x2.p", 'rb'))

# generate term frequency vectors
print "saving document vectors: MI"
topic.make_vectors(top_mi, vectors_mi, doc_cats, doc_terms, set_train, set_test, k_vals)

# generate term frequency vectors
print "saving document vectors: X2"
topic.make_vectors(top_x2, vectors_x2, doc_cats, doc_terms, set_train, set_test, k_vals)

# generate term frequency vectors
print "saving document vectors: TF-IDF"
topic.make_vectors(top_tf_idf, vectors_tf_idf, doc_cats, doc_terms, set_train, set_test, k_vals)