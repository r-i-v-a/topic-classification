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

# get paths to word count files
with open(files_counts, 'r') as file:
	files = [line.replace('\n', '') for line in file.readlines()]

# get document categories
print "getting document categories"
doc_cats, cats = topic.cats(datadir)

# get document term counts
print "getting document term counts"
doc_terms, terms = topic.count_lists(files, doc_cats, datadir)

# k = number of features to select = final vector size
k_vals = [10, 20]

# generate term frequency vectors
print "saving MI: document vectors"
topic.make_vectors(top_mi, vectors_mi, doc_cats, doc_terms, set_test, k_vals)

# generate term frequency vectors
print "saving X2: document vectors"
topic.make_vectors(top_x2, vectors_x2, doc_cats, doc_terms, set_test, k_vals)

# generate term frequency vectors
print "saving TF-IDF: document vectors"
topic.make_vectors(top_tfidf, vectors_tfidf, doc_cats, doc_terms, set_test, k_vals)