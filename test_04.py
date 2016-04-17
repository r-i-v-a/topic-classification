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

files_counts = "./files_counts.txt"
features = datadir + "/features"

# get paths to word count files
with open(files_counts, 'r') as file:
	files = [line.replace('\n', '') for line in file.readlines()]

# get document categories
print "getting document categories"
doc_cats, cats = topic.cats(datadir)

# get document term counts
print "getting document term counts"
doc_terms, terms = topic.count_lists(files, doc_cats, datadir)

# separate training and test sets
set_size = len(doc_cats.keys()) // 2
set_train = set(random.sample(doc_cats.keys(), set_size))
set_test = set(doc_cats.keys()) - set_train

print len(set_train)
print len(set_test)
print len(doc_cats)

# separate training and test sets
set_size = 20
set_train = set(random.sample(doc_cats.keys(), set_size))
set_test = set(random.sample(doc_cats.keys(), set_size))

print len(set_train)
print len(set_test)
print len(doc_cats)
