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

# test conversion of cat_id to integer
print cats