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

'''
# separate training and test sets
set_size = len(doc_cats.keys()) // 2
set_train = set(random.sample(doc_cats.keys(), set_size))
set_test = set(doc_cats.keys()) - set_train
'''

# ?
# separate training and test sets
set_size = 20
set_train = set(random.sample(doc_cats.keys(), set_size))
set_test = set(random.sample(doc_cats.keys(), set_size))

# save document data
with open(datadir + "/doc_cats.p", 'wb') as file:
	pickle.dump(doc_cats, file)
with open(datadir + "/doc_terms.p", 'wb') as file:
	pickle.dump(doc_terms, file)
with open(datadir + "/set_train.p", 'wb') as file:
	pickle.dump(set_train, file)
with open(datadir + "/set_test.p", 'wb') as file:
	pickle.dump(set_test, file)

# get vocabulary size
vocab_size = len(terms)
print "vocabulary size:", vocab_size

# for each term, category pair: compute MI, X2
mi_by_term = {}
x2_by_term = {}

for i, term in enumerate(terms):
	print "calculating MI, X2:", "{:.3f}".format(100 * i / vocab_size), '%'
	for cat_id in cats:
		counts = numpy.ones((2,2))

		# generate matrix of doc counts for term, class
		for doc_id in set_train:
			i = 0
			j = 0
			if term in doc_terms[doc_id]:
				i = 1
			if doc_cats[doc_id] == cat_id:
				j = 1
			counts[i,j] += 1

		# add mutual information score for current term
		mi = mutual_information.mi(counts)
		if term in mi_by_term:
			mi_by_term[term] += mi
		else:
			mi_by_term[term] = mi

		# add chi-squared score for current term
		x2 = chi_squared.x2(counts)
		if term in x2_by_term:
			x2_by_term[term] += x2
		else:
			x2_by_term[term] = x2

# divide by categories to get average
print "calculating average MI, X2 for terms"
num_cats = len(cats)

for term in mi_by_term:
	mi_by_term[term] /= num_cats
for term in x2_by_term:
	x2_by_term[term] /= num_cats

# for each term: compute TF-IDF
tfidf_by_term = tfidf.tfidf(list(terms), doc_terms, vocab_size)
	
# select top k features: mutual information
print "saving MI: ranked terms"
top_mi = sorted(mi_by_term.items(), key = lambda (k,v): v, reverse = True)
with open(features + "/top_mi.p", 'wb') as file:
	pickle.dump(top_mi, file)

# select top k features: chi-squared
print "saving X2: ranked terms"
top_x2 = sorted(x2_by_term.items(), key = lambda (k,v): v, reverse = True)
with open(features + "/top_x2.p", 'wb') as file:
	pickle.dump(top_x2, file)

# select top k features: TF-IDF
print "saving TF-IDF: ranked terms"
top_tfidf = sorted(tfidf_by_term, key = lambda (k,v): v, reverse = True)
with open(features + "/top_tfidf.p", 'wb') as file:
	pickle.dump(top_tfidf, file)

