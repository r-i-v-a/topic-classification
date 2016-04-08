#!/usr/bin/env python

from __future__ import division
import chi_squared
import cPickle as pickle
import mutual_information
import numpy
import topic

files_counts = "./files_counts.txt"
vectors_mi = "./vectors_mi/"
vectors_x2 = "./vectors_x2/"

# get paths to word count files
with open(files_counts, 'r') as file:
	files = [line.replace('\n', '') for line in file.readlines()]

# get document categories
print "getting document categories"
doc_cats, cats = topic.cats()

# get document term counts
print "getting document term counts"
doc_terms, terms = topic.count_lists(files, doc_cats)

# for each term, category pair: compute MI, X2
mi_by_term = {}
x2_by_term = {}

# get vocabulary size
num_terms = len(terms)
print "vocabulary size:", num_terms

for i, term in enumerate(terms):
	print "calculating MI, X2:", "{:.3f}".format(100 * i / num_terms), '%'
	for cat_id in cats:
		counts = numpy.ones((2,2))

		# generate matrix of doc counts for term, class
		for doc_id in doc_cats:
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

# k = number of features to select = final vector size
k_vals = [10, 20, 30]

# select top k: mutual information
print "saving MI: ranked terms"
top_mi = sorted(mi_by_term.items(), key = lambda (k,v): v, reverse = True)
with open("top_mi.p", 'wb') as file:
	pickle.dump(top_mi, file)

# generate term frequency vectors
print "saving MI: document vectors"
topic.make_vectors(top_mi, doc_cats, doc_terms, k_vals, vectors_mi)

# select top k: chi-squared
print "saving X2: ranked terms"
top_x2 = sorted(x2_by_term.items(), key = lambda (k,v): v, reverse = True)
with open("top_x2.p", 'wb') as file:
	pickle.dump(top_x2, file)

# generate term frequency vectors
print "saving X2: document vectors"
topic.make_vectors(top_x2, doc_cats, doc_terms, k_vals, vectors_x2)