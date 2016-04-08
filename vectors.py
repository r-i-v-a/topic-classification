#!/usr/bin/env python

from __future__ import division
import chi_squared
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
doc_cats, cats = topic.cats()

# get statistics by: document, category, global
doc_terms, terms = topic.count_lists(files, doc_cats)

# for each term, category pair: compute MI, X2
mi_by_term = {}
x2_by_term = {}

for term in terms:
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
num_cats = len(cats)

for term in mi_by_term:
	mi_by_term[term] /= num_cats

for term in x2_by_term:
	x2_by_term[term] /= num_cats

# k = number of features to select = final vector size
k_vals = [10, 20, 30]

# select top k: mutual information
mi_top = sorted(mi_by_term.items(), key = lambda (k,v): v, reverse = True)

# print out top 100
print "\nTOP 100: MUTUAL INFORMATION\n"
for item in mi_top[:100]:
	print item

# generate term frequency vectors
topic.make_vectors(mi_top, doc_cats, doc_terms, k_vals, vectors_mi)

# select top k: chi-squared
x2_top = sorted(x2_by_term.items(), key = lambda (k,v): v, reverse = True)

# print out top 100
print "\nTOP 100: CHI-SQUARED\n"
for item in x2_top[:100]:
	print item

# generate term frequency vectors
topic.make_vectors(x2_top, doc_cats, doc_terms, k_vals, vectors_x2)