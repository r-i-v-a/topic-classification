#!/usr/bin/env python

from __future__ import division
import chi_squared
import cPickle as pickle
import mutual_information
import numpy
import sys
import tf_idf
import topic

# select feature terms using mutual information, chi-squared, tf-idf

datadir = sys.argv[1]

files_counts = "./files_counts.txt"
features = datadir + "/features"

# get mapping from categories to documents
cat_docs = pickle.load(open(datadir + "/cat_docs.p", 'rb'))

# separate docs into training and test sets, balance categories
set_select = set()
set_svm_train = set()
set_svm_test = set()

for cat in cat_docs:
	add_select, add_svm = topic.split_set(cat_docs[cat])
	add_svm_train, add_svm_test = topic.split_set(add_svm)

	set_select.update(add_select)
	set_svm_train.update(add_svm_train)
	set_svm_test.update(add_svm_test)

# get paths to word count files
with open(files_counts, 'r') as file:
	files = [line.replace('\n', '') for line in file.readlines()]

# get document categories
print "getting document categories"
doc_cats, cats = topic.cats(cat_docs)

# get document term counts
print "getting document term counts"
doc_terms, terms = topic.count_lists(files, doc_cats, datadir)

# save document data
with open(datadir + "/doc_cats.p", 'wb') as file:
	pickle.dump(doc_cats, file)
with open(datadir + "/doc_terms.p", 'wb') as file:
	pickle.dump(doc_terms, file)
with open(datadir + "/set_select.p", 'wb') as file:
	pickle.dump(set_select, file)
with open(datadir + "/set_svm_train.p", 'wb') as file:
	pickle.dump(set_svm_train, file)
with open(datadir + "/set_svm_test.p", 'wb') as file:
	pickle.dump(set_svm_test, file)

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
		for doc_id in set_select:
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

# for each term: compute TF-IDF, FREQ
tf_idf_by_term, freq_by_term = tf_idf.tf_idf(list(terms), doc_terms, vocab_size, set_select)

# save (term, value) pairs: MI
print "saving (term, value) pairs: MI"
top_mi = sorted(mi_by_term.items(), key = lambda (k,v): v, reverse = True)
with open(features + "/top_mi.p", 'wb') as file:
	pickle.dump(top_mi, file)

# save (term, value) pairs: X2
print "saving (term, value) pairs: X2"
top_x2 = sorted(x2_by_term.items(), key = lambda (k,v): v, reverse = True)
with open(features + "/top_x2.p", 'wb') as file:
	pickle.dump(top_x2, file)

# save (term, value) pairs: TF-IDF
print "saving (term, value) pairs: TF-IDF"
top_tf_idf = sorted(tf_idf_by_term, key = lambda (k,v): v, reverse = True)
with open(features + "/top_tf_idf.p", 'wb') as file:
	pickle.dump(top_tf_idf, file)

# save (term, value) pairs: FREQ
print "saving (term, value) pairs: FREQ"
top_freq = sorted(freq_by_term, key = lambda (k,v): v, reverse = True)
with open(features + "/top_freq.p", 'wb') as file:
	pickle.dump(top_freq, file)