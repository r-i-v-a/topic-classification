#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import numpy
import random

# contains shared functions used in other scripts
# for feature selection and generating doc vectors

# get document categories
def cats(cat_docs):

	doc_cats = {}

	for cat_id in cat_docs:
		for doc_id in cat_docs[cat_id]:
			doc_cats[doc_id] = cat_id_to_num(cat_id)

	# update set of categories
	cats = set()
	cats.update(doc_cats.values())

	return doc_cats, cats

# get document term counts
def count_lists(files, doc_cats, datadir):

	# to return
	doc_terms = {}
	terms = set()

	# iterate over documents that have labels
	for doc_id in doc_cats:
		file_path = doc_id_to_path(doc_id, files, datadir)
		curr_file = {}

		# get term counts for current document
		with open(file_path, 'r') as file:
			for line in file:
				tokens = line.split()
				term = tokens[1]
				count = int(tokens[0])
				curr_file[term] = count

		# add current document to doc_terms
		doc_terms[doc_id] = curr_file

		# update set of terms
		terms.update(curr_file.keys())

	return doc_terms, terms

# generate term frequency vectors in sizes: k_vals
def make_vectors(top, dir, doc_cats, doc_terms, set_train, set_test, k_vals):

	# iterate over output vector sizes
	for k in k_vals:
		print "vector size:", k

		# make training set vectors
		x_train = numpy.zeros((len(set_train), k))
		y_train = numpy.zeros(len(set_train))

		# get term frequencies
		for i, doc_id in enumerate(set_train):
			y_train[i] = doc_cats[doc_id]
			for j, item in enumerate(top[:k]):
				term = item[0]
				if term in doc_terms[doc_id]:
					x_train[i,j] = doc_terms[doc_id][term]

		# save x-vectors as files
		target_file = dir + "/x_train_" + str(k) + ".npy"
		with open(target_file, 'wb') as file:
			numpy.save(file, x_train)

		# save y-vectors as files
		target_file = dir + "/y_train_" + str(k) + ".npy"
		with open(target_file, 'wb') as file:
			numpy.save(file, y_train)

		# make test set vectors
		x_test = numpy.zeros((len(set_test), k))
		y_test = numpy.zeros(len(set_test))

		# get term frequencies
		for i, doc_id in enumerate(set_test):
			y_test[i] = doc_cats[doc_id]
			for j, item in enumerate(top[:k]):
				term = item[0]
				if term in doc_terms[doc_id]:
					x_test[i,j] = doc_terms[doc_id][term]

		# save x-vectors as files
		target_file = dir + "/x_test_" + str(k) + ".npy"
		with open(target_file, 'wb') as file:
			numpy.save(file, x_test)

		# save y-vectors as files
		target_file = dir + "/y_test_" + str(k) + ".npy"
		with open(target_file, 'wb') as file:
			numpy.save(file, y_test)

# return path to counts file for doc ID
def doc_id_to_path(doc_id, files, datadir):
	return datadir + files[int(doc_id)-1]

# return integer ID for category
def cat_id_to_num(cat_id):
	return int(cat_id[3:])

# split set randomly in half
def split_set(original):
	new_size = len(original) // 2
	new_a = set(random.sample(original, new_size))
	new_b = set(original) - new_a
	return new_a, new_b