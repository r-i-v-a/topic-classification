#!/usr/bin/env python

from __future__ import division
import cPickle as pickle
import numpy

# get mapping from documents to categories
def cats():

	cat_docs = pickle.load(open("cat_docs.p", 'rb'))
	doc_cats = {}

	for cat_id in cat_docs:
		for doc_id in cat_docs[cat_id]:
			doc_cats[doc_id] = cat_id

	cats = set()
	cats.update(doc_cats.values())

	# TEST
	print doc_cats["05437"]
	print doc_cats["03374"]
	print doc_cats["00123"]
	print doc_cats["00024"]

	return doc_cats, cats

# get term counts by document for labeled documents
def count_lists(files, doc_cats):

	# to return
	doc_terms = {}
	terms = set()

	# iterate over documents that have labels
	for doc_id in doc_cats:
		file_path = doc_id_to_path(doc_id, files)
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
def make_vectors(top, doc_cats, doc_terms, k_vals, dir):
	print "\nsaving document vectors to ", dir

	# iterate over output vector sizes
	for k in k_vals:

		vec_x = numpy.zeros((len(doc_terms), k))
		vec_y = numpy.zeros(len(doc_terms))

		# get term frequencies
		for i, doc_id in enumerate(doc_cats):
			vec_y[i] = doc_cats[doc_id]
			for j, item in enumerate(top[:k]):
				term = item[0]
				if term in doc_terms[doc_id]:
					vec_x[i,j] = doc_terms[doc_id][term]

		# save x-vectors as files
		target_file = dir + "x_" + str(k) + ".p"
		with open(target_file, 'wb') as file:
			pickle.dump(vec_x, file)

		# save y-vectors as files
		target_file = dir + "y_" + str(k) + ".p"
		with open(target_file, 'wb') as file:
			pickle.dump(vec_y, file)

# return path to counts file for doc ID
def doc_id_to_path(i, files):
	return files[int(i)-1]
