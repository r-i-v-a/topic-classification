#!/usr/bin/env python

from __future__ import division
import numpy

# get tf-idf score for given term
def tf_idf(terms, doc_terms, vocab_size, set_train):
	count_terms = numpy.ones(len(terms))
	count_docs = numpy.ones(len(terms))

	total_terms = 0
	total_docs = len(set_train)

	for i, term in enumerate(terms):
		print "calculating TF-IDF, FREQ:", "{:.3f}".format(100 * i / vocab_size), '%'

		for doc_id in set_train:
			if term in doc_terms[doc_id]:
				total_terms += doc_terms[doc_id][term]
				count_terms[i] += doc_terms[doc_id][term]
				count_docs[i] += 1

	tf = count_terms / total_terms
	idf = total_docs / count_docs
	tf_idf = tf * idf

	tf_idf_by_term = zip(terms, list(tf_idf))
	freq_by_term = zip(terms, list(tf))

	return tf_idf_by_term, freq_by_term