#!/usr/bin/env python

from __future__ import division
import numpy

# get tf-idf score for given term
def tfidf(terms, doc_terms):
	terms = []
	count_terms = numpy.zeros(len(terms))
	count_docs = numpy.zeros(len(terms))

	total_terms = 0
	total_docs = len(doc_terms)

	for i, term in enumerate(terms):
		print "calculating TF-IDF:", "{:.3f}".format(100 * i / vocab_size), '%'
		terms[i] = term

		for doc_id in doc_terms:
			if term in doc_terms[doc_id]:
				total_terms += doc_terms[doc_id][term]
				count_terms[i] += doc_terms[doc_id][term]
				count_docs[i] += 1

	return tfidf_by_term