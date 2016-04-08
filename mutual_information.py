#!/usr/bin/env python

from __future__ import division
import numpy

# get mutual information for given term, class pairing
def mi(counts):

	# value to return
	result = 0

	# total number of documents
	n = counts.sum()

	# calculate mutual information
	for i in [0,1]:
		for j in [0,1]:
			result += (counts[i,j] / n) * numpy.log2((n * counts[i,j]) / (counts[i,:].sum() * counts[:,j].sum()))

	return result