#!/usr/bin/env python

from __future__ import division
import numpy

# get chi-squared for given term, class pairing
def x2(counts):

	# value to return
	result = 0

	# total number of documents
	n = counts.sum()

	# calculate chi-squared
	for i in [0,1]:
		for j in [0,1]:
			expected = counts[i,:].sum() * counts[:,j].sum() / n
			result += (counts[i,j] - expected) ** 2 / expected

	return result