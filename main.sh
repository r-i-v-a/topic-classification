#!/bin/bash

workdir=$1
datadir=$2

echo 'clearing output folders'
rm -r $datadir/counts/*
rm -r $datadir/features/*
rm -r $datadir/vectors_mi/*
rm -r $datadir/vectors_tfidf/*
rm -r $datadir/vectors_x2/*

echo 'making lists of file paths'
$workdir/files.py

echo 'making subdirectories for counts'
while read d; do
	mkdir -p $datadir/$d
done < $workdir/files_subdirectories.txt

# this part requires access to Fisher Speech Corpus
# transcript files on VUT server
echo 'getting term counts from each document'
paste files_source.txt files_counts.txt | while read fs fc; do
	tr -c "[a-z'_.]" "\n" < $fs | sed '/^[._]*$/d' | sort | uniq -c > $datadir/$fc
done

# this part does not need access to original transcript
# files, only locally stored data (word count files)
echo 'evaluating and selecting features'
$workdir/features.py $datadir

echo 'making document vectors'
$workdir/vectors.py $datadir