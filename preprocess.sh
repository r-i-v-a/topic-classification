#!/bin/bash

workdir=$1
datadir=$2
cd $workdir

echo 'clearing output folders'
rm -r counts/*
rm -r 'top'/*
rm -r vectors_mi/*
rm -r vectors_tfidf/*
rm -r vectors_x2/*

echo 'making lists of file paths'
$workdir/files.py

echo 'making subdirectories for counts'
while read d; do
	mkdir -p $datadir/$d
done < files_subdirectories.txt

echo 'getting term counts from each document'
paste files_source.txt files_counts.txt | while read fs fc; do
	tr -c "[a-z'_.]" "\n" < $fs | sed '/^[._]*$/d' | sort | uniq -c > $datadir/$fc
done