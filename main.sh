#!/bin/bash

echo 'clearing output folders'
rm -r counts/*
rm -r vectors_mi/*
rm -r vectors_x2/*

echo 'making lists of file paths'
./files.py

echo 'making subdirectories for counts'
while read d; do
	mkdir $d
done < files_subdirectories.txt

echo 'getting term counts from each document'
paste files_source.txt files_counts.txt | while read fs fc; do
	tr -c "[a-z'_.]" "\n" < $fs | sed '/^[._]*$/d' | sort | uniq -c > $fc
done

echo 'making document vectors'
./vectors.py