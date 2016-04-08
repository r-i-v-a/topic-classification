#!/bin/bash

# make lists of file paths
./files.py

# make subdirectories for counts
rm -r counts/*
while read d; do
	mkdir $d
done < files_subdirectories.txt

# get term counts from transcript files
paste files_source.txt files_counts.txt | while read fs fc; do
	tr -c "[a-z'_.]" "\n" < $fs | sed '/^[._]*$/d' | sort | uniq -c > $fc
done