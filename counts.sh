#!/bin/bash

while read d; do
	mkdir $d
done < files_subdirectories.txt

paste files_source.txt files_counts.txt | while read fs fc; do
	sed '/^#.*$/d' < $fs | sed 's/^.*?:/ /g' | tr -s "[:blank:]" "\n" | sed '/^$/d' | sort | uniq -c > $fc
done