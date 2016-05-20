#!/usr/bin/env python

# generate file name list: Fisher Corpus transcripts
with open("files_source.txt", "w") as file:
	for i in range(1, 5851):
		num_str = str(i).zfill(5)
		path = "/mnt/matylda2/data/FISHER/fe_03_p1_tran/data/trans/" + num_str[:3] + "/fe_03_" + num_str + ".txt"
		file.write(path + "\n")

# generate file name list: word count files
with open("files_counts.txt", "w") as file:
	for i in range(1, 5851):
		num_str = str(i).zfill(5)
		path = "/counts/" + num_str[:3] + "/" + num_str + ".txt"
		file.write(path + "\n")

# generate file name list: subdirectories for storing count files
with open("files_subdirectories.txt", "w") as file:
	for i in range(59):
		num_str = str(i).zfill(3)
		path = "/counts/" + num_str
		file.write(path + "\n")