#!/usr/bin/env python

with open("files_source.txt", "w") as file:
	for i in range(1, 5851):
		num_str = str(i).zfill(5)
		path = "/mnt/matylda2/data/FISHER/fe_03_pl_tran/data/trans/" + num_str[:3] + "/fe_03_" + num_str + ".txt"
		file.write(path + "\n")

with open("files_counts.txt", "w") as file:
	for i in range(1, 5851):
		num_str = str(i).zfill(5)
		path = "./counts/" + num_str[:3] + "/" + num_str + ".txt"
		file.write(path + "\n")