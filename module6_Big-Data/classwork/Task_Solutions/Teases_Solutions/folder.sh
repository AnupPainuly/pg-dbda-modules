#!/bin/bash


# Author:               Shubhashini
# Date Created:         24-05-2023
# Date Modified:        25-05-2023
# Description:          Implementing haystack problem
# Usage:                bash folder.sh
# Prerequisites:        NA

cd /home/cloudera/Desktop/haystack
for i in {1..500}do
	mkdir folder$i
	cd /home/cloudera/Desktop/haystack/folder$i
	for p in {1..100}
	do
		touch file$p
	done
	cd ..
done

