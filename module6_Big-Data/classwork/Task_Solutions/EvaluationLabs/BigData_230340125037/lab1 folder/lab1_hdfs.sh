#!/bin/bash

# Task 1: Create directories
hdfs dfs -mkdir test
hdfs dfs -mkdir test/test1
hdfs dfs -mkdir -p test/test2/test3

# Task 2: List directories
hdfs dfs -ls
hdfs dfs -ls -R

# Task 3: Remove directory
hdfs dfs -rm -R test/test2
hdfs dfs -ls -R

# Task 4: Upload a file to HDFS
cd /root/hdp/pigandhive/labs/Lab2.1
tail data.txt
hdfs dfs -put data.txt test/
hdfs dfs -ls test

# Task 5: Copy a file in HDFS
hdfs dfs -cp test/data.txt test/test1/data2.txt
hdfs dfs -ls -R test
hdfs dfs -rm test/test1/data2.txt

# Task 6: View the contents of a file in HDFS
hdfs dfs -cat test/data.txt
hdfs dfs -tail test/data.txt

# Task 7: Get a file from HDFS
hdfs dfs -get test/data.txt /tmp/
cd /tmp
ls data*

# Task 8: The getmerge command
hdfs dfs -put /root/hdp/pigandhive/labs/demos/small_blocks.txt test/
hdfs dfs -getmerge test /tmp/merged.txt

# Task 9: Specify block size and replication factor
hdfs dfs -D dfs.blocksize=1048576 -put data.txt data.txt
hdfs fsck /user/root/data.txt


