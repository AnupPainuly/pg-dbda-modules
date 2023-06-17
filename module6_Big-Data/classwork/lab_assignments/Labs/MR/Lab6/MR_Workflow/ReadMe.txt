Scenario-
-----------
A student has written a program which performs following activities on cloudera sandbox
1) Make a connectivity to MySql DB 
2) Collect the records from salaries table (The table created through Sqoop demo/lab)
and write them to a file on local file system
3) Putting the file created in Step 2 above to HDFS

Problem-
--------
Step 3 above is not working

Solution -
----------
Amit divided the code base from the student (Given as a Eclise Project)
into 2 projects and orchestrated the execution of the projects through a 
shell script workflow.sh

From Student -
--------------
Original code in the form of an Eclipse Project kept it in FromAshish/LoadData.zip