#!/bin/bash
LAB_HOME=/home/cloudera/hdp/pigandhive/labs/lab3
hdsh="hdfs dfs"
function splash_prompt(){
    #sleep 7 &
    PID=$!
    i=1
    sp="/__\|"
    echo -n ' '
    while [ -d /proc/$PID ]
    do
        printf %s "\b${sp:i++%${#sp}:1}"
    done
}

echo -e "choose an option from the below menu:
1. list the contents of hdfs storage
2. list the contents of hdfs storage recursively
3. put a file in home location of hdfs storage
4. create directory in the hdfs storage
5. view the file from hdfs storage
6. copy the file to temporary directory in hdfs
7. merge data
8. put file in hdfs with custom blocksize
9. seek the inconsistency in the file.  \n"


read -rp "select: " choice

case $choice in
    1)
        "$hdsh" -ls
        ;;

    2)
        "$hdsh" -ls -R
        ;;

    3)
        ls "$LAB_HOME"
        read -rp "select the file to put in hdfs:" putfile
        fileexists=$("$hdsh" -ls | grep -o "$putfile")
        if [ "$fileexists" == "$putfile" ]
        then
            echo "file already exists in the hdfs file system"
            exit
        else
            "$hdsh" -put "$LAB_HOME"/"$putfile" /user/cloudera &
            splash_prompt
        fi
        ;;

    4)
        read -rp "enter '/' separated  directory and subdirectory name to be created in HDFS storage: " dname
        "$hdsh" -mkdir "$dname"
        ;;

    5)
        "$hdsh" -ls -R
        read -rp "enter the file name to be viewed: " filename
        "$hdsh" -cat /user/cloudera "$filename"
        ;;
    6)
        "$hdsh" -ls -R
        read -rp "enter the full path of file/directory to get the file in temp: " gettotemp
        "$hdsh" -get /user/cloudera/"$gettotemp" /tmp
        ;;

    7)
        "$hdsh" -ls -R
        read -rp "enter the full path of directory to be merged : " mergetotemp
        "$hdsh" -get /user/cloudera/"$mergetotemp" /tmp
        echo " file merged succefully and put in /tmp folder"
        ;;

    8)
        ls "$LAB_HOME"
        read -rp "select the file to put in hdfs:" putfile
        fileexists=$("$hdsh" -ls | grep -o "$putfile")
        if [ "$fileexists" == "$putfile" ]
        then
            echo "file already exists in the hdfs file system"
            exit
        else
            read -rp "enter the block size: " size
            "$hdsh" -D dfs.blocksize="$size" -put "$LAB_HOME"/"$putfile" /user/cloudera &
        fi
        splash_prompt
        ;;

    9)
        ls "$LAB_HOME"
        read -rp "select the file to seek the inconsistencies in: " seekfile
        hdfs fsck /user/cloudera/"$seekfile"
        ;;

    *)
        echo "Invalid Option"
        ;;
esac
