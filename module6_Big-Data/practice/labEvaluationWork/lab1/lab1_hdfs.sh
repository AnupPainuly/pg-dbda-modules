#!/bin/bash
LAB_HOME=/home/cloudera/hdp/pigandhive/labs/lab3
if [ -d "$LAB_HOME" ]
then
    logger -s -p "info" directory $LAB_HOME exists on the local file system 2>> /tmp/automation_log
else
    echo "$LAB_home does not exist, Please create the aforementioned dir"
    exit
fi
hdsh="hdfs dfs"


#reset the cursor
function shutdown() {
    tput cnorm 
}

trap shutdown EXIT

#cursor position
function cursorBack() {
    echo -en "\033[$1D"
}

#splash prompt  when the commands are running
# The spinner function takes the PID of previously running process in bg and shows the spinner prompt until the process completes executing
function spinner() {
    local pid=$1 # Process Id of the previous running command
    local spin='-\|/'
    local charwidth=1
    local i=0
    tput civis # cursor invisible
    while kill -0 $pid 2>/dev/null; do
        local i=$(((i + $charwidth) % ${#spin}))
        printf "%s" "${spin:$i:$charwidth}"
        cursorBack 1
        sleep .1
    done
    tput cnorm
    wait $pid # capture exit code
    return $?
}

("$@") &

echo -e "choose an option from the below menu:
1. list the contents of hdfs storage
2. list the contents of hdfs storage recursively
3. put a file in home location of hdfs storage
4. create directory in the hdfs storage
5. view the file from hdfs storage
6. copy the file from hdfs to temporary directory
7. merge data
8. put file in hdfs with custom blocksize
9. seek the inconsistency in the file.
0. Exit  \n"


read -p "select:" choice

case $choice in
    1)
        $hdsh -ls &
        spinner $!
        ;;

    2)
        $hdsh -ls -R &
        spinner $!
        ;;

    3)
        ls $LAB_HOME
        read -p "select the file to put in hdfs:" putfile
        fileexists=`$hdsh -ls | grep -o "$putfile"`
        if [ "$fileexists" == "$putfile" ]
        then
            echo "file already exists in the hdfs file system"
            exit
        else
            $hdsh -put $LAB_HOME/$putfile /user/cloudera &
        fi
        spinner $!
        ;;

    4)
        read -p "enter '/' separated  directory and subdirectory name to be created in HDFS storage: " dirname
        $hdsh -mkdir $dirname &
        spinner $!

#creating a log file and logging info when file is successfully created
logger -s -p "info" directory $dirname created in the hdfs file system 2>> /tmp/automation_log
;;

5)
    $hdsh -ls -R
    read -p "enter the file name to be viewed: " filename
    $hdsh -cat /user/cloudera $filename &
    spinner $!
    ;;

6)
    $hdsh -ls -R
    read -p "enter the full path of file/directory to get the file in temp: " gettotemp
    $hdsh -get /user/cloudera/$gettotemp /tmp &
    spinner $!
    logger -s -p "info" $gettotemp moved to local file system 2>> /tmp/automation_log
    ;;

7)
    $hdsh -ls -R
    read -p "enter the full path of directory to be merged : " mergetotemp
    $hdsh -get /user/cloudera/$mergetotemp /tmp &
    spinner $!
    echo " file merged succefully and put in /tmp folder"
    ;;

8)
    ls $LAB_HOME
    read -p "select the file to put in hdfs:" putfile
    fileexists=`$hdsh -ls | grep -o "$putfile"`
    if [ "$fileexists" == "$putfile" ]
    then
        echo "file already exists in the hdfs file system"
        exit
    else
        read -p "enter the block size: " size
        $hdsh -D dfs.blocksize=$size -put $LAB_HOME/$putfile /user/cloudera &
    fi
    spinner $!
    ;;

9)
    ls $LAB_HOME
    read -p "select the file to seek the inconsistencies in: " seekfile
    hdfs fsck /user/cloudera/$seekfile
    spinner $!
    ;;

0)
    echo "program terminated"
    exit
    ;;

*)
    echo "Invalid Option"
    ;;
esac
