#!/bin/bash
: '
   ___  ___  _  __      ___  ____ ___  ________ ___ ______  ____ ___  ___  ____
  / _ \/ _ \/ |/ /___  |_  ||_  // _ \|_  / / // _ <  /_  |/ __// _ \/ _ \/_  /
 / ___/ , _/    /___/ / __/_/_ </ // //_ <_  _/ // / / __//__ \/ // / // / / / 
/_/  /_/|_/_/|_/     /____/____/\___/____//_/ \___/_/____/____/\___/\___/ /_/

'
<< 'metadata'
Author: Anup Painuly
Date Created: 04-06-2023 
Date Modified: 04-06-2023 19:47:25
Description: workflow for lab3
metadata


function workflow() {
    start=$(date +"%s")
    sed -i "12s/.*/    Date Modified: $(date +"%d-%m-%Y %H:%M:%S" -r "$0")/" "$0"
    hdfs dfs -mkdir -p /hire_data_input_files
    hdfs dfs -put /tmp/hire_data/*.csv /hire_data_input_files &> /dev/null
    hdfs dfs -rm -R /hire_data_output

    if [[ -z $1 ]]
    then
        echo argument of city to be filtered is missing
        exit 1
    fi
    echo "workflow started for lab3 project. Please be patient"

    #clean up build
    build_dir=./lab3/build/
    rm -rf "$build_dir" 
    mkdir -p "$build_dir" 

    #compilation
    bin_dir=./lab3/bin
    cd "$bin_dir" || exit 1
    rm -rf ./*
    export HADOOP_CLASSPATH=$(hadoop classpath)
    javac -cp "$HADOOP_CLASSPATH" -d . \
        ../src/hiredata/*.java &> /tmp/compilation_log

    #Build
    jar -cvf ../build/lab3.jar ./hiredata/* \
        -C ./hiredata/ .. &> /tmp/archival_logs
    jar -tf ../build/lab3.jar &>> /tmp/archival_logs

    #Run
    hdfs dfs -ls /hire_data_input_files &> /dev/null
    if [[ $? == 0 ]]
    then
        #read -rp "enter the output directory name to be created in hdfs" output_dir
        output_dir=/hire_data_output
        hadoop jar ../build/lab3.jar hiredata.SalaryFilter /hire_data_input_files \
            "$output_dir" "$1" &> /tmp/MapReduce_log
                else
                    echo consider putting input files in the hdfs for further processing
                    exit 1
    fi

    #sanity check
    output_file=$(hdfs dfs -ls $output_dir \
        | awk '{print $8}' \
        | grep "part")

    num_lines=$(hdfs dfs -cat "$output_file" | wc -l)

    if [[ $num_lines -gt 10 ]]
    then
        echo file preview:
        hdfs dfs -cat "$output_file" | head -5
        echo -e ".\n."
        hdfs dfs -cat "$output_file" | tail -5
    else 
        hdfs dfs -cat "$output_file"
    fi
end=$(date +"%s")
duration=$(awk "BEGIN {printf \"%.2f\", ($end-$start)/60}")
echo time elapsed "$duration" minutes
}
workflow "$1"

