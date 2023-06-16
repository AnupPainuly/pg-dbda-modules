#!/bin/bash
: '
   ___  ___  _  __      ___  ____ ___  ________ ___ ______  ____ ___  ___  ____
  / _ \/ _ \/ |/ /___  |_  ||_  // _ \|_  / / // _ <  /_  |/ __// _ \/ _ \/_  /
 / ___/ , _/    /___/ / __/_/_ </ // //_ <_  _/ // / / __//__ \/ // / // / / / 
/_/  /_/|_/_/|_/     /____/____/\___/____//_/ \___/_/____/____/\___/\___/ /_/

'
<< 'metadata'
Author: Anup Painuly
Date Created: 13-06-2023 
Date Modified: 13-06-2023 19:47:25
Description: workflow for lab5
Dependencies: ddl.hive, dml.hive, dql.hive, set_engine.hive, Data Set Dir: hire_data
metadata

# changing the Date Modified line dynamically inside the comment section
sed -i "12s/.*/    Date Modified: $(date +"%d-%m-%Y %H:%M:%S" -r "$0")/" "$0"
LOG_FILE="hive_evaluation.log"

function helper(){
    echo "Usage: lab5_workflow.sh [OPTIONS] [city]"
}

# check if arg are missing
if [[ -z "$1" ]] 
then
    echo Argument of city to be filtered is missing.
    helper
    exit 1
fi

# checking the count of dir and files of dataset
hdfs_file_count=$(hdfs dfs -count hire_data_input_files \
    | awk -F " " '{print $2}')
hdfs_dir_count=$(hdfs dfs -count hire_data_input_files \
    | awk -F " " '{print $1}')

# putting the dataset on hdfs if it does not already exist
if [[ "$hdfs_dir_count" -eq 0 ]];then
    hdfs dfs -mkdir -p hire_data_input_files
elif [[ "$hdfs_file_count" -eq 0 ]]; then
    echo Directory exists on the HDFS file system
    hdfs dfs -put ./hire_data/*.csv hire_data_input_files &> /dev/null
    echo Files copied to the HDFS file system.
fi


# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

# Function to check for errors
check_error() {
    if [ $1 -ne 0 ]; then
        log_message "Error: $2"
        exit 1
    fi
}

#set engine
beeline -u jdbc:hive2://localhost:10000 -f set_engine.hive
check_error $? "Failed to set engine"

log_message "exec engine changed successfully."

# Create table
beeline -u jdbc:hive2://localhost:10000 -f ddl.hive 
check_error $? "Failed to create table."

log_message "Table created successfully."

# Load data
beeline -u jdbc:hive2://localhost:10000 -f dml.hive 
check_error $? "Failed to load data into table."

log_message "Data loaded successfully."

# Select records
beeline -u jdbc:hive2://localhost:10000 -f dql.hive --hivevar city_var="$1"
check_error $? "Failed to select records from table."

log_message "Records selected successfully."

log_message "Script execution completed."



