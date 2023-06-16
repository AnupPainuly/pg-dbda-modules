#!/bin/bash
: '
   ___  ___  _  __      ___  ____ ___  ________ ___ ______  ____ ___  ___  ____
  / _ \/ _ \/ |/ /___  |_  ||_  // _ \|_  / / // _ <  /_  |/ __// _ \/ _ \/_  /
 / ___/ , _/    /___/ / __/_/_ </ // //_ <_  _/ // / / __//__ \/ // / // / / / 
/_/  /_/|_/_/|_/     /____/____/\___/____//_/ \___/_/____/____/\___/\___/ /_/

'
<< 'metadata'
Author: Anup Painuly
Date Created: 09-06-2023 
Date Modified: 13-06-2023 19:47:25
Description: workflow for lab4
Dependencies: ddl.hive, dml.hive, dql.hive, Data Set Dir: hire_data
metadata

# changing the Date Modified line dynamically inside the comment section
sed -i "12s/.*/    Date Modified: $(date +"%d-%m-%Y %H:%M:%S" -r "$0")/" "$0"
LOG_FILE="hive_evaluation.log"

if [[ -z "$1" ]] #check if arg are missing
then
    echo argument of city to be filtered is missing 
    exit 1
fi

hdfs dfs -mkdir -p hire_data_input_files
hdfs dfs -put ./hire_data/*.csv hire_data_input_files &> /dev/null

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



