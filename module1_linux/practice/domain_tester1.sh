#!/bin/bash

#Description: testing the subdomains by using domain and file list as variables

target=google.com
words=subdomains.txt
while read -r i #using -r to prevent read form mangling the backslashes
do
    if host "$i".$target &> /dev/null
    then
        echo "'$i'.$target: subdomain is alive"
    fi
done < $words #giving input to the while loop
