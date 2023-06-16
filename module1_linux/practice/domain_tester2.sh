#!/bin/bash
#Description: domain testing by using positional parameter

domain=$1
wordlist=$2
while read i
do 
    if host "$i".$domain &> /dev/null 
    then
        echo "'$i'.$domain: alive"
    fi
done < $wordlist
