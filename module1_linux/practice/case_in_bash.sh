#!/bin/bash
#Description: e.g. of case in bash scripting

input=$1
case $input in 
    1)
        echo "this is case one"
        ;;
    2)
        echo "this is case two"
        ;;
    *) 
        echo "this is default case"
        ;;
esac



