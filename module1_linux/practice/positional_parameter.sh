#!/bin/bash
#Description: e.g. for positional parameter

a=$1
b=$2
echo "this is first parameter: $a"
echo "this is second parameter: $b"
#shift # removes the first positional parameter so than we can use $1 again
shift 2 #removes the first positional parameter so than we can use $1 again
echo "this is first parameter: $1"
echo "this is second parameter: $2"

