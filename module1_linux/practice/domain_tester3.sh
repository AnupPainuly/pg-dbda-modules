#!/bin/bash
#Description: domain tester tool using function, case, postional arguments

function help(){
    echo "-d provide domain name"
    echo "-h/--help for more info"
}
function main(){
    while read i
    do 
        if host $i.$domain &> /dev/null
        then
            echo "$i.$domain: alive"
        fi
    done < "$wordlist"
}
flag=$1
while true
do
case $flag in
    -d)
        domain=$2
        shift
        if [ -z $domain ]
        then 
            echo "provide the domain"
            exit 127
        fi
        main
        ;;

    -h|--help)
        help
        ;;
    -w)
        wordlist=$2
        if [ -z $wordlist ]
        then
            echo "provide the wordlist"
            exit 127
        fi
        ;;
    *)
        echo "error: use -h/--help for more info"
        ;;
esac
done


