#!/bin/bash

function help(){
        echo "-d DOMAIN.TLD : Provide Target"
        echo "-h/--help : Help"
}

function main(){
while read sub;do
        if host "$sub.$domain" &> /dev/null;then
                echo "$sub.$domain: Alive"
        fi
#done < subs.txt
done < $wordlist 
}
#while true;do
for (( i=0; i<2; i++ ))
do
case $1 in
        "-d")
        domain=$2
        shift
        if [ -z $domain ];then
                echo "Provide domain name or use -h/--help"
                exit 127
        fi
        ;;
        "-w")
                wordlist=$2
                shift
                if [ -z $wordlist ];then
                        echo "Provide wordlist or use -h/--help"
                        exit 127
                fi
                main
                break
                ;;
        "-h"|"--help")
                help
                ;;
        *)
                echo "Error: use -h/--help"
                exit 127
        ;;
esac
shift
done
