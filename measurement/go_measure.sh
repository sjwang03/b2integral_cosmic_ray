#!/bin/bash

a=0
while [ $a -le 1000 ];do
    ./getWFM.sh 192.168.10.1 -Image
    a=$((a+1))
done




