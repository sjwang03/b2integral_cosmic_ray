#!/bin/bash

#for f in `ls wfmdata/*.isf`;do
#    echo -n "$f: "
#    cp $f tmp.bin
#    ./parce.exe
#    cp tmp.ascii ${f%isf}ascii
#    echo
#done

g++ parce_sample.cc -o parce.exe 

n=0

(while [ $n -lt 1000 ];do
    f=`printf "wfmdata/run%05d_4.isf" $n`
    if [ ! -e $f ];then break;fi

    echo -n "$n "
    for ch in 1 2 3 4;do
	f=`printf "wfmdata/run%05d_%d.isf" $n $ch`
    	cp $f tmp.bin
	./parce.exe
	if [ $n -lt 100 ];then cp tmp.ascii ${f%.isf}.ascii ;fi
    done
    echo
    n=$((n+1))
 done ) | tee data.dat_tmp

rm tmp.bin tmp.ascii

grep -v "saturated" data.dat_tmp  > data.dat
rm data.dat_tmp
    
