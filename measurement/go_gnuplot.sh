#!/bin/bash


mkdir -p pdfs/

n=0
while [ $n -lt 100 ];do
    f=`printf "wfmdata/run%05d_4.ascii" $n` 
    if [ ! -e $f ];then break;fi
    
    for ch in 1 2 3 4;do
	f=`printf "wfmdata/run%05d_%d.ascii" $n $ch`
	cp $f tmp$ch
    done
    gnuplot plot.plt
    #gnuplot plot_chk.plt

    nn=`printf "%05d" $n` 

    mv test.pdf pdfs/$nn.pdf

    n=$((n+1))
done

rm tmp1 tmp2 tmp3 tmp4
pdftk pdfs/*.pdf cat output all.pdf

