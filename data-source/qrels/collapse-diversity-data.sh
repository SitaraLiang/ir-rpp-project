#!/bin/sh

sort -t ' ' -k1,1 -k3,3 $1 | \
    awk '($NF>0)' | \
    awk '{
        if (q != $1){
            if (q!=""){
                print q"\t0\t"d"\t"r;
            }
            q="";
            d="";
            r=0;
        }
        if (d==$3){
            if (r<$NF){
                r=$NF
            }
        }else{
            if (q != ""){
                print q"\t0\t"d"\t"r;
            }
            q=$1; 
            d=$3; 
            r=$NF;
        }
    }
    END{
        if (q!=""){
            print q"\t0\t"d"\t"r;
        }
        q="";
        d="";
        r=0;
    }'