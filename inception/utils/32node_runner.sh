#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh local_runner.sh <learning_rate> <momentum> <compute_groups> <dir_name>" 
  exit
fi

echo "Running=> CGs: $3 lr: $1 momentum: $2"

OUTPUT="$4"
WORKDIR="/root/models/inception/utils/"
mkdir -p ${OUTPUT}


bash run-ps-32node.sh $1 $2 $3 > ${OUTPUT}/ps.out 2>&1 &
bash run-expr-32node.sh 0 0 $1 $2 $3 > ${OUTPUT}/w1.out 2>&1 &
bash run-expr-32node.sh 1 1 $1 $2 $3 > ${OUTPUT}/w2.out 2>&1 &
bash run-expr-32node.sh 2 2 $1 $2 $3 > ${OUTPUT}/w3.out 2>&1 &
bash run-expr-32node.sh 3 3 $1 $2 $3 > ${OUTPUT}/w4.out 2>&1 &

ssh node001 "cd ${WORKDIR}; bash run-expr-32node.sh 0 4 $1 $2 $3" > ${OUTPUT}/w5.out 2>&1 &
ssh node001 "cd ${WORKDIR}; bash run-expr-32node.sh 1 5 $1 $2 $3" > ${OUTPUT}/w6.out 2>&1 &
ssh node001 "cd ${WORKDIR}; bash run-expr-32node.sh 2 6 $1 $2 $3" > ${OUTPUT}/w7.out 2>&1 &
ssh node001 "cd ${WORKDIR}; bash run-expr-32node.sh 3 7 $1 $2 $3" > ${OUTPUT}/w8.out 2>&1 &

ssh node002 "cd ${WORKDIR}; bash run-expr-32node.sh 0 8 $1 $2 $3" > ${OUTPUT}/w9.out 2>&1 &
ssh node002 "cd ${WORKDIR}; bash run-expr-32node.sh 1 9 $1 $2 $3" > ${OUTPUT}/w10.out 2>&1 &
ssh node002 "cd ${WORKDIR}; bash run-expr-32node.sh 2 10 $1 $2 $3" > ${OUTPUT}/w11.out 2>&1 &
ssh node002 "cd ${WORKDIR}; bash run-expr-32node.sh 3 11 $1 $2 $3" > ${OUTPUT}/w12.out 2>&1 &

ssh node003 "cd ${WORKDIR}; bash run-expr-32node.sh 0 12 $1 $2 $3" > ${OUTPUT}/w13.out 2>&1 &
ssh node003 "cd ${WORKDIR}; bash run-expr-32node.sh 1 13 $1 $2 $3" > ${OUTPUT}/w14.out 2>&1 &
ssh node003 "cd ${WORKDIR}; bash run-expr-32node.sh 2 14 $1 $2 $3" > ${OUTPUT}/w15.out 2>&1 &
ssh node003 "cd ${WORKDIR}; bash run-expr-32node.sh 3 15 $1 $2 $3" > ${OUTPUT}/w16.out 2>&1 &

