#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh local_runner.sh <learning_rate> <momentum> <compute_groups> <dir_name>" 
  exit
fi

echo "Running=> CGs: $3 lr: $1 momentum: $2"

OUTPUT="$4"
WORKDIR="~/tf-test/inception/utils/"
mkdir -p ${OUTPUT}


sh run-ps-16node.sh $1 $2 $3 &> ${OUTPUT}/ps.out &
sh run-expr-16node.sh 0 0 $1 $2 $3 &> ${OUTPUT}/w1.out &
sh run-expr-16node.sh 1 1 $1 $2 $3 &> ${OUTPUT}/w2.out &
sh run-expr-16node.sh 0 2 $1 $2 $3 &> ${OUTPUT}/w3.out &
sh run-expr-16node.sh 1 3 $1 $2 $3 &> ${OUTPUT}/w4.out &
sh run-expr-16node.sh 0 4 $1 $2 $3 &> ${OUTPUT}/w5.out &
sh run-expr-16node.sh 1 5 $1 $2 $3 &> ${OUTPUT}/w6.out &
sh run-expr-16node.sh 0 6 $1 $2 $3 &> ${OUTPUT}/w7.out &
sh run-expr-16node.sh 1 7 $1 $2 $3 &> ${OUTPUT}/w8.out &
sh run-expr-16node.sh 0 8 $1 $2 $3 &> ${OUTPUT}/w9.out &
sh run-expr-16node.sh 1 9 $1 $2 $3 &> ${OUTPUT}/w10.out &
sh run-expr-16node.sh 0 10 $1 $2 $3 &> ${OUTPUT}/w11.out &
sh run-expr-16node.sh 1 11 $1 $2 $3 &> ${OUTPUT}/w12.out &
sh run-expr-16node.sh 0 12 $1 $2 $3 &> ${OUTPUT}/w13.out &
sh run-expr-16node.sh 1 13 $1 $2 $3 &> ${OUTPUT}/w14.out &
sh run-expr-16node.sh 0 14 $1 $2 $3 &> ${OUTPUT}/w15.out &
sh run-expr-16node.sh 1 15 $1 $2 $3 &> ${OUTPUT}/w16.out &











