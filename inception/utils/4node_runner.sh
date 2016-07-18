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


sh run-ps-4node.sh $1 $2 $3 &> ${OUTPUT}/ps.out &
sh run-expr-4node.sh 0 0 $1 $2 $3 &> ${OUTPUT}/w1.out &
sh run-expr-4node.sh 1 1 $1 $2 $3 &> ${OUTPUT}/w2.out &


ssh raiders3 "cd ${WORKDIR}; sh run-expr-4node.sh 0 2 $1 $2 $3" &> ${OUTPUT}/w3.out &
ssh raiders3 "cd ${WORKDIR}; sh run-expr-4node.sh 1 3 $1 $2 $3" &> ${OUTPUT}/w4.out &

