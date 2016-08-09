#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../
if [ "$#" -ne 3 ]; then
  echo "Illegal number of parameters";
  echo "Usage: sh run-expr.sh <learning_rate> <momentum> <sync>";
  exit
fi

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=32 \
--data_dir=/home/data/sample-data \
--job_name='ps' \
--task_id=0 \
--ps_hosts='master:2200' \
--worker_hosts='master:2201,master:2202,master:2203,master:2204,node001:2205,node001:2206,node001:2207,node001:2208,node002:2209,node002:2210,node002:2211,node002:2212,node003:2213,node003:2214,node003:2215,node003:2216' \
--initial_learning_rate=$1 \
--momentum=$2 \
--sync=True \
--compute_groups=$3
