#!/bin/bash
#bazel build inception/imagenet_distributed_train

LD_LIBRARY_PATH='/usr/local/cuda-7.5/lib64'

cd ../

if [ "$#" -ne 5 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh run-expr.sh <Device> <id> <learning_rate> <momentum> <compute_groups>"
  exit
fi


CUDA_VISIBLE_DEVICES=$1 bazel-bin/inception/imagenet_distributed_train \
--batch_size=32 \
--data_dir=/home/data/sample-data-$2 \
--job_name='worker' \
--task_id=$2 \
--ps_hosts='master:2200' \
--worker_hosts='master:2201,master:2202,master:2203,master:2204,node001:2205,node001:2206,node001:2207,node001:2208,node002:2209,node002:2210,node002:2211,node002:2212,node003:2213,node003:2214,node003:2215,node003:2216' \
--initial_learning_rate=$3 \
--momentum=$4 \
--sync=True \
--compute_groups=$5
