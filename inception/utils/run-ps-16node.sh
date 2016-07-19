#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../
if [ "$#" -ne 3 ]; then
  echo "Illegal number of parameters";
  echo "Usage: sh run-expr.sh <learning_rate> <momentum> <sync>";
  exit
fi

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=16 \
--data_dir=/lfs/local/0/daniter/imagenet-8 \
--job_name='ps' \
--task_id=0 \
--ps_hosts='localhost:2228' \
--worker_hosts='localhost:2201,localhost:2202,localhost:2203,localhost:2204,localhost:2205,localhost:2206,localhost:2207,localhost:2208,localhost:2209,localhost:2210,localhost:2211,localhost:2212,localhost:2213,localhost:2214,localhost:2215,localhost:2216' \
--initial_learning_rate=$1 \
--momentum=$2 \
--sync=True \
--compute_groups=$3
