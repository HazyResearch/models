#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../
if [ "$#" -ne 3 ]; then
  echo "Illegal number of parameters";
  echo "Usage: sh run-expr.sh <learning_rate> <momentum> <sync>";
  exit
fi

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=64 \
--data_dir=/lfs/local/0/daniter/imagenet-8 \
--job_name='ps' \
--task_id=0 \
--ps_hosts='localhost:2228' \
--worker_hosts='raiders1:2200,raiders1:2201,raiders3:2209,raiders3:2210'  \
--initial_learning_rate=$1 \
--momentum=$2 \
--sync=True \
--compute_groups=$3
