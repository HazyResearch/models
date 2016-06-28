#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../

if [ "$#" -ne 5 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh run-expr.sh <Device> <id> <learning_rate> <momentum> <sync>"
  exit
fi
CUDA_VISIBLE_DEVICES=$1 bazel-bin/inception/imagenet_distributed_train \
--batch_size=64 \
--data_dir=/lfs/local/0/daniter/sample-data \
--job_name='worker' \
--task_id=$2 \
--ps_hosts='raiders1:2228' \
--worker_hosts='raiders1:2226,raiders1:2227,raiders3:2229,raiders3:2230' \
--initial_learning_rate=$3 \
--momentum=$4 \
--sync=$5
