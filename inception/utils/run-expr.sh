#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../

if [ "$#" -ne 5 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh run-expr.sh <Device> <id> <learning_rate> <momentum> <sync>"
  exit
fi

if [ "$5" == "True" ]
then
  SIZE=16
else
  SIZE=16
fi

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=$SIZE \
--data_dir=/lfs/local/0/daniter/sample-data-$2 \
--job_name='worker' \
--task_id=$2 \
--ps_hosts='raiders1:2228' \
--worker_hosts='raiders1:2200,raiders1:2202,raiders1:2203,raiders1:2204,raiders1:2205,raiders1:2206,raiders1:2207,raiders1:2208,raiders3:2209,raiders3:2210,raiders3:2211,raiders3:2212,raiders3:2213,raiders3:22214,raiders3:2215,raiders3:2216' \
--initial_learning_rate=$3 \
--momentum=$4 \
--sync=$5
