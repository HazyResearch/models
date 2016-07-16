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
  SIZE=64
else
  SIZE=64
fi

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=$SIZE \
--data_dir=/lfs/local/0/daniter/imagenet-8 \
--job_name='worker' \
--task_id=$2 \
--ps_hosts='raiders1:2228' \
--worker_hosts='raiders1:2200,raiders1:2202,raiders3:2209,raiders3:2210' \
--initial_learning_rate=$3 \
--momentum=$4 \
--sync=$5
