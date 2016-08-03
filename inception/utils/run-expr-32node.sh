#!/bin/bash
#bazel build inception/imagenet_distributed_train

cd ../

if [ "$#" -ne 6 ]; then
  echo "Illegal number of parameters"
  echo "Usage: sh run-expr.sh <Device> <id> <learning_rate> <momentum> <compute_groups> <checkpoint-prefix>"
  exit
fi


CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=16 \
--data_dir=/lfs/local/0/daniter/16node-data-$2 \
--job_name='worker' \
--task_id=$2 \
--ps_hosts='raiders1:2200' \
--worker_hosts='raiders1:2201,raiders1:2202,raiders1:2203,raiders1:2204,raiders2:2205,raiders2:2206,raiders2:2207,raiders2:2208,raiders3:2209,raiders3:2210,raiders3:2211,raiders3:2212,raiders8:2213,raiders8:2214,raiders8:2215,raiders8:2216' \
--initial_learning_rate=$3 \
--momentum=$4 \
--sync=True \
--compute_groups=$5 \
--checkpoint_prefix=$6