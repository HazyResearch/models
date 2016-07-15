#!/bin/bash
#bazel build inception/imagenet_distributed_train

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=16 \
--data_dir=/lfs/local/0/daniter/sample-data \
--job_name='worker' \
--task_id=3 \
--ps_hosts='localhost:4228' \
--worker_hosts='raiders8:4221,raiders8:4222,raiders8:4223,raiders8:4224' \
--sync=True
