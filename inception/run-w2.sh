#!/bin/bash
#bazel build inception/imagenet_distributed_train

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=32 \
--data_dir=/lfs/local/0/daniter/sample-data \
--job_name='worker' \
--task_id=1 \
--ps_hosts='localhost:2222' \
--worker_hosts='localhost:2220,localhost:2221'
