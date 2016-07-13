#!/bin/bash
#bazel build inception/imagenet_distributed_train

CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_size=64 \
--data_dir=/lfs/local/0/daniter/sample-data \
--job_name='worker' \
--task_id=0 \
--ps_hosts='localhost:4228' \
--worker_hosts='raiders8:4226,raiders8:4227' \
--sync=True
