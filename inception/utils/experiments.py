#!/bin/python

from subprocess import call, Popen
import time

import signal
import sys, os

def kill_exp():
    call(["pkill", "-u", "daniter", "-f","imagenet_distributed_train"])
    call(["ssh", "raiders3","pkill -u daniter -f imagenet_distributed_train"])

def signal_handler(signal, frame):
        kill_exp()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

momentum = [0.3, 0.0, -0.3]
lr = [0.001, 0.005, 0.0005]

SYNC = "True"
ASYNC = "False"
exe = "./local_runner.sh"
prefix = "LongRun-16"

def run(m,l, sync):
    print("Testing (%s) momentum: %f, learning rate: %f" % (sync,m,l))
    if SYNC == sync:
        m += 0.6
    cmd = [exe, str(l), str(m), sync, prefix]
    print " ".join(cmd)
    call(cmd)
    time.sleep(30*60) 
    kill_exp()

if __name__=='__main__':
    print("Starting experiments:")
    for m in momentum:
        for l in lr:
            async_dir_name = "%s-%s-%s-%s" % (prefix, str(l), str(m), ASYNC)
            sync_dir_name = "%s-%s-%s-%s" % (prefix, str(l), str(m+0.6), SYNC)
            if os.path.exists(async_dir_name):
                print("Skipping " + async_dir_name)
            else:
                run(m,l, ASYNC)
            if os.path.exists(sync_dir_name):
                print("Skipping " + sync_dir_name)
            else:
                run(m,l, SYNC)
