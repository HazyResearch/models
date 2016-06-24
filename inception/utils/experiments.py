#!/bin/python

from subprocess import call
import time

import signal
import sys, os

def kill_exp():
    call(["pkill", "-u", "daniter", "-f","imagenet_distributed_train"])
    call(["ssh", "raiders5","pkill -u daniter -f imagenet_distributed_train"])
    #call(["pkill", "-f","test-runner.sh"])

def signal_handler(signal, frame):
        kill_exp()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

momentum = [0.9, 0.6, 0.3, 0.0]
lr = [0.01, 0.05, 0.1, 0.005]

SYNC = "True"
ASYNC = "False"
exe = "local_runner.sh"

def run(m,l, sync):
    print("Testing (%s) momentum: %f, learning rate: %f" % (sync,m,l))
    call(["sh", exe, str(l), str(m), sync])
    time.sleep(15*60) # change to 15 min
    kill_exp()

if __name__=='__main__':
    print("Starting experiments:")
    for m in momentum:
        for l in lr:
            async_dir_name = "output-%s-%s-%s" % (str(l), str(m), ASYNC)
            sync_dir_name = "output-%s-%s-%s" % (str(l), str(m), SYNC)
            if os.path.exists(async_dir_name):
                print("Skipping " + async_dir_name)
            else:
                run(m,l, ASYNC)
            if os.path.exists(sync_dir_name):
                print("Skipping " + sync_dir_name)
            else:
                pass
                #run(m,l, SYNC)
