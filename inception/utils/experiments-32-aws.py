#!/bin/python

from subprocess import call, Popen
import time

import signal
import sys, os

def kill_exp():
    call(["pkill", "-f","imagenet_distributed_train"])
    call(["ssh", "node001","pkill -f imagenet_distributed_train"])
    call(["ssh", "node002","pkill -f imagenet_distributed_train"])
    call(["ssh", "node003","pkill -f imagenet_distributed_train"])
    call(["ssh", "node004","pkill -f imagenet_distributed_train"])
    call(["ssh", "node005","pkill -f imagenet_distributed_train"])
    call(["ssh", "node006","pkill -f imagenet_distributed_train"])
    call(["ssh", "node007","pkill -f imagenet_distributed_train"])

def signal_handler(signal, frame):
        kill_exp()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

momentum = [0.0, 0.3]
lr = [ 0.025 ]
cgs = [2,4,8]

exe = "./32node_runner_aws.sh"
prefix = "IMAGENET1000-GPU-CG-"

def isFailed(folder):
    for log in os.listdir(folder):
        with open(folder+"/"+log) as f:
            if log == "ps.out":
                for line in f.readlines():
                    if "assertion failed" in line:
                        return True
            else:
                for line in f.readlines():
                    if "failed to connect to" in line:
                        return True
    return False

def run(m,l, cg):
    print("Testing cg:%d, momentum: %f, learning rate: %f" % (cg,m,l))
    dir_name = "%s-%s-%s" % (prefix+str(cg), str(l), str(m))
    if os.path.exists(dir_name):
        print("Skipping " + dir_name)
        return

    cmd = [exe, str(l), str(m), str(cg), dir_name]
    print " ".join(cmd)
    call(cmd)
    for i in range(3): # try retry twice
        time.sleep(2*60) # give 2 minutes to set up run
        if isFailed(dir_name):
            print("trying again")
            kill_exp()
            if i == 2:
                return
            call(cmd)
        else:
            break
    time.sleep(8*60)
    kill_exp()

if __name__=='__main__':
    print("Starting experiments:")
    for m in momentum:
        for l in lr:
            for cg in cgs:
                run(m, l, cg)

