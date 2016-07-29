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

momentum = [0.45, 0.6, 0.75, 0.9]
lr = [ 0.01, 0.005, 0.0025 ]
cgs = [1,2,4]

exe = "./4node_runner.sh"
prefix = "expr6-tf9-4Node-GPU-seed-42562-CG-"

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
                
    time.sleep(8*60) # 28 + 2 = 30 
    kill_exp()

if __name__=='__main__':
    print("Starting experiments:")
    for m in momentum:
        for l in lr:
            for cg in cgs:
                run(m, l, cg)

