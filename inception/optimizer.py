import time
import os
from subprocess import call, Popen, check_output
import numpy as np
import datetime

DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_CG = 1


exe = "utils/32node_runner.sh"
prefix = "OptimizerTest-"
path_to_data = "utils/"
INIT_RUNTIME = 1 # minutes
master_snapshot = "/lfs/raiders1/0/daniter/optimizer"
checkpoint_prefix = "model.ckpt"

cgs = [1,2]
lrs = [0.01, 0.005]
mus = [0.9, 0.6]


class Experiment():
	def __init__(self, lr, mu, cg, losses, checkpoint):
		self.lr = lr
		self.mu = mu
		self.cg = cg
		self.losses = losses
		self.checkpoint = checkpoint
		self.window_size = 1

	def __str__(self):
		return "lr: %f, mu: %f, cg: %d, checkpoint: %s" % (self.lr, self.mu, self.cg, self.checkpoint)

	def __repr__(self):
		return "lr: %f, mu: %f, cg: %d, checkpoint: %s" % (self.lr, self.mu, self.cg, self.checkpoint)

	def _get_raw_losses(self):
		return [l for s, l in self.losses]

	def get_final_loss(self):
		if self.window_size > len(self.losses):
			print("the window size is larger then the number of steps")
			return 999
		losses = self._get_raw_losses()
		window = np.ones(int(self.window_size))/float(self.window_size)
		conv = np.convolve(losses, window, 'same')
		conv = np.convolve(conv, window, 'same')[:-self.window_size]
		conv[:self.window_size] = np.nan
		return conv[-1]



def kill_all():
    call(["pkill", "-u", "daniter", "-f","imagenet_distributed_train"])
    call(["ssh", "raiders3","pkill -u daniter -f imagenet_distributed_train"])
    call(["ssh", "raiders8","pkill -u daniter -f imagenet_distributed_train"])
    call(["ssh", "raiders2","pkill -u daniter -f imagenet_distributed_train"])

def run(m,l, cg, iteration):
    print("Testing cg:%d, momentum: %f, learning rate: %f" % (cg,m,l))
    dir_name = "%s-%s-%s-%s" % (prefix+str(cg), str(l), str(m), str(iteration))
    if os.path.exists(path_to_data+dir_name):
        print("Skipping " + dir_name)
        return (-1, path_to_data+dir_name)

    cmd = [exe, str(l), str(m), str(cg), dir_name]
    print " ".join(cmd)
    call(cmd)
    return (0, path_to_data+dir_name)

def find_worker():
	try:
		pids = check_output(['pgrep', '-f', "python.*imagenet_distributed_train.*job_name=worker"])
		pids = pids.split("\n")
	except:
		return None
	return pids

def get_checkpoint(name):
	name = name.split("/")[1]
	files = os.listdir(master_snapshot)
	most_recent_checkpoint = None
	most_recent_time = 0
	for f in files:
		if name in f:
			time = os.path.getmtime(os.path.join(master_snapshot, f))
			if time > most_recent_time:
				most_recent_time = time
				most_recent_checkpoint = f

	most_recent_checkpoint = most_recent_checkpoint.replace(".meta", "")
	return os.path.join(master_snapshot, most_recent_checkpoint)



def cleanup(name):
	# check that all workers are down
	while find_worker():
		time.sleep(5)
		print("waiting for 5 sec for cheif to die")

	# kill paramserver
	print("killing all")
	kill_all()

	# check that snapshot was written
	ckpt = get_checkpoint(name)
	t = os.path.getmtime(ckpt)
	#assert time.time() - t < 60
	return ckpt, ckpt.split("-")[-1]

def create_checkpoint_file(checkpoint):
	format = 'model_checkpoint_path: "%s"\nall_model_checkpoint_paths: "%s"'
	with open(os.path.join(master_snapshot,"checkpoint"), "w") as f:
		f.write(format % (checkpoint, checkpoint))

def collect_losses(name):
	'''
	Implement loss writer to import into code
	'''
	losses = []
	for f in os.listdir(name):
		if 'ps.out' in f:
			continue     
		with open(os.path.join(name, f)) as log:
			for line in log.readlines():
				if line.startswith("INFO:tensorflow:Step:"):
					parts = line.split(" ")
					step = int(parts[1][:-1])
					loss = float(parts[3][:-1]) 
					losses.append((step,loss))
	losses.sort()
	return losses


if __name__=='__main__':

	main_checkpoint = None

	should_sleep = True
	# Start run with default params and timeout
	ret, name = run(DEFAULT_MOMENTUM, DEFAULT_LR, DEFAULT_CG, 0)
	if ret != 0:
		print("Failed to start run")
		should_sleep = False
	print("running : %s" % name)

	# monitor for errors / sleep until completed
	if should_sleep:
		time.sleep((INIT_RUNTIME+1)*60) # add 1 minute for init time of runs

	main_checkpoint, iteration = cleanup(name)
	print("new checkpoint : %s" % main_checkpoint)
	print("Current iteration at %s" % iteration)

	# while accuracy is not at threshold or timeout or iteration limit not reached:
	experiments = []
	while True:

		# do any moving of snapshots necessary NOTE: need checkpoint dir parser
		# for params mu, lr, cg:
		for cg in cgs:
			for mu in mus:
				for lr in lrs:
					# reset teh checkpoint before starting an expr
					create_checkpoint_file(main_checkpoint)
					should_sleep = True
					ret, name = run(mu, lr, cg, iteration) # TODO: use different model prefix for each
					if ret != 0:
						print("Failed to start run")
						should_sleep = False
					print("testing %s" % name)
					if should_sleep:
						time.sleep((INIT_RUNTIME+1)*60)
					currSnapshot, iteration = cleanup(name)
					losses = collect_losses(name)
					experiments.append(Experiment(lr, mu, cg, losses, currSnapshot))
					# Store snapshot and stats
					# collect losses and end snapshot

		# calculate best loss
		print experiments
		min_loss = 100
		best_exp = None
		for exp in experiments:
			loss = exp.get_final_loss()
			print("loss:")
			print(loss)
			if loss < min_loss:
				min_loss = loss
				best_exp = exp

		# set main snapshot and parameters
		main_checkpoint = best_exp.checkpoint
		create_checkpoint_file(main_checkpoint)
		iteration = main_checkpoint.split("-")[-1]
		# kick off longer run
		print("starting better run with params" + str(best_exp) + " at iteration: " + str(iteration))
		should_sleep = True
		ret, name = run(best_exp.mu, best_exp.lr, best_exp.cg, iteration)
		if ret != 0:
			print("Failed to start run")
			should_sleep = False
		print("running : %s" % name)

		if should_sleep:
			time.sleep((INIT_RUNTIME+1)*60)
