from stable_baselines3 import PPO 
from typing import Callable
import os
from environment import CarEnv
import time


print('This is the start of training script which opens an existing model and continues to train it')
print('setting folders for logs and models')
models_dir = "C:\\SelfDrive\\models\\1705442732"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()

env.reset()
print('Env has been reset as part of launch')
#print('Env action space:',env.action_space)
# point to where your model is saved
model_path = f"{models_dir}\\500000"
model = PPO.load(model_path, env=env)
print('Model action space:',model.action_space)
TIMESTEPS = 250_000 # how long is each training iteration - individual steps
iters = 0
while iters<2:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")