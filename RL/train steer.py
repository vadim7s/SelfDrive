'''
This was an attempt to limit "guessing" to just streering while the car
is driven at a constant speed

Outcome: the model trained to go straight as the best strategy
This was partially due to a lot of "drive straight" road situations
when this strategy is suitable.
The model has not learned to follow the lane when it curves or turns or ends

Also according to chat GPT, SB3 does not contain CNN layers in the way it does RL
so it is hard to expect it to perform well in computer vision

Where from here: this version added a pre-processing step on the camera vision
to apply another model to do CNN and get a more condensed array to pass into RL
Also picking towns or spawn points with more curvy lanes would help to make RL
learn to follow the lane 

Chat GPT suggestion on how to extract aand same a model from a middle layer of NN
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# Assuming you have a pre-existing model
original_model = load_model('your_model.h5')

# Choose the layer from which you want to save the model
desired_layer_name = 'desired_layer_name'
desired_layer_output = original_model.get_layer(desired_layer_name).output

# Create a new model that takes the input of the original model
# and outputs the output of the desired layer
model_to_save = Model(inputs=original_model.input, outputs=desired_layer_output)

# Save the new model
model_to_save.save('model_saved_from_desired_layer.h5')
'''


from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from carenv_steer_only_cnn import CarEnv
import time


print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarEnv()

env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)

TIMESTEPS = 500_000 # how long is each training iteration - individual steps
iters = 0
while iters<4:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")