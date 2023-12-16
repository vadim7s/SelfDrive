'''
just bits for cnn prep changes

'''

import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import carla
import sys
sys.path.append('C:/CARLA_0.9.13/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

from tensorflow.keras.models import load_model

SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

FIXED_DELTA_SECONDS = 0.2

SHOW_PREVIEW = True

class CarEnv(gym.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	CAMERA_POS_Z = 1.3 
	CAMERA_POS_X = 1.4
	PREFERRED_SPEED = 20 # what it says
	SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the




	def __init__(self):
		super(CarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

		self.action_space = spaces.MultiDiscrete([9])
        # First discrete variable with 9 possible actions for steering with middle being straight
        # Second discrete variable with 4 possible actions for throttle/braking
        #calculate image cropping to use with pre-trained CNN model
		self.height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
		self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
		self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
		self.new_height = HEIGHT - self.height_from
		self.new_width = self.width_to - self.width_from
		self.image_for_CNN = None

        		
		self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0,shape=(7,18,8), dtype=np.float32)
        })

	
		self.cnn_model = load_model('model_saved_from_CNN.h5',compile=False)
		self.cnn_model.compile()
	

			
	
			
	def apply_cnn(self,im):
		img = np.float32(im)
		img = img /255
		img = np.expand_dims(img, axis=0)
		cnn_applied = self.cnn_model([img,0],training=False)
		cnn_applied = np.squeeze(cnn_applied)
		return  cnn_applied.numpy()[0][0]


	def step(self, action):
		
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
		
		return  {'image': self.image_for_CNN}

