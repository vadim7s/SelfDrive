'''
1. New Controls Action space
2. New Observation space - add speed reading
3. Better reward structure

This version moves to have various steering and acceleration options in controls
Due to limitation of SB3 not allowing continuous action space:

Steering:
0 - max left = -1
1 - more left = -0.4
2 - a bit left = -0.1
3 - centred = 0
4 - a bit right = 0.1
5 - more right = 0.4
6 - max right = 1


Acceleration:

0 - none
1 - some = 0.3
2 - more = 0.7
3 - full power = 1

Accordingly, the dimensions will be:
Discrete(7) x Discrete(4)

Rewards

1. There is a minimum distance needed to move from original spawn point - to avoid going in circles
2. Reward staying within your lane
3. Reward speed in increments, e.g. 10kmh=0.1, 20kmh =0.5, 30 kmh=1.0

'''


import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import carla

N_DISCRETE_ACTIONS = 3

SECONDS_PER_EPISODE = 10

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

FIXED_DELTA_SECONDS = 0.1 # changed from 0.2 in Jan 2023

SHOW_PREVIEW = False

class CarEnv(gym.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	STEER_OPTIONS = 7
	THROTTLE_OPTIONS = 4
	MAX_SPEED=100


	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	
	def __init__(self):
		super(CarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
		self.action_space = spaces.Box(low =-1, high=1, shape=(2,),dtype=np.uint8)
		self.observation_space = spaces.Box(low=0, high=255,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(4.0)
		self.world = self.client.get_world()

		self.settings = self.world.get_settings()
		print('setting no rendering mode...')
		self.settings.no_rendering_mode = True
		#print('setting synch mode...')
		#self.settings.synchronous_mode = True # new in Jan 2023
		print('setting time interval...')
		self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
		print('applying all env settings...')
		self.world.apply_settings(self.settings)
		print('All env settings have been applied...')
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		print('Env Init has been completed...')


	def step(self, action):
		
		steer = round(action[0],1)
		if action[1]<-0.5:
			throttle = 0
		elif action[1]<-0:
			throttle= 0.4
		elif action[1]<0.8:
			throttle= 0.7
		else:
			throttle=1

		self.vehicle.apply_control(carla.VehicleControl(throttle=1.0*throttle, steer=self.STEER_AMT*steer))
		
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		
		if kmh>self.MAX_SPEED:
			kmh=self.MAX_SPEED
		
		distance_travelled = self.initial_location.distance(self.vehicle.get_location())

		if len(self.collision_hist) != 0:
			done = True
			reward = -200
			#self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.actor_list])
			
		elif kmh < 20 and distance_travelled<200:
			done = False
			reward = -1
		else:
			done = False
			reward = 3

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
			#self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.actor_list])
		return self.front_camera, reward, done, {}	#curly brackets - empty dictionary required by SB3 format

	def reset(self):
		print('Commencing env re-set...')
		self.collision_hist = []
		self.actor_list = []
		self.transform = random.choice(self.world.get_map().get_spawn_points())
		
		self.vehicle = None
		while self.vehicle is None:
			try:
        # connect
				self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
			except:
				pass
		self.actor_list.append(self.vehicle)
		self.initial_location = self.vehicle.get_location()
		self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
		self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.rgb_cam.set_attribute("fov", f"110")

		transform = carla.Transform(carla.Location(x=2.5, z=0.7))
		self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(4)

		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		print('Commencing reset loop until front cam is set...')

		while self.front_camera is None:
			time.sleep(0.01)
		
		self.episode_start = time.time()
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		
		print('Env re-set completed...')

		return self.front_camera  # fron camera and initial speed which is zero

	def process_img(self, image):
		i = np.array(image.raw_data)
		#print(i.shape)
		i2 = i.reshape((self.im_height, self.im_width, 4))
		i3 = i2[:, :, :3] # this is to ignore the 4rth Alpha channel - up to 3
		if self.SHOW_CAM:
			cv2.imshow("", i3)
			cv2.waitKey(1)
		self.front_camera = i3

	def collision_data(self, event):
		self.collision_hist.append(event)
	