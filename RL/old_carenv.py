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

FIXED_DELTA_SECONDS = 0.2

SHOW_PREVIEW = False

class CarEnv(gym.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	
	def __init__(self):
		super(CarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
		self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(4.0)
		self.world = self.client.get_world()

		self.settings = self.world.get_settings()
		self.settings.no_rendering_mode = True
		self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
		self.world.apply_settings(self.settings)
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
	
	def step(self, action):
		if action == 0:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
		elif action == 1:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
		elif action == 2:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

		if len(self.collision_hist) != 0:
			done = True
			reward = -200
			
		elif kmh < 20:
			done = False
			reward = -1
		else:
			done = False
			reward = 3

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
		return self.front_camera, reward, done, {}	#curly brackets - empty dictionary required by SB3 format

	def reset(self):
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

		while self.front_camera is None:
			time.sleep(0.01)
		
		self.episode_start = time.time()
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

		return self.front_camera

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
	