'''
this gets stuck possibly due to the navigational part
(from here - will try without navigation element)

this enviroment goes off the one with just front camera,
but adds an angle to the next waypoint - the car follows a route

The idea is to see how quickly the RL model will find 
to follow the angle and ignore the image data 

Also the car is driving at constant speed so throttle is not included

Note this is just an interim step to get our confidence
in the RL approach

After that we need to make the direction angle a lot further out from the car
so the car would have to learn from images not cut through corners

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
SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

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

        # Example for using image as input normalised to 0..1 (channel-first; channel-last also works):
		# adding a separate input of an angle to a close waypoint along the route
		self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32),
            'float_input': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })

		self.client = carla.Client("localhost", 2000)
		self.client.set_timeout(4.0)
		self.world = self.client.get_world()

		self.settings = self.world.get_settings()
		self.settings.no_rendering_mode = True
		self.settings.synchronous_mode = False
		self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
		self.world.apply_settings(self.settings)
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]
		self.route = None
	
	def select_random_route(self):
		'''
		retruns a random route for the car/veh
		out of the list of possible locations locs
		where distance is longer than 100 waypoints
		'''    
		point_a = self.vehicle.get_transform().location #we start at where the car is or last waypoint
		sampling_resolution = 1
		grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)
		# now let' pick the longest possible route
		min_distance = 100
		result_route = None
		route_list = []
		for loc in self.world.get_map().get_spawn_points(): # we start trying all spawn points 
															#but we just exclude first at zero index
			cur_route = grp.trace_route(point_a, loc.location)
			if len(cur_route) > min_distance:
				route_list.append(cur_route)
		result_route = random.choice(route_list)
		return result_route

	def get_closest_wp_forward(self):
		'''
		this function is to find the closest point looking forward
		if there in no points behind, then we get first available
		'''

		# first we create a list of angles and distances to each waypoint
		# yeah - maybe a bit wastefull
		points_ahead = []
		points_behind = []
		for i, wp in enumerate(self.route):
			#get angle
			vehicle_transform = self.vehicle.get_transform()
			wp_transform = wp[0].transform
			distance = ((wp_transform.location.y - vehicle_transform.location.y)**2 + (wp_transform.location.x - vehicle_transform.location.x)**2)**0.5
			angle = math.degrees(math.atan2(wp_transform.location.y - vehicle_transform.location.y,
								wp_transform.location.x - vehicle_transform.location.x)) -  vehicle_transform.rotation.yaw
			if angle>360:
				angle = angle - 360
			elif angle <-360:
				angle = angle + 360

			if angle>180:
				angle = -360 + angle
			elif angle <-180:
				angle = 360 - angle 
			if abs(angle)<=90:
				points_ahead.append([i,distance,angle])
			else:
				points_behind.append([i,distance,angle])
		# now we pick a point we need to get angle to 
		if len(points_ahead)==0:
			closest = min(points_behind, key=lambda x: x[1])
			if closest[2]>0:
				closest = [closest[0],closest[1],90]
			else:
				closest = [closest[0],closest[1],-90] 
		else:
			closest = min(points_ahead, key=lambda x: x[1])
			# move forward if too close
			for i, point in enumerate(points_ahead):
				if point[1]>=10 and point[1]<20:
					closest = point
					break
			return closest[2]/90.0, closest[1] # we convert angle to [-1 to +1] and also return distance
			
	def cleanup(self):
		for sensor in self.world.get_actors().filter('*sensor*'):
			sensor.destroy()
		for actor in self.world.get_actors().filter('*vehicle*'):
			actor.destroy()
		cv2.destroyAllWindows()
	
	def maintain_speed(self,s):
			''' 
			this is a very simple function to maintan desired speed
			s arg is actual current speed
			'''
			if s >= self.PREFERRED_SPEED:
				return 0
			elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
				return 0.7 # think of it as % of "full gas"
			else:
				return 0.3 # tweak this if the car is way over or under preferred speed 
	
	def step(self, action):
		self.step_counter +=1
		steer = action[0]
		
		# map steering actions
		if steer ==0:
			steer = - 0.9
		elif steer ==1:
			steer = -0.25
		elif steer ==2:
			steer = -0.1
		elif steer ==3:
			steer = -0.05
		elif steer ==4:
			steer = 0.0 
		elif steer ==5:
			steer = 0.05
		elif steer ==6:
			steer = 0.1
		elif steer ==7:
			steer = 0.25
		elif steer ==8:
			steer = 0.9
		# map throttle to maintain speed and apply steer and throttle	
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		estimated_throttle = self.maintain_speed(kmh)
		self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake = 0.0))
		
		if self.step_counter % 50 == 0:
			print('steer input from model:',steer)
		
		#self.world.tick()
		

		
		distance_travelled = self.initial_location.distance(self.vehicle.get_location())
		step_distance_gain = 0
		if self.distance_travelled_last < distance_travelled:
			step_distance_gain = distance_travelled - self.distance_travelled_last
			self.distance_travelled_last < distance_travelled
		# storing camera to return at the end in case the clean-up function destroys it
		cam = self.front_camera
		# showing image
		if self.SHOW_CAM:
			cv2.imshow('Sem Camera', cam)
			cv2.waitKey(1)

		# track steering lock duration
		lock_duration = 0
		if self.steering_lock == False:
			if steer<-0.6 or steer>0.6:
				self.steering_lock = True
				self.steering_lock_start = time.time()
		else:
			if steer<-0.6 or steer>0.6:
				lock_duration = time.time() - self.steering_lock_start
		# get angle and distance to the navigation route
		
		angle, distance = None, None
		while angle is None:
			try:
				# connect
				angle, distance = self.get_closest_wp_forward()
			except:
				pass
			
		#print('angle ',angle,' distance',distance)
		reward = 0
		done = False
		#punish for collision
		if len(self.collision_hist) != 0:
			done = True
			reward = reward-200
			self.cleanup()
		# punish for steer lock up
		if lock_duration>3:
			reward = reward - 100
			done = True
			self.cleanup()
		elif lock_duration > 1:
			reward = reward - 50
			
		# punish for deviating from the route
		route_loss =  distance - self.last_distance_to_route 
		if route_loss<-0.1:
			reward = reward + 10 #reward for getting closer to the route
		elif route_loss < 0.1:
			reward = reward + 1
		else:
			reward = reward - 2
		if distance > 20:
			reward = reward - 100
		# reward for making distance
		reward = reward + int(round(step_distance_gain*3,0))
		# check for episode duration
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
			self.cleanup()
		return  {'image': self.front_camera/255.0, 'float_input': angle}, reward, done, {}	#curly brackets - empty dictionary required by SB3 format

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
		self.initial_location = self.vehicle.get_location()
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.sem_cam.set_attribute("fov", f"90")

		
		camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(2)
		# showing camera at the spawn point
		if self.SHOW_CAM:
			cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Sem Camera', self.front_camera)
			cv2.waitKey(1)
		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		while self.front_camera is None:
			time.sleep(0.01)
		
		self.episode_start = time.time()
		self.steering_lock = False
		self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.distance_travelled_last = 0
		self.route = self.select_random_route()
		angle, distance_to_route = self.get_closest_wp_forward()
		self.last_distance_to_route = distance_to_route
		return  {'image': self.front_camera/255.0, 'float_input': angle}

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		i = np.array(image.raw_data)
		i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
		self.front_camera = i

	def collision_data(self, event):
		self.collision_hist.append(event)
	