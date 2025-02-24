'''
this is a close copy of image generation
but modified for displaying what a trained model predicts


Outcome: model predicts a very crapy picture

'''

import sys
from numpy import random
import numpy as np
import math
import logging
import os
import glob
import re
import cv2
import time
import torch
import torch.nn.functional as F
import torch.nn as nn


import torchvision.transforms as transforms
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


MAP = "Town10HD" #, "Town04", "Town05", "Town10HD"]
DISTANCE_SPECTATOR = 5
#camera mount offset on the car to mimic Tesla Model3
CAMERA_POS_Z = 1.3 
CAMERA_POS_X = 1.4 

CAM_W = '640'
CAM_H = '480'

NO_RENDERING = False

MAP_FOLDER = 'C://SelfDrive//2025 Map from vision//map_img//'
IMG_FOLDER = 'C://SelfDrive//2025 Map from vision//img//'
SEM_FOLDER = 'C://SelfDrive//2025 Map from vision//sem_img//'


MIN_MOVE_BETWEEN_IMAGES = 2 #meters required to move before taking next image
MAX_SECONDS_NOT_MOVING = 40 # how many max delay is allowed before restarting the current map and traffic (getting stuck)

IMAGES_PER_MAP = 20_000 # how many pairs expected per map
WEATHER_CHANGE_IMG_FREQUENCY = 100 #this must be less than above

CAR_COUNT = 50
WALKER_COUNT = 50


    
def rgb_callback(image,data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

def calculate_sides(hypotenuse, angle):

  # Convert the angle to radians
  angle_radians = math.radians(angle)

  # Calculate the opposite side using the sine function
  opposite_side = hypotenuse * math.sin(angle_radians)

  # Calculate the adjacent side using the cosine function
  adjacent_side = hypotenuse * math.cos(angle_radians)

  return opposite_side, adjacent_side

def adjust_spectator():
    '''
    optional function to make spectator view using above function
    '''
    vehicle_transform = sim_world.get_actors(world.vehicles_list)[0].get_transform()
    y,x = calculate_sides(DISTANCE_SPECTATOR, vehicle_transform.rotation.yaw )
    spectator_pos = carla.Transform(vehicle_transform.location + carla.Location(x=-x,y=-y,z=5 ),
                                            carla.Rotation( yaw = vehicle_transform.rotation.yaw,pitch = -25))
    spectator.set_transform(spectator_pos)  

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2))
        dec3 = self.dec3(F.interpolate(dec4, scale_factor=2))
        dec2 = self.dec2(F.interpolate(dec3, scale_factor=2))
        dec1 = self.dec1(F.interpolate(dec2, scale_factor=2))
        
        return self.final(dec1)

class World(object):
    def __init__(self,carla_world):
        self.world = carla_world
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
        self.synchronous_master = True
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.hero_car = None
        self.hero_transform = None
        self.last_img_pos = None
        self.camera_data = None
        self.camera_sem = None
        self.camera_rgb = None
        self.actors_with_transforms = []
        
        self.scale_offset = [0, 0]
        

        
    def attach_sensors(self):
            '''
            attaches rgb camera to hero car

            '''

            #normal rgb camera
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', CAM_W) # this ratio works in CARLA 9.13 on Windows
            camera_bp.set_attribute('image_size_y', CAM_H)
            camera_bp.set_attribute('fov', '90')
            camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
            self.camera_rgb = self.world.spawn_actor(camera_bp,camera_init_trans,attach_to=self.hero_car)
            image_w = int(CAM_W)
            image_h = int(CAM_H)
            self.camera_data = {'rgb_image': np.zeros((image_h,image_w,4))}
            # this actually opens a live stream from the camera
            self.camera_rgb.listen(lambda image: rgb_callback(image,self.camera_data))

    def destroy_traffic(self,client):
        '''
        destroy all previous traffic
        '''
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
    
    def generate_traffic(self,client,number_of_vehicles,number_of_walkers):
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)


        settings = self.world.get_settings()
        self.synchronous_master = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = NO_RENDERING
        self.world.apply_settings(settings)
        
        print('commencing spawning actors...')

        blueprints = get_actor_blueprints(self.world, 'vehicle.*', 'All')
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(self.world, 'walker.pedestrian.*','2')
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        print('commencing spawning vehicles...')

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            if n==0:
                blueprint = random.choice(self.world.get_blueprint_library().filter("vehicle.tesla.model3"))
            else:
                blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
    
        all_vehicle_actors = self.world.get_actors(self.vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)
    
        print('commencing spawning walkers...')

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)
        
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+Q to exit.' % (len(self.vehicles_list), len(self.walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)
        self.hero_car = self.world.get_actors(world.vehicles_list)[0]
    
    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])

transform = transforms.Compose([
    transforms.Resize((int(CAM_H), int(CAM_W))),  # Resize to match model input size
    transforms.ToTensor(),           # Convert to tensor
    #transforms.Normalize((0.5,), (0.5,))  # Normalize (adjust based on training)
])

#main code of loop
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device',device)
model = UNet().to(device) 

#load a trained model
state_dict = torch.load('C://SelfDrive//2025 Map from vision//unet_model_20250219_9.pth')  
model.load_state_dict(state_dict)


print('Loading town:',MAP)
client.load_world(MAP)
sim_world = client.get_world()
original_settings = sim_world.get_settings()

world = World(sim_world)

spectator = sim_world.get_spectator()
world.generate_traffic(client,number_of_vehicles=CAR_COUNT,number_of_walkers=WALKER_COUNT)
world.attach_sensors()
sim_world.tick()
#time.sleep(50)
#tick loop
img_counter = 0
time_grab_last_img = None
while True:
    sim_world.tick()
    time_grab = time.time_ns() # used for file names of images
    time_grab_current = time.time()
    #actors = sim_world.get_actors() # can use 
    world.actors_with_transforms = [(actor, actor.get_transform()) for actor in sim_world.get_actors()]
    world.hero_transform = world.hero_car.get_transform()
    if world.last_img_pos is None:
        world.last_img_pos = world.hero_transform
    if time_grab_last_img is None:
        time_grab_last_img = time_grab_current

    #optional - move spectator behind a car - use for debugging only and remove "no rendering" before hand
    #adjust_spectator() 

    # display RGB cam
    rgb_im = world.camera_data['rgb_image']
    rgb_im = np.clip(rgb_im, 0, 255).astype(np.uint8)
    rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGRA2BGR)
    
    input_tensor = transform(Image.fromarray(rgb_im))
    input_tensor = input_tensor.to(device)
    
    # Generate prediction
    #with torch.no_grad():
    output_tensor = model(input_tensor)

    # Convert output tensor to image
    predicted_map = output_tensor.squeeze(0).detach().cpu()
    predicted_map = transforms.ToPILImage()(predicted_map)

    predicted_map = np.array(predicted_map)

    # Ensure predicted_map is 3-channel (if it's grayscale, convert it to RGB)
    if len(predicted_map.shape) == 2:  # If single-channel (H, W)
        predicted_map = cv2.cvtColor(predicted_map, cv2.COLOR_GRAY2BGR)

    # Resize predicted_map to match rgb_im
    predicted_map = cv2.resize(predicted_map, (rgb_im.shape[1], rgb_im.shape[0]))  # Resize to (W, H)


    if cv2.waitKey(1) == ord('q'):
        break
    
    side_by_side = np.hstack((rgb_im, predicted_map))

    cv2.imshow('RGB Camera and predicted map',side_by_side)
    world.last_img_pos = world.hero_transform
    time_grab_last_img =  time.time()
    img_counter +=1
    if img_counter % WEATHER_CHANGE_IMG_FREQUENCY == 0:
        print('Changing weather now ...for every ',WEATHER_CHANGE_IMG_FREQUENCY,'images. Press q to exit')
        world.next_weather(reverse=False)
    # check for "being stuck"
    if (time_grab_current - time_grab_last_img) > MAX_SECONDS_NOT_MOVING:
        world.camera_sem.stop()
        world.camera_rgb.stop()
        world.destroy_traffic(client)
        print('detected being stuck, re-starting traffic in current location to continue..')
        time.sleep(5)
        world.generate_traffic(client,number_of_vehicles=CAR_COUNT,number_of_walkers=WALKER_COUNT)
        world.attach_sensors()
                

world.camera_rgb.stop()
world.destroy_traffic(client)

sim_world.tick()
sim_world.apply_settings(original_settings)

