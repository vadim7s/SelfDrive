''' 
This is a simple example of maintaining speed
A car is spawned without autopilot and the speed is controled by
comparing actual speed to PREFERRED_SPEED constant
If the speed is reached - set throttle to 0
if it is within a set threshold below Preferred speed (e.g. within 5 kmh) - set throttle to 0.3
if it is lower - set the threshold to a max value, say 0.8

That is very simplistic approach
in reality, you also need to use acceleration to achieve comfortable handling 
'''

import carla
import math
import time
import cv2
import numpy as np
import random



PREFERRED_SPEED = 15
# kmh down from max speed when full throttle is too much
SPEED_THRESHOLD = 5

#mount point of camera on the car
CAMERA_POS_Z = 1.6
CAMERA_POS_X = 0.9

HEIGHT = 360
WIDTH = 640


#adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
fontScale = 1
# white color
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2


client = carla.Client('localhost', 2000)
client.set_timeout(10)
client.load_world('Town05') 

time.sleep(5)

world = client.get_world()

traffic_manager = client.get_trafficmanager(8000)
settings = world.get_settings()
traffic_manager.set_synchronous_mode(True)
# option preferred speed
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('*model3*')

town_map = world.get_map()

good_roads = [37]
spawn_points = world.get_map().get_spawn_points()
good_spawn_points = []
for point in spawn_points:
    this_waypoint = world.get_map().get_waypoint(point.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
    if this_waypoint.road_id in good_roads:
        good_spawn_points.append(point)

start_point = random.choice(good_spawn_points)

vehicle = world.try_spawn_actor(vehicle_bp[0], start_point)

time.sleep(5)

#setting RGB Camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
camera_bp.set_attribute('image_size_y', '360')

camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
camera = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)

def camera_callback(image,data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

camera_data = {'image': np.zeros((image_h,image_w,4))}
camera.listen(lambda image: camera_callback(image,camera_data))

image = camera_data['image']

# show main camera
cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera',image)
#cv2.waitKey(1)

#main loop 
quit = False
#vehicle.set_autopilot(True)

def maintain_speed(s):
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.8
    else:
        return 0.3

while True:
    # Carla Tick
    world.tick()
    v = vehicle.get_velocity()
    a = vehicle.get_acceleration()
    speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2),0)
    acceleration = round(math.sqrt(a.x**2 + a.y**2 + a.z**2),1)

    estimated_throttle = maintain_speed(speed)
    vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle))
    if cv2.waitKey(1) == ord('q'):
        quit = True
        break
    image = camera_data['image']
    
    image = cv2.putText(image, "Speed: "+str(int(speed)), org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('RGB Camera',image)
    
            
#clean up
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
