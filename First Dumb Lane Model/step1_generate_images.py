#from CARLA camera tutorial on YouTube
# this approach make the camera image available with for a simple loop

import carla
import math
import time
import cv2
import numpy as np
import random


# max yaw angle from straight
YAW_ADJ_DEGREES = 35

PREFERRED_SPEED = 10 #optional

#mount point of camera on the car
CAMERA_POS_Z = 1.6
CAMERA_POS_X = 0.9

# I separately learned road id's covering the ring highway around Town 5
# not this would be different in other Towns/maps
good_roads = [12, 34, 35, 36, 37, 38, 1201, 1236, 2034, 2035, 2343, 2344]


# connect to sim
client = carla.Client('localhost', 2000)
client.set_timeout(15)

# load Town5 map
client.load_world('Town05') 

#transform car through waypoints in a loop while printing the angle onto the image

# sim settings
world = client.get_world()
traffic_manager = client.get_trafficmanager(8000)
settings = world.get_settings()
traffic_manager.set_synchronous_mode(True)
# option preferred speed
# traffic_manager.set_desired_speed(vehicle,float(PREFERRED_SPEED))
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# pick a car model - Tesla M3
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('*model3*')

town_map = world.get_map()

# optional - get all intersections to avoid cutting images near them
junction_list = []
waypoint_list = town_map.generate_waypoints(2.0)
for x in waypoint_list:
    if x.get_junction() is not None:
        junction_list.append(x.transform.location)
                
#limit spawn points to highways
spawn_points = town_map.get_spawn_points()
good_spawn_points = []
for point in spawn_points:
    this_waypoint = town_map.get_waypoint(point.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
    if this_waypoint.road_id in good_roads:
        good_spawn_points.append(point)

all_waypoint_pairs = town_map.get_topology()
# subset of lane start/end's which belong to good roads
good_lanes = []
for w in all_waypoint_pairs:
    if w[0].road_id in good_roads:
        good_lanes.append(w)        

transform = random.choice(good_spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp[0], transform)

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


#main loop 
quit = False

for lane in good_lanes:
    #loop within a lane
    if quit:
        break
    for wp in lane[0].next_until_lane_end(20):
        transform = wp.transform
        vehicle.set_transform(wp.transform)
        time.sleep(2) #these delays seem to be necessary for teh car to take the position before a shot is taken
        initial_yaw = wp.transform.rotation.yaw
        # do multiple shots of straight direction
        for i in range(5):
            # Carla Tick
            world.tick()
            
            trans = wp.transform
            angle_adj = random.randrange(-YAW_ADJ_DEGREES, YAW_ADJ_DEGREES, 1)
            trans.rotation.yaw = initial_yaw +angle_adj 
            vehicle.set_transform(trans)
            time.sleep(1)  #these delays seem to be necessary for teh car to take the position before a shot is taken
            if cv2.waitKey(1) == ord('q'):
                quit = True
                break
            img = camera_data['image']
            
            actual_angle = vehicle.get_transform().rotation.yaw - initial_yaw
            #fixing values like 361.5 to make them close to zero
            if actual_angle <-180:
                actual_angle +=360
            elif actual_angle >180:
                actual_angle -=360
            actual_angle = str(int(actual_angle))
            img = np.float32(img)
            img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
            time_grab = time.time_ns()
            cv2.imwrite('_out_ang/%06d_%s.png' % (time_grab, actual_angle), img_gry)
            #old way to screen - cv2.imshow('RGB Camera',img)
            
#clean up
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()