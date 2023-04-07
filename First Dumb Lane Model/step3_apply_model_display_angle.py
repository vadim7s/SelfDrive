#from CARLA camera tutorial on YouTube
# this approach make the camera image available with for a simple loop

import carla
import math
import time
import cv2
import numpy as np
import random

from tensorflow.keras.models import load_model


PREFERRED_SPEED = 10
#mount point of camera on the car
CAMERA_POS_Z = 1.6
CAMERA_POS_X = 0.9

HEIGHT = 360
WIDTH = 640

HEIGHT_REQUIRED_PORTION = 0.4 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.5

# image crop calcs  - same as in model build
height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
width_to = width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

#adding params to display text to image
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
fontScale = 1
# white color
color = (255, 255, 255)
# Line thickness of 2 px
thickness = 2

model = load_model('lane_model_360x640_02_20',compile=False)
model.compile()


client = carla.Client('localhost', 2000)
client.set_timeout(10)
client.load_world('Town05') 


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

traffic_manager.set_desired_speed(vehicle,float(PREFERRED_SPEED))

def predict_angle(im):
    # tweaks for prediction
    img = np.float32(im)
    img_gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    img_gry = cv2.resize(img_gry, (WIDTH,HEIGHT))
    # this version adds taking lower side of the image
    img_gry = img_gry[height_from:,width_from:width_to]
    img_gry = img_gry.astype(np.uint8)
    canny = cv2.Canny(img_gry,50,150)

    #cv2.imshow('processed image',canny)
    canny = canny /255
    input_for_model = canny[ :, :, None] 
    input_for_model = np.expand_dims(input_for_model, axis=0)
    #print('input shape: ',input_for_model.shape)
    return model.predict(input_for_model)[0][0] 


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

predicted_angle = predict_angle(image)

image = cv2.putText(image, str(predicted_angle), org, font, fontScale, color, thickness, cv2.LINE_AA)

# show main camera
cv2.namedWindow('RGB Camera',cv2.WINDOW_AUTOSIZE)
cv2.imshow('RGB Camera',image)
#cv2.waitKey(1)

#main loop 
quit = False
vehicle.set_autopilot(True)
while True:
    # Carla Tick
    world.tick()

    if cv2.waitKey(1) == ord('q'):
        quit = True
        break
    image = camera_data['image']
    
    predicted_angle = predict_angle(image)
    image = cv2.putText(image, str(predicted_angle), org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('RGB Camera',image)
    
            
#clean up
cv2.destroyAllWindows()
camera.stop()
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()