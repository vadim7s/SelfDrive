'''
This is a script to generate:
1. Normal RGB images
2. Semantic segmentation versions of RGB images (labels)

Under different weather and lighting conditions

Camera parameters are closest to real Tesla Model 3 front facing camera

The end purpose of this is to train a vision model using output images
so the model could be tested on real Tesla footage and would generate
a mask to show drivable surfave in front of the car

This version of code generate images with traffic

It is recommended to re-start the simulator each time you run this

You must create folder structure for output images BEFORE running
 In the folder where you launch jupyter notebook, these these
 out_sem/rgb    - this folder will contain RGB images
 out_sem/sem    - this will contain semantic images

 '''
import carla #the sim library itself
import cv2 #to work with images from cameras
import numpy as np #in this example to change image representation - re-shaping
import time
import random

client = carla.Client('localhost', 2000)
time.sleep(5)
client.set_timeout(25)
client.load_world('Town11')

world = client.get_world()
spawn_points = world.get_map().get_spawn_points()
vehicle_bp = world.get_blueprint_library().filter('*model3*')

# clean up
for actor in world.get_actors().filter('*vehicle*'):
    actor.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()

# ensure sync mode on 
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1
settings.no_rendering_mode = True
world.apply_settings(settings)
    
vehicle = world.try_spawn_actor(vehicle_bp[0], spawn_points[0])

# callbacks for cameras
def sem_callback(image,data_dict):
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))
    
def rgb_callback(image,data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

if vehicle == None:
    print("Re-start the sim and try again. No connection to the simulator.")
else:

    #lights always on
    vehicle.set_light_state(carla.VehicleLightState.LowBeam)

    #camera mount offset on the car - you can tweak these to each car to avoid any parts of the car being in the view
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4 

    #semantic camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.13 on Windows
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')

    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
    camera_sem = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)

    #normal rgb camera
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.13 on Windows
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    camera_init_trans = carla.Transform(carla.Location(z=CAMERA_POS_Z,x=CAMERA_POS_X))
    camera_rgb = world.spawn_actor(camera_bp,camera_init_trans,attach_to=vehicle)


    image_w = 640
    image_h = 480

    camera_data = {'sem_image': np.zeros((image_h,image_w,4)),
                   'rgb_image': np.zeros((image_h,image_w,4))}

    # this actually opens a live stream from the cameras
    camera_sem.listen(lambda image: sem_callback(image,camera_data))
    camera_rgb.listen(lambda image: rgb_callback(image,camera_data))


    # get all drivable locations on the map
    all_roads = world.get_map().get_topology()
    #loop for random weather conditios
    for i in range(10):
        print('Commencing cycle ',i)
        weather = carla.WeatherParameters(
                cloudiness=random.randint(0,100),
                precipitation=random.randint(0,100),
                sun_altitude_angle=random.randint(0,100),
                precipitation_deposits =random.randint(0,100),
                fog_density =random.choice([0.0,0.1,0.3,0.4,0.95]),
                wetness = random.randint(0,100))
        world.set_weather(weather)
        for waypoint in all_roads:
            world.tick()
            vehicle.set_transform(waypoint[0].transform)
            time.sleep(0.5)
            rgb_im = camera_data['rgb_image']
            sem_im = camera_data['sem_image']
            #show images
            im_h = cv2.hconcat([rgb_im,sem_im])
            cv2.imshow('2 cameras', im_h)
            if cv2.waitKey(1) == ord('q'):
                break
            #write images
            time_grab = time.time_ns()
            cv2.imwrite('out_sem/rgb/%06d.png' % time_grab, rgb_im)
            cv2.imwrite('out_sem/sem/%06d.png' % time_grab, sem_im)


    cv2.destroyAllWindows()
    camera_sem.stop() # this is the opposite of camera.listen
    camera_rgb.stop() 
    for actor in world.get_actors().filter('*vehicle*'):
        actor.destroy()
    for sensor in world.get_actors().filter('*sensor*'):
        sensor.destroy()