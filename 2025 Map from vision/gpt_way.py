'''
failed: traffic not moving, but could be used as a shell

'''
import carla
import random
import time
import cv2
import numpy as np
import os

# Constants
IMAGE_SAVE_PATH = "C://SelfDrive//2025 Map from vision//gpt_res"
TOTAL_IMAGE_PAIRS = 5000
IMAGES_PER_WEATHER_CHANGE = 200
MIN_MOVEMENT_THRESHOLD = 1.0  # meters
RESPAWN_STUCK_TIME = 10  # seconds
STUCK_THRESHOLD = 5  # meters movement in RESPAWN_STUCK_TIME
MAPS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
NUM_NPC_VEHICLES = 50

# Ensure save directory exists
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

# Initialize global variables
image_count = 0
vehicle_data = {}


def setup_client():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    return client


def set_random_weather(world):
    weather = carla.WeatherParameters(
        cloudiness=random.uniform(0, 100),
        precipitation=random.uniform(0, 50),
        sun_altitude_angle=random.uniform(0, 90),
        fog_density=random.uniform(0, 50),
    )
    world.set_weather(weather)


def spawn_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    return vehicle


def spawn_npc_vehicles(world, traffic_manager):
    blueprint_library = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for _ in range(NUM_NPC_VEHICLES):
        if spawn_points:
            vehicle_bp = random.choice(blueprint_library)
            spawn_point = random.choice(spawn_points)
            spawn_points.remove(spawn_point)
            npc_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if npc_vehicle:
                npc_vehicle.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.auto_lane_change(npc_vehicle, True)
                traffic_manager.random_left_lanechange_percentage(npc_vehicle, 50)
                traffic_manager.random_right_lanechange_percentage(npc_vehicle, 50)
                traffic_manager.vehicle_percentage_speed_difference(npc_vehicle, random.uniform(-30, 10))
                vehicles.append(npc_vehicle)
    return vehicles


def spawn_sensors(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    
    # Front-facing RGB camera
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", "1280")
    rgb_bp.set_attribute("image_size_y", "720")
    rgb_bp.set_attribute("fov", "90")
    rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
    rgb_camera = world.spawn_actor(rgb_bp, rgb_transform, attach_to=vehicle)
    
    # Top-down segmentation camera
    seg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    seg_bp.set_attribute("image_size_x", "1280")
    seg_bp.set_attribute("image_size_y", "720")
    seg_bp.set_attribute("fov", "90")
    seg_transform = carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))
    seg_camera = world.spawn_actor(seg_bp, seg_transform, attach_to=vehicle)
    
    return rgb_camera, seg_camera


def process_image(image, vehicle_id, is_segmentation=False):
    global image_count
    if vehicle_id not in vehicle_data:
        return
    
    # Convert CARLA image to numpy array
    img = np.reshape(np.array(image.raw_data), (image.height, image.width, 4))[:, :, :3]
    
    if is_segmentation:
        # Convert to easier-to-see RGB colors
        img = cv2.applyColorMap(img[:, :, 0], cv2.COLORMAP_JET)
    
    # Save images only if vehicle moved
    last_location = vehicle_data[vehicle_id]["last_location"]
    new_location = vehicle_data[vehicle_id]["vehicle"].get_location()
    
    if last_location.distance(new_location) > MIN_MOVEMENT_THRESHOLD:
        vehicle_data[vehicle_id]["last_location"] = new_location
        
        img_type = "seg" if is_segmentation else "rgb"
        filename = f"{IMAGE_SAVE_PATH}{image_count:06d}_{img_type}.png"
        cv2.imwrite(filename, img)
        
        if not is_segmentation:
            image_count += 1


def detect_stuck(vehicle):
    start_time = time.time()
    start_location = vehicle.get_location()
    
    while time.time() - start_time < RESPAWN_STUCK_TIME:
        time.sleep(1)
        current_location = vehicle.get_location()
        if start_location.distance(current_location) > STUCK_THRESHOLD:
            return False  # Vehicle is moving
    return True  # Vehicle is stuck


def run_simulation():
    global image_count
    client = setup_client()
    
    for town in MAPS:
        client.load_world(town)
        world = client.get_world()
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(True)
        
        set_random_weather(world)
        vehicle = spawn_vehicle(world)
        npc_vehicles = spawn_npc_vehicles(world, traffic_manager)
        rgb_camera, seg_camera = spawn_sensors(world, vehicle)
        
        vehicle_id = vehicle.id
        vehicle_data[vehicle_id] = {"vehicle": vehicle, "last_location": vehicle.get_location()}
        
        rgb_camera.listen(lambda image: process_image(image, vehicle_id, is_segmentation=False))
        seg_camera.listen(lambda image: process_image(image, vehicle_id, is_segmentation=True))
        
        while image_count < TOTAL_IMAGE_PAIRS:
            time.sleep(0.5)
            if image_count % IMAGES_PER_WEATHER_CHANGE == 0:
                set_random_weather(world)
            
            if detect_stuck(vehicle):
                print("Vehicle is stuck. Respawning.")
                for actor in world.get_actors().filter("*vehicle*"):
                    actor.destroy()
                for actor in world.get_actors().filter("*sensor*"):
                    actor.destroy()
                vehicle = spawn_vehicle(world)
                npc_vehicles = spawn_npc_vehicles(world, traffic_manager)
                rgb_camera, seg_camera = spawn_sensors(world, vehicle)
                vehicle_data[vehicle_id] = {"vehicle": vehicle, "last_location": vehicle.get_location()}
                
        # Cleanup
        vehicle.destroy()
        for npc in npc_vehicles:
            npc.destroy()
        rgb_camera.destroy()
        seg_camera.destroy()
    
    print("Image generation complete.")

if __name__ == "__main__":
    run_simulation()
