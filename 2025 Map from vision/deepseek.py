'''
failed but could be used as a shell

"C://SelfDrive//2025 Map from vision//deepseek//"
'''

import carla
import numpy as np
import cv2
import os
import random
import time

# Constants
NUM_IMAGE_PAIRS = 1000  # Total number of image pairs to generate
IMAGE_WIDTH = 1280  # Tesla Model 3-like camera dimensions
IMAGE_HEIGHT = 720
FOV = 90  # Field of view
WEATHER_CHANGE_INTERVAL = 50  # Change weather every 50 images
TIME_CHANGE_INTERVAL = 100  # Change time every 100 images
MIN_DISTANCE = 1.0  # Minimum distance (in meters) to move before capturing next image
MAPS = ["Town01", "Town02", "Town03", "Town04", "Town05"]  # Standard CARLA maps

# CARLA client and world setup
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# Function to change weather
def change_weather(world):
    weather = random.choice([carla.WeatherParameters.ClearNoon,
                             carla.WeatherParameters.CloudyNoon,
                             carla.WeatherParameters.WetNoon,
                             carla.WeatherParameters.WetCloudyNoon,
                             carla.WeatherParameters.MidRainyNoon,
                             carla.WeatherParameters.HardRainNoon,
                             carla.WeatherParameters.SoftRainNoon])
    world.set_weather(weather)

# Function to change time of day
def change_time(world):
    # CARLA does not have a direct method to set time of day.
    # Instead, we use the weather parameters to simulate time changes.
    weather = world.get_weather()
    weather.sun_altitude_angle = random.choice([30, 60, 90])  # Simulate morning, noon, evening
    world.set_weather(weather)

# Function to calculate distance between two locations
def calculate_distance(loc1, loc2):
    return np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)

# Function to check if a location is on a sidewalk
def is_location_on_sidewalk(world, location):
    waypoint = world.get_map().get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
    return waypoint is not None

# Function to spawn pedestrians
def spawn_pedestrians(world, num_pedestrians=10):
    pedestrian_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
    spawn_points = world.get_map().get_spawn_points()
    pedestrians = []
    attempts = 0

    while len(pedestrians) < num_pedestrians and attempts < 100:  # Limit attempts to avoid infinite loops
        spawn_point = random.choice(spawn_points)
        spawn_point.location.z += 1.0  # Ensure pedestrians are not on roads

        # Check if the spawn point is on a sidewalk
        if is_location_on_sidewalk(world, spawn_point.location):
            pedestrian = world.try_spawn_actor(pedestrian_bp, spawn_point)
            if pedestrian:
                pedestrians.append(pedestrian)
        attempts += 1

    return pedestrians

# Function to spawn vehicles
def spawn_vehicles(world, num_vehicles=20):
    vehicle_bp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for _ in range(num_vehicles):
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicles.append(vehicle)
    return vehicles

# Function to set up cameras
def setup_cameras(vehicle, world):
    # Front-facing RGB camera
    rgb_camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    rgb_camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    rgb_camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    rgb_camera_bp.set_attribute("fov", str(FOV))
    rgb_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Tesla-like position
    rgb_camera = world.spawn_actor(rgb_camera_bp, rgb_camera_transform, attach_to=vehicle)

    # Top-down semantic segmentation camera
    semantic_camera_bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    semantic_camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    semantic_camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    semantic_camera_bp.set_attribute("fov", str(FOV))
    semantic_camera_transform = carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))  # Top-down view
    semantic_camera = world.spawn_actor(semantic_camera_bp, semantic_camera_transform, attach_to=vehicle)

    return rgb_camera, semantic_camera

# Function to process and save images
def process_and_save_images(rgb_image, semantic_image, pair_count):
    # Convert CARLA images to numpy arrays
    rgb_array = np.frombuffer(rgb_image.raw_data, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]  # Remove alpha channel

    semantic_array = np.frombuffer(semantic_image.raw_data, dtype=np.uint8)
    semantic_array = np.reshape(semantic_array, (IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]  # Remove alpha channel

    # Save images
    cv2.imwrite(f"C://SelfDrive//2025 Map from vision//deepseek//rgb{pair_count:04d}.png", rgb_array)
    cv2.imwrite(f"C://SelfDrive//2025 Map from vision//deepseek//semantic{pair_count:04d}.png", semantic_array)

# Main loop
def main():
    # Create output directory
    os.makedirs("training_images", exist_ok=True)

    # Initialize counters
    pair_count = 0
    map_index = 0

    while pair_count < NUM_IMAGE_PAIRS:
        # Load map
        world = client.load_world(MAPS[map_index])
        time.sleep(5)  # Allow time for map to load

        # Set up traffic manager
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_random_device_seed(random.randint(0, 1000))

        # Spawn ego vehicle
        vehicle_bp = random.choice(world.get_blueprint_library().filter("vehicle.tesla.model3"))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        # Spawn other vehicles and pedestrians
        vehicles = spawn_vehicles(world)
        pedestrians = spawn_pedestrians(world)

        # Set up cameras
        rgb_camera, semantic_camera = setup_cameras(ego_vehicle, world)

        # Initialize variables for tracking movement
        last_location = ego_vehicle.get_location()

        # Image generation loop
        while pair_count < NUM_IMAGE_PAIRS:
            # Change weather and time at intervals
            if pair_count % WEATHER_CHANGE_INTERVAL == 0:
                change_weather(world)
            if pair_count % TIME_CHANGE_INTERVAL == 0:
                change_time(world)

            # Check if ego vehicle has moved at least MIN_DISTANCE
            current_location = ego_vehicle.get_location()
            distance_moved = calculate_distance(last_location, current_location)
            if distance_moved < MIN_DISTANCE:
                time.sleep(0.1)
                continue

            # Capture images
            rgb_image = rgb_camera.listen()
            semantic_image = semantic_camera.listen()

            # Process and save images
            process_and_save_images(rgb_image, semantic_image, pair_count)
            pair_count += 1
            last_location = current_location

            # Check for collisions or stuck vehicle
            if ego_vehicle.is_at_traffic_light() or ego_vehicle.get_velocity().length() < 0.1:
                print("Vehicle stuck or collision detected. Resetting...")
                break

        # Clean up
        ego_vehicle.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
        for pedestrian in pedestrians:
            pedestrian.destroy()
        rgb_camera.destroy()
        semantic_camera.destroy()

        # Switch to next map
        map_index = (map_index + 1) % len(MAPS)

    print("Image generation complete.")

if __name__ == "__main__":
    main()