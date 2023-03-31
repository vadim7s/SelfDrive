'''

working example how to get a route from A to B
where A and B are specific locations in town 5
expressed as carla.Location(z,y,z)

This plots the route in the simulation iself
so it must be cleaned up or world reloaded

single lane route example in town 5 (no lane changes, but the road bends)
a = carla.Location(x=-247.802231, y=-102.741714, z=10.000187)
b = carla.Location(x=211.036575, y=14.105213, z=0.000000)
'''



import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

#add a specific folder - must be changed for something more generic
sys.path.append('C:/CARLA_0.9.14/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world('Town05')
#world = client.get_world()
amap = world.get_map()
sampling_resolution = 2
#dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
grp = GlobalRoutePlanner(amap, sampling_resolution)
#grp.setup()
#spawn_points = world.get_map().get_spawn_points()
#a = carla.Location(spawn_points[50].location)
a = carla.Location(x=-247.802231, y=-102.741714, z=10.000187)

b = carla.Location(x=211.036575, y=14.105213, z=0.000000)
#Location(spawn_points[100].location)
w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
i = 0
for w in w1:
    if i % 10 == 0:
        world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
            persistent_lines=True)
    else:
        world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
            persistent_lines=True)
    i += 1