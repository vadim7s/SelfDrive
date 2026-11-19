"""Phase A: capture matching RGB + ground-truth semantic frames from CARLA (Tesla-matched camera, full 1280x960),
to score a real-world segmenter against CARLA's perfect masks.
  python seg_capture.py [--per 12] [--out data\\seg_phaseA\\carla]
Ego on Traffic Manager autopilot in traffic; per town x weather saves --per pairs, one every 1.5 s.
Shuts CARLA down at the end."""
import argparse
import json
import os
import queue
import random

import numpy as np
from PIL import Image

import config as C
C.CARLA_ARGS = C.VISION_CARLA_ARGS                    # real rendering (Epic), like vision collection
import carla_server as srv
import carla

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
WEATHERS = ["ClearNoon", "WetCloudySunset", "ClearNight"]


def cam_bp(lib, kind):
    bp = lib.find(kind)
    bp.set_attribute("image_size_x", str(C.TESLA_CAM["width"]))
    bp.set_attribute("image_size_y", str(C.TESLA_CAM["height"]))
    bp.set_attribute("fov", f"{C.TESLA_CAM['fov']:.2f}")
    bp.set_attribute("sensor_tick", "0.0")
    return bp


def grab(q, frame):
    while True:
        img = q.get(timeout=30)
        if img.frame >= frame:
            return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(C.DATA_ROOT, "seg_phaseA", "carla"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--towns", default=",".join(TOWNS))
    ap.add_argument("--weathers", default=",".join(WEATHERS))
    ap.add_argument("--every", type=int, default=30, help="ticks between saved frames (2 = 10 Hz sequence)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    try:
        if not srv.ensure_running():
            raise SystemExit("CARLA did not start")
        client = carla.Client(C.HOST, C.PORT)
        client.set_timeout(120.0)
        for town in args.towns.split(","):
            world = client.load_world(town)
            s = world.get_settings()
            s.synchronous_mode, s.fixed_delta_seconds = True, C.FIXED_DELTA
            world.apply_settings(s)
            tm = client.get_trafficmanager(C.TM_PORT)
            tm.set_synchronous_mode(True)
            lib = world.get_blueprint_library()
            pts = world.get_map().get_spawn_points()
            for wname in args.weathers.split(","):
                world.set_weather(getattr(carla.WeatherParameters, wname))
                rng.shuffle(pts)
                actors = []
                ego = world.spawn_actor(lib.find("vehicle.tesla.model3"), pts[0])
                actors.append(ego)
                cars = [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
                for p in pts[1:45]:
                    a = world.try_spawn_actor(rng.choice(cars), p)
                    if a:
                        a.set_autopilot(True, C.TM_PORT)
                        actors.append(a)
                ego.set_autopilot(True, C.TM_PORT)
                tf = carla.Transform(carla.Location(x=C.TESLA_CAM["x"], y=C.TESLA_CAM["y"], z=C.TESLA_CAM["z"]),
                                     carla.Rotation(pitch=C.TESLA_CAM["pitch"]))
                qs = []
                for kind in ("sensor.camera.rgb", "sensor.camera.semantic_segmentation"):
                    cam = world.spawn_actor(cam_bp(lib, kind), tf, attach_to=ego)
                    q = queue.Queue()
                    cam.listen(q.put)
                    actors.append(cam)
                    qs.append(q)
                saved = 0
                for step in range(60 + args.per * args.every):
                    frame = world.tick()
                    rgb, sem = grab(qs[0], frame), grab(qs[1], frame)
                    if step >= 60 and step % args.every == 0 and saved < args.per:
                        a = np.frombuffer(rgb.raw_data, np.uint8).reshape(rgb.height, rgb.width, 4)[:, :, 2::-1]
                        t = np.frombuffer(sem.raw_data, np.uint8).reshape(sem.height, sem.width, 4)[:, :, 2]
                        name = f"{town}_{wname}_{saved:03d}" if args.every < 30 else f"{town}_{wname}_{saved:02d}"
                        tf_e, v = ego.get_transform(), ego.get_velocity()
                        json.dump(dict(x=tf_e.location.x, y=tf_e.location.y, z=tf_e.location.z, yaw=tf_e.rotation.yaw,
                                       pitch=tf_e.rotation.pitch, roll=tf_e.rotation.roll, frame=frame,
                                       speed=(v.x ** 2 + v.y ** 2) ** 0.5),
                                  open(os.path.join(args.out, name + "_pose.json"), "w"))
                        Image.fromarray(a).save(os.path.join(args.out, name + "_rgb.jpg"), quality=92)
                        Image.fromarray(t).save(os.path.join(args.out, name + "_tag.png"))
                        saved += 1
                for a in actors:
                    if a.type_id.startswith("sensor"):
                        a.stop()
                client.apply_batch_sync([carla.command.DestroyActor(a) for a in actors])
                world.tick()
                print(f"{town} {wname}: {saved} pairs", flush=True)
            s.synchronous_mode, s.fixed_delta_seconds = False, None
            world.apply_settings(s)
            tm.set_synchronous_mode(False)
    finally:
        srv.shutdown_if_configured()
        print("CARLA running:", bool(srv.processes_running()), flush=True)


if __name__ == "__main__":
    main()
