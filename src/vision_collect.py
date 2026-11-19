"""Stage 3 data + driving: windscreen-camera frames labelled with the teacher's actions.

    python vision_collect.py
        collect until config.VISION_TARGET_FRAMES (supervised: restarts CARLA on a hang and carries on)
    ... vision_collect.py --driver student --student <student.pt> --town Town02 --episodes 10 --no-save
        test the student driving on its own (scores like evaluate.py)
    ... vision_collect.py --driver mix --student <student.pt> --beta 0.5
        DAgger: student and teacher share the wheel, teacher labels every frame

Every frame's label is ALWAYS the teacher's action for that moment, whoever is driving.
Teacher mode adds short steering pushes (config NOISE_*) so the data shows how to recover.
Per episode folder: 000000.jpg ... + labels.csv + meta.json. A folder is written as <name>.tmp
and renamed when the episode ends, so a crash never leaves a half episode.
CARLA is shut down at the end.
"""
import argparse
import csv
import json
import os
import pickle
import queue
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

import config as C

C.CARLA_ARGS = C.VISION_CARLA_ARGS            # rendering server (also used by restarts below)
C.TICK_TIMEOUT_S = C.VISION_TICK_TIMEOUT_S
import carla_server as srv                    # noqa: E402
import carla                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_COLS = ["frame", "steer", "accel", "wheel", "speed", "command", "noise", "lat", "head_err", "v_target",
              # pose + traffic state (Phase C: map truth top-down + situation labels are derived offline from these)
              "x", "y", "z", "yaw", "route_idx", "lead_gap", "lead_speed", "red_dist", "yield_dist"]


# ----------------------------------------------------------------------------------- data on disk
def done_episodes(root):
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if d.startswith("ep_") and not d.endswith(".tmp"))


def frames_on_disk(root, town=None):
    """Frames in finished episodes (of one town only, if given: the target is per town)."""
    n = 0
    for d in done_episodes(root):
        try:
            with open(os.path.join(root, d, "meta.json"), encoding="utf-8") as f:
                m = json.load(f)
            if town is None or m.get("town") == town:
                n += m["frames"]
        except (OSError, ValueError, KeyError):
            pass
    return n


def clean_partial(root):
    if os.path.isdir(root):
        for d in os.listdir(root):
            if d.endswith(".tmp"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)


# ----------------------------------------------------------------------------------- camera
def camera_params(rng, jitter):
    t = dict(C.TESLA_CAM)
    if jitter:
        for k, amp in C.CAM_JITTER.items():
            t[k] += rng.uniform(-amp, amp)
    return t


def attach_camera(env, cam):
    bp = env.world.get_blueprint_library().find("sensor.camera.rgb")
    w, h = C.VISION_IMG
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", f"{cam['fov']:.2f}")
    bp.set_attribute("sensor_tick", "0.0")
    tf = carla.Transform(carla.Location(x=cam["x"], y=cam["y"], z=cam["z"]), carla.Rotation(pitch=cam["pitch"]))
    actor = env.world.spawn_actor(bp, tf, attach_to=env.vehicle)
    q = queue.Queue()
    actor.listen(q.put)
    env.sensors.append(actor)                 # env destroys it at the next reset / close
    return q


def image_at(q, frame, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            img = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if img.frame >= frame:
            a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            return np.ascontiguousarray(a[:, :, 2::-1])        # BGRA -> RGB
    raise RuntimeError(f"camera frame {frame} did not arrive")


# ----------------------------------------------------------------------------------- weather
def weather_table(which):
    day = [(w, 1.0) for w in C.VISION_DAY_WEATHERS]
    extra = list(C.VISION_EXTRA_WEATHERS.items())
    rows = {"day": day, "extra": extra, "all": day + extra}[which]
    return [r[0] for r in rows], [r[1] for r in rows]


def set_weather(env, name):
    if name in C.VISION_CUSTOM_WEATHERS:
        base = carla.WeatherParameters.ClearNoon            # shared preset: copy, never modify it
        w = carla.WeatherParameters()
        for k in ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
                  "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
                  "wetness", "fog_falloff", "scattering_intensity", "mie_scattering_scale",
                  "rayleigh_scattering_scale", "dust_storm"):
            if hasattr(base, k):
                setattr(w, k, getattr(base, k))
        for k, v in C.VISION_CUSTOM_WEATHERS[name].items():
            setattr(w, k, v)
    else:
        w = getattr(carla.WeatherParameters, name)
    env.world.set_weather(w)
    L = carla.VehicleLightState
    lights = L.Position | L.LowBeam if any(s in name for s in C.HEADLIGHTS_IN) else L.NONE
    env.vehicle.set_light_state(carla.VehicleLightState(lights))


# ----------------------------------------------------------------------------------- drivers
class Teacher:
    def __init__(self, path):
        from stable_baselines3 import PPO
        from train import vecnorm_for
        with open(vecnorm_for(path), "rb") as f:
            self.vecnorm = pickle.load(f)
        self.model = PPO.load(path, device="cpu")

    def act(self, obs):
        a, _ = self.model.predict(self.vecnorm.normalize_obs(obs), deterministic=True)
        return np.asarray(a, dtype=np.float32)


class Pusher:
    """Occasional steering push (recovery data). Returns the offset to add this step."""
    def __init__(self, rng, steer=None, steps=None):
        self.rng, self.left, self.size = rng, 0, 0.0
        self.steer, self.steps = steer or C.NOISE_STEER, steps or C.NOISE_STEPS

    def __call__(self):
        if self.left == 0 and self.rng.random() < C.NOISE_PROB:
            self.left = self.rng.randint(*self.steps)
            self.size = self.rng.choice((-1, 1)) * self.rng.uniform(*self.steer)
        if self.left:
            self.left -= 1
            return self.size
        return 0.0


# ----------------------------------------------------------------------------------- one session
def opt(v):
    return "" if v is None else f"{v:.2f}"


def worker(args):
    if args.v2:                                           # real-car camera geometry + 640x480 (segmentation stage)
        C.TESLA_CAM, C.VISION_IMG = C.TESLA_CAM_V2, C.VISION_IMG_V2
    from carla_env import CarlaDriveEnv, CarlaFailure
    from evaluate import smooth_line
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    save = not args.no_save
    root = args.out
    if save:
        os.makedirs(root, exist_ok=True)
        clean_partial(root)
    teacher = Teacher(args.teacher)
    student = None
    if args.driver != "teacher":
        import torch
        if torch.load(args.student, map_location="cpu", weights_only=False).get("settings", {}).get("kind") == "bev":
            from bev_model import BevStudentDriver      # top-down student (Phase D): needs the --v2 camera
            if not args.v2:
                raise SystemExit("the top-down student needs --v2 (real-car camera at 640x480)")
            student = BevStudentDriver(args.student)
            student.gps = args.gps_offset              # fixed GPS error test (route-curve student)
        else:
            from vision_model import StudentDriver
            student = StudentDriver(args.student)

    env = None
    results = []
    trace = trace_f = None
    if args.trace:
        trace_f = open(args.trace, "w", newline="", encoding="utf-8")
        trace = csv.writer(trace_f)
        trace.writerow(["episode", "step", "x", "y", "route_idx", "route_len", "cmd", "steer_cmd", "teacher_steer",
                        "wheel", "speed", "lat", "junction"])
    try:
        env = CarlaDriveEnv(town=args.town, render=True, seed=rng.randrange(1 << 30))
        if args.steady:                                        # pure pursuit on straights, PPO in bends
            from teacher_steady import SteadyTeacher
            teacher = SteadyTeacher(teacher, env)
        while True:
            if save and args.episodes is None and frames_on_disk(root, args.town) >= args.target:
                print(f"target reached: {frames_on_disk(root, args.town)} frames", flush=True)
                return 0
            if args.episodes is not None and len(results) >= args.episodes:
                break
            obs, _ = env.reset()
            names, weights = weather_table(args.weather_set)
            weather = rng.choices(names, weights)[0]
            set_weather(env, weather)
            cam = camera_params(rng, jitter=save and not args.no_jitter)
            q = attach_camera(env, cam)
            env.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            env._tick()                                        # a camera delivers only after a tick
            obs, _ = env._observe()
            push = Pusher(rng, args.noise_steer, args.noise_steps) if (args.driver == "teacher" and save and not args.no_noise) else None
            takeover = False                                   # mix: teacher holds the wheel while far off-centre
            if student is not None:
                if getattr(student, "needs_pose", False):
                    student.reset(cam)                         # top-down student: this episode's (jittered) camera
                else:
                    student.reset()                            # turn-distance tracker
            name = f"ep_{time.strftime('%Y%m%d_%H%M%S')}_{rng.randrange(10**6):06d}"
            tmp = os.path.join(root, name + ".tmp")
            if save:
                os.makedirs(tmp)
            rows, n, info = [], 0, {}
            for step in range(C.MAX_EPISODE_STEPS + 10):
                try:
                    img = image_at(q, env.world.get_snapshot().frame)
                except RuntimeError as e:
                    env._fail(str(e))                          # -> CarlaFailure -> supervisor restarts
                label = teacher.act(obs)                       # what the teacher would do now
                cmd = env.command()
                wheel, speed = float(env.prev_action[0]), float(env.speed)
                kw = {}
                if student is not None and getattr(student, "needs_pose", False):
                    tf0 = env.vehicle.get_transform()       # ego motion for the top-down memory
                    kw = dict(pose=(tf0.location.x, tf0.location.y, tf0.rotation.yaw), route=(env.route, env.idx))
                if args.driver == "teacher":
                    action = label.copy()
                elif args.driver == "student":
                    action = student.act(img, speed, wheel, cmd, env.v_target, **kw)
                else:                                          # mix (DAgger)
                    s_act = student.act(img, speed, wheel, cmd, env.v_target, **kw)    # every step: keeps its tracker in step
                    if args.takeover:
                        lat = abs(float(obs[1]))                   # lateral offset / half-lane
                        takeover = lat > args.takeover or (takeover and lat > args.takeover / 3)
                    action = label.copy() if (takeover or rng.random() <= args.beta) else s_act
                noise = push() if push else 0.0
                action[0] = float(np.clip(action[0] + noise, -1, 1))
                if trace is not None:
                    loc = env.vehicle.get_location()
                    wp = env.world.get_map().get_waypoint(loc)
                    trace.writerow([len(results) + 1, step, f"{loc.x:.1f}", f"{loc.y:.1f}", env.idx, len(env.route),
                                    cmd, f"{action[0]:.3f}", f"{label[0]:.3f}", f"{wheel:.3f}", f"{speed:.2f}",
                                    f"{obs[1]:.3f}", int(wp.is_junction)])
                if save and step >= C.VISION_SKIP_STEPS:
                    Image.fromarray(img).save(os.path.join(tmp, f"{n:06d}.jpg"), quality=90)
                    if getattr(student, "last_cls", None) is not None:   # top-down student segmented it already
                        os.makedirs(os.path.join(tmp, "seg"), exist_ok=True)
                        Image.fromarray(student.last_cls).save(os.path.join(tmp, "seg", f"{n:06d}.png"))
                    tf = env.vehicle.get_transform()
                    rows.append([n, f"{label[0]:.4f}", f"{label[1]:.4f}", f"{wheel:.4f}", f"{speed:.3f}",
                                 cmd, int(noise != 0.0), f"{obs[1]:.3f}", f"{obs[2]:.3f}",   # lat/half-lane, heading/pi
                                 f"{env.v_target:.3f}",                                      # cruise speed m/s
                                 f"{tf.location.x:.2f}", f"{tf.location.y:.2f}", f"{tf.location.z:.2f}",
                                 f"{tf.rotation.yaw:.2f}", env.idx, opt(env.lead_gap), f"{env.lead_speed:.2f}",
                                 opt(env.red_dist), opt(env.yield_dist)])
                    n += 1
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            results.append(info)
            print(f"episode {len(results)}: {info.get('reason', ''):<10} route {info.get('route_completion', 0):5.0%} "
                  f"of {info.get('route_m', 0):.0f} m | {weather:<13} | frames {n} | {smooth_line(info)}"
                  + (f" | hit {getattr(env, 'collided_with', '')}" if info.get("reason") == "collision" else ""), flush=True)
            if save and info.get("reason") in ("collision", "off_route") and args.driver == "teacher":
                cut = min(len(rows), C.TRIM_FAILED_STEPS)      # the teacher's own mistake: don't teach it
                for r in rows[len(rows) - cut:]:
                    os.remove(os.path.join(tmp, f"{r[0]:06d}.jpg"))
                rows, n = rows[:len(rows) - cut], n - cut
            if save:
                with open(os.path.join(tmp, "labels.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(LABEL_COLS)
                    w.writerows(rows)
                meta = dict(frames=n, town=args.town, weather=weather, camera=cam, driver=args.driver, steady=args.steady,
                            beta=args.beta, teacher=args.teacher, student=args.student,
                            end=info.get("reason"), route_completion=info.get("route_completion"),
                            route_m=info.get("route_m"))
                with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=1)
                np.savez_compressed(os.path.join(tmp, "route.npz"), route=np.asarray(env.route, np.float32),
                                    opt=np.asarray(env.route_opt), junction=np.asarray(env.route_junction, bool))
                os.replace(tmp, os.path.join(root, name))
                print(f"  total on disk: {frames_on_disk(root, args.town)} / {args.target} frames", flush=True)
    except CarlaFailure as e:
        print("CARLA failure:", e, flush=True)
        return C.EXIT_RESTART_CARLA
    finally:
        if trace_f:
            trace_f.close()
        if env:
            try:
                env.close()
            except Exception:  # noqa: BLE001 - dead server
                pass
    summarise(results)
    return 0


def summarise(results):
    if not results:
        return
    from collections import Counter
    from evaluate import SMOOTH_KEYS, smooth_line
    print("\nsummary:", dict(Counter(r.get("reason") for r in results)))
    print(f"mean route completion {np.mean([r['route_completion'] for r in results]):.0%}, "
          f"mean |lat| {np.mean([r['mean_lat_abs'] for r in results]):.2f} m")
    avg = {k: float(np.mean([r[k] for r in results])) for k in SMOOTH_KEYS}
    avg["max_lat_acc"] = float(np.max([r["max_lat_acc"] for r in results]))
    print("smoothness (lower = smoother):", smooth_line(avg))


# ----------------------------------------------------------------------------------- supervisor
def supervise(args):
    """Run worker sessions until the frame target is on disk; restart CARLA on hangs/crashes."""
    from run_training import kill_tree, read_heartbeat
    srv.log_event(f"vision collection start: {frames_on_disk(args.out, args.town)} / {args.target} frames on disk")
    restarts = 0
    passthru = [a for a in sys.argv[1:]]
    srv.kill()                                # never reuse a Low-quality training server
    time.sleep(3)
    while frames_on_disk(args.out, args.town) < args.target:
        if not srv.ensure_running():
            srv.restart("CARLA did not start")
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--worker"] + passthru, cwd=HERE)
        started = time.time()
        while p.poll() is None:
            time.sleep(15)
            hb = read_heartbeat()
            hb_time = hb["time"] if hb and hb.get("pid") == p.pid else started
            if time.time() - hb_time > C.STALE_S:
                srv.log_event("vision collection FROZEN - killing worker and CARLA")
                kill_tree(p.pid)
                srv.kill()
                p.wait()
                break
        if p.returncode == 0 and frames_on_disk(args.out, args.town) >= args.target:
            break
        restarts += 1
        if restarts > C.MAX_RESTARTS:
            srv.log_event(f"vision collection: giving up after {restarts - 1} restarts")
            return 1
        srv.restart(f"vision collection worker ended (code {p.returncode}) - restart #{restarts}")
        time.sleep(5)
    srv.log_event(f"vision collection finished: {frames_on_disk(args.out, args.town)} frames")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", choices=("teacher", "student", "mix"), default="teacher")
    ap.add_argument("--student", default=None, help="student .pt (driver student / mix)")
    ap.add_argument("--beta", type=float, default=0.5, help="mix: chance the teacher drives a step")
    ap.add_argument("--teacher", default=C.VISION_TEACHER)
    ap.add_argument("--town", default=C.VISION_TOWN)
    ap.add_argument("--episodes", type=int, default=None, help="stop after N episodes (default: frame target)")
    ap.add_argument("--target", type=int, default=C.VISION_TARGET_FRAMES)
    ap.add_argument("--out", default=C.VISION_DATA_DIR)
    ap.add_argument("--no-save", action="store_true", help="drive and score only")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--v2", action="store_true", help="Phase C: real-car camera (TESLA_CAM_V2) at 640x480")
    ap.add_argument("--noise-steer", type=lambda v: tuple(float(x) for x in v.split(",")), default=None,
                    help="teacher push size range, e.g. 0.1,0.25 (default config NOISE_STEER)")
    ap.add_argument("--noise-steps", type=lambda v: tuple(int(x) for x in v.split(",")), default=None,
                    help="teacher push length range in steps, e.g. 6,15 (default config NOISE_STEPS)")
    ap.add_argument("--takeover", type=float, default=0.0,
                    help="mix: teacher drives while |lateral offset| > this many half-lanes (until back under 1/3 of it)")
    ap.add_argument("--no-jitter", action="store_true", help="exact Tesla camera, no randomisation")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gps-offset", type=lambda v: tuple(float(x) for x in v.split(",")), default=(0.0, 0.0),
                    help="route-curve student: fixed GPS error 'sideways,along' in m (+ = curve drawn right / closer)")
    ap.add_argument("--weather-set", choices=("all", "day", "extra"), default="all",
                    help="day = noon/sunset only (v1 conditions), extra = night/fog/dawn/heavy rain only")
    ap.add_argument("--steady", action="store_true",
                    help="steady-hand teacher: smooth pure-pursuit steering on straight road (teacher_steady.py)")
    ap.add_argument("--trace", default=None, help="write a per-step CSV (position, command, steering, junction)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.driver != "teacher" and not args.student:
        raise SystemExit("--student <path to student.pt> is needed for driver student / mix")

    if args.worker:
        sys.exit(worker(args))
    try:
        if args.no_save or args.episodes is not None:      # short job: one session, no supervisor
            srv.kill()                                     # fresh rendering server
            time.sleep(3)
            if not srv.start():
                raise SystemExit("CARLA did not start")
            code = worker(args)
        else:
            code = supervise(args)
    finally:
        srv.shutdown_if_configured()
    sys.exit(code)


if __name__ == "__main__":
    main()
