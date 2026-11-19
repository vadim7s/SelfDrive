"""Record a smooth 30 s milestone clip of a trained model driving (for YouTube).

Why not evaluate.py: its view jumps at 10 Hz and runs at whatever speed the PC manages.
Here a chase camera rides on CARLA's spring arm (natural lag), EVERY simulation frame is
saved, and the video is written at exactly the sim rate (20 fps) -> real-time, smooth playback.
A picture-in-picture shows the Tesla-matched windscreen camera (what the vision model will see).

    python record_clip.py --model <zip> --label "Teacher v3"

Defaults: latest run's final model, Town02, 30 s, output in data\\milestone_clips.
It restarts CARLA at Epic quality for the recording and shuts it down at the end.
Refuses to run while training is active (it would kill the training server).
"""
import argparse
import json
import os
import pickle
import queue
import re
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import carla
from stable_baselines3 import PPO

import config as C
import carla_server as srv
from train import latest_checkpoint, latest_run, vecnorm_for

CLIPS_DIR = os.path.join(C.DATA_ROOT, "milestone_clips")
FPS = round(1.0 / C.FIXED_DELTA)                 # 20: one video frame per simulation frame
RED_KEEP = int(0.5 * FPS)                        # route clips: keep the first 0.5 s of a red-light wait, cut the rest
W, H = 1920, 1080                                # chase camera = the video frame
PIP_W, PIP_H = 576, 432                          # windscreen camera inset (4:3 like the Tesla)
MIN_ROUTE_M = 320.0                              # enough road for 30 s without a cut
SKIP_FRAMES = 10                                 # drop the first frames after each spawn


def training_active():
    try:
        with open(C.HEARTBEAT_FILE, encoding="utf-8") as f:
            hb = json.load(f)
    except (OSError, ValueError):
        return False
    if time.time() - hb.get("time", 0) >= 120:
        return False
    pid = hb.get("pid")                           # eval also writes heartbeats: only a LIVE writer counts
    if not pid:
        return True
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return str(pid) in r.stdout


def start_recording_server():
    """Fresh CARLA at Epic quality (training uses Low)."""
    C.CARLA_ARGS = ["-RenderOffScreen", "-quality-level=Epic"]
    C.TICK_TIMEOUT_S, C.TIMEOUT_S = 60.0, 60.0    # first rendered frames compile shaders: slow
    srv.kill()
    time.sleep(3)
    if not srv.start():
        raise SystemExit("CARLA did not start")


def attach_cameras(env, student=False):
    """Spawn chase + windscreen cameras on the current vehicle. Returns their image queues."""
    bp_lib = env.world.get_blueprint_library()

    def cam(w, h, fov, tf, attach):
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(w))
        bp.set_attribute("image_size_y", str(h))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.0")    # every simulation frame
        actor = env.world.spawn_actor(bp, tf, attach_to=env.vehicle, attachment_type=attach)
        q = queue.Queue()
        actor.listen(q.put)
        env.sensors.append(actor)                 # env destroys it on the next reset / close
        return q

    chase_tf = carla.Transform(carla.Location(x=-6.5, z=2.6), carla.Rotation(pitch=3.0))   # was -12, then -7: more road ahead
    t = C.TESLA_CAM
    tesla_tf = carla.Transform(carla.Location(x=t["x"], y=t["y"], z=t["z"]), carla.Rotation(pitch=t["pitch"]))
    q_chase = cam(W, H, 90, chase_tf, carla.AttachmentType.SpringArmGhost)
    q_tesla = cam(t["width"], t["height"], t["fov"], tesla_tf, carla.AttachmentType.Rigid)
    q_small = cam(*C.VISION_IMG, t["fov"], tesla_tf, carla.AttachmentType.Rigid) if student else None
    return q_chase, q_tesla, q_small    # q_small: exactly what the vision student was trained on


def to_rgb(img):
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, 2::-1]                         # BGRA -> RGB


def collect(q, buf, upto, timeout=30.0):
    """Move images from q into buf until frame `upto` has arrived."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            img = q.get(timeout=0.5)
        except queue.Empty:
            continue
        buf[img.frame] = img
        if img.frame >= upto:
            return
    raise RuntimeError(f"camera frame {upto} did not arrive")


def load_font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class Hud:
    def __init__(self, label, town, subtitle="RL teacher (privileged state)"):
        self.label, self.town, self.subtitle = label, town, subtitle
        self.big, self.mid, self.small = load_font(40), load_font(30), load_font(26)

    def draw(self, chase, tesla, speed_kmh, steer, t_s, topdown=None, narration=""):
        frame = Image.fromarray(np.ascontiguousarray(chase))
        d = ImageDraw.Draw(frame, "RGBA")
        if topdown is not None:                   # the top-down student's own map (what it sees and remembers)
            th, tw = topdown.shape[:2]
            tx, ty = 30, H - 150 - th
            cap = "bird's-eye view  (blue = satnav route)"
            d.rectangle((tx - 4, ty - 40, tx + max(tw, d.textlength(cap, font=self.small) + 12) + 4, ty), fill=(0, 0, 0, 170))
            d.rectangle((tx - 4, ty, tx + tw + 4, ty + th + 4), fill=(0, 0, 0, 170))
            d.text((tx + 6, ty - 36), cap, font=self.small, fill=(255, 255, 255))
            frame.paste(Image.fromarray(topdown), (tx, ty))
        if narration:                             # what the student thinks it is doing
            f = self.mid
            tw_ = d.textlength(narration, font=f)
            nx = 470 + (W - PIP_W - 30 - 20 - 470 - tw_) / 2     # centred in the gap between speed box and windscreen inset
            d.rounded_rectangle((nx - 20, H - 104, nx + tw_ + 20, H - 50), radius=12, fill=(0, 0, 0, 160))
            d.text((nx, H - 98), narration, font=f, fill=(255, 235, 140))
        # title + town
        d.rounded_rectangle((30, 28, 30 + 40 + d.textlength(self.label, font=self.big), 128),
                            radius=14, fill=(0, 0, 0, 140))
        d.text((50, 36), self.label, font=self.big, fill=(255, 255, 255))
        d.text((50, 88), f"{self.town}  |  {self.subtitle}", font=self.small,
               fill=(200, 200, 200))
        # speed + steering bar
        x0, y0 = 30, H - 130
        d.rounded_rectangle((x0, y0, x0 + 420, y0 + 100), radius=14, fill=(0, 0, 0, 140))
        d.text((x0 + 20, y0 + 10), f"{speed_kmh:4.0f} km/h", font=self.big, fill=(255, 255, 255))
        bx, by, bw = x0 + 20, y0 + 70, 380
        d.rectangle((bx, by, bx + bw, by + 12), fill=(255, 255, 255, 60))
        cx = bx + bw / 2
        sx = cx + steer * bw / 2
        d.rectangle((min(cx, sx), by, max(cx, sx), by + 12), fill=(80, 200, 255))
        d.line((cx, by - 4, cx, by + 16), fill=(255, 255, 255), width=2)
        d.text((x0 + 250, y0 + 22), "steering", font=self.small, fill=(200, 200, 200))
        # windscreen camera inset
        inset = Image.fromarray(np.ascontiguousarray(tesla)).resize((PIP_W, PIP_H), Image.BILINEAR)
        px, py = W - PIP_W - 30, H - PIP_H - 30
        d.rectangle((px - 4, py - 40, px + PIP_W + 4, py + PIP_H + 4), fill=(0, 0, 0, 170))
        d.text((px + 6, py - 36), "windscreen camera (Tesla-matched)", font=self.small, fill=(255, 255, 255))
        frame.paste(inset, (px, py))
        d.text((W - 150, 36), f"{t_s:4.1f} s", font=self.small, fill=(255, 255, 255))
        return np.asarray(frame)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()



def mid_turn(env):
    """True while the ego is inside a junction on its route (or within 3 points of entering one): never cut a clip there."""
    j = getattr(env, "route_junction", None)
    if j is None or not len(j):
        return False
    i = min(env.idx, len(j) - 1)
    return bool(any(j[i:i + 4]))


def red_wait(env):
    """True while the ego is stopped for a red light, directly or queued behind a car waiting at one."""
    if float(env.speed) > 0.5:
        return False
    red = getattr(env, "red_dist", None)
    gap = getattr(env, "lead_gap", None)
    return red is not None and (red < 15.0 or (gap is not None and gap < 10.0))   # red < 0 = stopped at the line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model .zip (default: latest run's newest save)")
    ap.add_argument("--student", default=None, help="vision student .pt: it drives instead of the teacher")
    ap.add_argument("--label", default=None, help="title shown in the clip and used in the file name")
    ap.add_argument("--town", default=C.TEST_TOWN)
    ap.add_argument("--steady", action="store_true", help="teacher with the steady hand on straights (teacher_steady.py)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--routes", type=int, default=0, help="record exactly N routes (each up to --route-seconds) instead of --seconds")
    ap.add_argument("--route-seconds", type=float, default=15.0)
    ap.add_argument("--prefer-turns", action="store_true", help="pick routes with a junction turn early on")
    ap.add_argument("--out", default=None, help="output .mp4 (default: milestone_clips folder, stamp + label)")
    args = ap.parse_args()

    if training_active():
        raise SystemExit("training is running (fresh heartbeat) - record after it finishes")
    model_path = args.student or args.model or latest_checkpoint(latest_run())
    label = args.label or os.path.basename(os.path.dirname(model_path))
    os.makedirs(CLIPS_DIR, exist_ok=True)
    out = args.out or os.path.join(CLIPS_DIR, f"{time.strftime('%Y%m%d_%H%M')}_{slug(label)}.mp4")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg not found on PATH")

    start_recording_server()
    from carla_env import CarlaDriveEnv            # after the timeout overrides above
    env, enc = None, None
    try:
        env = CarlaDriveEnv(town=args.town, render=True, seed=args.seed)
        bev_student = False
        if args.student:
            import torch
            bev_student = torch.load(args.student, map_location="cpu", weights_only=False).get(
                "settings", {}).get("kind") == "bev"
            if bev_student:                        # top-down student: real-car camera, segmenter, map, narration
                C.TESLA_CAM, C.VISION_IMG = C.TESLA_CAM_V2, C.VISION_IMG_V2
                from bev_model import BevStudentDriver, Narrator, route_in_view, topdown_picture
                student, narrator = BevStudentDriver(args.student), Narrator()
                hud = Hud(label, args.town, "camera to bird's-eye view, with memory")
            else:
                from vision_model import StudentDriver
                student = StudentDriver(args.student)
                hud = Hud(label, args.town, "vision student (camera only)")
        else:
            student = None
            with open(vecnorm_for(model_path), "rb") as f:
                vecnorm = pickle.load(f)           # observation scaling the model was trained with
            model = PPO.load(model_path, device="cpu")
            hud = Hud(label, args.town, ("RL speed + lane tracker steering" if C.STEADY_BENDS == "track" else "RL teacher + steady hand")
                      if args.steady else "RL teacher (privileged state)")
            steady = None
            if args.steady:                        # pure pursuit on straight road, PPO in bends
                from teacher_steady import SteadyTeacher

                class _Ppo:
                    def act(self, o):
                        return model.predict(vecnorm.normalize_obs(o), deterministic=True)[0]
                steady = SteadyTeacher(_Ppo(), env)
        enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                                "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)

        per_route = int(args.route_seconds * FPS) if args.routes else None
        need = args.routes * per_route if args.routes else int(args.seconds * FPS)
        written, cuts, endings, routes_done = 0, 0, [], 0
        seen, red_cut = 0, 0                       # frames rendered incl. cut ones (for the wait cap); seconds cut
        driven = 0                                 # frames that count to the clip length (waits at reds excluded)
        wait_cap = 3 * per_route if per_route else None   # a route never runs past 3x its length in real frames
        while (driven if per_route else written) < need and (not args.routes or routes_done < args.routes):
            for _ in range(30):                    # a route long enough for the clip (with a turn early on, if asked)
                obs, _ = env.reset()
                if len(env.route) * C.ROUTE_RESOLUTION < MIN_ROUTE_M:
                    continue
                if args.prefer_turns:
                    early = int(args.route_seconds * 6.0 / C.ROUTE_RESOLUTION)
                    if not any(o in (1, 2) for o in env.route_opt[5:early]):
                        continue
                break
            route_start, route_driven, held_run = seen, 0, 0
            q_chase, q_tesla, q_small = attach_cameras(env, student=bool(student))
            buf_c, buf_t, last, skip = {}, {}, -1, SKIP_FRAMES
            if student:
                student.reset()                    # turn-distance tracker / map memory
                if bev_student:
                    narrator = Narrator()
                env._tick()                        # cameras deliver only after a tick
                obs, _ = env._observe()
            while (driven if per_route else written) < need and (per_route is None or ((route_driven < per_route or mid_turn(env)) and seen - route_start < wait_cap)):
                if student:
                    small = {}
                    collect(q_small, small, env.world.get_snapshot().frame)
                    kw = {}
                    if bev_student:
                        tf0 = env.vehicle.get_transform()
                        kw = dict(pose=(tf0.location.x, tf0.location.y, tf0.rotation.yaw), route=(env.route, env.idx))
                    act = student.act(np.ascontiguousarray(to_rgb(small[max(small)])), float(env.speed),
                                      float(env.prev_action[0]), env.command(), env.v_target, **kw)
                    if bev_student:
                        path = route_in_view(env.route, env.idx, *kw["pose"])     # satnav route, for viewers
                        td_pic, narr = topdown_picture(student.last[2], path=path), narrator.line(student.last)
                elif steady:
                    act = steady.act(obs)
                else:
                    act, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
                obs, _, term, trunc, info = env.step(act)
                now = env.world.get_snapshot().frame
                held = per_route and red_wait(env)   # stopped for a red light (or queued behind a car at one)
                collect(q_chase, buf_c, now)
                collect(q_tesla, buf_t, now)
                for fr in sorted(set(buf_c) & set(buf_t)):
                    if fr <= last:
                        continue
                    last = fr
                    if skip:
                        skip -= 1
                        continue
                    seen += 1
                    held_run = held_run + 1 if held else 0
                    if held_run > RED_KEEP:          # waiting at a red: cut until the car moves off
                        red_cut += 1
                        if seen - route_start >= wait_cap:
                            break
                        continue
                    img = hud.draw(to_rgb(buf_c[fr]), to_rgb(buf_t[fr]), info["speed_kmh"],
                                   float(env.prev_action[0]), written / FPS,
                                   *((td_pic, narr) if bev_student else ()))
                    enc.stdin.write(img.tobytes())
                    written += 1
                    if not held:
                        driven += 1
                        route_driven += 1
                    if (driven if per_route else written) >= need or (per_route and ((route_driven >= per_route and not mid_turn(env)) or seen - route_start >= wait_cap)):
                        break
                buf_c = {k: v for k, v in buf_c.items() if k > last}
                buf_t = {k: v for k, v in buf_t.items() if k > last}
                if term or trunc:
                    endings.append(info.get("reason", "") + (f" [{env.collided_with}]" if info.get("reason") == "collision" else ""))
                    cuts += 1
                    break
            routes_done += 1
            print(f"recorded {written / FPS:.1f} s ({routes_done} routes, {red_cut / FPS:.0f} s of red-light waits cut)", flush=True)

        enc.stdin.close()
        enc.wait()
        meta = dict(model=model_path, label=label, town=args.town, seconds=args.seconds, fps=FPS,
                    driver="student" if student else ("teacher_steady" if args.steady else "teacher"), cuts=cuts, endings=endings, routes=routes_done, settings=dict(STEADY_BENDS=getattr(C, "STEADY_BENDS", None), STEER_SMOOTH=C.STEER_SMOOTH,
                                                              CURVE_LIMIT=C.CURVE_LIMIT))
        with open(out.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("clip:", out, "| route changes in clip:", cuts, endings)
    finally:
        if enc and enc.poll() is None:
            enc.stdin.close()
            enc.wait()
        if env:
            env.close()
        srv.shutdown_if_configured()


if __name__ == "__main__":
    main()
