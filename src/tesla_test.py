"""Open-loop test of a vision student on REAL TeslaCam footage (sim-to-real check).

The real car was driven by a human; we can't let the student steer it. So per frame (10 Hz) we feed
the student the real image + real speed + real wheel angle, and compare what it WOULD steer with
what the car actually did. Compared as path curvature (1/m, + = right), so steering ratios don't matter:
  actual    = yaw rate / speed, from the Tesla's own heading telemetry
  predicted = tan(steer x CARLA max wheel angle) / wheelbase  (CARLA Model 3 geometry)
Nav command: "oracle" = left/right if the car really turns > 30 deg within the next 20 m, else follow
(like a sat-nav prompt); "follow" = never told about turns (pure lane following).

    python tesla_test.py --student <best.pt>

Outputs in data\\tesla_test (no GPS in any of them; the overlay video IS real footage):
  <name>.csv  per-step values   <name>.png  plot   <name>.mp4  footage with actual vs predicted steering
"""
import argparse
import csv
import io
import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as C
from vision_model import StudentDriver

CLIP = C.TESLA_CLIP
TEL = os.path.join(C.DATA_ROOT, "tesla_cam", "telemetry.csv")
OUT_DIR = os.path.join(C.DATA_ROOT, "tesla_test")
CALIB = os.path.join(OUT_DIR, "carla_steer_geometry.json")   # written by tesla_calib.py (CARLA's real numbers)
HZ = 10
CMD_FOLLOW, CMD_LEFT, CMD_RIGHT = 0, 1, 2


def geometry():
    if os.path.exists(CALIB):
        with open(CALIB, encoding="utf-8") as f:
            g = json.load(f)
        return g["max_steer_deg"], g["wheelbase_m"], "CARLA physics"
    return 70.0, 2.875, "default guess (run tesla_calib.py)"


def read_frames(clip, w, h):
    ff = shutil.which("ffmpeg")
    raw = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", clip, "-vf", f"fps={HZ},scale={w}:{h}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def telemetry(path, n_steps, src_fps):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    speed = np.array([float(r["speed_mps"]) for r in rows])
    head = np.unwrap(np.radians([float(r["heading_deg"]) for r in rows]))
    wheel_deg = np.array([float(r["steering_wheel_deg"]) for r in rows])
    t = np.arange(len(rows)) / src_fps
    # yaw rate over a 0.5 s centred window, smoothed (heading is GPS/IMU fused, a bit noisy)
    k = int(src_fps * 0.25)
    yaw = (np.roll(head, -k) - np.roll(head, k)) / (2 * k / src_fps)
    yaw[:k], yaw[-k:] = yaw[k], yaw[-k - 1]
    curv = yaw / np.maximum(speed, 1.0)
    dist = np.concatenate([[0], np.cumsum(speed[1:] / src_fps)])
    ts = np.arange(n_steps) / HZ
    idx = np.clip(np.round(ts * src_fps).astype(int), 0, len(rows) - 1)
    # oracle command: heading change over the next 20 m of driving
    cmd = np.zeros(n_steps, int)
    for i, j in enumerate(idx):
        ahead = np.searchsorted(dist, dist[j] + C.CMD_LOOKAHEAD_M)
        ahead = min(ahead, len(rows) - 1)
        d = np.degrees(head[ahead] - head[j])
        cmd[i] = CMD_RIGHT if d > 30 else CMD_LEFT if d < -30 else CMD_FOLLOW
    return dict(t=ts, speed=speed[idx], curv=curv[idx], wheel_deg=wheel_deg[idx], cmd=cmd)


def jpeg(rgb):
    """Same compression the training frames went through."""
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, "JPEG", quality=90)
    return np.asarray(Image.open(b).convert("RGB"))


def stats(a, p, mask=None):
    if mask is not None:
        a, p = a[mask], p[mask]
    if len(a) < 3:
        return dict(n=int(len(a)))
    big = np.abs(a) > 0.01
    return dict(n=int(len(a)), corr=float(np.corrcoef(a, p)[0, 1]), mae=float(np.mean(np.abs(a - p))),
                sign_agree=float(np.mean(np.sign(a[big]) == np.sign(p[big]))) if big.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--clip", default=CLIP)
    ap.add_argument("--tel", default=TEL)
    ap.add_argument("--name", default=None, help="output name (default: student folder name)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    name = args.name or os.path.basename(os.path.dirname(args.student))
    max_deg, wb, geo_src = geometry()
    max_rad = np.radians(max_deg)

    frames = read_frames(args.clip, *C.VISION_IMG)
    ffp = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", args.clip], capture_output=True, text=True).stdout
    num, den = ffp.strip().split("/")
    tel = telemetry(args.tel, len(frames), float(num) / float(den))
    # the car's real wheel angle in CARLA units (the student's "current wheel" input)
    wheel_in = np.clip(np.arctan(tel["curv"] * wb) / max_rad, -1, 1)

    drv = StudentDriver(args.student)
    pred = {}
    # oracle / follow: student gets the real wheel angle (as in CARLA). camera_only: wheel input 0,
    # so everything must come from the image (the honest sim-to-real test: no copying the wheel).
    # mirrored: Australia drives on the LEFT, CARLA on the right. Flip the image (-> right-hand traffic),
    # swap left/right prompts, negate the wheel in and the steering out.
    swap = {CMD_FOLLOW: CMD_FOLLOW, CMD_LEFT: CMD_RIGHT, CMD_RIGHT: CMD_LEFT, 3: 3}
    for mode in ("oracle", "follow", "camera_only", "mirrored"):
        steer, speed_cmd = [], []
        drv.reset()                                    # turn-distance tracker (same dead reckoning as CARLA)
        for i, fr in enumerate(frames):
            cmd = CMD_FOLLOW if mode == "follow" else int(tel["cmd"][i])
            wheel = 0.0 if mode == "camera_only" else float(wheel_in[i])
            img = jpeg(fr)
            if mode == "mirrored":
                a = drv.act(np.ascontiguousarray(img[:, ::-1]), float(tel["speed"][i]), -wheel, swap[cmd])
                a[0] = -a[0]
            else:
                a = drv.act(img, float(tel["speed"][i]), wheel, cmd)
            steer.append(a[0]); speed_cmd.append(a[1])
        steer = np.array(steer)
        # the car turns the wheel only STEER_SMOOTH of the way to the command each step (as in CARLA):
        # filter the student's OWN commands (no real-wheel data mixed in)
        sm = np.zeros_like(steer)
        prev = float(np.clip(np.arctan(tel["curv"][0] * wb) / max_rad, -1, 1))
        for i, s in enumerate(steer):
            prev = prev + C.STEER_SMOOTH * (s - prev)
            sm[i] = prev
        pred[mode] = dict(steer=steer, speed_cmd=np.array(speed_cmd), curv=np.tan(sm * max_rad) / wb,
                          curv_raw=np.tan(steer * max_rad) / wb)

    # target: what the car does LEAD steps later (wheel smoothing already modelled; tyres/yaw lag ~0.2 s)
    LEAD = 2
    a_now = tel["curv"]
    a = np.concatenate([a_now[LEAD:], np.repeat(a_now[-1], LEAD)])
    turn = tel["cmd"] != CMD_FOLLOW
    report = dict(student=args.student, clip=os.path.basename(args.clip), steps=len(frames), hz=HZ,
                  target=f"real curvature {LEAD / HZ:.1f} s later",
                  geometry=dict(max_steer_deg=max_deg, wheelbase_m=wb, source=geo_src),
                  wheel_vs_yaw_corr=float(np.corrcoef(tel["wheel_deg"], a_now)[0, 1]),
                  turn_steps=int(turn.sum()))
    for mode, p in pred.items():
        report[mode] = dict(all=stats(a, p["curv"]), straight=stats(a, p["curv"], ~turn),
                            turns=stats(a, p["curv"], turn),
                            straight_bias=float(np.mean(p["curv"][~turn] - a[~turn])),   # + = leans right
                            jitter=float(np.mean(np.abs(np.diff(p["curv"])))))
    # baselines: "always straight" and "keep the current wheel" (a student that just copies its wheel input)
    report["baseline_zero"] = dict(all=dict(mae=float(np.mean(np.abs(a)))))
    report["baseline_hold_wheel"] = dict(all=stats(a, a_now), turns=stats(a, a_now, turn))

    with open(os.path.join(OUT_DIR, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "speed_mps", "cmd_oracle", "actual_curv_now", "actual_curv_target", "pred_curv_oracle",
                    "pred_curv_follow", "pred_curv_camera_only", "student_steer_oracle", "actual_wheel_carla_units"])
        for i in range(len(frames)):
            w.writerow([f"{tel['t'][i]:.1f}", f"{tel['speed'][i]:.2f}", int(tel["cmd"][i]), f"{a_now[i]:.4f}",
                        f"{a[i]:.4f}", f"{pred['oracle']['curv'][i]:.4f}", f"{pred['follow']['curv'][i]:.4f}",
                        f"{pred['camera_only']['curv'][i]:.4f}", f"{pred['oracle']['steer'][i]:.3f}",
                        f"{wheel_in[i]:.3f}"])
    with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(tel["t"], a, "k", lw=2, label=f"real car, {LEAD / HZ:.1f} s later")
    ax[0].plot(tel["t"], pred["oracle"]["curv"], "tab:blue", label="student (camera + wheel + turn prompt)")
    ax[0].plot(tel["t"], pred["camera_only"]["curv"], "tab:red", alpha=.6, label="student, camera only")
    ax[0].plot(tel["t"], pred["mirrored"]["curv"], "tab:green", label="student, mirrored (left-hand traffic fix)")
    ax[0].set_ylabel("path curvature 1/m  (+ right)")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[0].grid(alpha=.3)
    ax[0].set_title(f"{name} on real TeslaCam footage: corr {report['oracle']['all'].get('corr', 0):.2f}, "
                    f"camera only {report['camera_only']['all'].get('corr', 0):.2f}, "
                    f"mirrored {report['mirrored']['all'].get('corr', 0):.2f}")
    ax[1].plot(tel["t"], tel["speed"] * 3.6, "k")
    ax[1].set_ylabel("km/h")
    ax[1].set_xlabel("s")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=110)

    if not args.no_video:
        overlay_video(args.clip, os.path.join(OUT_DIR, name + ".mp4"), tel, a, pred["oracle"]["curv"], name)
    print(json.dumps(report, indent=1))
    print("out:", os.path.join(OUT_DIR, name), "(.csv .json .png" + ("" if args.no_video else " .mp4") + ")")


def overlay_video(clip, out, tel, actual, predicted, name):
    W, H = 960, 720
    frames = read_frames(clip, W, H)
    ff = shutil.which("ffmpeg")
    enc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    try:
        big = ImageFont.truetype("segoeuib.ttf", 28)
        small = ImageFont.truetype("segoeui.ttf", 22)
    except OSError:
        big = small = ImageFont.load_default()
    scale = 0.12                                      # curvature at full bar width (~8 m radius)
    names = {0: "follow", 1: "turn left", 2: "turn right"}
    for i, fr in enumerate(frames[:len(actual)]):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle((0, H - 175, W, H), fill=(0, 0, 0, 150))
        d.text((20, H - 170), f"{name}  |  real Tesla footage  |  {tel['speed'][i] * 3.6:.0f} km/h  |  "
               f"command: {names[int(tel['cmd'][i])]}", font=small, fill=(255, 255, 255))
        cx, bw = W / 2, W / 2 - 60
        for row, (lab, v, col) in enumerate((("real car", actual[i], (255, 255, 255)),
                                             ("student", predicted[i], (80, 200, 255)))):
            y = H - 100 + row * 45
            d.rectangle((60, y, W - 60, y + 18), fill=(255, 255, 255, 40))
            x = cx + np.clip(v / scale, -1, 1) * bw
            d.rectangle((min(cx, x), y, max(cx, x), y + 18), fill=col)
            d.line((cx, y - 4, cx, y + 22), fill=(255, 255, 255), width=2)
            d.text((62, y - 26), lab, font=small, fill=col)
        d.text((W - 210, 15), f"{tel['t'][i]:5.1f} s", font=big, fill=(255, 255, 255))
        enc.stdin.write(np.asarray(im).tobytes())
    enc.stdin.close()
    enc.wait()


if __name__ == "__main__":
    main()
