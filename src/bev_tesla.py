"""Phase B on real footage: top-down view from the Tesla clip.
1) segment frames at 6 fps (640x480) once -> cache   2) fit the camera pitch so lane lines on straight stretches
come out parallel   3) lane width check   4) video: camera + segmentation | top-down, earlier frames filling the
blind spots (moved by our own speed + heading from telemetry).
Uses ONLY speed + heading from telemetry (never the GPS columns).  Outputs are PRIVATE (real footage):
data\\seg_phaseA\\tesla_PRIVATE\\
  python bev_tesla.py [clip] [telemetry.csv]"""
import csv
import math
import os
import subprocess
import sys

import cv2
import numpy as np

import config as C
from bev_project import BevProjector, Fuser, UNSEEN, RES, CX, CY, colorize, footprints

CLIP = C.TESLA_CLIP
TEL = os.path.join(C.DATA_ROOT, "tesla_cam", "telemetry.csv")
OUT = os.path.join(C.DATA_ROOT, "seg_phaseA", "tesla_PRIVATE")
FPS = 6
STACK = 6                                    # frames fused (1 s)
LANE_M = 3.3                                 # typical Australian suburban lane width (m)


def frames(clip, w=1280, h=960):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", clip, "-vf", f"fps={FPS}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def segment_all(clip, cache):
    if os.path.isfile(cache):
        return np.load(cache)["cls"]
    from seg_check import Segmenter
    seg = Segmenter()
    cls = np.stack([seg(f, (640, 480)) for f in frames(clip)])
    np.savez_compressed(cache, cls=cls)
    return cls


def telemetry(path, n, clip_s):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    sp = np.array([float(r["speed_mps"]) for r in rows])
    hd = np.unwrap(np.radians([float(r["heading_deg"]) for r in rows]))
    st = np.array([float(r["steering_wheel_deg"]) for r in rows])
    t = np.linspace(0, clip_s, len(rows))
    tf = np.arange(n) / FPS
    return np.interp(tf, t, sp), np.degrees(np.interp(tf, t, hd)), np.interp(tf, t, st)


def line_angles(mask):
    m = mask.astype(np.uint8) * 255
    ls = cv2.HoughLinesP(m, 1, np.pi / 180, 25, minLineLength=40, maxLineGap=15)
    out = []
    for x1, y1, x2, y2 in ([] if ls is None else ls.reshape(-1, 4)):
        a = math.degrees(math.atan2(x2 - x1, -(y2 - y1)))
        a = (a + 90) % 180 - 90                          # 0 = straight ahead
        if abs(a) < 30:
            out.append((a, math.hypot(x2 - x1, y2 - y1), (x1 + x2) / 2 - CX))
    return out


def spread(cls_list, cam, pitch):
    proj = BevProjector(dict(cam, pitch=pitch))
    s = []
    for c in cls_list:
        al = line_angles(proj.classes(c) == 2)
        left = [a for a, w, x in al if x < -5]
        right = [a for a, w, x in al if x > 5]
        if left and right:
            s.append(np.median(right) - np.median(left))     # >0: lines open up away from us, <0: they close in
    return float(np.median(s)) if s else float("nan"), len(s)


def lane_width(cls_list, proj):
    ws = []
    for c in cls_list:
        top = proj.classes(c) == 2
        band = top[int(CY - 15 / RES):int(CY - 6 / RES)]                  # 6-15 m ahead
        cols = np.where(band.mean(0) > 0.25)[0]
        if len(cols) < 2:
            continue
        groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
        cent = sorted(float(g.mean()) for g in groups)
        ws += [(b - a) * RES for a, b in zip(cent, cent[1:]) if 2.4 < (b - a) * RES < 4.8]
    return (float(np.median(ws)), len(ws)) if ws else (float("nan"), 0)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else CLIP
    tel = sys.argv[2] if len(sys.argv) > 2 else TEL
    os.makedirs(OUT, exist_ok=True)
    cls = segment_all(clip, os.path.join(OUT, "tesla_cls_640.npz"))
    n = len(cls)
    sp, hd, st = telemetry(tel, n, n / FPS)
    straight = [i for i in range(n) if abs(st[i]) < 4 and abs(hd[min(i + 3, n - 1)] - hd[max(i - 3, 0)]) < 2 and sp[i] > 3]
    sample = [cls[i] for i in straight[::2]]
    cam = dict(C.TESLA_CAM)
    print(f"{n} frames, {len(straight)} on straight road; fitting pitch on {len(sample)}", flush=True)
    best = None
    for p in np.arange(-3.0, 5.01, 0.5):
        sd, k = spread(sample, cam, p)
        print(f"  pitch {p:+.1f} deg: lines open/close by {sd:+6.2f} deg  ({k} frames with lines both sides)", flush=True)
        if not math.isnan(sd) and (best is None or abs(sd) < abs(best[1])):
            best = (p, sd)
    pitch = best[0]
    for p in np.arange(pitch - 0.5, pitch + 0.51, 0.1):
        sd, _ = spread(sample, cam, p)
        if not math.isnan(sd) and abs(sd) < abs(best[1]):
            best = (p, sd)
    pitch = round(float(best[0]), 2)
    proj = BevProjector(dict(cam, pitch=pitch))
    lw, k = lane_width(sample, proj)
    print(f"best pitch {pitch:+.2f} deg (config {cam['pitch']:+.1f}); lane width at 6-15 m: {lw:.2f} m from {k} pairs "
          f"(Australian lanes ~3.3-3.5 m)", flush=True)
    z = round(cam["z"] * LANE_M / lw, 2) if k else cam["z"]         # height scales every distance alike
    proj = BevProjector(dict(cam, pitch=pitch, z=z))
    lw2, _ = lane_width(sample, proj)
    print(f"camera height fitted to a {LANE_M} m lane: {z:.2f} m (config {cam['z']:.2f}); lane width now {lw2:.2f} m",
          flush=True)
    import json
    json.dump(dict(cam, pitch=pitch, z=z, source="bev_tesla.py fit: parallel lines + 3.3 m lane"),
              open(os.path.join(OUT, "tesla_cam_fit.json"), "w"), indent=1)

    vid = os.path.join(OUT, "tesla_topdown_PRIVATE.mp4")
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1066x500",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", vid],
                           stdin=subprocess.PIPE)
    from seg_check import COLORS
    camfit = dict(cam, pitch=pitch, z=z)
    fuser = Fuser()
    for i, rgb in enumerate(frames(clip)):
        if i >= n:
            break
        top = proj.classes(cls[i])
        veh = footprints(camfit, cls[i])
        d = sp[i] / FPS if i else 0.0                    # how far we moved since the last frame
        dy = hd[i] - hd[i - 1] if i else 0.0             # + = turned right (clockwise)
        fused = fuser.step(top, veh, d * math.cos(math.radians(dy / 2)), d * math.sin(math.radians(dy / 2)), dy)
        c = cv2.resize(cls[i], (1280, 960), interpolation=cv2.INTER_NEAREST)
        ov = np.where((c > 0)[..., None], (rgb * 0.45 + COLORS[c] * 0.55).astype(np.uint8), rgb)
        left = cv2.resize(ov[:, :, ::-1], (666, 500))
        right = colorize(fused)
        cv2.putText(right, f"{sp[i] * 3.6:4.0f} km/h", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        enc.stdin.write(np.ascontiguousarray(np.concatenate([left, right], 1)).tobytes())
        if i % 60 == 30:
            cv2.imwrite(os.path.join(OUT, f"topdown_{i // FPS:03d}s.jpg"), np.concatenate([left, right], 1))
    enc.stdin.close()
    enc.wait()
    print("video (PRIVATE, real footage):", vid)


if __name__ == "__main__":
    main()
