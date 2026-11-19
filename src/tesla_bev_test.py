"""Open-loop sim-to-real test of a BIRD'S-EYE-VIEW student (K5+) on REAL TeslaCam footage.

A human drove the car; the student can't steer it.  Per frame (10 Hz) we build exactly what it gets in CARLA:
  real camera frame -> segmenter -> flat-ground projection -> fused top-down map (moved by our own motion)
  + speed + satnav command + route curve, and compare the steering it WOULD give with what the car really did
  (path curvature 1/m, + = right, so steering ratios don't matter).
Honesty: the route curve is taken from where the human actually drove (dead reckoning from speed + heading; no GPS
columns are read), i.e. a perfect satnav.  Mode 'no_route' removes it (map + command only) to show what vision alone
does.  Australia drives on the LEFT, CARLA on the right: 'mirrored' flips the image, the route and the commands
(and negates the answer), which is the fair comparison; 'as_is' is shown for reference.
  python tesla_bev_test.py --student <best.pt> [--name K5] [--no-video]
Outputs (NO GPS in any): data\\tesla_test\\<name>.json/.csv/.png
Video (REAL FOOTAGE, PRIVATE): data\\seg_phaseA\\tesla_PRIVATE\\<name>_tesla_PRIVATE.mp4"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess

import numpy as np
import torch

import config as C
from bev_dataset import BevInput
from bev_model import WINDOW, load_student, scalars, topdown_picture
from bev_route import ROUTE_BIT, route_layer, route_vector

CLIP = C.TESLA_CLIP
TEL = os.path.join(C.DATA_ROOT, "tesla_cam", "telemetry.csv")
OUT_DIR = os.path.join(C.DATA_ROOT, "tesla_test")
PRIV = os.path.join(C.DATA_ROOT, "seg_phaseA", "tesla_PRIVATE")
CALIB = os.path.join(OUT_DIR, "carla_steer_geometry.json")
HZ, W, H = 10, 1280, 960
FOLLOW, LEFT, RIGHT = 0, 1, 2
FUSER, WORM_M = "current", 4.5           # set from --fuser / --worms
LEAD = 2                                   # compare with what the car does 0.2 s later (tyre / yaw lag)


def frames(clip):
    p = subprocess.Popen([shutil.which("ffmpeg"), "-loglevel", "error", "-i", clip, "-vf", f"fps={HZ},scale={W}:{H}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def segment(clip, mirrored):
    cache = os.path.join(PRIV, f"tesla_cls_10hz{'_mirror' if mirrored else ''}.npz")
    if os.path.isfile(cache):
        return np.load(cache)["cls"]
    from seg_check import Segmenter
    seg = Segmenter()
    out = np.stack([seg(np.ascontiguousarray(f[:, ::-1]) if mirrored else f, (640, 480)) for f in frames(clip)])
    np.savez_compressed(cache, cls=out)
    return out


def telemetry(path, n):
    """Speed + heading only (never the GPS columns).  Dead-reckoned pose in a CARLA-like frame: x = north,
    y = east, yaw = compass heading (clockwise), so + yaw change = right turn as in CARLA."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    src_hz = len(rows) / 59.8334 if len(rows) > 2000 else 36.0
    sp = np.array([float(r["speed_mps"]) for r in rows])
    hd = np.unwrap(np.radians([float(r["heading_deg"]) for r in rows]))
    k = int(src_hz * 0.25)
    yaw_rate = (np.roll(hd, -k) - np.roll(hd, k)) / (2 * k / src_hz)
    yaw_rate[:k], yaw_rate[-k:] = yaw_rate[k], yaw_rate[-k - 1]
    curv = yaw_rate / np.maximum(sp, 1.0)
    idx = np.clip(np.round(np.arange(n) / HZ * src_hz).astype(int), 0, len(rows) - 1)
    dt = 1.0 / src_hz
    x = np.concatenate([[0], np.cumsum(sp[1:] * np.cos(hd[1:]) * dt)])
    y = np.concatenate([[0], np.cumsum(sp[1:] * np.sin(hd[1:]) * dt)])
    dist = np.concatenate([[0], np.cumsum(sp[1:] * dt)])
    cmd = np.zeros(n, int)
    for i, j in enumerate(idx):
        a = min(np.searchsorted(dist, dist[j] + C.CMD_LOOKAHEAD_M), len(rows) - 1)
        d = math.degrees(hd[a] - hd[j])
        cmd[i] = RIGHT if d > 30 else LEFT if d < -30 else FOLLOW
    # route = the driven path, resampled every 2 m of travel (route points are 2 m apart in CARLA)
    s = np.arange(0, dist[-1], 2.0)
    route = np.stack([np.interp(s, dist, x), np.interp(s, dist, y)], 1)
    ridx = np.clip((dist[idx] / 2.0).astype(int), 0, len(route) - 1)
    return dict(t=np.arange(n) / HZ, speed=sp[idx], curv=curv[idx], x=x[idx], y=y[idx],
                yaw=np.degrees(hd[idx]), cmd=cmd, route=route, ridx=ridx)


def paint_filter(cls):
    """Real roads carry painted symbols (bicycles, arrows, hatching) that the segmenter calls 'lane marking'; CARLA
    roads hardly have any, so on the map they become phantom lines.  Keep only lane-line-like marking pieces (long and
    thin in the camera view); the rest becomes road."""
    import cv2
    out = cls.copy()
    m = (cls == 2).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if len(xs) < 5:
            out[lab == k] = 1
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        long_, short = max(w, h), max(1.0, min(w, h))
        fill = len(xs) / max(1.0, w * h)
        if not (long_ / short > 5 and long_ > 30 and fill > 0.25):
            out[lab == k] = 1
    return out


def run(net, dev, cls, tel, mirrored, use_route):
    """Same steps as BevStudentDriver.act, on cached class maps."""
    sgn = -1.0 if mirrored else 1.0
    route = tel["route"] * np.array([1.0, sgn])
    swap = {FOLLOW: FOLLOW, LEFT: RIGHT, RIGHT: LEFT}
    if FUSER == "vote":
        from bev_project import VoteFuser
        bev = BevInput(dict(C.TESLA_CAM_V2), VoteFuser(decay=0.95, min_ev=0.3, stick=1.0), WORM_M)
    else:
        bev = BevInput(dict(C.TESLA_CAM_V2), None, WORM_M)
    feats, scal, cmds = [], [], []
    steer, maps, reads = [], [], []
    for i in range(len(cls)):
        x, y, yaw = tel["x"][i], sgn * tel["y"][i], sgn * tel["yaw"][i]
        top = bev.step(cls[i], x, y, yaw)
        cmd = int(tel["cmd"][i])
        cmd = swap[cmd] if mirrored else cmd
        rv = np.zeros(17, np.float32)
        if net.route and use_route:
            rv = route_vector(route, int(tel["ridx"][i]), x, y, yaw)
            top = top + ROUTE_BIT * route_layer(route, int(tel["ridx"][i]), x, y, yaw).astype(np.uint8)
        maps.append(top)
        with torch.no_grad():
            feats.append(net.step_features(torch.from_numpy(top)[None].to(dev))[0])
            sc = scalars(float(tel["speed"][i]), None, cmd) + (rv.tolist() if net.route else [])
            scal.append(torch.tensor(sc, device=dev))
            cmds.append(cmd)
            f, s_, c_ = feats[-WINDOW:], scal[-WINDOW:], cmds[-WINDOW:]
            d, sit, aux = net.forward_feats(torch.stack(f)[None], torch.stack(s_)[None],
                                            torch.tensor([c_], device=dev))
        steer.append(sgn * float(d[0, -1, 0]))
        reads.append((torch.softmax(sit[0, -1].float(), -1).cpu().numpy(), aux[0, -1].float().cpu().numpy(), top))
    run.reads = reads
    return np.array(steer), maps


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
    ap.add_argument("--name", default="bev_student")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--paint-filter", action="store_true", help="drop painted road symbols (keep lane lines)")
    ap.add_argument("--fuser", default=None, choices=["current", "vote"], help="default: what the student trained with")
    ap.add_argument("--worms", action="store_true", help="cars 1 m deep on the map instead of 4.5 m")
    a = ap.parse_args()
    global FUSER, WORM_M
    FUSER = a.fuser or torch.load(a.student, map_location="cpu", weights_only=False).get("settings", {}).get("fuser", "current")
    WORM_M = 1.0 if a.worms else 4.5
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRIV, exist_ok=True)
    g = json.load(open(CALIB, encoding="utf-8"))
    max_rad, wb = math.radians(g["max_steer_deg"]), g["wheelbase_m"]
    dev = torch.device("cuda")
    net = load_student(a.student, dev).eval()

    cls_m = segment(a.clip, True)
    cls_a = segment(a.clip, False)
    n = min(len(cls_m), len(cls_a))
    tel = telemetry(a.tel, n)
    target = np.concatenate([tel["curv"][LEAD:], np.repeat(tel["curv"][-1], LEAD)])
    turn = tel["cmd"] != FOLLOW
    moving = tel["speed"] > 1.0

    def to_curv(st):
        sm, prev = np.zeros_like(st), 0.0
        for i, s in enumerate(st):
            prev = prev + C.STEER_SMOOTH * (s - prev)
            sm[i] = prev
        return np.tan(np.clip(sm, -1, 1) * max_rad) / wb

    if a.paint_filter:
        cls_m = np.stack([paint_filter(c) for c in cls_m])
        cls_a = np.stack([paint_filter(c) for c in cls_a])
    modes = {"mirrored": (cls_m, True, True), "mirrored_no_route": (cls_m, True, False), "as_is": (cls_a, False, True)}
    pred, maps = {}, {}
    for m, (cl, mir, ur) in modes.items():
        st, mp = run(net, dev, cl[:n], tel, mir, ur)
        pred[m] = dict(steer=st, curv=to_curv(st))
        if m == "mirrored":
            maps, reads = mp, run.reads
        print(m, "done", flush=True)
    rep = dict(student=a.student, clip=os.path.basename(a.clip), steps=n, hz=HZ, turn_steps=int(turn.sum()),
               target=f"real path curvature {LEAD / HZ:.1f} s later", route="the driven path (perfect satnav)",
               baseline_zero_mae=float(np.mean(np.abs(target[moving]))))
    for m, p in pred.items():
        rep[m] = dict(all=stats(target, p["curv"], moving), turns=stats(target, p["curv"], turn & moving),
                      straight=stats(target, p["curv"], ~turn & moving))
    json.dump(rep, open(os.path.join(OUT_DIR, a.name + ".json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, a.name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "speed_mps", "cmd", "actual_curv_target"] + [f"pred_curv_{m}" for m in pred])
        for i in range(n):
            w.writerow([f"{tel['t'][i]:.1f}", f"{tel['speed'][i]:.2f}", int(tel["cmd"][i]), f"{target[i]:.4f}"] +
                       [f"{pred[m]['curv'][i]:.4f}" for m in pred])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(tel["t"], target, "k", lw=2, label="the human driver")
    ax.plot(tel["t"], pred["mirrored"]["curv"], "tab:blue", label="student (map + route curve)")
    ax.plot(tel["t"], pred["mirrored_no_route"]["curv"], "tab:orange", alpha=.7, label="student, no route curve")
    ax.set_ylabel("path curvature 1/m (+ right)")
    ax.set_xlabel("s")
    ax.grid(alpha=.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{a.name} on real footage: corr {rep['mirrored']['all'].get('corr', 0):.2f} "
                 f"(no route {rep['mirrored_no_route']['all'].get('corr', 0):.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, a.name + ".png"), dpi=110)
    print(json.dumps({k: v for k, v in rep.items() if k in pred or k == "baseline_zero_mae"}, indent=1))
    if not a.no_video:
        path_video(a.clip, os.path.join(PRIV, f"{a.name}_paths_PRIVATE.mp4"), tel, pred["mirrored"]["curv"], reads,
                   maps, n)


def project(cam, fwd, right):
    """Ground point (fwd, right) in the car frame -> camera pixel (u, v), or None if behind the camera.
    Inverse of bev_project.ground_point (flat ground, same camera model the student's map uses)."""
    f = (cam["width"] / 2) / math.tan(math.radians(cam["fov"]) / 2)
    p = math.radians(cam["pitch"])
    X, Y, Z = fwd - cam["x"], right, -cam["z"]
    cf = X * math.cos(p) + Z * math.sin(p)
    cu = -X * math.sin(p) + Z * math.cos(p)
    if cf < 1.0:
        return None
    return cam["width"] / 2 + f * Y / cf, cam["height"] / 2 - f * cu / cf


def path_video(clip, out, tel, curv, reads, maps, n, horizon_s=2.5):
    """PRIVATE review video for the real-footage segment: the raw camera view with
      white = where the human driver actually went over the next 2.5 s (dead reckoning from speed + heading)
      blue  = the path the student would take from here (its steering held for 2.5 s)
    + the student's own commentary line (its situation / distance read-outs) and a small map inset."""
    import cv2
    from bev_model import Narrator
    from PIL import Image, ImageDraw, ImageFont
    cam = dict(C.TESLA_CAM_V2)
    fnt = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 30)
    fs = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 22)
    nar = Narrator()
    OW, OH = 1280, 960
    enc = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{OW}x{OH}", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt",
                            "yuv420p", out], stdin=subprocess.PIPE)
    k_h = int(horizon_s * HZ)
    for i, fr in enumerate(frames(clip)):
        if i >= n:
            break
        img = fr.copy()
        v = max(float(tel["speed"][i]), 2.0)
        L = min(max(v * horizon_s, 12.0), 35.0)
        # driver: real future positions in the current car frame
        h = math.radians(tel["yaw"][i])
        drv = []
        for j in range(i, min(i + k_h + 1, n)):
            dx, dy = tel["x"][j] - tel["x"][i], tel["y"][j] - tel["y"][i]
            drv.append((dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)))
        # student: an arc with its current curvature (+ = right)
        k = float(curv[i])
        stu = []
        for s in np.linspace(0, L, 30):
            th = k * s
            stu.append((s, 0.0) if abs(k) < 1e-5 else (math.sin(th) / k, (1 - math.cos(th)) / k))

        def ribbon(pts, col, width_m, alpha):
            left, right_ = [], []
            for (f1, r1), (f2, r2) in zip(pts, pts[1:]):
                a = math.atan2(r2 - r1, f2 - f1)
                nf, nr = -math.sin(a) * width_m / 2, math.cos(a) * width_m / 2
                pl, pr = project(cam, f1 - nf, r1 - nr), project(cam, f1 + nf, r1 + nr)
                if pl and pr:
                    left.append(pl)
                    right_.append(pr)
            if len(left) > 1:
                poly = np.array(left + right_[::-1], np.int32)
                over = img.copy()
                cv2.fillPoly(over, [poly], col, cv2.LINE_AA)
                img[:] = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)

        ribbon(drv, (255, 255, 255), 1.8, 0.35)
        ribbon(stu, (77, 163, 255), 0.5, 0.85)
        pim = Image.fromarray(img)
        d = ImageDraw.Draw(pim)
        probs = reads[i][0].copy()
        for a_, b_ in ((1, 2), (4, 5), (7, 8)):            # it ran mirrored: its 'left' is a real right
            probs[[a_, b_]] = probs[[b_, a_]]
        line = nar.line((probs, reads[i][1], None))
        tb = d.textbbox((0, 0), line, font=fnt)
        d.rounded_rectangle([30, 26, 30 + tb[2] + 40, 26 + 56], radius=14, fill=(12, 16, 22))
        d.text((50, 34), line, font=fnt, fill=(120, 255, 170))
        d.rounded_rectangle([30, OH - 74, 520, OH - 26], radius=12, fill=(12, 16, 22))
        d.rectangle([50, OH - 58, 90, OH - 42], fill=(240, 240, 240))
        d.text((100, OH - 66), "me", font=fs, fill=(240, 240, 240))
        d.rectangle([170, OH - 54, 210, OH - 46], fill=(77, 163, 255))
        d.text((220, OH - 66), "the AI's plan", font=fs, fill=(77, 163, 255))
        d.text((380, OH - 66), f"{tel['speed'][i] * 3.6:3.0f} km/h", font=fs, fill=(200, 200, 200))
        img = np.asarray(pim).copy()
        inset = cv2.resize(topdown_picture(maps[i].copy() % 8, w=200)[:, ::-1], (200, 312))   # un-mirrored for viewers
        img[OH - 312 - 90:OH - 90, OW - 230:OW - 30] = inset
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
    enc.stdin.close()
    enc.wait()
    print("video (PRIVATE, real footage):", out)


def video(clip, out, tel, target, pred, maps, cls, n):
    """PRIVATE review video: real frame + segmentation | the student's map (mirrored, as it sees it) | steering."""
    import cv2
    from seg_check import COLORS
    enc = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "1280x540", "-r", str(HZ), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt",
                            "yuv420p", out], stdin=subprocess.PIPE)
    for i, f in enumerate(frames(clip)):
        if i >= n:
            break
        c = cv2.resize(cls[i], (W, H), interpolation=cv2.INTER_NEAREST)
        ov = np.where((c > 0)[..., None], (f * 0.5 + COLORS[c] * 0.5).astype(np.uint8), f)
        left = cv2.resize(ov, (720, 540))
        mid = cv2.resize(topdown_picture(maps[i].copy() % 8, w=240), (240, 375))
        canvas = np.full((540, 1280, 3), 14, np.uint8)
        canvas[:, :720] = left
        canvas[40:415, 740:980] = mid
        rl = route_layer(tel["route"] * np.array([1, -1]), int(tel["ridx"][i]), tel["x"][i], -tel["y"][i],
                         -tel["yaw"][i])
        rl = cv2.resize(rl.astype(np.uint8), (240, 375), interpolation=cv2.INTER_NEAREST).astype(bool)
        canvas[40:415, 740:980][rl] = (60, 150, 255)
        cv2.putText(canvas, "what the student sees (mirrored)", (740, 30), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for k, (lab, v, col) in enumerate([("driver", target[i], (240, 240, 240)),
                                           ("student", pred["mirrored"]["curv"][i], (77, 163, 255))]):
            y = 450 + k * 40
            cv2.putText(canvas, lab, (1000, y), 0, 0.55, col, 1, cv2.LINE_AA)
            x0 = 1100
            cv2.rectangle(canvas, (x0, y - 12), (x0 + 160, y - 2), (40, 46, 58), -1)
            xv = int(x0 + 80 + np.clip(v / 0.08, -1, 1) * 80)
            cv2.rectangle(canvas, (min(x0 + 80, xv), y - 12), (max(x0 + 80, xv), y - 2), col, -1)
        cv2.putText(canvas, f"{tel['speed'][i] * 3.6:3.0f} km/h", (1000, 60), 0, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        enc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    enc.stdin.close()
    enc.wait()
    print("video (PRIVATE, real footage):", out)


if __name__ == "__main__":
    main()
