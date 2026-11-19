"""Score map-memory variants BEFORE any retrain: the student's map vs CARLA's map truth on recorded drives, plus
frame-to-frame flicker (the suspected cause of the weave on real roads), and flicker on the real Tesla clip.
No CARLA needed (uses saved segmentation + poses).
  python map_quality.py [n episodes]
Variants: current Fuser | VoteFuser | VoteFuser + car 'worms' (cars 1 m deep instead of 4.5 m)."""
import config as C
import csv
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

from bev_dataset import BevInput
from bev_map import BevMap
from bev_project import UNSEEN, VoteFuser, shift_rotate

DATA = os.path.join(C.DATA_ROOT, "vision_data_v2")
ZONE = (slice(None), slice(40, 360))           # the part of the 500 x 400 map the student gets (+-16 m)
VARIANTS = {"current": lambda: (None, 4.5),
            "vote": lambda: (VoteFuser(), 4.5),
            "slow": lambda: (VoteFuser(decay=0.95, min_ev=0.3), 4.5),
            "slow+stick": lambda: (VoteFuser(decay=0.95, min_ev=0.3, stick=0.5), 4.5),
            "slow+stick1": lambda: (VoteFuser(decay=0.95, min_ev=0.3, stick=1.0), 4.5),
            "slow+stick+worms": lambda: (VoteFuser(decay=0.95, min_ev=0.3, stick=0.5), 1.0)}


def dilate(m, r=3):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def motion(prev, cur):
    a = math.radians(prev[2])
    dx, dy = cur[0] - prev[0], cur[1] - prev[1]
    return (dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a),
            (cur[2] - prev[2] + 180) % 360 - 180)


class Score:
    def __init__(self):
        self.s = {k: [] for k in ("iou", "line_p", "line_r", "flicker", "cover")}

    def add_truth(self, view, truth):
        v, t = view[ZONE], truth[ZONE]
        seen = (v != UNSEEN) & (v != 4)
        road_p, road_t = seen & ((v == 1) | (v == 2)), seen & ((t == 1) | (t == 2))
        if (road_p | road_t).sum():
            self.s["iou"].append((road_p & road_t).sum() / (road_p | road_t).sum())
        lp, lt = seen & (v == 2), seen & (t == 2)
        if lp.sum():
            self.s["line_p"].append((lp & dilate(lt)).sum() / lp.sum())
        if lt.sum():
            self.s["line_r"].append((lt & dilate(lp)).sum() / lt.sum())
        self.s["cover"].append(seen.mean())

    def add_flicker(self, view, prev_view, mv):
        if prev_view is None:
            return
        pv = shift_rotate(prev_view, *mv)[ZONE]
        v = view[ZONE]
        both = (v < 4) & (pv < 4)
        if both.sum():
            self.s["flicker"].append((v[both] != pv[both]).mean())

    def summary(self):
        return {k: (float(np.mean(x)) if x else float("nan")) for k, x in self.s.items()}


def carla_eps(n):
    eps = []
    for ep in sorted(glob.glob(os.path.join(DATA, "ep_*")))[::7]:
        m = json.load(open(os.path.join(ep, "meta.json")))
        if m.get("driver") == "teacher" and os.path.isdir(os.path.join(ep, "seg")):
            eps.append((ep, m))
        if len(eps) >= n:
            break
    return eps


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    maps = {}
    res = {v: Score() for v in VARIANTS}
    for ep, m in carla_eps(n):
        town = m["town"]
        maps.setdefault(town, BevMap(town))
        rows = list(csv.DictReader(open(os.path.join(ep, "labels.csv"))))[:200]
        for name, mk in VARIANTS.items():
            fuser, worm = mk()
            bi = BevInput(m["camera"], fuser, worm)
            prev, prev_view = None, None
            for k, r in enumerate(rows):
                pose = (float(r["x"]), float(r["y"]), float(r["yaw"]))
                cls = np.asarray(Image.open(os.path.join(ep, "seg", f"{int(r['frame']):06d}.png")))
                bi.step(cls, *pose)
                view = bi.fuser_view
                if k >= 20:                                    # let the memory fill first
                    truth = maps[town].render_classes(pose[0], pose[1], pose[2], float(r["z"]))
                    res[name].add_truth(view, truth)
                    res[name].add_flicker(view, prev_view, motion(prev, pose))
                prev, prev_view = pose, view
        print(os.path.basename(ep), town, "done", flush=True)
    print("\nCARLA, vs map truth (student's +-16 m crop; vehicles excluded):")
    print(f"{'variant':12s} {'road IoU':>9s} {'line prec':>9s} {'line rec':>9s} {'flicker':>8s} {'covered':>8s}")
    for name, sc in res.items():
        s = sc.summary()
        print(f"{name:12s} {s['iou']:9.3f} {s['line_p']:9.3f} {s['line_r']:9.3f} {s['flicker']:8.3f} {s['cover']:8.2f}")

    # real footage: flicker only (no truth), motion from speed + heading
    from tesla_bev_test import PRIV, TEL, telemetry
    import config as C
    cls_t = np.load(os.path.join(PRIV, "tesla_cls_10hz.npz"))["cls"]
    tel = telemetry(TEL, len(cls_t))
    print("\nReal Tesla clip (flicker = share of map cells that change class between frames, after our motion):")
    for name, mk in VARIANTS.items():
        fuser, worm = mk()
        bi = BevInput(dict(C.TESLA_CAM_V2), fuser, worm)
        sc, prev, prev_view = Score(), None, None
        for i in range(len(cls_t)):
            pose = (tel["x"][i], tel["y"][i], tel["yaw"][i])
            bi.step(cls_t[i], *pose)
            view = bi.fuser_view
            if i >= 20:
                sc.add_flicker(view, prev_view, motion(prev, pose))
                sc.s["cover"].append(((view[ZONE] != UNSEEN) & (view[ZONE] != 4)).mean())
            prev, prev_view = pose, view
        s = sc.summary()
        print(f"  {name:12s} flicker {s['flicker']:.3f}  covered {s['cover']:.2f}")


if __name__ == "__main__":
    main()
