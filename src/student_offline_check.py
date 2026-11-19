"""Offline check (no CARLA): does a student copy the teacher's steering on HELD-OUT recorded teacher frames?
Slope ~1 here but ~0.3 when driving => the problem is drift into unseen states (closed loop), not learning.
  python student_offline_check.py <best.pt> <data dir> [episodes per town]"""
import config as C
import csv
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
from PIL import Image

from vision_model import StudentDriver


def main():
    student_path, root = sys.argv[1], sys.argv[2]
    per_town = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    wheel_mode = sys.argv[4] if len(sys.argv) > 4 else "teacher"   # teacher | own (student's own last output) | zero
    drv = StudentDriver(student_path)
    by_town = defaultdict(list)
    for d in sorted(os.listdir(root)):
        ep = os.path.join(root, d)
        if d.startswith("ep_") and zlib.crc32(d.encode()) % 10 == 0 and os.path.isfile(os.path.join(ep, "meta.json")):
            by_town[json.load(open(os.path.join(ep, "meta.json"), encoding="utf-8"))["town"]].append(ep)
    for town, eps in sorted(by_town.items()):
        s_all, t_all = [], []
        for ep in eps[:per_town]:
            drv.reset()
            own = 0.0
            for r in csv.DictReader(open(os.path.join(ep, "labels.csv"), encoding="utf-8")):
                img = os.path.join(ep, f"{int(r['frame']):06d}.jpg")
                if not os.path.isfile(img):
                    continue
                wheel = {"teacher": float(r["wheel"]), "own": own, "zero": 0.0}[wheel_mode]
                out = drv.act(np.asarray(Image.open(img).convert("RGB")), float(r["speed"]), wheel,
                              int(r["command"]), float(r.get("v_target") or 0) or None)
                s_all.append(float(out[0]))
                own = float(out[0])
                t_all.append(float(r["steer"]))
        s, t = np.array(s_all), np.array(t_all)
        m = np.abs(t) > 0.1
        slope = np.polyfit(t[m], s[m], 1)[0] if m.sum() > 10 else float("nan")
        print(f"[wheel={wheel_mode}] {town:9s} {len(eps[:per_town])} held-out routes, {len(t)} frames | steer err {np.abs(s - t).mean():.3f} | "
              f"slope student/teacher {slope:.2f} | mean|teacher| {np.abs(t).mean():.3f} mean|student| {np.abs(s).mean():.3f}",
              flush=True)


if __name__ == "__main__":
    main()
