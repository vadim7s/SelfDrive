"""Phase B time-stack check on a CARLA 10 Hz sequence (seg_capture.py --every 2): the fused top-down map
(earlier frames moved by our TRUE motion from the poses) vs the map truth.  Shows how much of the area around
the car gets covered and how well it matches, single frame vs fused.  Also writes a side-by-side video:
camera | fused top-down from the segmenter | map truth.
  python bev_stack_check.py [folder]"""
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image

import config as C
from bev_map import BevMap
from bev_project import BevProjector, Fuser, UNSEEN, RES, CY, colorize, footprints
from seg_check import Segmenter

OUT = os.path.join(C.DATA_ROOT, "seg_phaseA", "bev_check")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.DATA_ROOT, "seg_phaseA", "carla_seq")
    poses = sorted(glob.glob(os.path.join(src, "*_pose.json")))
    town = os.path.basename(poses[0]).split("_")[0]
    bev, proj, seg, fuser = BevMap(town), BevProjector(C.TESLA_CAM), Segmenter(), Fuser()
    fwd = (CY - np.mgrid[0:500, 0:400][0]) * RES
    zone = np.abs(fwd) < 20                                     # 20 m ahead and behind, full width
    vid = os.path.join(OUT, f"stack_{town}.mp4")
    enc = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, (533 + 400 + 400, 500))
    stats, prev = [], None
    for k, pf in enumerate(poses):
        p = json.load(open(pf))
        rgb = np.asarray(Image.open(pf.replace("_pose.json", "_rgb.jpg")).convert("RGB"))
        cls = seg(rgb, (640, 480))
        top = proj.classes(cls)
        if prev is None:
            d_f = d_r = d_y = 0.0
        else:                                                   # our true motion, in the previous car frame
            a = math.radians(prev["yaw"])
            dx, dy = p["x"] - prev["x"], p["y"] - prev["y"]
            d_f, d_r = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
            d_y = (p["yaw"] - prev["yaw"] + 180) % 360 - 180
        fused = fuser.step(top, footprints(C.TESLA_CAM, cls), d_f, d_r, d_y)
        prev = p
        T = bev.render_classes(p["x"], p["y"], p["yaw"], p["z"]) >= 1
        for name, P in (("single frame", top), ("fused", fused)):
            seen = (P != UNSEEN) & zone
            road = ((P == 1) | (P == 2) | (P == 4)) & seen
            iou = (road & T).sum() / max((road | (T & seen)).sum(), 1)
            stats.append((name, seen.sum() / zone.sum(), iou))
        tv = np.zeros(T.shape + (3,), np.uint8)
        tv[:] = (14, 14, 14)
        tcls = bev.render_classes(p["x"], p["y"], p["yaw"], p["z"])
        tv[tcls == 1], tv[tcls == 2], tv[tcls == 3] = (58, 56, 54), (235, 235, 235), (150, 150, 160)
        cam = np.zeros((500, 533, 3), np.uint8)
        cam[50:450] = cv2.resize(rgb[:, :, ::-1], (533, 400))
        enc.write(np.concatenate([cam, colorize(fused), tv], 1))
    enc.release()
    for name in ("single frame", "fused"):
        a = np.array([(c, i) for n, c, i in stats if n == name])
        print(f"{name:13s}: covers {100 * a[:, 0].mean():4.0f}% of the 40 m x 40 m area around the car | "
              f"road IoU vs map truth {a[:, 1].mean():.2f}")
    print("video (camera | fused top-down | map truth):", vid)


if __name__ == "__main__":
    main()
