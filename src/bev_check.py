"""Phase B check on CARLA: does the camera -> top-down projection land on the real road geometry?
For every posed frame (seg_capture.py output with *_pose.json):
  A = CARLA's perfect camera mask projected top-down   (tests the projection geometry alone)
  B = segmenter output projected top-down              (tests the full real-world pipeline)
  T = map truth from the road file at the same pose    (bev_map.render_classes)
Scores within the camera's view and 3-30 m ahead: road IoU, lane-line hit rate within 0.3 m (both ways),
and the sideways shift that best lines the projected road up with the truth.
  python bev_check.py [folder]  -> table + pictures in data\\seg_phaseA\\bev_check"""
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import config as C
from bev_map import BevMap
from bev_project import BevProjector, UNSEEN, RES, CY, colorize
from seg_check import C_LUT, Segmenter

OUT = os.path.join(C.DATA_ROOT, "seg_phaseA", "bev_check")
TOL_PX = int(round(0.3 / RES))


def lines_score(pred_line, true_line, area):
    """share of predicted line pixels near a true line, share of true line pixels near a predicted line."""
    tp = true_line & area
    pp = pred_line & area
    if pp.sum() < 20 or tp.sum() < 20:
        return float("nan"), float("nan")
    d_true = cv2.distanceTransform((~tp).astype(np.uint8), cv2.DIST_L2, 3)
    d_pred = cv2.distanceTransform((~pp).astype(np.uint8), cv2.DIST_L2, 3)
    return float((d_true[pp] <= TOL_PX).mean()), float((d_pred[tp] <= TOL_PX).mean())


def best_shift(pred_road, true_road, area):
    best, arg = -1, 0
    for sft in range(-15, 16):                                    # +-1.5 m
        sh = np.roll(true_road, sft, axis=1)
        inter = (pred_road & sh & area).sum()
        union = ((pred_road | sh) & area).sum()
        if union and inter / union > best:
            best, arg = inter / union, sft
    return arg * RES


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.DATA_ROOT, "seg_phaseA", "carla_pose")
    os.makedirs(OUT, exist_ok=True)
    proj = BevProjector(C.TESLA_CAM)
    seg = Segmenter()
    maps = {}
    fwd = (CY - np.mgrid[0:500, 0:400][0]) * RES
    band = (fwd > 3) & (fwd < 30)
    rows = defaultdict(list)
    for k, pf in enumerate(sorted(glob.glob(os.path.join(src, "*_pose.json")))):
        pose = json.load(open(pf))
        town = os.path.basename(pf).split("_")[0]
        if town not in maps:
            maps[town] = BevMap(town)
        rgb = np.asarray(Image.open(pf.replace("_pose.json", "_rgb.jpg")).convert("RGB"))
        tag = np.asarray(Image.open(pf.replace("_pose.json", "_tag.png")))
        T = maps[town].render_classes(pose["x"], pose["y"], pose["yaw"], pose["z"])
        A = proj.classes(C_LUT[tag])
        B = proj.classes(seg(rgb, (640, 480)))
        area = band & (A != UNSEEN)
        true_road = T >= 1
        true_line = T == 2
        for name, P in (("A perfect mask", A), ("B segmenter", B)):
            road = (P == 1) | (P == 2)
            iou = (road & true_road & area).sum() / max(((road | true_road) & area).sum(), 1)
            prec, rec = lines_score(P == 2, true_line, area)
            rows[name].append((iou, prec, rec, best_shift(road, true_road, area)))
        if k % 4 == 0:
            tv = np.zeros(T.shape + (3,), np.uint8)
            tv[T == 1], tv[T == 2], tv[T == 3] = (58, 56, 54), (235, 235, 235), (150, 150, 160)
            ov = tv.copy()
            ov[(B == 2)] = (40, 200, 240)                         # segmenter lines (yellow) on the truth
            cam = cv2.resize(rgb[:, :, ::-1], (400 * 4 // 3, 400))
            pad = np.zeros((500, cam.shape[1], 3), np.uint8)
            pad[50:450] = cam
            cv2.imwrite(os.path.join(OUT, os.path.basename(pf).replace("_pose.json", ".jpg")),
                        np.concatenate([pad, colorize(B), tv, ov], 1))
    print(f"{'':16s} {'road IoU':>9s} {'lines near truth':>17s} {'truth lines found':>18s} {'sideways shift':>15s}")
    for name, r in rows.items():
        a = np.array(r, dtype=float)
        print(f"{name:16s} {np.nanmean(a[:, 0]):9.2f} {np.nanmean(a[:, 1]):17.2f} {np.nanmean(a[:, 2]):18.2f} "
              f"{np.nanmedian(a[:, 3]):+12.2f} m   ({len(r)} frames)")
    print("pictures (camera | segmenter top-down | map truth | segmenter lines on truth):", OUT)


if __name__ == "__main__":
    main()
