"""Situation labels (Phase C) derived OFFLINE from a recorded episode (labels.csv + route.npz from vision_collect --v2).
One class per frame, by priority: red > give way > junction turn > approaching junction > lane change > following > bend > straight.
Also per frame: curvature ahead (1/m, + = right), distance to the next junction, stop distance.
  python situation.py <data dir> [n episodes for timelines]  -> class counts + timeline pictures"""
import config as C
import csv
import glob
import math
import os
import sys
from collections import Counter

import numpy as np

CLASSES = ["straight", "left bend", "right bend", "approaching junction", "junction left", "junction right",
           "junction straight", "lane change left", "lane change right", "following car", "stopping for red",
           "giving way"]
NARRATE = ["following the lane", "left bend", "right bend", "approaching junction", "turning left at junction",
           "turning right at junction", "straight through junction", "changing lane left", "changing lane right",
           "following car", "stopping for red light", "giving way"]
RES = 2.0                         # route point spacing (config.ROUTE_RESOLUTION)
BEND_R = 150.0                    # radius below this = bend
LEFT, RIGHT, STRAIGHT, CH_L, CH_R = 1, 2, 3, 5, 6


def num(v):
    return None if v in ("", None) else float(v)


def curvature(route, i, ahead_m=15.0):
    """Mean curvature (1/m) over the next ahead_m along the route; + = turning right (CARLA yaw is clockwise)."""
    n = max(2, int(ahead_m / RES))
    j = min(i + n, len(route) - 2)
    if j <= i:
        return 0.0
    h0 = math.atan2(route[i + 1, 1] - route[i, 1], route[i + 1, 0] - route[i, 0])
    h1 = math.atan2(route[j + 1, 1] - route[j, 1], route[j + 1, 0] - route[j, 0])
    return ((h1 - h0 + math.pi) % (2 * math.pi) - math.pi) / ((j - i) * RES)


def junction_turn(opt, junc, i):
    """Turn option (1/2/3) of the junction stretch that contains route index i."""
    a = i
    while a > 0 and junc[a - 1]:
        a -= 1
    b = i
    while b < len(junc) - 1 and junc[b + 1]:
        b += 1
    for k in range(max(0, a - 6), min(len(opt), b + 2)):          # option is often set just before the junction
        if opt[k] in (LEFT, RIGHT, STRAIGHT):
            return int(opt[k])
    return STRAIGHT


def label_episode(ep):
    rows = list(csv.DictReader(open(os.path.join(ep, "labels.csv"), encoding="utf-8")))
    r = np.load(os.path.join(ep, "route.npz"))
    route, opt, junc = r["route"][:, :2], r["opt"], r["junction"]
    out = []
    for row in rows:
        i = min(int(row["route_idx"]), len(route) - 2)
        v = float(row["speed"])
        red, yld, gap = num(row["red_dist"]), num(row["yield_dist"]), num(row["lead_gap"])
        stop_zone = max(10.0, v * v / (2 * 2.5) + 8.0)
        nxt = next((k for k in range(i, len(junc)) if junc[k]), None)
        to_j = (nxt - i) * RES if nxt is not None else float("inf")
        kap = curvature(route, i)
        if red is not None and red < stop_zone:
            c = 10
        elif yld is not None and yld < stop_zone:
            c = 11
        elif junc[i]:
            c = {LEFT: 4, RIGHT: 5}.get(junction_turn(opt, junc, i), 6)
        elif to_j < max(20.0, 3.0 * v):
            c = 3
        elif any(o in (CH_L, CH_R) for o in opt[i:i + 5]):
            c = 7 if CH_L in opt[i:i + 5] else 8
        elif gap is not None and gap < max(12.0, 2.5 * v):
            c = 9
        elif abs(kap) > 1.0 / BEND_R:
            c = 2 if kap > 0 else 1
        else:
            c = 0
        out.append(dict(situation=c, curvature=kap, to_junction=min(to_j, 999.0),
                        stop=min([d for d in (red, yld) if d is not None], default=999.0)))
    return rows, out


def timeline(ep, path):
    import cv2
    rows, lab = label_episode(ep)
    n = len(rows)
    w, h = max(600, min(n, 1800)), 260
    img = np.full((h, w, 3), 255, np.uint8)
    pal = np.array([(200, 200, 200), (255, 170, 90), (90, 170, 255), (120, 220, 255), (255, 80, 40), (40, 80, 255),
                    (60, 200, 60), (255, 120, 200), (200, 120, 255), (150, 150, 0), (0, 0, 220), (0, 140, 255)], np.uint8)
    xs = (np.arange(n) * w / n).astype(int)
    for k in range(n):
        img[0:60, xs[k]:xs[min(k + 1, n - 1)] + 1] = pal[lab[k]["situation"]]
    sp = np.array([float(r["speed"]) for r in rows])
    st = np.array([float(r["steer"]) for r in rows])
    for arr, top, scale, col in ((sp, 70, max(sp.max(), 1), (0, 0, 0)), (st, 165, 1.0, (200, 0, 0))):
        y = (top + 45 - arr / scale * 45).astype(int) if arr is sp else (top + 45 - arr * 45).astype(int)
        cv2.polylines(img, [np.stack([xs, y], 1).astype(np.int32)], False, col, 1, cv2.LINE_AA)
    cv2.putText(img, "speed", (4, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, "steer", (4, 177), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    used = sorted(set(d["situation"] for d in lab))
    lx = 60
    for c in used:
        cv2.rectangle(img, (lx, 240), (lx + 12, 252), tuple(int(v) for v in pal[c]), -1)
        cv2.putText(img, CLASSES[c], (lx + 15, 251), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        lx += 20 + 7 * len(CLASSES[c])
    cv2.imwrite(path, img)


def main():
    root = sys.argv[1]
    n_tl = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    eps = sorted(d for d in glob.glob(os.path.join(root, "ep_*")) if os.path.isfile(os.path.join(d, "route.npz")))
    cnt = Counter()
    for ep in eps:
        _, lab = label_episode(ep)
        cnt.update(d["situation"] for d in lab)
    tot = sum(cnt.values())
    print(f"{len(eps)} episodes, {tot} frames")
    for c in range(len(CLASSES)):
        print(f"  {CLASSES[c]:22s} {cnt[c]:7d}  {100 * cnt[c] / max(tot, 1):5.1f}%")
    out = os.path.join(root, "_timelines")
    os.makedirs(out, exist_ok=True)
    for ep in eps[:n_tl]:
        timeline(ep, os.path.join(out, os.path.basename(ep) + ".png"))
    print("timelines:", out)


if __name__ == "__main__":
    main()
