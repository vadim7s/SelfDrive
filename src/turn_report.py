"""Lane keeping in junction turns, from vision_collect.py --trace files.
    python turn_report.py [trace.csv ...]
Trace 'lat' = offset from the route (lane centre) in half-lane widths, + = right, - = left (towards oncoming traffic,
CARLA drives on the right). Car half-width ~0.95 m of a ~1.75 m half lane, so |lat| > 0.45 = a wheel over the lane line.
Per turn (junction block entered with command left/right, +-1 s around it):
  out%    turns where a wheel crosses into the oncoming side (lat < -0.45)
  worst   most-left lat (oncoming side), mean over turns, and the single worst
  kerb    turns that go past the right lane line (lat > +0.45)
"""
import config as C
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHEEL_OVER = 0.45


def turns(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    eps = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    out = {1: [], 2: []}                     # 1 = left, 2 = right: list of (min lat, max lat)
    for ep in eps.values():
        j = np.array([int(r["junction"]) for r in ep])
        c = np.array([int(r["cmd"]) for r in ep])
        lat = np.array([float(r["lat"]) for r in ep])
        i = 0
        while i < len(ep):
            if j[i] and (i == 0 or not j[i - 1]):
                k = i
                while k < len(ep) and j[k]:
                    k += 1
                cmd = c[max(0, i - 1)] if c[max(0, i - 1)] in (1, 2) else c[i]
                if cmd in (1, 2) and k < len(ep):           # finished turns only
                    a, b = max(0, i - 10), min(len(ep), k + 10)
                    out[cmd].append((lat[a:b].min(), lat[a:b].max()))
                i = k
            else:
                i += 1
    return out


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "runs", "trace_teacher_*.csv")))
    print(f"   {'trace':<34} {'turn':<5} {'n':>3} {'out%':>5} {'worst mean':>10} {'worst':>6} {'kerb%':>6} {'rmax':>6}")
    for f in files:
        t = turns(f)
        name = os.path.splitext(os.path.basename(f))[0]
        for cmd, lab in ((1, "left"), (2, "right")):
            v = np.array(t[cmd]) if t[cmd] else None
            if v is None:
                continue
            print(f"   {name:<34} {lab:<5} {len(v):3d} {100 * (v[:, 0] < -WHEEL_OVER).mean():5.0f} "
                  f"{v[:, 0].mean():10.2f} {v[:, 0].min():6.2f} {100 * (v[:, 1] > WHEEL_OVER).mean():6.0f} {v[:, 1].mean():6.2f}")


if __name__ == "__main__":
    main()
