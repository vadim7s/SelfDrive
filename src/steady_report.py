"""Straight-road steadiness from vision_collect.py --trace files (default: runs\\trace_teacher_*.csv).
    python steady_report.py [trace.csv ...]
Straight = follow command, not in a junction, 2 s average wheel angle < 0.03, speed > 5 m/s.
  wobble    mean |wheel - its 2 s average|  (lower = steadier; real driver ~0)
  flips/km  wheel direction changes (> 0.005) per km on straight road
  |lat|     mean distance from the lane centre on straight road, in half-lane widths
"""
import config as C
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def straight_stats(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = dict(wob=[], flips=0, km=0.0, lat=[])
    eps = {}
    for r in rows:
        eps.setdefault(r["episode"], []).append(r)
    k = np.ones(21) / 21
    for ep in eps.values():
        if len(ep) < 40:
            continue
        w = np.array([float(r["wheel"]) for r in ep])
        sp = np.array([float(r["speed"]) for r in ep])
        c = np.array([int(r["cmd"]) for r in ep])
        j = np.array([int(r["junction"]) for r in ep])
        lat = np.array([float(r["lat"]) for r in ep])
        slow = np.convolve(w, k, "same")
        m = (c == 0) & (j == 0) & (np.abs(slow) < 0.03) & (sp > 5)
        m[:10] = m[-10:] = False
        if m.sum() < 10:
            continue
        out["wob"].append(np.abs(w - slow)[m])
        dw = np.diff(w)
        big = (np.abs(dw) > 0.005) & m[1:]
        sgn = np.sign(dw[big])
        out["flips"] += int((np.diff(sgn) != 0).sum())
        out["km"] += float((sp[m] * 0.1).sum()) / 1000
        out["lat"].append(np.abs(lat[m]))
    if not out["wob"]:
        return None
    return dict(wobble=float(np.concatenate(out["wob"]).mean()), flips_km=out["flips"] / max(out["km"], 1e-3),
                lat=float(np.concatenate(out["lat"]).mean()), km=out["km"])


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "runs", "trace_teacher_*.csv")))
    print(f"   {'trace':<36} {'km':>5} {'wobble':>7} {'flips/km':>9} {'|lat|':>8}")
    for f in files:
        st = straight_stats(f)
        name = os.path.splitext(os.path.basename(f))[0]
        if st is None:
            print(f"   {name:<36} (no straight road)")
        else:
            print(f"   {name:<36} {st['km']:5.1f} {st['wobble']:7.4f} {st['flips_km']:9.0f} {st['lat']:8.2f}")


if __name__ == "__main__":
    main()
