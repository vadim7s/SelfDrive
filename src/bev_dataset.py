"""Phase D input cache: every segmented frame -> fused HEADING-UP top-down class map (the student's input).
Per episode: <ep>\\bev.npz  bev = N x 250 x 160 uint8 (0.2 m/px; 40 m ahead, 10 m behind, +-16 m sideways),
classes 0-6 as the segmenter, 7 = not seen.  Same code path as live driving (BevInput), so sim and real match.
Re-running skips finished episodes.  No CARLA.
  python bev_dataset.py <data dir>"""
import config as C
import glob
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from bev_project import UNSEEN, BevProjector, Fuser, footprints

SIDE_PX = 40                     # crop 4 m each side of the 0.1 m grid (400 -> 320 px = +-16 m)
NOT_SEEN = 7
OUT_H, OUT_W = 250, 160


class BevInput:
    """Camera class map + ego motion -> fused top-down input map (OUT_H x OUT_W, classes 0-7)."""
    def __init__(self, cam, fuser=None, worm_m=4.5):
        """fuser: None = Fuser (latest frame overwrites); bev_project.VoteFuser() = evidence voting.
        worm_m: how far a seen car extends away from us on the map (4.5 = a car length; ~1 = a thin 'worm')."""
        self.cam, self.proj, self.prev = cam, BevProjector(cam), None
        self.fuser, self.worm_m = fuser or Fuser(), worm_m

    def step(self, cls, x, y, yaw):
        d_f = d_r = d_y = 0.0
        if self.prev is not None:
            px, py, pyaw = self.prev
            a = math.radians(pyaw)
            dx, dy = x - px, y - py
            d_f, d_r = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
            d_y = (yaw - pyaw + 180) % 360 - 180
        self.prev = (x, y, yaw)
        view = self.fuser.step(self.proj.classes(cls), footprints(self.cam, cls, length_m=self.worm_m), d_f, d_r, d_y)
        self.fuser_view = view                                   # full-resolution map, for map-quality checks
        return shrink(view)


FUSERS = ("current", "vote")


def cache_name(fuser="current"):
    """Map cache per episode: bev.npz (original memory) or bev_vote.npz (evidence-voting memory)."""
    return "bev.npz" if fuser == "current" else f"bev_{fuser}.npz"


def make_input(cam, fuser="current"):
    """The student's map builder for a named memory.  'vote' = VoteFuser tuned 18 Nov on CARLA truth + real flicker."""
    if fuser == "vote":
        from bev_project import VoteFuser
        return BevInput(cam, VoteFuser(decay=0.95, min_ev=0.3, stick=1.0))
    return BevInput(cam)


def shrink(view):
    v = view[::2, SIDE_PX:-SIDE_PX:2].copy()
    v[v == UNSEEN] = NOT_SEEN
    return v


def do_episode(ep, fuser=None):
    fuser = fuser or os.environ.get("FSD_FUSER", "current")
    out = os.path.join(ep, cache_name(fuser))
    if os.path.isfile(out):
        return 0
    import csv
    cam = json.load(open(os.path.join(ep, "meta.json"), encoding="utf-8"))["camera"]
    rows = list(csv.DictReader(open(os.path.join(ep, "labels.csv"), encoding="utf-8")))
    b = make_input(cam, fuser)
    maps = np.empty((len(rows), OUT_H, OUT_W), np.uint8)
    for k, r in enumerate(rows):
        cls = np.asarray(Image.open(os.path.join(ep, "seg", f"{int(r['frame']):06d}.png")))
        maps[k] = b.step(cls, float(r["x"]), float(r["y"]), float(r["yaw"]))
    np.savez_compressed(out + ".tmp.npz", bev=maps)
    os.replace(out + ".tmp.npz", out)
    return len(rows)


def main():
    root = sys.argv[1]
    if len(sys.argv) > 2:                                       # python bev_dataset.py <dir> vote
        os.environ["FSD_FUSER"] = sys.argv[2]
    eps = sorted(d for d in glob.glob(os.path.join(root, "ep_*")) if os.path.isdir(os.path.join(d, "seg")))
    t0, n = time.time(), 0
    with Pool(3) as p:
        for k, m in enumerate(p.imap_unordered(do_episode, eps), 1):
            n += m
            if k % 10 == 0:
                print(f"  {k}/{len(eps)} episodes, {n} frames, {time.time() - t0:.0f} s", flush=True)
    print(f"[{os.environ.get('FSD_FUSER', 'current')}] bev cache done: {len(eps)} episodes, {n} new frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
