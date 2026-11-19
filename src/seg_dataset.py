"""Segment every frame of a collected dataset once (Phase C): <episode>\\seg\\<frame>.png = our 7-class map (640x480).
Re-running continues (skips frames already done).  No CARLA needed.
  python seg_dataset.py <data dir>"""
import config as C
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

from seg_check import Segmenter


def main():
    root = sys.argv[1]
    eps = sorted(d for d in glob.glob(os.path.join(root, "ep_*")) if os.path.isfile(os.path.join(d, "labels.csv")))
    seg, t0, done = Segmenter(), time.time(), 0
    todo = sum(len(glob.glob(os.path.join(e, "*.jpg"))) for e in eps)
    for ep in eps:
        out = os.path.join(ep, "seg")
        os.makedirs(out, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(ep, "*.jpg"))):
            p = os.path.join(out, os.path.basename(f)[:-4] + ".png")
            done += 1
            if os.path.isfile(p):
                continue
            rgb = np.asarray(Image.open(f).convert("RGB"))
            Image.fromarray(seg(rgb)).save(p)
            if done % 2000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{todo} frames, {rate:.1f}/s, ~{(todo - done) / max(rate, 1e-3) / 60:.0f} min left", flush=True)
    print(f"segmented {todo} frames in {len(eps)} episodes", flush=True)


if __name__ == "__main__":
    main()
