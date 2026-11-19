"""Phase A check: real-world segmenter (Mask2Former, Mapillary Vistas) on CARLA frames (scored against CARLA's
ground-truth tags) and on the Tesla clip (overlays to look at).  No CARLA needed.
  python seg_check.py carla       -> IoU per class, per input resolution, per weather + overlays + speed
  python seg_check.py tesla [clip] -> overlay frames every 2 s + overlay video (PRIVATE: real footage)
Outputs: data\\seg_phaseA\\"""
import config as C
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

MODEL = C.SEG_MODEL
OUT = os.path.join(C.DATA_ROOT, "seg_phaseA")
CLIP = C.TESLA_CLIP
FF = "ffmpeg"

CLASSES = ["background", "road", "lane marking", "kerb/sidewalk", "vehicle", "pedestrian/cyclist", "traffic light"]
COLORS = np.array([(0, 0, 0), (128, 64, 128), (255, 230, 0), (244, 35, 232), (0, 60, 255), (220, 20, 60),
                   (250, 170, 30)], np.uint8)
VISTAS = {1: (7, 8, 10, 13, 14, 36, 41, 43), 2: (23, 24), 3: (2, 9, 11, 15),
          4: (54, 55, 56, 57, 58, 59, 60, 61, 62), 5: (19, 20, 21, 22, 52), 6: (48,)}
CARLA_TAGS = {1: (1,), 2: (24,), 3: (2,), 4: (14, 15, 16, 17, 18), 5: (12, 13, 19), 6: (7,)}


def lut(mapping, n):
    t = np.zeros(n, np.uint8)
    for c, ids in mapping.items():
        t[list(ids)] = c
    return t


V_LUT, C_LUT = lut(VISTAS, 256), lut(CARLA_TAGS, 256)


class Segmenter:
    def __init__(self):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).cuda().half().eval()

    @torch.no_grad()
    def __call__(self, rgb, size=None):
        """rgb HxWx3 uint8 -> class map (our 7 classes) at the input size.  size=(w,h): downscale first."""
        im = Image.fromarray(rgb)
        if size and im.size != size:
            im = im.resize(size, Image.BILINEAR)
        x = self.proc(images=im, return_tensors="pt", do_resize=False)
        out = self.net(pixel_values=x["pixel_values"].cuda().half())
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[im.size[::-1]])[0]
        return V_LUT[seg.cpu().numpy().astype(np.uint8)]


def overlay(rgb, cls, alpha=0.5):
    col = COLORS[cls]
    m = (cls > 0)[..., None]
    return np.where(m, (rgb * (1 - alpha) + col * alpha).astype(np.uint8), rgb)


def iou_counts(pred, gt):
    r = {}
    for c in range(1, len(CLASSES)):
        p, g = pred == c, gt == c
        r[c] = (int((p & g).sum()), int((p | g).sum()), int(g.sum()))
    return r


def run_carla(seg):
    src = os.path.join(OUT, "carla")
    files = sorted(glob.glob(os.path.join(src, "*_rgb.jpg")))
    res = {"1280x960": (1280, 960), "640x480": (640, 480), "320x240": (320, 240)}
    tot = {k: defaultdict(lambda: [0, 0, 0]) for k in res}
    by_w = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    speed = defaultdict(list)
    os.makedirs(os.path.join(OUT, "carla_overlays"), exist_ok=True)
    for i, f in enumerate(files):
        rgb = np.asarray(Image.open(f).convert("RGB"))
        gt = C_LUT[np.asarray(Image.open(f.replace("_rgb.jpg", "_tag.png")))]
        weather = os.path.basename(f).split("_")[1]
        for name, size in res.items():
            torch.cuda.synchronize()
            t0 = time.time()
            pred = seg(rgb, size)
            torch.cuda.synchronize()
            speed[name].append(time.time() - t0)
            if pred.shape != gt.shape:
                pred = np.asarray(Image.fromarray(pred).resize(gt.shape[::-1], Image.NEAREST))
            for c, (a, b, g) in iou_counts(pred, gt).items():
                t = tot[name][c]
                t[0] += a; t[1] += b; t[2] += g
                if name == "1280x960":
                    t = by_w[weather][c]
                    t[0] += a; t[1] += b; t[2] += g
            if name == "1280x960" and i % 6 == 0:
                img = np.concatenate([rgb, overlay(rgb, pred), overlay(rgb, gt)], 1)
                Image.fromarray(img).resize((1920, 480)).save(
                    os.path.join(OUT, "carla_overlays", os.path.basename(f).replace("_rgb", "_rgb_pred_truth")), quality=88)
    report = {}
    print(f"\nCARLA: {len(files)} frames.  IoU per class (segmenter vs CARLA truth)")
    print(f"{'input size':10s} " + " ".join(f"{c[:13]:>13s}" for c in CLASSES[1:]) + "   s/frame")
    for name in res:
        ious = {CLASSES[c]: (v[0] / v[1] if v[2] else float("nan")) for c, v in sorted(tot[name].items())}
        report[name] = dict(iou=ious, s_per_frame=float(np.median(speed[name])))
        print(f"{name:10s} " + " ".join(f"{x:13.2f}" for x in ious.values()) + f"   {np.median(speed[name]):.2f}")
    print("\nby weather (1280x960):")
    for w, d in sorted(by_w.items()):
        ious = [v[0] / v[1] if v[2] else float("nan") for c, v in sorted(d.items())]
        report["weather_" + w] = dict(zip(CLASSES[1:], ious))
        print(f"{w:16s} " + " ".join(f"{x:13.2f}" for x in ious))
    json.dump(report, open(os.path.join(OUT, "carla_iou.json"), "w"), indent=1)
    print("overlays (camera | segmenter | CARLA truth):", os.path.join(OUT, "carla_overlays"))


def run_tesla(seg, clip):
    out = os.path.join(OUT, "tesla_PRIVATE")
    os.makedirs(out, exist_ok=True)
    w, h = 1280, 960
    p = subprocess.Popen([FF, "-loglevel", "error", "-i", clip, "-vf", "fps=6", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    enc = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * 2 // 2}x{h // 2}",
                            "-r", "6", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                            os.path.join(out, "tesla_seg_overlay.mp4")], stdin=subprocess.PIPE)
    frac, n, t = np.zeros(len(CLASSES)), 0, []
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        t0 = time.time()
        cls = seg(rgb)
        t.append(time.time() - t0)
        frac += np.bincount(cls.ravel(), minlength=len(CLASSES)) / cls.size
        side = np.concatenate([rgb, overlay(rgb, cls, 0.55)], 1)
        side = np.asarray(Image.fromarray(side).resize((w, h // 2)))
        enc.stdin.write(side.tobytes())
        if n % 12 == 0:
            Image.fromarray(side).save(os.path.join(out, f"tesla_{n // 6:03d}s.jpg"), quality=88)
        n += 1
    enc.stdin.close()
    enc.wait()
    frac /= max(n, 1)
    print(f"\nTesla: {n} frames at 6 fps, {np.median(t):.2f} s/frame. Mean share of image per class:")
    for c, f in zip(CLASSES, frac):
        print(f"  {c:20s} {100 * f:5.1f}%")
    print("PRIVATE overlays + video (real footage, don't publish):", out)


if __name__ == "__main__":
    seg = Segmenter()
    if sys.argv[1] == "carla":
        run_carla(seg)
    else:
        run_tesla(seg, sys.argv[2] if len(sys.argv) > 2 else CLIP)
