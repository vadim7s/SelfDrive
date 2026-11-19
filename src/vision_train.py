"""Train the vision student on collected frames (behaviour cloning of the teacher).

    python vision_train.py
    ... --data <dir> --data <dir2>   several datasets (e.g. original + DAgger rounds)
    ... --quick                      tiny smoke run (few batches, 1 epoch)

Output: data\\vision_runs\\student_<stamp>\\ best.pt (lowest validation steer error),
last.pt, train_log.csv, TensorBoard logs. Validation = 10% of the routes (whole routes held out).
No CARLA involved (uses the GPU only).
"""
import argparse
import csv
import json
import os
import random
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config as C
from vision_model import IN_H, IN_W, CmdDistance, Student, measurements, n_meas

W_STEER, W_ACCEL = 1.0, 0.25          # steering matters most (speed is mostly the curve limit)


def load_index(dirs):
    items, n_eps = [], 0
    for root in dirs:
        for d in sorted(os.listdir(root)):
            ep = os.path.join(root, d)
            if not d.startswith("ep_") or d.endswith(".tmp") or not os.path.isfile(os.path.join(ep, "labels.csv")):
                continue
            n_eps += 1
            val = zlib.crc32(d.encode()) % 10 == 0          # stable 10% of routes held out
            with open(os.path.join(ep, "labels.csv"), encoding="utf-8") as f:
                track = CmdDistance()                       # metres to the manoeuvre, rebuilt from speed
                for r in csv.DictReader(f):
                    cmd, speed = int(r["command"]), float(r["speed"])
                    d = track.step(cmd, speed)
                    items.append((os.path.join(ep, f"{int(r['frame']):06d}.jpg"), float(r["steer"]),
                                  float(r["accel"]), speed, float(r["wheel"]), cmd, val,
                                  float("nan") if d is None else d,
                                  float(r.get("v_target") or C.TARGET_SPEED)))   # old data: town cruise speed
    return items, n_eps


SWAP_CMD = {0: 0, 1: 2, 2: 1, 3: 3}          # mirrored image: a left turn becomes a right turn


def real_camera_aug(im):
    """Sim-to-real: real dashcams differ in zoom, mounting roll, colour, noise and compression."""
    import io
    from PIL import Image
    w, h = im.size
    if random.random() < 0.5:                                # zoom 0-10% (FOV / mounting differences)
        z = random.uniform(0.9, 1.0)
        cw, ch = w * z, h * z
        x0 = (w - cw) / 2
        y0 = float(np.clip((h - ch) / 2 + random.uniform(-0.03, 0.03) * h, 0, h - ch))   # + small pitch change
        im = im.resize((w, h), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
    if random.random() < 0.3:                                # camera roll +-2 deg
        im = im.rotate(random.uniform(-2, 2), Image.BILINEAR, expand=False).crop((0, 0, w, h))
    a = np.asarray(im, dtype=np.float32)
    if random.random() < 0.5:                                # white balance / tint
        a = a * np.random.uniform(0.9, 1.1, 3)
    if random.random() < 0.3:                                # gamma (exposure curve)
        a = 255.0 * (np.clip(a, 0, 255) / 255.0) ** random.uniform(0.75, 1.3)
    if random.random() < 0.3:                                # sensor noise
        a = a + np.random.normal(0, random.uniform(2, 8), a.shape)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.4:                                # stronger compression (dashcam H.264)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=random.randint(30, 80))
        im = Image.open(b).convert("RGB")
    return im


def balance_weights(items):
    """Sampling weights ~ 1/sqrt(frequency) of the |steer| bin: sharp turns are rare in the data, so a
    plain average teaches the student to under-steer them (Town02 trace: 0.25 where the teacher used 0.6)."""
    bins = np.array([0.05, 0.1, 0.2, 0.35])
    b = np.digitize(np.abs([it[1] for it in items]), bins)
    freq = np.bincount(b, minlength=len(bins) + 1).astype(float)
    w = 1.0 / np.sqrt(freq[b] / len(items))
    print("steer bins |s|<.05/.1/.2/.35/+ :", freq.astype(int).tolist(), "-> relative weights",
          np.round(1.0 / np.sqrt(freq / freq[0]), 2).tolist(), flush=True)
    return torch.as_tensor(w, dtype=torch.double)


class Frames(Dataset):
    def __init__(self, items, train, strong_aug=False, mirror=0.0, no_wheel=False, cmd_dist=False, road_speed=False):
        self.items, self.train, self.strong_aug, self.mirror = items, train, strong_aug, mirror
        self.no_wheel, self.cmd_dist, self.road_speed = no_wheel, cmd_dist, road_speed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image, ImageEnhance, ImageFilter
        path, steer, accel, speed, wheel, cmd, _, dist, v_target = self.items[i]
        im = Image.open(path).convert("RGB").resize((IN_W, IN_H), Image.BILINEAR)
        if self.train:                                       # lighting / camera-quality variety
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
            im = ImageEnhance.Color(im).enhance(random.uniform(0.6, 1.4))
            if random.random() < 0.2:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
            if self.strong_aug:
                im = real_camera_aug(im)
            if self.mirror and random.random() < self.mirror:   # left-hand traffic (e.g. Australia)
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                steer, wheel, cmd = -steer, -wheel, SWAP_CMD[cmd]
        img = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)
        if self.no_wheel:                                    # camera decides alone (no copying the lagging wheel)
            wheel = 0.0
        meas = torch.tensor(measurements(speed, wheel, dist if self.cmd_dist else None,
                                         v_target if self.road_speed else None), dtype=torch.float32)
        y = torch.tensor([steer, accel], dtype=torch.float32)
        return img, meas, torch.tensor(cmd), y


def run_epoch(net, loader, dev, opt=None, scaler=None, max_batches=None):
    train = opt is not None
    net.train(train)
    tot = np.zeros(3)
    n = 0
    for b, (img, meas, cmd, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        img = img.to(dev, non_blocking=True).float() / 255.0
        meas, cmd, y = meas.to(dev), cmd.to(dev), y.to(dev)
        with torch.set_grad_enabled(train), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out = net(img, meas, cmd).float()
            l_s = F.l1_loss(out[:, 0], y[:, 0])
            l_a = F.l1_loss(out[:, 1], y[:, 1])
            loss = W_STEER * l_s + W_ACCEL * l_a
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        k = len(y)
        tot += np.array([loss.item(), l_s.item(), l_a.item()]) * k
        n += k
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=3)        # i7-7700: 4 cores
    ap.add_argument("--init", default=None, help="start from this student .pt (e.g. for DAgger rounds)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strong-aug", action="store_true", help="sim-to-real camera augmentation")
    ap.add_argument("--mirror", type=float, default=0.0, help="chance to mirror a frame (left-hand traffic)")
    ap.add_argument("--balance", action="store_true", help="oversample sharp-steering frames")
    ap.add_argument("--no-wheel", action="store_true", help="wheel-angle input always 0 (StudentDriver follows)")
    ap.add_argument("--cmd-dist", action="store_true",
                    help="extra input: metres to the announced turn (fixes turning too early); "
                         "frames whose distance is unknown are dropped")
    ap.add_argument("--road-speed", action="store_true",
                    help="extra input: cruise speed from the road's speed limit (speed labels are relative to it)")
    args = ap.parse_args()
    dirs = args.data or [C.VISION_DATA_DIR]

    items, n_eps = load_index(dirs)
    if args.cmd_dist:
        n0 = len(items)
        items = [it for it in items if not np.isnan(it[7])]
        print(f"--cmd-dist: dropped {n0 - len(items)} frames whose turn distance is unknown "
              f"(announcement began before the first saved frame)", flush=True)
    tr = [it for it in items if not it[6]]
    va = [it for it in items if it[6]]
    if args.quick:
        random.shuffle(tr)
        tr, va, args.epochs = tr[:2048], va[:512] or tr[:512], 1
    if not tr or not va:
        raise SystemExit(f"not enough data in {dirs}: {len(tr)} train / {len(va)} val frames")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{n_eps} routes, {len(tr)} train / {len(va)} validation frames, device {dev}", flush=True)

    run = os.path.join(C.VISION_RUNS_DIR, time.strftime("student_%Y%m%d_%H%M%S") + ("_quick" if args.quick else ""))
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(dict(vars(args), data=dirs, routes=n_eps, train_frames=len(tr), val_frames=len(va),
                       input=[IN_W, IN_H], w_steer=W_STEER, w_accel=W_ACCEL), f, indent=1)

    sampler = None
    if args.balance:
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(balance_weights(tr), num_samples=len(tr), replacement=True)
    dl_tr = DataLoader(Frames(tr, True, args.strong_aug, args.mirror, args.no_wheel, args.cmd_dist, args.road_speed), args.batch,
                       shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                       pin_memory=dev.type == "cuda", persistent_workers=args.workers > 0, drop_last=True)
    dl_va = DataLoader(Frames(va, False, no_wheel=args.no_wheel, cmd_dist=args.cmd_dist, road_speed=args.road_speed), args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=dev.type == "cuda", persistent_workers=args.workers > 0)

    net = Student(pretrained=args.init is None, n_meas=n_meas(vars(args))).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(run)

    best = float("inf")
    with open(os.path.join(run, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_steer_l1", "val_loss", "val_steer_l1", "val_accel_l1", "minutes"])
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            mb = 16 if args.quick else None
            trl = run_epoch(net, dl_tr, dev, opt, scaler, max_batches=mb)
            val = run_epoch(net, dl_va, dev, max_batches=mb)
            sched.step()
            mins = (time.time() - t0) / 60
            log.writerow([ep, *np.round(trl[:2], 5), *np.round(val, 5), round(mins, 1)])
            f.flush()
            for k, v in (("train/loss", trl[0]), ("train/steer_l1", trl[1]), ("val/loss", val[0]),
                         ("val/steer_l1", val[1]), ("val/accel_l1", val[2])):
                tb.add_scalar(k, v, ep)
            ck = dict(model=net.state_dict(), epoch=ep, val_steer_l1=float(val[1]), settings=vars(args))
            torch.save(ck, os.path.join(run, "last.pt"))
            tag = ""
            if val[1] < best:
                best = val[1]
                torch.save(ck, os.path.join(run, "best.pt"))
                tag = "  <- best"
            print(f"epoch {ep}/{args.epochs}: train steer err {trl[1]:.4f} | val steer err {val[1]:.4f} "
                  f"speed err {val[2]:.4f} | {mins:.0f} min{tag}", flush=True)
    tb.close()
    print("student:", os.path.join(run, "best.pt"))


if __name__ == "__main__":
    main()
