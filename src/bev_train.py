"""Train the top-down student (Phase D) on cached fused maps (bev_dataset.py) + situation labels (situation.py).

  python bev_train.py --data data\\vision_data_v2 [--data <dagger dir>] [--epochs 25] [--quick]

Samples = windows of WINDOW frames; loss on the last LOSS_LAST frames (they have enough memory behind them).
Weights: standing still at a red light x0.2; rarer situations up-weighted (1/sqrt frequency).
Augmentation (memory training): lines ahead erased for random stretches; occasional dropped frames.
Output: data\\vision_runs\\bev_<stamp>\\ best.pt / last.pt / train_log.csv.  GPU only, no CARLA."""
import argparse
import csv
import json
import os
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from bev_dataset import NOT_SEEN
from bev_model import AUX_NAMES, N_SIT, WINDOW, BevStudent, aux_targets, load_student, scalars
from bev_route import N_RV, ROUTE_BIT, degrade_plan, route_layer, route_vector
from situation import CLASSES, label_episode

LOSS_LAST = 8
W_STEER, W_ACCEL, W_SIT, W_AUX = 1.0, 0.25, 0.2, 0.5
TURN_W = 0.0                                 # --turn-weight: steer loss x (1 + TURN_W * |teacher steer|) against under-steer
ROUTE = False                                # --route: add the GPS-grade satnav curve layer (bev_route.py)
HARD = 0.3                                   # "hard steer" frames for the strength read-out
CY_ROW = 200                                 # car row in the 250-row map (10 m behind at 0.2 m/px)


def num(v):
    return None if v in ("", None) else float(v)


CACHE = "bev.npz"                       # set from --fuser (bev_dataset.cache_name)


def load(dirs):
    eps = []
    for root in dirs:
        for d in sorted(os.listdir(root)):
            ep = os.path.join(root, d)
            if not (d.startswith("ep_") and os.path.isfile(os.path.join(ep, CACHE))):
                continue
            rows, lab = label_episode(ep)
            bev = np.load(os.path.join(ep, CACHE))["bev"]
            assert len(bev) == len(rows), ep
            sc = np.array([scalars(float(r["speed"]), float(r["v_target"]), int(r["command"])) for r in rows], np.float32)
            y = np.array([[float(r["steer"]), float(r["accel"])] for r in rows], np.float32)
            sit = np.array([l["situation"] for l in lab], np.int64)
            aux = np.array([aux_targets(l["curvature"], l["to_junction"], l["stop"], num(r["lead_gap"]),
                                        float(r["lat"]), float(r["head_err"])) for r, l in zip(rows, lab)], np.float32)
            still = np.array([float(r["speed"]) < 0.5 for r in rows])
            route = np.load(os.path.join(ep, "route.npz"))["route"]
            pose = np.array([[float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["route_idx"])] for r in rows])
            eps.append(dict(name=d, bev=bev, sc=sc, cmd=sc[:, 2:].argmax(1), y=y, sit=sit, aux=aux, still=still,
                            route=route, pose=pose,
                            val=zlib.crc32(d.encode()) % 10 == 0))
    return eps


def frame_weights(eps):
    cnt = np.bincount(np.concatenate([e["sit"] for e in eps if not e["val"]]), minlength=N_SIT).astype(float)
    cw = 1.0 / np.sqrt(np.maximum(cnt, 1) / cnt.sum())
    cw /= cw[0]
    for e in eps:
        e["w"] = (cw[e["sit"]] * np.where(e["still"] & (e["sit"] == 10), 0.2, 1.0)).astype(np.float32)
    return cw


def windows(eps, val, stride):
    """(episode index, end frame) for every window; ends every `stride` frames."""
    out = []
    for i, e in enumerate(eps):
        if e["val"] == val:
            out += [(i, t) for t in range(LOSS_LAST - 1, len(e["bev"]), stride)]
    return out


def batch(eps, idx, aug, rng):
    B, T = len(idx), WINDOW
    bev = np.full((B, T, 250, 160), NOT_SEEN, np.uint8)
    sc = np.zeros((B, T, 6 + (N_RV if ROUTE else 0)), np.float32)
    cmd = np.zeros((B, T), np.int64)
    y = np.zeros((B, T, 2), np.float32)
    sit = np.zeros((B, T), np.int64)
    aux = np.zeros((B, T, 6), np.float32)
    w = np.zeros((B, T), np.float32)
    for b, (i, t) in enumerate(idx):
        e = eps[i]
        a = max(0, t - T + 1)
        k = t + 1 - a
        s = slice(T - k, T)
        bev[b, s], sc[b, s, :6], cmd[b, s], y[b, s] = e["bev"][a:t + 1], e["sc"][a:t + 1], e["cmd"][a:t + 1], e["y"][a:t + 1]
        sit[b, s], aux[b, s] = e["sit"][a:t + 1], e["aux"][a:t + 1]
        w[b, T - LOSS_LAST:] = e["w"][t + 1 - LOSS_LAST:t + 1] if t + 1 >= LOSS_LAST else 0
        if aug:
            if rng.random() < 0.4:                            # worn paint: lines ahead vanish for 1-2 s
                L = rng.integers(10, 21)
                s0 = rng.integers(max(0, T - L - LOSS_LAST), T - 1)
                seg = bev[b, s0:s0 + L, :CY_ROW]
                seg[seg == 2] = 1
            drop = rng.random(T) < 0.05                      # dropped frames: the camera saw nothing
            bev[b, drop] = NOT_SEEN
        if ROUTE:                                            # satnav curve: GPS-grade in training, clean in validation
            plan = degrade_plan(k, rng) if aug else [(0.0, 0.0, (0.0, 0.0))] * k
            if plan is not None:
                for j in range(k):
                    px, py, pyaw, pidx = e["pose"][a + j]
                    lat, lon, jit = plan[j]
                    bev[b, T - k + j] += ROUTE_BIT * route_layer(e["route"], int(pidx), px, py, pyaw, lat, lon, jit).astype(np.uint8)
                    sc[b, T - k + j, 6:] = route_vector(e["route"], int(pidx), px, py, pyaw, lat, lon, jit)
    return [torch.from_numpy(v) for v in (bev, sc, cmd, y, sit, aux, w)]


def run(net, eps, idx, dev, opt=None, scaler=None, bs=24, rng=None, max_b=None):
    train = opt is not None
    net.train(train)
    if train:
        idx = [idx[k] for k in rng.permutation(len(idx))]
    tot, n, hit, nsit = np.zeros(4 + len(AUX_NAMES)), 0.0, 0.0, 0.0
    s_pred, s_true = 0.0, 0.0                 # turn strength: student vs teacher on hard-steer frames
    for k in range(0, len(idx), bs):
        if max_b and k // bs >= max_b:
            break
        bev, sc, cmd, y, sit, aux, w = (v.to(dev, non_blocking=True) for v in batch(eps, idx[k:k + bs], train, rng))
        with torch.set_grad_enabled(train), torch.autocast(dev.type, enabled=True):
            d, s, a = net(bev, sc, cmd)
        d, s, a = d.float(), s.float(), a.float()
        ws = w.sum().clamp(min=1)
        w_st = w * (1 + TURN_W * y[..., 0].abs())
        l_st = ((d[..., 0] - y[..., 0]).abs() * w_st).sum() / w_st.sum().clamp(min=1)
        l_ac = ((d[..., 1] - y[..., 1]).abs() * w).sum() / ws
        m = (w > 0).float()
        l_sit = (F.cross_entropy(s.flatten(0, 1), sit.flatten(), reduction="none") * m.flatten()).sum() / m.sum().clamp(min=1)
        l_aux_each = (F.smooth_l1_loss(a, aux, reduction="none") * m[..., None]).sum((0, 1)) / m.sum().clamp(min=1)
        loss = W_STEER * l_st + W_ACCEL * l_ac + W_SIT * l_sit + W_AUX * l_aux_each.sum()
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
        # unweighted steer error on the loss frames (comparable with the camera students' val steer err)
        mm = m.bool()
        tot += np.array([loss.item(), (d[..., 0] - y[..., 0]).abs()[mm].mean().item(), l_ac.item(), l_sit.item(),
                         *l_aux_each.detach().cpu().numpy()])
        n += 1
        hit += (s.argmax(-1) == sit)[mm].float().sum().item()
        nsit += mm.sum().item()
        hm = mm & (y[..., 0].abs() >= HARD)
        s_pred += (d[..., 0] * y[..., 0].sign())[hm].sum().item()
        s_true += y[..., 0].abs()[hm].sum().item()
    run.strength = s_pred / max(s_true, 1e-6)
    return tot / max(n, 1), hit / max(nsit, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--turn-weight", type=float, default=0.0)
    ap.add_argument("--route", action="store_true", help="add the satnav route-curve layer")
    ap.add_argument("--fuser", default="current", help="map memory the caches were built with: current | vote")
    args = ap.parse_args()
    global TURN_W, ROUTE, CACHE
    from bev_dataset import cache_name
    TURN_W, ROUTE, CACHE = args.turn_weight, args.route, cache_name(args.fuser)
    print("map cache:", CACHE, flush=True)
    t0 = time.time()
    eps = load(args.data)
    cw = frame_weights(eps)
    tr, va = windows(eps, False, LOSS_LAST), windows(eps, True, LOSS_LAST)
    print(f"{len(eps)} episodes ({sum(e['val'] for e in eps)} validation), {len(tr)} train / {len(va)} val windows, "
          f"loaded in {time.time() - t0:.0f} s", flush=True)
    print("situation weights:", {CLASSES[c]: round(float(cw[c]), 2) for c in range(N_SIT)}, flush=True)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    net = load_student(args.init, dev, route=ROUTE) if args.init else BevStudent(route=ROUTE).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = 1 if args.quick else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler()
    rng = np.random.default_rng(0)
    run_dir = os.path.join(C.VISION_RUNS_DIR, time.strftime("bev_%Y%m%d_%H%M%S") + ("_quick" if args.quick else ""))
    os.makedirs(run_dir, exist_ok=True)
    json.dump(dict(vars(args), window=WINDOW, loss_last=LOSS_LAST, kind="bev"),
              open(os.path.join(run_dir, "settings.json"), "w"), indent=1)
    best = float("inf")
    with open(os.path.join(run_dir, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_steer", "val_loss", "val_steer", "val_accel", "val_sit_ce", "val_sit_acc",
                      *("val_" + a for a in AUX_NAMES), "minutes"])
        for ep in range(1, epochs + 1):
            mb = 20 if args.quick else None
            trl, _ = run(net, eps, tr, dev, opt, scaler, rng=rng, max_b=mb)
            val, acc = run(net, eps, va, dev, rng=rng, max_b=mb)
            sched.step()
            mins = (time.time() - t0) / 60
            log.writerow([ep, round(trl[0], 4), round(trl[1], 4), *np.round(val[:4], 4), round(acc, 3),
                          *np.round(val[4:], 4), round(mins, 1)])
            f.flush()
            ck = dict(model=net.state_dict(), epoch=ep, val_steer=float(val[1]), settings=dict(vars(args), kind="bev"))
            torch.save(ck, os.path.join(run_dir, "last.pt"))
            tag = ""
            if val[1] < best:
                best = val[1]
                torch.save(ck, os.path.join(run_dir, "best.pt"))
                tag = "  <- best"
            print(f"epoch {ep}/{epochs}: train steer {trl[1]:.4f} | val steer {val[1]:.4f} speed {val[2]:.4f} "
                  f"situation {acc:.0%} | turn strength {run.strength:.2f} | aux {np.round(val[4:], 3).tolist()} | {mins:.0f} min{tag}", flush=True)
    print("student:", os.path.join(run_dir, "best.pt"))


if __name__ == "__main__":
    main()
