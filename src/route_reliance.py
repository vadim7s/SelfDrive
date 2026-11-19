"""Does a route-curve student actually USE the curve, and does vision still win for lane position?  Offline, no CARLA.
Held-out teacher drives, steering predicted with: clean curve / no curve / curve shifted 1 m right (GPS error).
  python route_reliance.py <student .pt> [data dir]
Healthy: the curve raises turn strength (hard turns) and changes steering at junctions; a 1 m shift moves steering
only a little (vision holds the lane)."""
import os
import config as C
import sys

import numpy as np
import torch

from bev_model import WINDOW, load_student
from bev_route import N_RV, ROUTE_BIT, route_layer, route_vector
from bev_train import load
from situation import CLASSES


@torch.no_grad()
def predict(net, e, mode, dev):
    bev = e["bev"].copy()
    rv = np.zeros((len(bev), N_RV), np.float32)
    if mode != "none":
        lat = 1.0 if mode == "shift" else 0.0
        for t in range(len(bev)):
            x, y, yaw, idx = e["pose"][t]
            bev[t] += ROUTE_BIT * route_layer(e["route"], int(idx), x, y, yaw, lat).astype(np.uint8)
            rv[t] = route_vector(e["route"], int(idx), x, y, yaw, lat)
    b = torch.from_numpy(bev).to(dev)
    with torch.autocast("cuda"):
        f = torch.cat([net.enc(b[i:i + 256]) for i in range(0, len(b), 256)]).float()
    sc = torch.from_numpy(np.concatenate([e["sc"], rv], 1)).to(dev)
    cmd = torch.from_numpy(e["cmd"]).to(dev)
    out = np.zeros(len(b), np.float32)
    for t0 in range(0, len(b), 128):
        ts = range(t0, min(t0 + 128, len(b)))
        idx = torch.tensor([[max(0, t - WINDOW + 1 + j) for j in range(WINDOW)] for t in ts], device=dev)
        d, _, _ = net.forward_feats(f[idx], sc[idx], cmd[idx])
        out[t0:t0 + len(ts)] = d[:, -1, 0].cpu().numpy()
    return out


def main():
    dev = torch.device("cuda")
    net = load_student(sys.argv[1], dev).eval()
    import bev_train
    from bev_dataset import cache_name
    bev_train.CACHE = cache_name(torch.load(sys.argv[1], map_location="cpu", weights_only=False)
                                 .get("settings", {}).get("fuser", "current"))
    assert net.route, "not a route-curve student"
    eps = [e for e in load([sys.argv[2] if len(sys.argv) > 2 else os.path.join(C.DATA_ROOT, "vision_data_v2")]) if e["val"]]
    y = np.concatenate([e["y"][:, 0] for e in eps])
    sit = np.concatenate([e["sit"] for e in eps])
    P = {m: np.concatenate([predict(net, e, m, dev) for e in eps]) for m in ("clean", "none", "shift")}
    hard = np.abs(y) >= 0.3
    for m, p in P.items():
        print(f"{m:6s}: L1 {np.abs(p - y).mean():.4f} | turn strength {(p[hard] * np.sign(y[hard])).mean() / np.abs(y[hard]).mean():.2f}")
    print(f"mean |steer change|  curve vs none {np.abs(P['clean'] - P['none']).mean():.4f} | "
          f"1 m shift {np.abs(P['shift'] - P['clean']).mean():.4f} (signed {np.mean(P['shift'] - P['clean']):+.4f}, + = right)")
    for c in np.unique(sit):
        m = sit == c
        if m.sum() > 100:
            print(f"  {CLASSES[c]:22s} n={m.sum():5d}  curve effect {np.abs(P['clean'] - P['none'])[m].mean():.4f}"
                  f"  1 m shift effect {np.abs(P['shift'] - P['clean'])[m].mean():.4f}")


if __name__ == "__main__":
    main()
