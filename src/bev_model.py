"""Top-down student (Phase D): fused segmented top-down maps -> steer + speed, with a situation read-out.

  map (8 one-hot classes, 250 x 160, heading-up) -> small CNN -> per-frame feature
  + [speed, road cruise speed, satnav command]                  (NO wheel angle, NO yaw rate: copycat lesson)
  -> GRU over the last WINDOW frames (memory: "I am in the middle of a left turn")
  -> driving heads, one per command: steer, accel
  -> read-out heads: situation (12 classes), curvature ahead, metres to junction, metres to stop,
     gap to lead car, lateral offset, heading error.
Training and driving both run the GRU from zero over exactly the last WINDOW frames, so they match."""
import collections
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

N_CLS, N_CMD, N_SIT, N_AUX = 8, 4, 12, 6
WINDOW = 32                                   # 3.2 s at 10 Hz
AUX_NAMES = ["curvature", "to_junction", "stop", "lead_gap", "lat", "head_err"]


def aux_targets(curv, to_j, stop, gap, lat, head):
    """Raw labels -> scaled targets (~-1..1)."""
    return [float(np.clip(curv * 20.0, -1.5, 1.5)), min(to_j, 50.0) / 50.0, min(stop, 50.0) / 50.0,
            (min(gap, 30.0) if gap is not None else 30.0) / 30.0, float(np.clip(lat, -2, 2)), float(np.clip(head * 4, -2, 2))]


def scalars(speed, v_target, cmd):
    s = [speed / C.TARGET_SPEED, (v_target or C.TARGET_SPEED) / C.TARGET_SPEED]
    oh = [0.0] * N_CMD
    oh[cmd] = 1.0
    return s + oh


N_SCAL = 2 + N_CMD


class BevEncoder(nn.Module):
    """route=True: one more input layer, the satnav route curve (bit 3 of the map, see bev_route.py)."""
    def __init__(self, feat=256, route=False):
        super().__init__()
        self.route = route
        def blk(i, o, k=3, s=2):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(blk(N_CLS + int(route), 32, 5), blk(32, 64), blk(64, 128), blk(128, 128),
                                  nn.Conv2d(128, 32, 1), nn.ReLU(inplace=True))          # -> 32 x 16 x 10
        self.fc = nn.Sequential(nn.Linear(32 * 16 * 10, feat), nn.ReLU(inplace=True))

    def forward(self, cls):
        """cls: N x H x W uint8/long class maps -> N x feat."""
        dt = torch.float16 if torch.is_autocast_enabled() else torch.float32
        r = None
        if self.route:
            r, cls = (cls >= 8)[:, None].to(dt), cls % 8
        x = (cls[:, None] == torch.arange(N_CLS, device=cls.device, dtype=cls.dtype)[None, :, None, None]).to(dt)
        if r is not None:                                                    # one-hot (+ route), light on memory
            x = torch.cat([x, r], 1)
        return self.fc(self.conv(x).flatten(1))


class BevStudent(nn.Module):
    """route=True: the satnav curve comes in twice - as a map layer AND as N_RV numbers appended to the scalars
    (sc = [N_SCAL scalars | N_RV route vector]); the vector feeds the memory and the steering heads directly."""
    def __init__(self, feat=256, hid=256, route=False):
        super().__init__()
        from bev_route import N_RV
        self.route = route
        rv = 64 if route else 0
        self.enc = BevEncoder(feat, route)
        self.scal = nn.Sequential(nn.Linear(N_SCAL, 32), nn.ReLU(inplace=True))
        if route:
            self.rv = nn.Sequential(nn.Linear(N_RV, 64), nn.ReLU(inplace=True), nn.Linear(64, rv), nn.ReLU(inplace=True))
        self.gru = nn.GRU(feat + 32 + rv, hid, batch_first=True)
        self.drive = nn.ModuleList([nn.Sequential(nn.Linear(hid + feat + rv, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
                                    for _ in range(N_CMD)])
        self.sit = nn.Linear(hid, N_SIT)
        self.aux = nn.Linear(hid, N_AUX)

    def step_features(self, cls):
        return self.enc(cls)

    def forward_feats(self, f, sc, cmd):
        """f: B x T x feat, sc: B x T x N_SCAL, cmd: B x T long -> drive B x T x 2, sit B x T x 12, aux B x T x 6."""
        if self.route:
            r = self.rv(sc[..., N_SCAL:])
            h, _ = self.gru(torch.cat([f, self.scal(sc[..., :N_SCAL]), r], -1))
            z = torch.cat([h, f, r], -1)                                # memory + what is seen now + where to go
        else:
            h, _ = self.gru(torch.cat([f, self.scal(sc), ], -1))
            z = torch.cat([h, f], -1)                                   # memory + what is seen now
        d = torch.stack([m(z) for m in self.drive], 2)                  # B x T x N_CMD x 2
        d = torch.gather(d, 2, cmd[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        return torch.tanh(d), self.sit(h), self.aux(h)

    def forward(self, cls, sc, cmd):
        B, T = cls.shape[:2]
        f = self.enc(cls.flatten(0, 1)).view(B, T, -1)
        return self.forward_feats(f, sc, cmd)


def load_student(path, dev, route=None):
    """Checkpoint -> BevStudent.  route=None: as saved; route=True on a no-route checkpoint: add the route layer
    with zero weights (starts out driving exactly like the old student)."""
    sd = torch.load(path, map_location=dev, weights_only=False)["model"]
    has = "rv.0.weight" in sd
    net = BevStudent(route=has if route is None else route).to(dev)
    if net.route and not has:
        w = sd["enc.conv.0.0.weight"]
        sd["enc.conv.0.0.weight"] = torch.cat([w, torch.zeros_like(w[:, :1])], 1)
    net.load_state_dict(sd)
    return net


class BevStudentDriver:
    """Live driving: camera RGB -> segmenter -> fused top-down -> student.  needs_pose = True (ego motion for the
    map memory; the same motion the Tesla gets from speed + heading telemetry)."""
    needs_pose = True

    def __init__(self, path):
        from seg_check import Segmenter
        self.dev = torch.device("cuda")
        self.net = load_student(path, self.dev)
        self.net.eval()
        st = torch.load(path, map_location="cpu", weights_only=False).get("settings", {})
        self.fuser = st.get("fuser", "current")               # map memory it was trained with (bev_dataset.FUSERS)
        self.gps = (0.0, 0.0)                             # fixed GPS error (sideways, along) for robustness tests
        self.seg = Segmenter()
        self.reset()

    def reset(self, cam=None):
        """cam: this episode's camera (dict like config.TESLA_CAM); None = config.TESLA_CAM (no jitter)."""
        self.cam = cam
        self.bev = None
        self.last_cls = None                              # last camera-view class map (DAgger saves it)
        self.feats, self.scal, self.cmds = (collections.deque(maxlen=WINDOW) for _ in range(3))
        self.last = None                                  # (situation probs, aux, top-down map) for narration / clips

    @torch.no_grad()
    def act(self, rgb, speed, wheel, cmd, v_target=None, pose=None, route=None):
        """route: (planned route array, current index) - used only by a route-curve student."""
        from bev_dataset import make_input
        if self.bev is None:
            self.bev = make_input(dict(self.cam or C.TESLA_CAM), self.fuser)
        self.last_cls = self.seg(rgb)
        top = self.bev.step(self.last_cls, *pose)
        rv = None
        if self.net.route:
            from bev_route import ROUTE_BIT, route_layer, route_vector
            r, i = route if route is not None else (None, 0)
            rv = route_vector(r, i, *pose, lat=self.gps[0], lon=self.gps[1])
            if r is not None:
                top = top + ROUTE_BIT * route_layer(r, i, *pose, lat=self.gps[0], lon=self.gps[1]).astype(np.uint8)
        f = self.net.step_features(torch.from_numpy(top)[None].to(self.dev))
        self.feats.append(f[0])
        sc = scalars(speed, v_target, cmd) + ([] if rv is None else rv.tolist())
        self.scal.append(torch.tensor(sc, device=self.dev))
        self.cmds.append(cmd)
        d, s, a = self.net.forward_feats(torch.stack(list(self.feats))[None], torch.stack(list(self.scal))[None],
                                         torch.tensor([list(self.cmds)], device=self.dev))
        self.last = (torch.softmax(s[0, -1], -1).cpu().numpy(), a[0, -1].cpu().numpy(), top)
        return d[0, -1].float().cpu().numpy().astype(np.float32)


class Narrator:
    """Situation read-out -> one steady on-screen line (smoothed over ~0.5 s so it doesn't flicker)."""
    def __init__(self, alpha=0.2):
        from situation import NARRATE
        self.names, self.alpha, self.p = NARRATE, alpha, None

    def line(self, last):
        if last is None:
            return ""
        probs, aux, _ = last
        self.p = probs if self.p is None else (1 - self.alpha) * self.p + self.alpha * probs
        c = int(self.p.argmax())
        text = self.names[c]
        stop, to_j, gap = aux[2] * 50.0, aux[1] * 50.0, aux[3] * 30.0
        if c in (10, 11) and stop < 40:
            text += f"  ·  stop line {max(stop, 0):.0f} m"
        elif c == 3 and to_j < 45:
            text += f"  ·  junction in {max(to_j, 0):.0f} m"
        elif c == 9 and gap < 28:
            text += f"  ·  gap {max(gap, 0):.0f} m"
        return text


def route_in_view(route, idx, x, y, yaw_deg, ahead_m=40.0, res=2.0):
    """Planned route points ahead (world x, y) -> (fwd, right) metres in our heading-up car frame."""
    a = math.radians(yaw_deg)
    pts = np.asarray(route[max(0, idx - 1):idx + int(ahead_m / res) + 1], np.float32)[:, :2]
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.stack([dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)], 1)


def topdown_picture(top, w=270, path=None):
    """Student's input map (classes 0-7) -> Tesla-style RGB picture, heading-up, with our car drawn in.
    path: intended route as (fwd, right) metres (satnav line for viewers; the student itself only gets the command)."""
    import cv2
    from bev_dataset import NOT_SEEN
    from bev_project import UNSEEN, colorize
    m = top % 8                                                # drop the route-curve bit (student input only)
    m[m == NOT_SEEN] = UNSEEN
    img = colorize(m, car=False)[:, :, ::-1]                   # BGR -> RGB
    h = int(round(w * img.shape[0] / img.shape[1]))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    s = w / 160.0                                              # px per map cell (0.2 m)
    cx, cy = 80 * s, 200 * s                                   # our car: centre column, 10 m from the bottom
    l, b = 4.7 / 0.2 * s, 1.9 / 0.2 * s
    if path is not None and len(path) > 1:
        p = np.stack([cx + path[:, 1] / 0.2 * s, cy - path[:, 0] / 0.2 * s], 1)
        inside = (path[:, 0] > -1.0) & (path[:, 0] < 38.0) & (np.abs(path[:, 1]) < 14.0)
        stop = np.argmin(inside[1:]) + 1 if not inside[1:].all() else len(p)   # end at the first exit from the view
        p = p[:stop][inside[:stop]]
        if len(p) > 1:
            q = p.astype(np.int32)
            over = img.copy()
            cv2.polylines(over, [q], False, (60, 150, 255), max(3, int(1.2 / 0.2 * s / 2)), cv2.LINE_AA)
            img = cv2.addWeighted(over, 0.75, img, 0.25, 0)
            tip, base = p[-1], p[max(0, len(p) - 3)]
            d = tip - base
            n = float(np.hypot(*d))
            if n > 1e-3:                                       # arrowhead in the intended direction
                d /= n
                L, Wd = 9 * s, 6 * s
                left = tip - d * L + np.array([-d[1], d[0]]) * Wd
                right = tip - d * L - np.array([-d[1], d[0]]) * Wd
                cv2.fillConvexPoly(img, np.array([tip, left, right], np.int32), (60, 150, 255), cv2.LINE_AA)
    cv2.rectangle(img, (int(cx - b / 2), int(cy - l / 2)), (int(cx + b / 2), int(cy + l / 2)), (230, 230, 230), -1)
    return np.ascontiguousarray(img)
