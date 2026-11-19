"""Camera -> top-down (bird's-eye) projection, flat-ground assumption.
Same grid as bev_map.BevMap: car-centred, HEADING-UP, 0.1 m/px, 400 x 500 px (+-20 m sideways, 40 m ahead, 10 m behind).
  proj = BevProjector(cam)              cam = dict(width, height, fov, x, z, pitch)   (config.TESLA_CAM style)
  top = proj.classes(class_map)         camera-view class map -> top-down class map (UNSEEN where the camera can't see)
  img = colorize(top)                   Tesla-style picture
Time stack: shift_rotate(top_prev, d_fwd, d_right, d_yaw_deg) moves an older top-down map into the current car frame."""
import math

import cv2
import numpy as np

UNSEEN = 255
W, H, RES, BEHIND_M = 400, 500, 0.1, 10.0
CX, CY = W / 2, H - BEHIND_M / RES                 # car origin pixel
MAX_RANGE_M = 45.0                                  # further ground is too stretched to trust

# colours (BGR) for our 7 classes: background, road, lane marking, kerb/sidewalk, vehicle, pedestrian, traffic light
STYLE = np.array([(30, 30, 30), (58, 56, 54), (235, 235, 235), (150, 150, 160), (60, 140, 230), (60, 60, 220),
                  (40, 180, 250)], np.uint8)
UNSEEN_C = (14, 14, 14)


def grid_to_vehicle():
    """Vehicle-frame ground coordinates (fwd, right) in metres of every top-down pixel."""
    j, i = np.mgrid[0:H, 0:W].astype(np.float32)
    return (CY - j) * RES, (i - CX) * RES


class BevProjector:
    def __init__(self, cam):
        w, h = cam["width"], cam["height"]
        f = (w / 2) / math.tan(math.radians(cam["fov"]) / 2)
        p = math.radians(cam["pitch"])
        fwd, right = grid_to_vehicle()
        dx, dz = fwd - cam["x"], -cam["z"]               # camera -> ground point
        cf = dx * math.cos(p) + dz * math.sin(p)         # along the (pitched) optical axis
        cu = -dx * math.sin(p) + dz * math.cos(p)        # up in the camera
        ok = (cf > 0.5) & (np.hypot(fwd, right) < MAX_RANGE_M)
        u = np.where(ok, w / 2 + f * right / np.maximum(cf, 1e-3), -1)
        v = np.where(ok, h / 2 - f * cu / np.maximum(cf, 1e-3), -1)
        ok &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        self.mapx = np.where(ok, u, -1).astype(np.float32)
        self.mapy = np.where(ok, v, -1).astype(np.float32)
        self.seen = ok
        self.size = (w, h)

    def classes(self, cls):
        """cls: camera-view class map (any size with the camera's aspect) -> top-down class map."""
        sx, sy = cls.shape[1] / self.size[0], cls.shape[0] / self.size[1]
        out = cv2.remap(cls, self.mapx * sx, self.mapy * sy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=UNSEEN)
        out[~self.seen] = UNSEEN
        return out

    def image(self, rgb):
        sx, sy = rgb.shape[1] / self.size[0], rgb.shape[0] / self.size[1]
        return cv2.remap(rgb, self.mapx * sx, self.mapy * sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def shift_rotate(top, d_fwd, d_right, d_yaw_deg):
    """Move a top-down map made d_fwd / d_right metres and d_yaw_deg earlier into the current car frame.
    (d_* = how far the car moved / turned since that map, in that map's frame; yaw + = clockwise / to the right.)"""
    a = math.radians(d_yaw_deg)
    c, s = math.cos(a), math.sin(a)
    fwd, right = grid_to_vehicle()                       # point in the CURRENT frame
    pf = d_fwd + fwd * c - right * s                      # same point in the OLD frame
    pr = d_right + fwd * s + right * c
    mx = (CX + pr / RES).astype(np.float32)
    my = (CY - pf / RES).astype(np.float32)
    return cv2.remap(top, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=UNSEEN)


def colorize(top, car=True, tesla=True):
    """tesla=True: parking-assist look like bev_map: dark road, white lane lines, a light ROAD-EDGE line where the
    road meets kerb / off-road (instead of a grey sidewalk area), vehicles orange, the rest near-black."""
    img = np.empty(top.shape + (3,), np.uint8)
    img[:] = UNSEEN_C
    m = top != UNSEEN
    img[m] = STYLE[np.minimum(top[m], len(STYLE) - 1)]
    if tesla:
        road = ((top == 1) | (top == 2) | (top == 4)).astype(np.uint8)
        off = ((top == 0) | (top == 3)).astype(np.uint8)
        img[top == 3] = STYLE[0]
        edge = road & cv2.dilate(off, np.ones((3, 3), np.uint8))
        edge = cv2.dilate(edge, np.ones((2, 2), np.uint8)).astype(bool)
        img[edge] = (150, 150, 160)
        img[top == 2] = STYLE[2]
        img[top == 4] = (60, 140, 230)
    if car:
        l, w = 4.7 / RES, 1.9 / RES
        cv2.rectangle(img, (int(CX - w / 2), int(CY - l / 2)), (int(CX + w / 2), int(CY + l / 2)), (220, 220, 220), -1)
    return img


def ground_point(cam, u, v):
    """Camera pixel(s) -> flat-ground point (fwd, right) in metres in the vehicle frame (nan above the horizon)."""
    w, h = cam["width"], cam["height"]
    f = (w / 2) / math.tan(math.radians(cam["fov"]) / 2)
    p = math.radians(cam["pitch"])
    r_, u_ = (np.asarray(u, float) - w / 2) / f, -(np.asarray(v, float) - h / 2) / f
    fwd = math.cos(p) - u_ * math.sin(p)
    up = math.sin(p) + u_ * math.cos(p)
    t = np.where(up < -1e-3, -cam["z"] / np.where(up < -1e-3, up, -1), np.nan)
    return cam["x"] + t * fwd, t * r_


def footprints(cam, cls, length_m=4.5, max_range=MAX_RANGE_M, step=3):
    """Vehicles as footprints: where a vehicle's lowest pixels touch the ground, extended length_m away from us.
    cls = camera-view class map (any size with the camera's aspect).  Returns a top-down bool mask."""
    out = np.zeros((H, W), np.uint8)
    veh = cls == 4
    contact = veh & ~np.vstack([veh[1:], np.zeros((1, veh.shape[1]), bool)])   # vehicle pixel with no vehicle below
    vs, us = np.nonzero(contact[:, ::step])
    us = us * step
    if len(us) == 0:
        return out.astype(bool)
    sx, sy = cam["width"] / cls.shape[1], cam["height"] / cls.shape[0]
    fwd, right = ground_point(cam, us * sx, (vs + 1) * sy)
    ok = np.isfinite(fwd) & (np.hypot(fwd, right) < max_range)
    for fw, rt in zip(fwd[ok], right[ok]):
        dx, dy = fw - cam["x"], rt
        n = math.hypot(dx, dy) or 1.0
        a = (int(CX + rt / RES), int(CY - fw / RES))
        b = (int(CX + (rt + dy / n * length_m) / RES), int(CY - (fw + dx / n * length_m) / RES))
        cv2.line(out, a, b, 1, max(2, int(0.5 / RES)))
    return out.astype(bool)


class Fuser:
    """Time stack: road / lines / kerbs remembered from earlier frames (moved by our own motion);
    vehicles only from the current frame (they move).  Only the near range (trusted) updates the memory."""
    def __init__(self, trust_m=25.0, skip_m=4.5):
        self.mem = np.full((H, W), UNSEEN, np.uint8)
        fwd, right = grid_to_vehicle()
        self.near = (np.hypot(fwd, right) < trust_m) & (fwd > skip_m)   # rows at the bonnet edge are unreliable

    def step(self, top, veh_mask, d_fwd=0.0, d_right=0.0, d_yaw=0.0):
        self.mem = shift_rotate(self.mem, d_fwd, d_right, d_yaw)
        static = top.copy()
        static[(top == 4) | (top == 5) | ~self.near] = UNSEEN      # vehicles / people don't go into memory
        self.mem = np.where(static != UNSEEN, static, self.mem)
        view = self.mem.copy()
        view[veh_mask] = 4
        return view


def shift_rotate_f(ev, d_fwd, d_right, d_yaw_deg):
    """shift_rotate for float evidence maps (H x W x K), bilinear, empty (0) outside."""
    a = math.radians(d_yaw_deg)
    c, s = math.cos(a), math.sin(a)
    fwd, right = grid_to_vehicle()
    pf = d_fwd + fwd * c - right * s
    pr = d_right + fwd * s + right * c
    mx = (CX + pr / RES).astype(np.float32)
    my = (CY - pf / RES).astype(np.float32)
    return cv2.remap(ev, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class VoteFuser:
    """Fuser v2 (Vadim's idea, 18 Nov): instead of 'the latest frame overwrites the memory', every cell collects
    EVIDENCE per class over frames (moved by our own motion), weighted by how far the pixel was when seen (near =
    reliable, far = stretched), and fading (DECAY per frame) so old evidence gives way.  A cell shows the class with
    most evidence once there is enough of it.  Lane lines get extra weight (they are thin and jitter by a pixel or two,
    so their votes spread).  Vehicles still come only from the current frame.  Same interface as Fuser."""
    K = 4                                                        # background, road, lane line, kerb

    def __init__(self, trust_m=30.0, skip_m=4.5, decay=0.85, min_ev=0.6, line_w=2.0, far_w=0.25, stick=0.0):
        self.ev = np.zeros((H, W, self.K), np.float32)
        fwd, right = grid_to_vehicle()
        d = np.hypot(fwd, right)
        self.near = (d < trust_m) & (fwd > skip_m)
        self.w = np.where(self.near, np.clip(1.0 - (d - 8.0) / (trust_m - 8.0) * (1 - far_w), far_w, 1.0), 0)
        self.w = self.w.astype(np.float32)
        self.decay, self.min_ev = decay, min_ev
        self.cw = np.array([1.0, 1.0, line_w, 1.0], np.float32)
        self.stick, self.prev = stick, None                      # stick: keep the shown class unless beaten by this

    def step(self, top, veh_mask, d_fwd=0.0, d_right=0.0, d_yaw=0.0):
        self.ev = shift_rotate_f(self.ev, d_fwd, d_right, d_yaw) * self.decay
        ok = (top < self.K) & self.near
        r, c = np.nonzero(ok)
        self.ev[r, c, top[r, c]] += self.w[r, c]
        score = self.ev * self.cw
        best = score.argmax(-1).astype(np.uint8)
        if self.stick > 0 and self.prev is not None:             # hysteresis: a new class must clearly win
            pv = shift_rotate(self.prev, d_fwd, d_right, d_yaw)
            ok_prev = pv < self.K
            pvc = np.where(ok_prev, pv, 0)
            prev_score = np.take_along_axis(score, pvc[..., None].astype(np.int64), -1)[..., 0]
            keep = ok_prev & (prev_score * (1 + self.stick) >= score.max(-1))
            best = np.where(keep, pvc, best).astype(np.uint8)
        view = np.where(self.ev.sum(-1) >= self.min_ev, best, UNSEEN).astype(np.uint8)
        self.prev = view.copy()
        view[veh_mask] = 4
        return view
