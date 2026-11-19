"""Steady-hand teacher: smooth, human-like steering on straight road.

The PPO teacher weaves on straights (training labels: command wanders +-0.05 around its 2 s average,
~150-220 wiggles/km in CARLA). Real drivers hold the wheel almost still and correct slowly.

On straight road (route ahead from the map, like the curve speed limit) the steering comes from a
long-lookahead pure-pursuit controller (the classic human driver model: aim at a point 1-2 s ahead,
smooth by construction). In bends and junction turns the PPO teacher keeps full control; in between
the two are cross-faded by how straight the road ahead is. Speed always stays with the teacher.
"""
import math

import numpy as np

import config as C


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def straightness(env):
    """1.0 = route straight from STEADY_BEHIND_M behind to STEADY_AHEAD_M ahead, 0.0 = a real bend."""
    res = C.ROUTE_RESOLUTION
    a = max(0, env.idx - int(C.STEADY_BEHIND_M / res))
    b = min(len(env.route), env.idx + int(C.STEADY_AHEAD_M / res) + 3)
    yaw = env.route[a:b, 2]
    if len(yaw) < 3:
        return 0.0
    k = np.abs([wrap_pi(yaw[j + 2] - yaw[j]) for j in range(len(yaw) - 2)]) / (2 * res)
    kmax = float(k.max())
    # lane change ahead: route points step sideways while their heading stays straight -> PPO does it
    pts = env.route[a:b, :2]
    d = np.diff(pts, axis=0)
    side = np.abs(-np.sin(yaw[:-1]) * d[:, 0] + np.cos(yaw[:-1]) * d[:, 1])
    if float(side.max()) > C.STEADY_MAX_SIDE_STEP_M:
        return 0.0
    return float(np.clip((C.STEADY_K_BEND - kmax) / (C.STEADY_K_BEND - C.STEADY_K_STRAIGHT), 0.0, 1.0))


def pure_pursuit(env):
    """Steer command [-1, 1] that drives an arc through the route point a look-ahead distance away."""
    tf = env.vehicle.get_transform()
    x, y, yaw = tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)
    ld = max(C.STEADY_MIN_LOOKAHEAD_M, C.STEADY_LOOKAHEAD_S * float(env.speed))
    j = min(len(env.route) - 1, env.idx + int(round(ld / C.ROUTE_RESOLUTION)))
    tx, ty = env.route[j, 0], env.route[j, 1]
    ld = max(1.0, math.hypot(tx - x, ty - y))
    alpha = wrap_pi(math.atan2(ty - y, tx - x) - yaw)           # CARLA: y to the right, + = target on the right
    delta = math.atan(2.0 * C.STEADY_WHEELBASE_M * math.sin(alpha) / ld)
    return float(np.clip(delta / math.radians(C.STEADY_MAX_STEER_DEG), -1.0, 1.0))


class CleanPath:
    """The route centreline cleaned for tracking: duplicate points (the planner repeats the point at every
    junction border) removed; arc length, tangent heading and curvature (from point POSITIONS) per vertex."""
    def __init__(self, route):
        keep = [0]
        for j in range(1, len(route)):
            if math.hypot(*(route[j, :2] - route[keep[-1], :2])) > 0.5:
                keep.append(j)
        self.p = route[keep, :2].astype(np.float64)
        self.of = np.searchsorted(keep, np.arange(len(route)), side="right") - 1   # route idx -> clean idx
        d = np.diff(self.p, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.seg_h = np.arctan2(d[:, 1], d[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        m = len(self.p)
        tan = np.empty(m)
        k = np.zeros(m)
        tan[0], tan[-1] = self.seg_h[0], self.seg_h[-1]
        for j in range(1, m - 1):
            dh = wrap_pi(self.seg_h[j] - self.seg_h[j - 1])
            tan[j] = self.seg_h[j - 1] + dh / 2
            k[j] = dh / max((self.seg_len[j - 1] + self.seg_len[j]) / 2, 0.5)
        self.tan = tan
        self.k = np.clip(np.convolve(k, np.ones(3) / 3, "same"), -C.TRACK_K_MAX, C.TRACK_K_MAX)

    def project(self, x, y, idx):
        """(lateral offset + = right, tangent heading, arc length) of the point's projection on the path."""
        c = int(self.of[min(idx, len(self.of) - 1)])
        best = None
        for j in range(max(0, c - 3), min(len(self.seg_len), c + 4)):
            L = self.seg_len[j]
            ux, uy = math.cos(self.seg_h[j]), math.sin(self.seg_h[j])
            dx, dy = x - self.p[j, 0], y - self.p[j, 1]
            t = min(max((dx * ux + dy * uy) / L, 0.0), 1.0)
            ex, ey = dx - t * L * ux, dy - t * L * uy
            dist = math.hypot(ex, ey)
            if best is None or dist < best[0]:
                best = (dist, j, t, -uy * dx + ux * dy)
        _, j, t, lat = best
        h = self.tan[j] + t * wrap_pi(self.tan[j + 1] - self.tan[j])
        return lat, h, self.s[j] + t * self.seg_len[j]

    def curvature_at(self, s):
        return float(np.interp(s, self.s, self.k))


def _path(env):
    cache = getattr(env, "_clean_path", None)
    if cache is None or cache[0] is not env.route:
        cache = (env.route, CleanPath(env.route))
        env._clean_path = cache
    return cache[1]


def lane_tracker(env):
    """Precise lane-centre tracking for bends and junction turns (sim telemetry: lane centreline + its
    curvature from the map). Steer = curvature feed-forward taken TRACK_PREVIEW_S ahead (covers the
    ~0.3 s wheel lag, see STEER_SMOOTH) + feedback on lateral offset and heading (Stanley-style).
    The PPO teacher cuts the inside of turns (left turns: wheels in the oncoming lane in ~90% of turns)."""
    tf = env.vehicle.get_transform()
    vx, vy, yaw = tf.location.x, tf.location.y, math.radians(tf.rotation.yaw)
    v = float(env.speed)
    path = _path(env)
    lat, tan, s = path.project(vx, vy, env.idx)
    head = wrap_pi(yaw - tan)                                              # + = pointing right of the path
    kappa = path.curvature_at(s + C.TRACK_PREVIEW_S * v + C.TRACK_PREVIEW_M)
    delta = (math.atan(C.STEADY_WHEELBASE_M * kappa)
             - C.TRACK_K_HEAD * head
             - math.atan(C.TRACK_K_LAT * lat / (v + C.TRACK_V0)))
    return float(np.clip(delta / math.radians(C.STEADY_MAX_STEER_DEG), -1.0, 1.0))


def on_line(env):
    """1.0 = car near the route centre and parallel to it; 0.0 = clearly off (after a push or drift):
    then PPO makes the firm correction, as a driver would; the slow pure pursuit only holds the line."""
    tf = env.vehicle.get_transform()
    wx, wy, wyaw = env.route[env.idx, 0], env.route[env.idx, 1], env.route[env.idx, 2]
    lat = abs(-math.sin(wyaw) * (tf.location.x - wx) + math.cos(wyaw) * (tf.location.y - wy))
    head = abs(math.degrees(wrap_pi(math.radians(tf.rotation.yaw) - wyaw)))
    w_lat = (C.STEADY_LAT_PPO_M - lat) / (C.STEADY_LAT_PPO_M - C.STEADY_LAT_OK_M)
    w_head = (C.STEADY_HEAD_PPO_DEG - head) / (C.STEADY_HEAD_PPO_DEG - C.STEADY_HEAD_OK_DEG)
    return float(np.clip(min(w_lat, w_head), 0.0, 1.0))


class SteadyTeacher:
    """Wraps the PPO teacher (anything with .act(obs)). Needs the env for route + vehicle pose."""
    def __init__(self, teacher, env):
        self.teacher, self.env = teacher, env
        self.last_weight = 0.0

    def act(self, obs):
        a = np.array(self.teacher.act(obs), dtype=np.float32)
        if C.STEADY_BENDS == "track":            # lane tracker everywhere except calm straights
            w = straightness(self.env) * on_line(self.env)
            bend = lane_tracker(self.env)
        else:                                    # "ppo": the PPO teacher steers bends and corrections
            w = straightness(self.env) * on_line(self.env)
            bend = float(a[0])
        self.last_weight = w
        a[0] = np.float32(w * pure_pursuit(self.env) + (1.0 - w) * bend) if w > 0.0 else np.float32(bend)
        return a
