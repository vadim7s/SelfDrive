"""Satnav route curve as a layer on the top-down map (Phase D round 4).

A real car's navigation knows the shape of the road ahead (how sharp the turn is, where the exit is), but only to
GPS / map accuracy: a few metres sideways and along the road.  So the student gets a GPS-GRADE curve, not the exact
lane the teacher drives (an exact curve would just be copied - the copycat lesson - and fail on real GPS):
  - points every 4 m (road-level shape, not lane-level detail)
  - training: a slowly drifting sideways offset (+-1.5 m), an along-road offset (+-3 m), small per-frame jitter,
    and in 20% of windows no curve at all (just the left / right command)
  - driving: a fixed offset can be set to test robustness (--gps-offset).
Encoding: the layer rides in bit 3 of the uint8 map (map = classes 0-7 | 8 * route), so caches, batches and the
live driver keep one tensor.  Grid = the student's map: 250 x 160 at 0.2 m, car at row 200, column 80."""
import math

import cv2
import numpy as np

RES, CAR_ROW, CAR_COL = 0.2, 200, 80
ROUTE_BIT = 8
SPACING = 2                       # every 2nd route point (route points are 2 m apart) -> 4 m
BEHIND, AHEAD = 3, 26             # route points behind / ahead of the current one (~6 m back, ~52 m on)
THICK = 3                         # px (0.6 m)

# the same curve as numbers (straight into the memory + steering heads; a thin line on the map alone was ignored):
# 8 points along the route ahead (arc length, m), each (fwd, right) / 20 in the car frame, + a "route present" flag
RV_ARC = (4, 8, 12, 16, 20, 26, 32, 40)
N_RV = 2 * len(RV_ARC) + 1

# training degradation
P_DROP, LAT_MAX, LAT_DRIFT, LON_MAX, JITTER = 0.2, 1.5, 0.02, 3.0, 0.15


def route_layer(route, idx, x, y, yaw_deg, lat=0.0, lon=0.0, jit=(0.0, 0.0)):
    """route: n x >=2 (world x, y, ...), idx: current route index, pose of our car.
    GPS error moves where we THINK we are, so the curve shifts as a whole in the car frame:
    lat: + = curve drawn further right, lon: + = curve drawn closer (we think we are further on), jit: (fwd, right).
    -> 250 x 160 bool."""
    out = np.zeros((250, 160), np.uint8)
    route = np.asarray(route)
    k0, k1 = max(0, idx - BEHIND), min(len(route), idx + AHEAD + 1)
    if k1 - k0 < 2:
        return out.astype(bool)
    pts = route[k0:k1:SPACING, :2]
    if (k1 - 1 - k0) % SPACING:
        pts = np.vstack([pts, route[k1 - 1:k1, :2]])
    a = math.radians(yaw_deg)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    fwd = dx * math.cos(a) + dy * math.sin(a) - lon + jit[0]
    right = -dx * math.sin(a) + dy * math.cos(a) + lat + jit[1]
    q = np.stack([CAR_COL + right / RES, CAR_ROW - fwd / RES], 1)
    q = np.clip(q, -1000, 1000).round().astype(np.int32)
    cv2.polylines(out, [q], False, 1, THICK)
    return out.astype(bool)


def degrade_plan(T, rng):
    """Per-window GPS error: None (no curve) or per-frame (lat, lon, jitter) for T frames."""
    if rng.random() < P_DROP:
        return None
    lat0, drift = rng.uniform(-LAT_MAX, LAT_MAX), rng.uniform(-LAT_DRIFT, LAT_DRIFT)
    lat = np.clip(lat0 + drift * np.arange(T), -2.0, 2.0)
    lon = rng.uniform(-LON_MAX, LON_MAX)
    jit = rng.normal(0, JITTER, (T, 2))
    return [(float(lat[t]), lon, tuple(jit[t])) for t in range(T)]


def route_vector(route, idx, x, y, yaw_deg, lat=0.0, lon=0.0, jit=(0.0, 0.0)):
    """The curve as N_RV numbers, with the same GPS error convention as route_layer.  None route -> all zeros."""
    v = np.zeros(N_RV, np.float32)
    if route is None:
        return v
    route = np.asarray(route)
    k = np.minimum(idx + np.array(RV_ARC) // 2, len(route) - 1)        # route points are 2 m apart
    a = math.radians(yaw_deg)
    dx, dy = route[k, 0] - x, route[k, 1] - y
    v[0:-1:2] = (dx * math.cos(a) + dy * math.sin(a) - lon + jit[0]) / 20.0
    v[1:-1:2] = (-dx * math.sin(a) + dy * math.cos(a) + lat + jit[1]) / 20.0
    v[-1] = 1.0
    return v
