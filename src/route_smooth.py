"""Clean + smooth a planned route before anyone drives on it.

CARLA's GlobalRoutePlanner output has three defects (route_dump.py, 15 Nov):
  - the same point is repeated at every junction border (0 m steps -> fake curvature, stalled route index)
  - a lane change is ONE 12.5 m diagonal jump between lanes (two ~16 deg kinks: full-lock steering,
    off-route ends in Town03/04/05)
  - uneven spacing (1.3 - 4 m)
Here: duplicates dropped, each lane change replaced by a smooth S-curve (cubic Hermite, LC_LENGTH_M long,
tangent = lane heading at both ends), then the path is resampled every ROUTE_RESOLUTION m. Heading (yaw)
comes from the point positions. Lane width / RoadOption / junction flag are carried over from the nearest
original point (by arc length), so navigation commands are unchanged.
"""
import math

import numpy as np

import config as C


def _hermite(p0, t0, p1, t1, n):
    s = np.linspace(0.0, 1.0, n)[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def smooth_route(xy, lane_w, opt, junction):
    """xy (n,2), lane_w (n,), opt (n,) RoadOption ints, junction (n,) bools ->
    (route (m,4) = x, y, yaw rad, lane width;  opt (m,);  junction (m,))."""
    xy = np.asarray(xy, dtype=np.float64)
    keep = [0]
    for j in range(1, len(xy)):
        if math.hypot(*(xy[j] - xy[keep[-1]])) > 0.5:
            keep.append(j)
    p = xy[keep]
    src = np.array(keep, dtype=np.float64)              # original index of every polyline point
    if len(p) < 3:
        p = np.vstack([xy[0], xy[-1]]) if len(xy) > 1 else np.vstack([xy[0], xy[0] + [1.0, 0.0]])
        src = np.array([0.0, len(xy) - 1.0])
    # lane changes: a single long segment (planner jumps lanes in one step)
    d = np.diff(p, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    jumps = [j for j in range(len(seg)) if seg[j] > C.LC_JUMP_M]
    pieces, last = [], 0                                # pieces of (points, src)
    for j in jumps:
        a = j
        while a > last and s[j] - s[a] < C.LC_PRE_M:   # start the lane change this far before the jump
            a -= 1
        b = j + 1
        while b < len(p) - 1 and s[b] - s[j + 1] < C.LC_POST_M:
            b += 1
        if a < last:
            continue
        pieces.append((p[last:a], src[last:a]))
        h_a = math.atan2(*(p[a] - p[max(a - 1, 0)])[::-1]) if a > 0 else math.atan2(*(p[a + 1] - p[a])[::-1])
        h_b = math.atan2(*(p[min(b + 1, len(p) - 1)] - p[b])[::-1]) if b < len(p) - 1 else math.atan2(*(p[b] - p[b - 1])[::-1])
        L = math.hypot(*(p[b] - p[a]))
        n = max(3, int(L / 0.5))
        curve = _hermite(p[a], L * np.array([math.cos(h_a), math.sin(h_a)]),
                         p[b], L * np.array([math.cos(h_b), math.sin(h_b)]), n)
        pieces.append((curve[:-1], np.linspace(src[a], src[b], n)[:-1]))
        last = b
    pieces.append((p[last:], src[last:]))
    P = np.vstack([q for q, _ in pieces if len(q)])
    S = np.concatenate([q for _, q in pieces if len(q)])
    # resample every ROUTE_RESOLUTION m
    d = np.diff(P, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    good = np.concatenate([[True], seg > 1e-6])
    P, S = P[good], S[good]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    m = max(2, int(arc[-1] / C.ROUTE_RESOLUTION) + 1)
    t = np.linspace(0.0, arc[-1], m)
    x, y, sr = np.interp(t, arc, P[:, 0]), np.interp(t, arc, P[:, 1]), np.interp(t, arc, S)
    # kinks sharper than any real junction turn (short lane jumps the length test misses): relax locally
    for _ in range(C.KINK_ITERS):
        dx1, dy1 = x[1:-1] - x[:-2], y[1:-1] - y[:-2]
        dx2, dy2 = x[2:] - x[1:-1], y[2:] - y[1:-1]
        k = np.abs(np.arctan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2)) / C.ROUTE_RESOLUTION
        bad = np.where(k > C.KINK_K_MAX)[0] + 1
        if not len(bad):
            break
        for j in bad:
            for q in (j - 1, j, j + 1):
                if 0 < q < m - 1:
                    x[q] += 0.5 * ((x[q - 1] + x[q + 1]) / 2 - x[q])
                    y[q] += 0.5 * ((y[q - 1] + y[q + 1]) / 2 - y[q])
    yaw = np.empty(m)
    yaw[1:-1] = np.arctan2(y[2:] - y[:-2], x[2:] - x[:-2])
    yaw[0] = math.atan2(y[1] - y[0], x[1] - x[0])
    yaw[-1] = math.atan2(y[-1] - y[-2], x[-1] - x[-2])
    k = np.clip(np.rint(sr).astype(int), 0, len(xy) - 1)
    route = np.stack([x, y, yaw, np.asarray(lane_w, dtype=np.float64)[k]], axis=1)
    return route, np.asarray(opt)[k], np.asarray(junction)[k]
