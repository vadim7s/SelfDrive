"""Tesla-style top-down map from CARLA's road files (no CARLA server needed).
Car-centred, HEADING-UP: our car is fixed pointing up, the world rotates around it.
Focus: lane markings (solid / dashed, white / yellow) and road boundaries; drivable area as dark grey.
  python bev_map.py build                 -> cache every town's road geometry (data\\bev_maps)
  python bev_map.py demo <trace.csv> [episode]  -> map video along a recorded drive (x, y per step)
Library: BevMap(town).render(x, y, yaw_deg, route=None, cars=None) -> HxWx3 uint8"""
import config as C
import csv
import math
import os
import sys

import cv2
import numpy as np

XODR = os.path.join(C.CARLA_ROOT, "CarlaUE4", "Content", "Carla", "Maps", "OpenDrive")
CACHE = os.path.join(C.DATA_ROOT, "bev_maps")
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
LEVEL_M = 4.0                                 # roads more than this above/below us are another level (bridge)
DASH_M = 3.0                                  # dashed marking: 3 m line, 3 m gap

# style (BGR for cv2)
BG = (18, 18, 18)
ROAD = (58, 56, 54)
WHITE = (235, 235, 235)
YELLOW = (40, 200, 240)
EDGE = (150, 150, 160)
ROUTE = (255, 140, 40)
CAR = (220, 220, 220)
OTHER = (200, 120, 60)
SOLID, DASHED, EDGE_K = 0, 1, 2


def build(town):
    import carla
    m = carla.Map(town, open(os.path.join(XODR, town + ".xodr"), encoding="utf-8").read())
    quads, lines, kinds, cols, qz, lz = [], [], [], [], [], []
    seen = set()                              # a boundary shared by two lanes is drawn once

    def side_pts(wp, sign):
        r = wp.transform.get_right_vector()
        loc, h = wp.transform.location, sign * wp.lane_width / 2
        return (loc.x + r.x * h, loc.y + r.y * h)

    LT = carla.LaneType
    SURF = {LT.Driving, LT.Parking, LT.Shoulder, LT.Bidirectional, LT.Biking}   # road surface up to the kerb

    def is_edge(wp, sign):
        nb = wp.get_right_lane() if sign > 0 else wp.get_left_lane()
        return nb is None or nb.lane_type not in SURF

    wps = m.generate_waypoints(1.0)
    lanes, keys = list(wps), set()
    for wp in wps:                            # add parking / shoulder / bike lanes beside the driving lanes
        for step in (lambda w: w.get_right_lane(), lambda w: w.get_left_lane()):
            nb = step(wp)
            for _ in range(4):
                if nb is None or nb.lane_type not in SURF or nb.lane_type == LT.Driving:
                    break
                k = (nb.road_id, nb.section_id, nb.lane_id, round(nb.s))
                if k not in keys:
                    keys.add(k)
                    lanes.append(nb)
                nb = step(nb)
    jboxes, jids = [], set()
    for wp in wps:
        if wp.is_junction and wp.get_junction().id not in jids:
            jids.add(wp.get_junction().id)
            jboxes.append(wp.get_junction().bounding_box)

    def in_junction(a, b):
        p = carla.Location(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2, z=0.0)
        return any(bb.contains(carla.Location(x=p.x, y=p.y, z=bb.location.z), carla.Transform()) for bb in jboxes)

    for wp in lanes:
        driving = wp.lane_type == LT.Driving
        for n in wp.next(1.0):
            if n.road_id != wp.road_id or n.lane_id != wp.lane_id:
                if wp.transform.location.distance(n.transform.location) > 1.5:
                    continue
            l0, r0, l1, r1 = side_pts(wp, -1), side_pts(wp, 1), side_pts(n, -1), side_pts(n, 1)
            quads.append((l0, r0, r1, l1))
            qz.append(wp.transform.location.z)
            if wp.is_junction:
                continue                      # no painted lines or kerbs drawn inside junctions
            for sign, mk, a, b in ((-1, wp.left_lane_marking, l0, l1), (1, wp.right_lane_marking, r0, r1)):
                t = str(mk.type)
                if not driving and not is_edge(wp, sign):
                    continue                      # parking / shoulder lanes only draw the kerb
                if sign < 0 and not is_edge(wp, sign):          # a shared left line: one owner only
                    nb = wp.get_left_lane()
                    if nb.lane_type == LT.Driving and (nb.lane_id * wp.lane_id > 0 or wp.lane_id > 0):
                        continue                  # same-direction neighbour draws it as ITS right line; centre line from the - side
                if is_edge(wp, sign):
                    kind = EDGE_K
                elif t in ("Solid", "SolidSolid", "SolidBroken", "BrokenSolid"):
                    kind = SOLID
                elif t in ("Broken", "BrokenBroken"):
                    if int(wp.s / DASH_M) % 2:
                        continue
                    kind = DASHED
                else:
                    continue
                if in_junction(a, b):
                    continue
                key = (round((a[0] + b[0]) / 2 * 2), round((a[1] + b[1]) / 2 * 2))   # 0.5 m cells
                if key in seen:
                    continue
                seen.add(key)
                lines.append((a, b))
                lz.append(wp.transform.location.z)
                kinds.append(kind)
                cols.append(1 if str(mk.color) == "Yellow" else 0)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, town + ".npz"), quads=np.array(quads, np.float32),
                        lines=np.array(lines, np.float32), kinds=np.array(kinds, np.uint8),
                        cols=np.array(cols, np.uint8), qz=np.array(qz, np.float32), lz=np.array(lz, np.float32))
    print(f"{town}: {len(quads)} road pieces, {len(lines)} line pieces", flush=True)


class BevMap:
    def __init__(self, town, size=(400, 500), m_per_px=0.1, behind_m=10.0):
        d = np.load(os.path.join(CACHE, town + ".npz"))
        self.quads, self.lines, self.kinds, self.cols = d["quads"], d["lines"], d["kinds"], d["cols"]
        self.qz, self.lz = d["qz"], d["lz"]                # road height: bridges / overpasses are other levels
        self.qc = self.quads.mean(1)
        self.lc = self.lines.mean(1)
        self.w, self.h = size
        self.ss = 2                                   # draw at 2x without anti-aliasing, then shrink: no seams
        self.res = m_per_px / self.ss
        self.cx, self.cy = self.w * self.ss / 2, (self.h - behind_m / m_per_px) * self.ss
        self.reach = math.hypot(self.w, self.h) * m_per_px

    def _to_px(self, pts, x, y, c, s):
        dx, dy = pts[..., 0] - x, pts[..., 1] - y
        fwd, right = dx * c + dy * s, -dx * s + dy * c          # CARLA: x forward, y right (yaw clockwise)
        return np.stack([self.cx + right / self.res, self.cy - fwd / self.res], -1)

    def render(self, x, y, yaw_deg, z=None, route=None, cars=None):
        """z = our car's height (m): only roads within LEVEL_M of it are drawn (None = all levels)."""
        img = np.full((self.h * self.ss, self.w * self.ss, 3), BG, np.uint8)
        ss = self.ss
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        near = np.hypot(self.qc[:, 0] - x, self.qc[:, 1] - y) < self.reach
        if z is not None:
            near &= np.abs(self.qz - z) < LEVEL_M
        q = np.round(self._to_px(self.quads[near], x, y, c, s) * 8).astype(np.int32)
        for p in q:                                   # one by one: a single fillPoly call cancels overlaps
            cv2.fillConvexPoly(img, p, ROAD, cv2.LINE_8, shift=3)
        near = np.hypot(self.lc[:, 0] - x, self.lc[:, 1] - y) < self.reach
        if z is not None:
            near &= np.abs(self.lz - z) < LEVEL_M
        L = np.round(self._to_px(self.lines[near], x, y, c, s) * 8).astype(np.int32)
        for (a, b), k, col in zip(L, self.kinds[near], self.cols[near]):
            if k == EDGE_K:
                cv2.line(img, tuple(a), tuple(b), EDGE, 3 * ss, cv2.LINE_8, shift=3)
            else:
                cv2.line(img, tuple(a), tuple(b), YELLOW if col else WHITE, 2 * ss, cv2.LINE_8, shift=3)
        if route is not None and len(route) > 1:
            p = np.round(self._to_px(np.asarray(route, np.float32), x, y, c, s) * 8).astype(np.int32)
            cv2.polylines(img, [p], False, ROUTE, 6 * ss, cv2.LINE_8, shift=3)
        for (ox, oy, oyaw, ln, wd) in (cars or []):
            self._box(img, ox, oy, oyaw, ln, wd, x, y, c, s, yaw_deg, OTHER)
        self._box(img, x, y, yaw_deg, 4.7, 1.9, x, y, c, s, yaw_deg, CAR)
        return cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def render_classes(self, x, y, yaw_deg, z=None):
        """Truth class map on the same grid at 1x: 0 off-road, 1 road, 2 lane line, 3 road edge."""
        out = np.zeros((self.h, self.w), np.uint8)
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        cx, cy, res = self.cx / self.ss, self.cy / self.ss, self.res * self.ss

        def px(pts):
            dx, dy = pts[..., 0] - x, pts[..., 1] - y
            return np.stack([cx + (-dx * s + dy * c) / res, cy - (dx * c + dy * s) / res], -1)

        near = np.hypot(self.qc[:, 0] - x, self.qc[:, 1] - y) < self.reach
        if z is not None:
            near &= np.abs(self.qz - z) < LEVEL_M
        for p in np.round(px(self.quads[near]) * 8).astype(np.int32):
            cv2.fillConvexPoly(out, p, 1, cv2.LINE_8, shift=3)
        near = np.hypot(self.lc[:, 0] - x, self.lc[:, 1] - y) < self.reach
        if z is not None:
            near &= np.abs(self.lz - z) < LEVEL_M
        for (a, b), k in zip(np.round(px(self.lines[near]) * 8).astype(np.int32), self.kinds[near]):
            cv2.line(out, tuple(a), tuple(b), 3 if k == EDGE_K else 2, 2, cv2.LINE_8, shift=3)
        return out

    def _box(self, img, ox, oy, oyaw, ln, wd, x, y, c, s, yaw_deg, color):
        a = math.radians(oyaw)
        f, r = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        cen = np.array([ox, oy])
        pts = np.array([cen + f * ln / 2 + r * wd / 2, cen + f * ln / 2 - r * wd / 2,
                        cen - f * ln / 2 - r * wd / 2, cen - f * ln / 2 + r * wd / 2], np.float32)
        p = np.round(self._to_px(pts, x, y, c, s) * 8).astype(np.int32)
        cv2.fillPoly(img, [p], color, cv2.LINE_8, shift=3)


def demo(trace, episode=None):
    rows = list(csv.DictReader(open(trace, encoding="utf-8")))
    town = next(t for t in TOWNS if t in os.path.basename(trace))
    episode = episode or rows[0]["episode"]
    pts = np.array([(float(r["x"]), float(r["y"])) for r in rows if r["episode"] == str(episode)])
    bev = BevMap(town)
    import carla
    m = carla.Map(town, open(os.path.join(XODR, town + ".xodr"), encoding="utf-8").read())
    zs = [m.get_waypoint(carla.Location(x=float(px), y=float(py), z=0.0)).transform.location.z for px, py in pts]
    out = os.path.join(CACHE, f"demo_{os.path.basename(trace)[:-4]}_ep{episode}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (bev.w, bev.h))
    yaws, yaw = [], 0.0
    for i in range(len(pts)):
        j = min(i + 3, len(pts) - 1)
        d = pts[j] - pts[max(i - 3, 0)]
        if np.hypot(*d) > 0.3:
            yaw = math.degrees(math.atan2(d[1], d[0]))           # heading from motion (trace has no yaw)
        yaws.append(yaw)
        vw.write(bev.render(pts[i, 0], pts[i, 1], yaw, z=zs[i], route=pts[i:i + 200]))
    vw.release()
    for k in (0.25, 0.5, 0.75):
        i = int(len(pts) * k)
        cv2.imwrite(out.replace(".mp4", f"_{int(k * 100)}.png"), bev.render(pts[i, 0], pts[i, 1], yaws[i], z=zs[i], route=pts[i:i + 200]))
    print("demo:", out)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        for t in (sys.argv[2:] or TOWNS):
            build(t)
    else:
        demo(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
