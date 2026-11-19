"""Traffic for the ego car: NPC count per town, traffic-light stops and car following along the ROUTE.

Rule-based speed caps (like the map-based curve speed limit), used by carla_env._pedals for every driver:
  - red / yellow light ahead on the route: v <= sqrt(2 * LIGHT_DECEL * (distance to stop line - margin))
  - lead vehicle on the route (also in bends, by projection onto the path): keep a TIME_GAP_S gap
The lead seen by the policy (obs 6-7) is the nearer of the lead vehicle and a red stop line (as a standing
obstacle), so the observation size is unchanged and PPO can learn to stop for both.
"""
import math

import numpy as np

import config as C


def sense_range(env, base, decel):
    """Look-ahead (m): the base range, or more when our stopping distance at this speed needs it."""
    return max(base, env.speed ** 2 / (2.0 * decel) + C.SENSE_EXTRA_M)


def npc_count(town):
    return int(C.NPC_PER_TOWN.get(town, C.NUM_NPC_VEHICLES)) if C.TRAFFIC else 0


def configure_tm(tm):
    tm.set_global_distance_to_leading_vehicle(C.TM_GAP_M)
    tm.global_percentage_speed_difference(C.TM_SPEED_DIFF_PCT)    # + = slower than the speed limit
    if C.TM_HYBRID_RADIUS_M:
        tm.set_hybrid_physics_mode(True)                          # far NPCs: cheap physics (speed)
        tm.set_hybrid_physics_radius(C.TM_HYBRID_RADIUS_M)


def route_stops(world, route):
    """[(route index, traffic light actor)] for every stop line on the route (matched by position + heading)."""
    out = []
    xy = route[:, :2]
    for tl in world.get_actors().filter("traffic.traffic_light"):
        try:
            stops = tl.get_stop_waypoints()
        except RuntimeError:
            continue
        for wp in stops:
            loc = wp.transform.location
            d = np.hypot(xy[:, 0] - loc.x, xy[:, 1] - loc.y)
            j = int(np.argmin(d))
            if d[j] > C.STOP_MATCH_M:
                continue
            h = math.radians(wp.transform.rotation.yaw)
            if abs((h - route[j, 2] + math.pi) % (2 * math.pi) - math.pi) > math.radians(35):
                continue
            out.append((j, tl))
    out.sort(key=lambda t: t[0])
    return out


def next_red(env):
    """Distance (m) along the route to the next red/yellow stop line ahead, or None."""
    import carla
    for j, tl in env.route_stop_lines:
        if j < env.idx:
            continue
        dist = (j - env.idx) * C.ROUTE_RESOLUTION - C.STOP_MARGIN_M
        if dist > sense_range(env, C.LIGHT_LOOKAHEAD_M, C.LIGHT_DECEL):
            return None
        st = tl.get_state()
        if st == carla.TrafficLightState.Red:
            return dist
        if st == carla.TrafficLightState.Yellow and dist > env.speed ** 2 / (2 * C.LIGHT_DECEL):
            return dist                                   # can still stop comfortably -> stop
        return None                                       # green (or too late to stop on yellow): go
    return None


def lead_on_route(env, vx, vy):
    """(gap m, lead speed m/s along the route) of the nearest NPC whose BODY is on our path ahead (any heading:
    cars ahead in our lane, cars cutting in, cars standing across a junction), else (None, 0).
    gap = how far our front bumper can go before our body would touch it."""
    n = len(env.route)
    rng = sense_range(env, C.LEAD_RANGE_M, C.FOLLOW_DECEL)
    hi = min(n, env.idx + int(rng / C.ROUTE_RESOLUTION) + 1)
    if hi - env.idx < 2 or not env.npcs:
        return None, 0.0
    pts = env.route[env.idx:hi]
    s_ahead = np.arange(len(pts)) * C.ROUTE_RESOLUTION
    h_ego = env.route[env.idx, 2]
    reach = env.ego_half_w + C.LEAD_SIDE_MARGIN_M
    best, vlead = None, 0.0
    for npc in env.npcs:
        l = npc.get_location()
        if abs(l.x - vx) > rng + 5 or abs(l.y - vy) > rng + 5:
            continue
        if (l.x - vx) * math.cos(h_ego) + (l.y - vy) * math.sin(h_ego) < 0:
            continue                                      # centre behind us: follower, not a lead
        bb = npc.bounding_box.extent
        h = math.radians(npc.get_transform().rotation.yaw)
        d = _seg_dist(pts[:, 0], pts[:, 1], l.x, l.y, math.cos(h), math.sin(h), bb.x) - bb.y
        on = np.where(d < reach)[0]
        if not len(on):
            continue
        k = int(on[0])
        gap = s_ahead[k] - env.ego_half_len
        if best is None or gap < best:
            v = npc.get_velocity()
            yaw = pts[k, 2]
            best, vlead = gap, v.x * math.cos(yaw) + v.y * math.sin(yaw)
    return best, vlead


def _seg_dist(px, py, cx, cy, fx, fy, ex):
    """Distance from points (px, py) to the segment centre +- ex along unit (fx, fy) (a car's long axis)."""
    rx, ry = px - cx, py - cy
    along = np.clip(rx * fx + ry * fy, -ex, ex)
    return np.hypot(rx - along * fx, ry - along * fy)


def yield_dist(env, vx, vy):
    """Distance (m) along the route to stop before an NPC that will cross / merge into our path when we get
    there (unprotected left turns, junction cross traffic, merges), or None.
    NPC prediction: constant speed + constant turn rate (TM cars turn in junctions), as a body (segment along
    its long axis); conflict when its body comes within our half width + its half width + margin of a path
    point we reach at about the same time."""
    n = len(env.route)
    hi = min(n, env.idx + int(sense_range(env, C.YIELD_LOOKAHEAD_M, C.YIELD_DECEL) / C.ROUTE_RESOLUTION) + 1)
    if hi - env.idx < 2 or not env.npcs:
        return None
    pts = env.route[env.idx:hi]
    near = env.route[max(0, env.idx - int(10 / C.ROUTE_RESOLUTION)):hi]   # incl. just behind us
    s_ahead = np.arange(len(pts)) * C.ROUTE_RESOLUTION
    v_ego = max(env.speed, C.YIELD_MIN_EGO_SPEED)
    t_ego = s_ahead / v_ego                                       # when we reach each path point
    dt = C.YIELD_DT_S
    best = None
    for npc in env.npcs:
        l = npc.get_location()
        yr = max(C.YIELD_RANGE_M, sense_range(env, C.YIELD_LOOKAHEAD_M, C.YIELD_DECEL) + 15)
        if abs(l.x - vx) > yr or abs(l.y - vy) > yr:
            continue
        v = npc.get_velocity()
        sp = math.hypot(v.x, v.y)
        if sp < 0.5:
            continue                                              # parked / waiting: the lead logic handles it
        h = math.atan2(v.y, v.x)
        dn = np.hypot(near[:, 0] - l.x, near[:, 1] - l.y)
        kn = int(np.argmin(dn))
        same = abs((h - near[kn, 2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(30)
        k0 = max(0, env.idx - int(10 / C.ROUTE_RESOLUTION))
        lo_i, hi_i = sorted((env.idx, k0 + kn))
        lane_change = bool(np.isin(env.route_opt[lo_i:hi_i + 1], (5, 6)).any())
        if same and dn[kn] < near[kn, 3] / 2 and not lane_change:
            continue                                              # in our lane, same way: lead / follower, not a yield
        h_ego = env.route[env.idx, 2]
        behind = (l.x - vx) * math.cos(h_ego) + (l.y - vy) * math.sin(h_ego) < 0
        if behind and abs((h - h_ego + math.pi) % (2 * math.pi) - math.pi) < math.radians(60):
            continue                                              # behind us, going our way: it gives way to us
                                                                  # (braking for it = being hit from behind)
        w = float(np.clip(math.radians(npc.get_angular_velocity().z), -C.YIELD_MAX_TURN, C.YIELD_MAX_TURN))
        bb = npc.bounding_box.extent
        reach = env.ego_half_w + bb.y + C.YIELD_MARGIN_M
        px, py = l.x, l.y
        for i in range(int(C.YIELD_HORIZON_S / dt) + 1):
            t = i * dt
            fx, fy = math.cos(h), math.sin(h)
            d = _seg_dist(pts[:, 0], pts[:, 1], px, py, fx, fy, bb.x)
            hit = np.where((d < reach) & (np.abs(t_ego - t) <= C.YIELD_TIME_S))[0]
            if len(hit):
                k = int(hit[0])
                stop = s_ahead[k] - reach - env.ego_half_len
                if stop >= 0.5:                                   # < 0.5: already committed, keep going
                    best = stop if best is None else min(best, stop)
                break
            px, py, h = px + sp * fx * dt, py + sp * fy * dt, h + w * dt
    return best


def speed_cap(env):
    """Highest allowed speed from lights + lead vehicle (inf = no cap). Sets env.lead_* for the observation."""
    cap = float("inf")
    gap, vl = env.lead_gap, max(0.0, env.lead_speed)     # crossing / oncoming lead = standing obstacle
    if gap is not None:
        g = gap - C.MIN_GAP_M
        cap = min(cap, max(0.0, min(math.sqrt(max(0.0, vl * vl + 2 * C.FOLLOW_DECEL * g)),
                                    vl + (g - C.TIME_GAP_S * vl) / C.TIME_GAP_S + 0.5)))
    if getattr(env, "yield_dist", None) is not None:
        cap = min(cap, math.sqrt(max(0.0, 2 * C.YIELD_DECEL * env.yield_dist)) if env.yield_dist > 0.3 else 0.0)
    if env.red_dist is not None:
        cap = min(cap, math.sqrt(max(0.0, 2 * C.LIGHT_DECEL * env.red_dist)) if env.red_dist > 0.3 else 0.0)
    return cap
