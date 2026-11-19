"""CARLA 0.9.16 Gymnasium environment: route following with a compact "privileged" state.

Observation (float32, 8 + 2*N_LOOKAHEAD values)
  0  speed / target speed
  1  lateral offset from route centre / half lane width   (+ = right of centre)
  2  heading error / pi                                    (+ = pointing right of route)
  3  sin(heading error)
  4  current wheel angle (after STEER_SMOOTH)   5  previous speed command
  6  lead vehicle distance / LEAD_RANGE (1 = none)
  7  lead relative speed / target speed (0 = none)
  8.. next N route points in the car's frame: (forward m / 30, right m / 30) pairs

Action: [steer, accel] in [-1, 1]; accel > 0 = throttle, accel < 0 = brake.

CARLA health: every server call has a timeout. If CARLA hangs or crashes, the env raises
CarlaFailure. train.py then saves the model and exits with EXIT_RESTART_CARLA, and
run_training.py restarts CARLA and resumes. (Reconnecting inside the same process after a
CARLA restart aborts libcarla, so recovery is done by restarting the training process.)
A heartbeat file is written for run_training.py / watch_carla.py.
"""
import json
import math
import os
import random
import sys
import time
import weakref

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla

import config as C
import carla_server as srv
from route_smooth import smooth_route
import traffic

sys.path.append(os.path.join(C.CARLA_ROOT, "PythonAPI", "carla"))
from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402

# CARLA 0.9.16 bug: Vector3D.make_unit_vector() now requires an epsilon argument, but the
# bundled route planner calls it without one. Give it a default here (CARLA files untouched).
_make_unit = carla.Vector3D.make_unit_vector
try:
    carla.Vector3D(1, 0, 0).make_unit_vector()
except Exception:  # noqa: BLE001 - Boost.Python.ArgumentError
    carla.Vector3D.make_unit_vector = lambda self, eps=1e-6: _make_unit(self, eps)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CarlaFailure(RuntimeError):
    """CARLA hung, crashed or needs a restart. Save, exit, let run_training.py restart it."""


class CarlaDriveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, town=C.TRAIN_TOWN, render=False, num_npc=None, seed=None, multi_town=False):
        self.town, self.render_on, self.num_npc = town, render, num_npc   # None = traffic.npc_count(town)
        self.route_stop_lines, self.lead_gap, self.lead_speed, self.red_dist, self.yield_dist = [], None, 0.0, None, None
        self.ego_half_len, self.ego_half_w = 2.4, 0.95
        self.v_target, self.v_limit = C.TARGET_SPEED, C.TARGET_SPEED
        self.multi_town = multi_town                 # teacher training: rotate through C.TRAIN_TOWNS
        if multi_town:
            self._town_state = self._load_town_state()
            self.town = C.TRAIN_TOWNS[self._town_state["idx"] % len(C.TRAIN_TOWNS)]
        self.rng = random.Random(seed)
        self.seed_val = seed or 0

        obs_dim = 8 + 2 * C.N_LOOKAHEAD
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.vehicle, self.sensors, self.npcs = None, [], []
        self.route = None
        self.restarts = int(os.environ.get("CARLA_RESTARTS", "0"))     # set by run_training.py
        self.episodes, self.total_steps = 0, 0
        self._hb_t, self._hb_steps = time.time(), 0

        if C.AUTO_START and not srv.ensure_running():
            raise CarlaFailure("CARLA could not be started")
        try:
            self._connect()
        except RuntimeError as e:
            if self.multi_town:
                self._town_connect_failed()
            self._fail(f"connect failed: {e}")
        if self.multi_town and self._town_state.get("connect_fails"):
            self._town_state["connect_fails"] = 0
            self._save_town_state()
        self._heartbeat(event="connected", force=True)

    # ------------------------------------------------------------------ connection / recovery
    def _connect(self):
        self.client = carla.Client(C.HOST, C.PORT)
        self.client.set_timeout(C.TIMEOUT_S)
        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith(self.town):
            self._heartbeat(event=f"loading {self.town}", force=True)
            self.client.set_timeout(C.LOAD_MAP_TIMEOUT_S)        # switching towns can take > 30 s
            self.world = self.client.load_world(self.town)
            self.client.set_timeout(C.TIMEOUT_S)
        self._setup_world()

    def _setup_world(self):
        self._orig_settings = self.world.get_settings()
        self._orig_settings.synchronous_mode = False
        self._orig_settings.fixed_delta_seconds = None
        self._orig_settings.no_rendering_mode = False
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = C.FIXED_DELTA
        s.no_rendering_mode = bool(C.NO_RENDERING and not self.render_on)
        self.world.apply_settings(s)

        self.tm = self.client.get_trafficmanager(C.TM_PORT)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(self.seed_val)
        traffic.configure_tm(self.tm)

        self.map = self.world.get_map()
        self.grp = GlobalRoutePlanner(self.map, C.ROUTE_RESOLUTION)
        self.spawn_points = self.map.get_spawn_points()
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle, self.sensors, self.npcs = None, [], []
        self.world.tick(C.TICK_TIMEOUT_S)

    # ------------------------------------------------------------------ town rotation (teacher training)
    @staticmethod
    def _load_town_state():
        try:
            with open(C.TOWN_STATE_FILE, encoding="utf-8") as f:
                st = json.load(f)
            return {"idx": int(st["idx"]), "episodes": int(st["episodes"]),
                    "connect_fails": int(st.get("connect_fails", 0))}
        except (OSError, ValueError, KeyError):
            return {"idx": 0, "episodes": 0, "connect_fails": 0}

    def _town_connect_failed(self):
        """A town that keeps failing to load must not stall training (Town03, 14 Nov: 5 h of time-outs).
        After TOWN_MAX_CONNECT_FAILS in a row the rotation skips it."""
        st = self._town_state
        st["connect_fails"] = st.get("connect_fails", 0) + 1
        if st["connect_fails"] >= C.TOWN_MAX_CONNECT_FAILS:
            bad = self.town
            self._town_state = {"idx": (st["idx"] + 1) % len(C.TRAIN_TOWNS), "episodes": 0, "connect_fails": 0}
            self.town = C.TRAIN_TOWNS[self._town_state["idx"]]
            srv.log_event(f"SKIPPING {bad}: failed to load {C.TOWN_MAX_CONNECT_FAILS} times in a row -> {self.town}")
        self._save_town_state()

    def _save_town_state(self, town=None):
        tmp = C.TOWN_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(self._town_state, town=town or self.town), f)
        os.replace(tmp, C.TOWN_STATE_FILE)

    def _next_town(self):
        """Move the rotation to the next town and hand over for a CARLA restart. Loading a map inside a
        running training process crashed CARLA 0.9.16 (smoke test 14 Nov); a fresh CARLA loads the
        town at connect time, the path every collection run already uses."""
        self._town_state = {"idx": (self._town_state["idx"] + 1) % len(C.TRAIN_TOWNS), "episodes": 0,
                            "connect_fails": 0}
        nxt = C.TRAIN_TOWNS[self._town_state["idx"]]
        self._save_town_state(nxt)
        self._fail(f"town switch {self.town} -> {nxt} (planned)")

    def _fail(self, reason):
        """Record the failure and hand over to train.py / run_training.py."""
        self.vehicle, self.sensors, self.npcs = None, [], []   # don't touch actors on a dead server
        srv.log_event(f"FAILURE: {reason} (episode {self.episodes}, steps {self.total_steps})")
        self._heartbeat(event=f"failure: {reason}"[:200], force=True)
        raise CarlaFailure(reason)

    def _tick(self):
        self.world.tick(C.TICK_TIMEOUT_S)       # raises RuntimeError if CARLA doesn't answer in time

    def _heartbeat(self, event="", force=False):
        if not force and self.total_steps - self._hb_steps < C.HEARTBEAT_EVERY_STEPS:
            return
        now = time.time()
        sps = (self.total_steps - self._hb_steps) / max(1e-6, now - self._hb_t)
        self._hb_t, self._hb_steps = now, self.total_steps
        srv.write_heartbeat(total_steps=self.total_steps, episodes=self.episodes,
                            steps_per_s=round(sps, 1), restarts=self.restarts,
                            town=self.town, event=event)

    # ------------------------------------------------------------------ actors
    def _destroy(self):
        try:
            for s in self.sensors:
                if s.is_alive:
                    s.stop()
            ids = [a.id for a in self.sensors + self.npcs + ([self.vehicle] if self.vehicle else [])
                   if a.is_alive]
            if ids:
                self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        finally:
            self.vehicle, self.sensors, self.npcs = None, [], []

    def _plan_route(self):
        for _ in range(60):
            a, b = self.rng.sample(self.spawn_points, 2)
            trace = self.grp.trace_route(a.location, b.location)
            if not C.ROUTE_MIN_M <= len(trace) * C.ROUTE_RESOLUTION <= C.ROUTE_MAX_M:
                continue
            if C.ROUTE_START_CHECK and not self._route_starts_at(a, trace):
                continue                                        # planner snapped the start to another lane / direction
            return a, trace
        return a, trace

    @staticmethod
    def _route_starts_at(spawn, trace):
        w0 = trace[0][0].transform.location
        if math.hypot(w0.x - spawn.location.x, w0.y - spawn.location.y) > C.ROUTE_START_MAX_M:
            return False
        far = next((wp for wp, _ in trace if wp.transform.location.distance(w0) > 4.0), None)
        if far is None:
            return False
        h = math.degrees(math.atan2(far.transform.location.y - w0.y, far.transform.location.x - w0.x))
        return abs((h - spawn.rotation.yaw + 180) % 360 - 180) <= C.ROUTE_START_MAX_DEG

    def _set_route(self, trace):
        wps = [wp for wp, _ in trace]
        xy = [(w.transform.location.x, w.transform.location.y) for w in wps]
        opt = [o.value for _, o in trace]                       # RoadOption per point
        if C.ROUTE_SMOOTH:                                      # duplicates out, smooth lane changes, even spacing
            self.route, self.route_opt, self.route_junction = smooth_route(
                xy, [w.lane_width for w in wps], opt, [w.is_junction for w in wps])
            self.route_opt = self.route_opt.astype(np.int64)
        else:
            self.route = np.array([[w.transform.location.x, w.transform.location.y,
                                    math.radians(w.transform.rotation.yaw), w.lane_width] for w in wps],
                                  dtype=np.float64)
            self.route_opt = np.array(opt, dtype=np.int64)
            self.route_junction = np.array([w.is_junction for w in wps])
        self.idx = 0

    # navigation command for the vision student (like a sat-nav instruction): the next junction
    # manoeuvre within CMD_LOOKAHEAD_M, else "follow lane". RoadOption: 1 LEFT 2 RIGHT 3 STRAIGHT.
    CMD_FOLLOW, CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT = 0, 1, 2, 3

    def command(self):
        n = int(C.CMD_LOOKAHEAD_M / C.ROUTE_RESOLUTION)
        for opt in self.route_opt[self.idx:self.idx + n]:
            if opt in (1, 2, 3):
                return int(opt)
        return self.CMD_FOLLOW

    def _npc_target(self):
        return traffic.npc_count(self.town) if self.num_npc is None else int(self.num_npc)

    def _spawn_npcs(self):
        bps = [b for b in self.bp_lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        pts = [p for p in self.spawn_points if p.location.distance(self.start_loc) > C.NPC_MIN_DIST_M]   # (a fresh actor reports (0,0,0) until a tick)
        self.rng.shuffle(pts)
        batch = [carla.command.SpawnActor(self.rng.choice(bps), p)
                 .then(carla.command.SetAutopilot(carla.command.FutureActor, True, C.TM_PORT))
                 for p in pts[:self._npc_target()]]
        for r in self.client.apply_batch_sync(batch, True):
            if not r.error:
                self.npcs.append(self.world.get_actor(r.actor_id))
        if not C.TM_AUTO_LANE_CHANGE:
            for a in self.npcs:                                   # no voluntary overtakes: TM changes lanes
                self.tm.auto_lane_change(a, False)                # into the ego car without looking

    def _setup_episode(self):
        self._destroy()
        bp = self.bp_lib.find(C.VEHICLE)
        bp.set_attribute("role_name", "hero")
        for _ in range(20):
            start, trace = self._plan_route()
            self.vehicle = self.world.try_spawn_actor(bp, start)
            if self.vehicle:
                self.start_loc = start.location
                break
        if self.vehicle is None:
            raise RuntimeError("could not spawn the ego vehicle")
        self._set_route(trace)

        col = self.world.spawn_actor(self.bp_lib.find("sensor.other.collision"), carla.Transform(),
                                     attach_to=self.vehicle)
        me = weakref.ref(self)
        col.listen(lambda e: me() and me()._on_collision(e))
        self.sensors.append(col)
        self.route_stop_lines = traffic.route_stops(self.world, self.route) if C.TRAFFIC else []
        self.lead_gap, self.lead_speed, self.red_dist, self.yield_dist = None, 0.0, None, None
        self.ego_half_len = self.vehicle.bounding_box.extent.x
        self.ego_half_w = self.vehicle.bounding_box.extent.y
        if self._npc_target():
            self._spawn_npcs()
        for _ in range(10):                                   # let the car settle on the road
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            self._tick()

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        if C.RESTART_EVERY_EPISODES and self.episodes >= C.RESTART_EVERY_EPISODES:
            self._fail(f"preventive restart after {self.episodes} episodes")
        try:
            if self.multi_town and self._town_state["episodes"] >= C.TOWN_SWITCH_EPISODES:
                self._next_town()
            self._setup_episode()
        except CarlaFailure:
            raise
        except RuntimeError as e:
            self._fail(f"reset failed: {e}")

        self.episodes += 1
        if self.multi_town:
            self._town_state["episodes"] += 1
            self._save_town_state()
        self.collided, self.collided_with = False, ""
        self.steps, self.slow_steps = 0, 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.stats = {"lat_abs": 0.0, "speed": 0.0, "dsteer": 0.0, "reversals": 0,
                      "lat_acc_abs": 0.0, "lat_acc_max": 0.0, "lat_jerk_abs": 0.0, "dist_m": 0.0}
        self.prev_dsteer, self.prev_lat_acc = 0.0, 0.0
        self.speed, self.ep_throttle = 0.0, []
        self.v_target, self.v_limit = C.TARGET_SPEED, C.TARGET_SPEED
        obs, st0 = self._observe()
        self.prev_lat = st0["lat"]
        return obs, {}

    def _on_collision(self, event):
        if not self.collided:
            try:
                self.collided_with = event.other_actor.type_id
                o, e = event.other_actor, self.vehicle
                et, ot = e.get_transform(), o.get_transform()
                h = math.radians(et.rotation.yaw)
                dx, dy = ot.location.x - et.location.x, ot.location.y - et.location.y
                fwd, left = dx * math.cos(h) + dy * math.sin(h), -(-dx * math.sin(h) + dy * math.cos(h))
                ov = o.get_velocity()
                rel_h = math.degrees(wrap_pi(math.radians(ot.rotation.yaw) - h))
                self.collided_with += (f" | other at fwd {fwd:+.1f} m left {left:+.1f} m, heading {rel_h:+.0f} deg,"
                                       f" {math.hypot(ov.x, ov.y):.1f} m/s | ego {self.speed:.1f} m/s idx {self.idx}"
                                       f" junction {bool(self.route_junction[min(self.idx, len(self.route) - 1)])}"
                                       f" | lead {self.lead_gap} yield {self.yield_dist} red {self.red_dist}")
            except Exception:
                self.collided_with = "?"
        self.collided = True

    def _pedals(self, a):
        """action[1] -> (throttle, brake)."""
        if not C.SPEED_CONTROL:
            dz = C.BRAKE_DEADZONE
            return max(a, 0.0), ((-a - dz) / (1.0 - dz) if a < -dz else 0.0)
        v_des = (a + 1.0) / 2.0 * C.V_MAX_FACTOR * self.v_target
        if C.ROAD_SPEED_TARGET:
            v_des = min(v_des, max(self.v_limit, C.TARGET_SPEED))
        if C.CURVE_LIMIT:
            v_des = min(v_des, self._curve_speed_limit())
        if C.TRAFFIC_SPEED_CAPS:
            v_des = min(v_des, traffic.speed_cap(self))          # red lights + car ahead
        v = self.speed
        err = v_des - v
        if v_des < 0.3:                                       # asked to stop
            return 0.0, (0.5 if v > 0.3 else 0.2)
        if err > 0:
            return min(C.THROTTLE_MAX, C.THROTTLE_FF + C.SPEED_KP * err), 0.0
        if err < -1.0:
            return 0.0, min(1.0, C.BRAKE_KP * (-err - 1.0))
        return C.THROTTLE_FF * 0.5, 0.0                        # hold speed

    def _road_target(self):
        """Cruise speed (m/s): the road's speed limit less the same margin the NPCs keep (so we flow with
        traffic instead of being run into from behind on highways), never below TARGET_SPEED (towns)."""
        if not C.ROAD_SPEED_TARGET or self.vehicle is None:
            return C.TARGET_SPEED
        try:
            lim = float(self.vehicle.get_speed_limit())              # km/h of the road we are on
        except RuntimeError:
            return self.v_target
        if lim <= 0:
            return self.v_target
        self.v_limit = lim / 3.6                                     # hard cap: never above the sign
        kmh = min(lim * (1.0 - C.TM_SPEED_DIFF_PCT / 100.0), C.ROAD_SPEED_MAX_KMH)
        return max(C.TARGET_SPEED, kmh / 3.6)

    def _curve_speed_limit(self):
        """Highest speed that keeps every bend within CURVE_LOOKAHEAD_M at CURVE_LAT_ACC,
        allowing gentle braking (CURVE_DECEL) over the distance to each bend."""
        res = C.ROUTE_RESOLUTION
        n = int(traffic.sense_range(self, C.CURVE_LOOKAHEAD_M, C.CURVE_DECEL) / res)   # >= braking distance
        yaw = self.route[self.idx:self.idx + n + 3, 2]
        limit = float("inf")
        for j in range(len(yaw) - 2):
            k = abs(wrap_pi(yaw[j + 2] - yaw[j])) / (2 * res)          # curvature, 1/m
            if k < 1e-3:
                continue
            v_bend = math.sqrt(C.CURVE_LAT_ACC / k)
            v_now = math.sqrt(v_bend ** 2 + 2 * C.CURVE_DECEL * j * res)  # can still brake to v_bend
            limit = min(limit, v_now)
        return max(C.CURVE_MIN_SPEED, limit)

    def step(self, action):
        # steering smoothing: the wheel only moves part of the way to the commanded angle each step,
        # like a real steering rack. Removes rapid back-and-forth by construction.
        steer_cmd = float(np.clip(action[0], -1, 1))
        prev = float(self.prev_action[0])
        steer = prev + C.STEER_SMOOTH * (steer_cmd - prev)          # plain float: CARLA rejects numpy types
        accel = float(np.clip(action[1], -1, 1))
        throttle, brake = self._pedals(accel)
        self.ep_throttle.append(throttle)
        try:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer))
            for _ in range(C.FRAME_SKIP):
                self._tick()
            if self.render_on:
                self._follow_camera()
            prev_idx = self.idx
            obs, st = self._observe()
        except RuntimeError as e:
            self._fail(f"step failed: {e}")      # CARLA hung (tick timeout) or died

        progress_m = (self.idx - prev_idx) * C.ROUTE_RESOLUTION
        self.steps += 1
        self.total_steps += 1
        self._heartbeat()

        hw = st["half_width"]
        dsteer = steer - self.prev_action[0]
        lat_acc = st["speed"] * st["yaw_rate"]                  # m/s^2, centripetal
        r = (C.R_PROGRESS * progress_m
             + C.R_SPEED * min(st["speed"], self.v_target) / self.v_target
             - C.R_LATERAL * (st["lat"] / hw) ** 2
             - C.R_HEADING * abs(st["head_err"]) / math.pi
             - C.R_STEER_RATE * abs(dsteer)
             - C.R_LAT_ACC * (lat_acc / C.LAT_ACC_COMFORT) ** 2
             - C.R_ACCEL_RATE * abs(accel - self.prev_action[1])
             - C.R_OVERSPEED * max(0.0, (st["speed"] - self.v_target) / self.v_target))
        # ideal-line shaping: reward every step that brings the car closer to the lane centre (potential-based,
        # so it speeds learning without changing what is optimal), but only at a smooth sideways rate;
        # a wheel over a lane line costs a lot
        dlat = float(np.clip(abs(st["lat"]) - abs(self.prev_lat), -0.5, 0.5))
        lat_vel = (st["lat"] - self.prev_lat) / (C.FIXED_DELTA * C.FRAME_SKIP)
        over = max(0.0, abs(st["lat"]) - (hw - C.CAR_HALF_WIDTH_M))
        r += (- C.R_TOWARD_LINE * dlat / hw
              - C.R_LAT_VEL * max(0.0, abs(lat_vel) - C.LAT_VEL_OK) ** 2
              - C.R_LANE_OUT * (over > 0) - C.R_LANE_OUT_M * over)
        self.prev_lat = st["lat"]

        # smoothness metrics (reported, not rewarded directly)
        dt = C.FIXED_DELTA * C.FRAME_SKIP
        s = self.stats
        s["dsteer"] += abs(dsteer)
        if abs(dsteer) > 0.02 and dsteer * self.prev_dsteer < 0:  # steering changed direction = wiggle
            s["reversals"] += 1
        if abs(dsteer) > 0.02:
            self.prev_dsteer = dsteer
        s["lat_acc_abs"] += abs(lat_acc)
        s["lat_acc_max"] = max(s["lat_acc_max"], abs(lat_acc))
        s["lat_jerk_abs"] += abs(lat_acc - self.prev_lat_acc) / dt
        s["dist_m"] += progress_m
        self.prev_lat_acc = lat_acc

        terminated, truncated, reason = False, False, ""
        waiting = any(d is not None and d < 10.0 for d in (self.red_dist, self.yield_dist, self.lead_gap))   # red / give way / queue: not stuck
        self.slow_steps = self.slow_steps + 1 if (self.steps > C.WARMUP_STEPS and st["speed"] < C.STUCK_SPEED
                                                  and st["lead_dist"] > 10.0 and not waiting) else 0
        if self.collided:
            r, terminated, reason = r + C.R_COLLISION, True, "collision"
        elif abs(st["lat"]) > C.OFF_ROUTE_M:
            r, terminated, reason = r + C.R_OFF_ROUTE, True, "off_route"
        elif self.idx >= len(self.route) - 3:
            r, terminated, reason = r + C.R_ROUTE_DONE, True, "route_done"
        elif self.slow_steps >= C.STUCK_STEPS:
            r, terminated, reason = r + C.R_STUCK, True, "stuck"
        elif self.steps >= C.MAX_EPISODE_STEPS:
            truncated, reason = True, "time_limit"

        self.prev_action = np.array([steer, accel], dtype=np.float32)
        self.stats["lat_abs"] += abs(st["lat"])
        self.stats["speed"] += st["speed"]
        info = {"route_completion": self.idx / max(1, len(self.route) - 1),
                "speed_kmh": st["speed"] * 3.6, "lat": st["lat"]}
        if terminated or truncated:
            info.update(reason=reason, town=self.town,
                        mean_lat_abs=self.stats["lat_abs"] / self.steps,
                        mean_speed_kmh=self.stats["speed"] / self.steps * 3.6,
                        route_m=len(self.route) * C.ROUTE_RESOLUTION,
                        mean_dsteer=s["dsteer"] / self.steps,
                        reversals_per_km=s["reversals"] / max(0.05, s["dist_m"] / 1000.0),
                        mean_lat_acc=s["lat_acc_abs"] / self.steps,
                        max_lat_acc=s["lat_acc_max"],
                        mean_lat_jerk=s["lat_jerk_abs"] / self.steps,
                        # diagnostics: what was the car doing at the end?
                        end_speed_kmh=st["speed"] * 3.6,
                        end_lat=st["lat"],
                        throttle_last5s=float(np.mean(self.ep_throttle[-50:])),
                        at_junction=bool(self.map.get_waypoint(
                            self.vehicle.get_location()).is_junction))
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------ state
    def _observe(self):
        tf = self.vehicle.get_transform()
        vx, vy = tf.location.x, tf.location.y
        yaw = math.radians(tf.rotation.yaw)
        v = self.vehicle.get_velocity()
        speed = math.hypot(v.x, v.y)
        self.speed = speed

        lo, hi = self.idx, min(len(self.route), self.idx + 25)
        d = np.hypot(self.route[lo:hi, 0] - vx, self.route[lo:hi, 1] - vy)
        self.idx = lo + int(np.argmin(d))
        wx, wy, wyaw, lane_w = self.route[self.idx]

        lat = -math.sin(wyaw) * (vx - wx) + math.cos(wyaw) * (vy - wy)
        head_err = wrap_pi(yaw - wyaw)
        cy, sy = math.cos(yaw), math.sin(yaw)

        pts = []
        for k in range(1, C.N_LOOKAHEAD + 1):
            j = min(len(self.route) - 1, self.idx + k * C.LOOKAHEAD_STEP)
            dx, dy = self.route[j, 0] - vx, self.route[j, 1] - vy
            pts += [(cy * dx + sy * dy) / 30.0, (-sy * dx + cy * dy) / 30.0]

        self.v_target = self._road_target()
        # lead = nearest of: vehicle ahead in our lane (along the route, so also in bends) / red stop line
        self.lead_gap, self.lead_speed = traffic.lead_on_route(self, vx, vy)
        self.red_dist = traffic.next_red(self) if self.route_stop_lines else None
        self.yield_dist = traffic.yield_dist(self, vx, vy) if C.TRAFFIC_SPEED_CAPS else None
        lead_dist, lead_rel = C.LEAD_RANGE_M, 0.0
        if self.lead_gap is not None and self.lead_gap < lead_dist:
            lead_dist, lead_rel = max(0.0, self.lead_gap), self.lead_speed - speed
        for stop in (self.red_dist, self.yield_dist):            # red light / give way: a standing obstacle
            if stop is not None and stop < lead_dist:
                lead_dist, lead_rel = max(0.0, stop), -speed

        obs = np.array([speed / self.v_target,
                        lat / (lane_w / 2.0),
                        head_err / math.pi,
                        math.sin(head_err),
                        self.prev_action[0], self.prev_action[1],
                        lead_dist / C.LEAD_RANGE_M,
                        lead_rel / self.v_target] + pts, dtype=np.float32)
        yaw_rate = math.radians(self.vehicle.get_angular_velocity().z)   # deg/s -> rad/s
        st = {"speed": speed, "lat": lat, "head_err": head_err, "yaw_rate": yaw_rate,
              "half_width": lane_w / 2.0, "lead_dist": lead_dist}
        return obs, st

    def _follow_camera(self):
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        loc = tf.location - carla.Location(x=fwd.x * 8, y=fwd.y * 8) + carla.Location(z=4)
        self.world.get_spectator().set_transform(
            carla.Transform(loc, carla.Rotation(pitch=-15, yaw=tf.rotation.yaw)))

    def close(self):
        try:
            self._destroy()
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self._orig_settings)
        except RuntimeError:
            pass
