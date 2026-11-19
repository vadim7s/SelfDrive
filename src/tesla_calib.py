"""Measure how CARLA's Model 3 turns for a given steer command (for tesla_test.py's curvature mapping).

Drives at ~8 m/s with constant steer s (wheel fully settled) and measures path curvature = yaw rate / speed.
Writes data\\tesla_test\\carla_steer_geometry.json: physics max steer angle, wheelbase, and
an effective max angle fitted so that curvature = tan(s x angle) / wheelbase matches what the car really does.
Refuses to run if CARLA is already up (never touch another job's server). Shuts CARLA down at the end.

    python tesla_calib.py
"""
import json
import math
import os
import time

import numpy as np
import carla

import config as C
import carla_server as srv

OUT = os.path.join(C.DATA_ROOT, "tesla_test", "carla_steer_geometry.json")


def main():
    if srv.processes_running():
        raise SystemExit("CARLA is already running (another job) - not touching it")
    if not srv.start():
        raise SystemExit("CARLA did not start")
    env = None
    try:
        from carla_env import CarlaDriveEnv
        env = CarlaDriveEnv(town="Town04", render=False, seed=5)    # Town04: wide roads / highway
        rows = []
        for steer in (0.03, 0.06, 0.1, 0.15, -0.06, -0.1) * 2:        # each twice, different spawns
            env.reset()
            v = env.vehicle
            wl = v.get_physics_control().wheels
            wheelbase = wl[0].position.distance(wl[2].position) / 100.0   # front-left to rear-left, cm -> m
            max_deg = wl[0].max_steer_angle
            for _ in range(40):                                   # 4 s run-up, straight, speed controller
                v.apply_control(carla.VehicleControl(throttle=0.6, steer=0.0))
                env._tick(); env._tick()
            yaws, speeds = [], []
            for k in range(60):                                   # 3 s at constant steer, direct control
                vel = v.get_velocity()
                spd = math.hypot(vel.x, vel.y)
                thr = float(np.clip(0.3 * (8.0 - spd) + 0.25, 0, 0.7))
                v.apply_control(carla.VehicleControl(throttle=thr, steer=float(steer)))
                env._tick()
                if k >= 30:                                       # settled
                    yaws.append(math.radians(v.get_angular_velocity().z))
                    speeds.append(spd)
            if np.mean(speeds) < 3:
                print(f"steer {steer:+.2f}: car not moving ({np.mean(speeds):.1f} m/s) - skipped", flush=True)
                continue
            curv = float(np.mean(yaws) / np.mean(speeds))
            rows.append(dict(steer=steer, curvature=curv, speed=float(np.mean(speeds))))
            print(f"steer {steer:+.2f}: speed {np.mean(speeds):.1f} m/s, curvature {curv:+.4f} 1/m", flush=True)
        if len(rows) < 4:
            raise SystemExit("too few valid samples - geometry NOT written")
        # fit effective angle: curvature = tan(steer * A) / wheelbase
        st = np.array([r["steer"] for r in rows])
        cu = np.array([r["curvature"] for r in rows])
        cands = np.radians(np.arange(5, 90, 0.25))
        err = [np.mean((np.tan(st * A) / wheelbase - cu) ** 2) for A in cands]
        eff = float(np.degrees(cands[int(np.argmin(err))]))
        out = dict(max_steer_deg=eff, wheelbase_m=wheelbase, physics_max_steer_deg=max_deg,
                   sign_note="angular_velocity.z > 0 with steer > 0 means + = right", samples=rows)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"physics max steer {max_deg:.1f} deg, wheelbase {wheelbase:.3f} m, effective {eff:.1f} deg -> {OUT}")
    finally:
        if env:
            env.close()
        srv.shutdown_if_configured()


if __name__ == "__main__":
    main()
