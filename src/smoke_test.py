"""Checks the environment end to end, no learning.

1. A hand-written driver: if it completes routes with sensible rewards, the env is sound
   and any later training problem is in the RL, not the plumbing.
2. A random policy like PPO's starting point (Gaussian, std 1, clipped): it must keep
   rolling, not stall. If random actions mostly end "stuck", the agent can learn to park.

    python smoke_test.py
"""
import math
import time
from collections import Counter

import numpy as np

import carla_server as srv
import config as C
from carla_env import CarlaDriveEnv

SPEED_CMD = 2.0 / C.V_MAX_FACTOR - 1.0      # action[1] that asks for exactly the target speed


def scripted_driver(obs):
    """Steer toward the nearest look-ahead point; hold target speed."""
    fx, fy = obs[8], obs[9]             # 1st look-ahead point (~6 m): no corner cutting
    steer = np.clip(1.6 * math.atan2(fy, fx), -1, 1)
    if C.SPEED_CONTROL:
        accel = SPEED_CMD
    else:
        accel = 0.6 if obs[0] < 0.95 else (-0.2 if obs[0] > 1.1 else 0.1)
    return np.array([steer, accel], dtype=np.float32)


def random_policy(rng):
    return np.clip(rng.normal(0.0, 1.0, size=2), -1, 1).astype(np.float32)


def run(env, policy, label, episodes):
    ends = Counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        total, t0, n = 0.0, time.time(), 0
        while True:
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            n += 1
            if term or trunc:
                break
        ends[info["reason"]] += 1
        print(f"{label} {ep + 1}: {info['reason']:<10} route {info['route_completion']:5.0%} "
              f"of {info['route_m']:.0f} m | steps {n} | return {total:7.1f} | "
              f"mean |lat| {info['mean_lat_abs']:.2f} m | {info['mean_speed_kmh']:.1f} km/h | "
              f"{n / (time.time() - t0):.0f} steps/s")
    return ends


def main():
    env = CarlaDriveEnv(town=C.TRAIN_TOWN, seed=1)
    print("observation size:", env.observation_space.shape[0], "| speed control:", C.SPEED_CONTROL)
    rng = np.random.default_rng(0)
    try:
        run(env, scripted_driver, "scripted", 3)
        ends = run(env, lambda _o: random_policy(rng), "random  ", 10)
        print("random policy endings:", dict(ends),
              "-> OK, it doesn't stall" if ends["stuck"] <= 2 else "-> WARNING: random actions stall the car")
    finally:
        env.close()
        srv.shutdown_if_configured()


if __name__ == "__main__":
    main()
