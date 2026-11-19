"""Watch and score a trained policy (rendering on, chase camera).

Start CARLA with a window (start_carla_window.bat), then:
    python evaluate.py                  # latest run, Town01
    python evaluate.py --town Town02    # unseen-town test
    python evaluate.py --model <path to .zip> --episodes 10
"""
import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import carla_server as srv
import config as C
from carla_env import CarlaDriveEnv
from train import latest_checkpoint, latest_run, vecnorm_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--town", default=C.TRAIN_TOWN)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--npc", type=int, default=None, help="NPC count (default: config NPC_PER_TOWN)")
    ap.add_argument("--keep-carla", action="store_true", help="leave CARLA running afterwards")
    ap.add_argument("--no-render", action="store_true",
                    help="score only, no rendering (use with the headless training server)")
    args = ap.parse_args()

    model_path = args.model or latest_checkpoint(latest_run())
    if not model_path:
        raise SystemExit("no trained model found - run train.py first")
    print("model:", model_path, "| town:", args.town)
    try:
        evaluate(args, model_path)
    finally:
        srv.shutdown_if_configured(keep=args.keep_carla)


def evaluate(args, model_path):

    env = DummyVecEnv([lambda: CarlaDriveEnv(town=args.town, render=not args.no_render, num_npc=args.npc, seed=123)])
    env = VecNormalize.load(vecnorm_for(model_path), env)
    env.training, env.norm_reward = False, False
    model = PPO.load(model_path, env=env, device="cpu")

    results = []
    obs = env.reset()
    while len(results) < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        if dones[0]:
            i = infos[0]
            results.append(i)
            print(f"episode {len(results)}: {i['reason']:<10} route {i['route_completion']:5.0%} "
                  f"of {i['route_m']:.0f} m | mean |lat| {i['mean_lat_abs']:.2f} m | "
                  f"mean speed {i['mean_speed_kmh']:.1f} km/h | {smooth_line(i)}")
    env.close()

    print("\nsummary:", dict(Counter(r["reason"] for r in results)))
    print(f"mean route completion {np.mean([r['route_completion'] for r in results]):.0%}, "
          f"mean |lat| {np.mean([r['mean_lat_abs'] for r in results]):.2f} m")
    avg = {k: float(np.mean([r[k] for r in results])) for k in SMOOTH_KEYS}
    avg["max_lat_acc"] = float(np.max([r["max_lat_acc"] for r in results]))
    print("smoothness (lower = smoother):", smooth_line(avg))


SMOOTH_KEYS = ("mean_dsteer", "reversals_per_km", "mean_lat_acc", "max_lat_acc", "mean_lat_jerk")


def smooth_line(i):
    return (f"steer change/step {i['mean_dsteer']:.3f}, wiggles/km {i['reversals_per_km']:.0f}, "
            f"lat acc mean {i['mean_lat_acc']:.2f} max {i['max_lat_acc']:.1f} m/s2, "
            f"lat jerk {i['mean_lat_jerk']:.1f} m/s3")


if __name__ == "__main__":
    main()
