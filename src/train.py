"""Stage 1: PPO on the compact privileged state (Town01, empty roads by default).

Normally started by run_training.py (which restarts CARLA and resumes on failures).
Direct use (no auto-recovery):
    python train.py              # new run
    python train.py --resume     # continue the latest run
Exit codes: 0 = finished/stopped, EXIT_RESTART_CARLA (3) = CARLA failed, model saved.
Watch progress:
    tensorboard.exe --logdir runs
Ctrl+C saves the model before exiting.
"""
import argparse
import glob
import os
import sys
import time
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config as C
from carla_env import CarlaDriveEnv, CarlaFailure


class EpisodeStats(BaseCallback):
    """Logs route completion and why episodes ended, to TensorBoard and the console."""

    def __init__(self):
        super().__init__()
        self.reasons, self.done_eps, self.stuck = Counter(), 0, []
        self.completion, self.smooth = [], {}

    def _on_step(self):
        for info in self.locals["infos"]:
            if "reason" in info:
                self.done_eps += 1
                self.reasons[info["reason"]] += 1
                self.completion.append(info["route_completion"])
                self.logger.record_mean("drive/route_completion", info["route_completion"])
                if "town" in info:
                    self.logger.record_mean(f"town/{info['town']}_completion", info["route_completion"])
                self.logger.record_mean("drive/mean_lat_abs_m", info["mean_lat_abs"])
                self.logger.record_mean("drive/mean_speed_kmh", info["mean_speed_kmh"])
                for k in ("mean_dsteer", "reversals_per_km", "mean_lat_acc", "max_lat_acc", "mean_lat_jerk"):
                    if k in info:
                        self.logger.record_mean(f"smooth/{k}", info[k])
                        self.smooth.setdefault(k, []).append(info[k])
                for k in ("collision", "off_route", "stuck", "route_done", "time_limit", "carla_restart"):
                    self.logger.record_mean(f"end/{k}", float(info["reason"] == k))
                if info["reason"] == "stuck" and "throttle_last5s" in info:
                    self.stuck.append(info)
                    self.logger.record_mean("stuck/throttle_last5s", info["throttle_last5s"])
                    self.logger.record_mean("stuck/at_junction", float(info["at_junction"]))
                if self.done_eps % 20 == 0:
                    tot = sum(self.reasons.values())
                    line = (f"[{self.num_timesteps:>8} steps] last 20 eps: "
                            + ", ".join(f"{k} {v / tot:.0%}" for k, v in self.reasons.most_common())
                            + f" | mean route {sum(self.completion) / len(self.completion):.0%}")
                    if self.stuck:
                        n = len(self.stuck)
                        line += (f" | stuck: throttle {sum(s['throttle_last5s'] for s in self.stuck) / n:.2f}, "
                                 f"at junction {sum(s['at_junction'] for s in self.stuck) / n:.0%}, "
                                 f"|lat| {sum(abs(s['end_lat']) for s in self.stuck) / n:.1f} m")
                    if self.smooth:
                        m = {k: sum(v) / len(v) for k, v in self.smooth.items()}
                        line += (f" | steer chg {m['mean_dsteer']:.3f}, wiggles/km {m['reversals_per_km']:.0f}, "
                                 f"lat acc {m['mean_lat_acc']:.2f} m/s2")
                    print(line, flush=True)
                    self.reasons.clear()
                    self.stuck, self.completion, self.smooth = [], [], {}
        return True


def latest_run():
    runs = sorted(glob.glob(os.path.join(C.RUNS_DIR, "ppo_*")), key=os.path.getmtime)
    return runs[-1] if runs else None


def latest_checkpoint(run):
    final = os.path.join(run, "final_model.zip")
    ckpts = sorted(glob.glob(os.path.join(run, "checkpoints", "ppo_*_steps.zip")), key=os.path.getmtime)
    cands = [p for p in [final] + ckpts if os.path.exists(p)]
    return max(cands, key=os.path.getmtime) if cands else None


def vecnorm_for(model_path):
    if model_path.endswith("final_model.zip"):
        return os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    folder, name = os.path.split(model_path)            # ppo_25000_steps.zip
    return os.path.join(folder, name.replace("ppo_", "ppo_vecnormalize_", 1).replace(".zip", ".pkl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="continue the latest run")
    ap.add_argument("--steps", type=int, default=None, help="total steps for this run")
    ap.add_argument("--init", default=None,
                    help="start a NEW run from this model's weights (fine-tuning, e.g. new rewards)")
    args = ap.parse_args()

    try:
        env = DummyVecEnv([lambda: Monitor(CarlaDriveEnv(town=C.TRAIN_TOWN, seed=C.SEED, multi_town=True))])
    except CarlaFailure as e:
        print("CARLA not usable:", e)
        sys.exit(C.EXIT_RESTART_CARLA)

    if args.resume and latest_run() and latest_checkpoint(latest_run()):
        run = latest_run()
        ckpt = latest_checkpoint(run)
        print("resuming", ckpt)
        env = VecNormalize.load(vecnorm_for(ckpt), env)
        model = PPO.load(ckpt, env=env, tensorboard_log=run, device="cpu")
        reset_steps = False
    else:
        run = os.path.join(C.RUNS_DIR, time.strftime("ppo_%Y%m%d_%H%M%S"))
        os.makedirs(run, exist_ok=True)
        if args.init:
            # keep the policy's skill, keep its observation scaling; reward scaling re-adapts
            print("fine-tuning from", args.init)
            env = VecNormalize.load(vecnorm_for(args.init), env)
            env.training = True
            model = PPO.load(args.init, env=env, tensorboard_log=run, device="cpu",
                             custom_objects={"learning_rate": C.FINETUNE_LR})
            model._setup_lr_schedule()                     # apply the lower fine-tune learning rate
            print("learning rate", model.learning_rate)
            with open(os.path.join(run, "init_from.txt"), "w", encoding="utf-8") as f:
                f.write(args.init)
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO("MlpPolicy", env, verbose=1, seed=C.SEED, tensorboard_log=run,
                        device="cpu", **C.PPO)
        reset_steps = True
    print("run folder:", run)

    # the run's step target is stored in the run folder so a resume after a CARLA restart keeps it
    target_file = os.path.join(run, "target_steps.txt")
    if args.steps is None:
        args.steps = (int(open(target_file).read()) if os.path.exists(target_file)
                      else C.FINETUNE_STEPS if args.init else C.TOTAL_TIMESTEPS)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(str(args.steps))

    callbacks = [EpisodeStats(),
                 CheckpointCallback(C.CHECKPOINT_EVERY, os.path.join(run, "checkpoints"), "ppo",
                                    save_vecnormalize=True)]
    # SB3 adds total_timesteps on top of the steps already done when resuming -> pass the remainder
    remaining = args.steps - (0 if reset_steps else model.num_timesteps)
    if remaining <= 0:
        print(f"already trained {model.num_timesteps} of {args.steps} steps - done")
        sys.exit(0)

    exit_code, carla_dead = 0, False
    try:
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=reset_steps,
                    tb_log_name="tb")
    except KeyboardInterrupt:
        print("interrupted - saving")
    except CarlaFailure as e:
        print(f"CARLA failure at {model.num_timesteps} steps: {e} - saving, CARLA needs a restart")
        exit_code, carla_dead = C.EXIT_RESTART_CARLA, True
    finally:
        model.save(os.path.join(run, "final_model"))
        env.save(os.path.join(run, "vecnormalize.pkl"))
        if not carla_dead:
            env.close()
        print(f"saved to {run} at {model.num_timesteps} steps")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
