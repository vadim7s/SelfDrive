"""Supervisor: the one command for long training runs. Keeps CARLA and training alive.

    python run_training.py         # resume latest run (or start one)
    python run_training.py --new   # start a fresh run
    ... run_training.py --finetune   # new run from config.FINETUNE_FROM's weights (reward changes)

What it does
  - starts CARLA if needed, then runs train.py as a child process
  - train.py exits with EXIT_RESTART_CARLA when CARLA hangs/crashes (model saved first)
    -> restart CARLA, resume from that save
  - train.py crashes hard (no save) -> restart CARLA, resume from the last checkpoint
  - no heartbeat for STALE_S (a freeze with no error) -> kill training + CARLA, restart, resume
  - exit code 0 from train.py = training finished -> stop
Every restart is logged to runs\\carla_events.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import config as C
import carla_server as srv
from train import latest_checkpoint, latest_run

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
POLL_S = 15
STATUS_EVERY_S = 120


def read_heartbeat():
    try:
        with open(C.HEARTBEAT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def kill_tree(pid):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def run_child(resume, restarts, init=None):
    cmd = [PY, "-u", os.path.join(HERE, "train.py")] + (["--resume"] if resume else [])
    if init and not resume:
        cmd += ["--init", init]
    env = dict(os.environ, CARLA_RESTARTS=str(restarts))
    p = subprocess.Popen(cmd, cwd=HERE, env=env)
    started, last_status = time.time(), time.time()
    while p.poll() is None:
        time.sleep(POLL_S)
        hb = read_heartbeat()
        hb_time = hb["time"] if hb and hb.get("pid") == p.pid else started
        age = time.time() - hb_time
        if age > C.STALE_S:
            srv.log_event(f"FROZEN: no heartbeat for {age:.0f}s - killing training (pid {p.pid}) and CARLA")
            kill_tree(p.pid)
            srv.kill()
            p.wait()
            return "frozen"
        if time.time() - last_status > STATUS_EVERY_S and hb and hb.get("pid") == p.pid:
            print(f"[supervisor {time.strftime('%H:%M:%S')}] steps {hb['total_steps']} this session, "
                  f"{hb['steps_per_s']} steps/s, restarts so far {restarts}", flush=True)
            last_status = time.time()
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="store_true", help="start a fresh run instead of resuming")
    ap.add_argument("--finetune", action="store_true",
                    help="new run starting from config.FINETUNE_FROM (e.g. after reward changes)")
    args = ap.parse_args()

    init = C.FINETUNE_FROM if args.finetune else None
    resume = not args.new and not args.finetune and latest_run() is not None
    restarts = 0
    srv.log_event(f"supervisor start ({'resume' if resume else 'fine-tune' if init else 'new run'})")
    run_before = latest_run()
    while True:
        if not srv.ensure_running():
            srv.restart("CARLA did not start")
        result = run_child(resume, restarts, init)
        # only resume once the child has actually created its run with a save in it;
        # otherwise a failure before that would "resume" an older, unrelated run
        new_run = latest_run()
        created = new_run is not None and new_run != run_before and latest_checkpoint(new_run)
        if result == 0:
            srv.log_event("training finished")
            break
        restarts += 1
        if restarts > C.MAX_RESTARTS:
            srv.log_event(f"giving up after {restarts - 1} restarts")
            sys.exit(1)
        why = {C.EXIT_RESTART_CARLA: "CARLA failure (model saved)", "frozen": "training froze"}.get(
            result, f"training crashed with exit code {result} (resuming from last checkpoint)")
        srv.restart(f"{why} - restart #{restarts}")
        resume = resume or bool(created)
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("supervisor stopped (training child gets Ctrl+C too and saves)")
        time.sleep(10)                      # give the child time to save before CARLA goes
    finally:
        srv.shutdown_if_configured()
