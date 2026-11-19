"""Live status of CARLA + training (view only; run_training.py does the restarting).

Run in a second terminal:
    python watch_carla.py

Every 10 s: CARLA process up?, CARLA memory, does the server answer, heartbeat age,
env steps/s, steps, episodes, restarts, and the last event (failure/restart).
"""
import json
import subprocess
import time

import config as C
import carla_server as srv

INTERVAL_S = 10


def carla_memory_gb():
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq CarlaUE4-Win64-Shipping.exe"],
                         capture_output=True, text=True).stdout
    total = 0.0
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 5 and parts[4].rstrip('"').endswith("K"):
            digits = "".join(ch for ch in parts[4] if ch.isdigit())
            total += int(digits) / 1024 / 1024 if digits else 0.0
    return total


def read_heartbeat():
    try:
        with open(C.HEARTBEAT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    print(f"watching CARLA on port {C.PORT}; heartbeat {C.HEARTBEAT_FILE}. Ctrl+C to stop.")
    while True:
        procs = bool(srv.processes_running())
        mem = carla_memory_gb() if procs else 0.0
        answers = srv.server_responds(5.0) if procs else False
        hb = read_heartbeat()
        age = time.time() - hb["time"] if hb else None

        if not procs:
            status = "CARLA NOT RUNNING"
        elif not answers:
            status = "CARLA NOT ANSWERING"
        elif age is not None and age > 60:
            status = f"NO HEARTBEAT {age:.0f}s"
        else:
            status = "OK"

        hb_txt = (f"hb {age:4.0f}s ago | {hb['steps_per_s']:5.1f} steps/s | steps {hb['total_steps']:>8} | "
                  f"eps {hb['episodes']:>5} | restarts {hb['restarts']}" if hb else "no heartbeat yet")
        event = f" | last: {hb['event']}" if hb and hb.get("event") else ""
        print(f"{time.strftime('%H:%M:%S')}  {status:<20} | CARLA {'up' if procs else 'down'} "
              f"{mem:4.1f} GB | {hb_txt}{event}", flush=True)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
