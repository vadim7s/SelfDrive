"""Start / stop / health-check the CARLA server process, plus heartbeat + event logging."""
import json
import os
import socket
import subprocess
import time

import carla

import config as C


def log_event(msg):
    os.makedirs(os.path.dirname(C.EVENTS_LOG), exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print("[carla]", line)
    with open(C.EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_heartbeat(**fields):
    """Best effort: a heartbeat write must never crash training.

    os.replace is atomic (readers never see half a file), but on Windows it fails with
    'Access is denied' while another process (supervisor, watcher, antivirus) has the target open.
    So retry briefly, then just skip this beat - the next one comes 50 steps later.
    """
    fields["time"] = time.time()
    fields["pid"] = os.getpid()
    tmp = f"{C.HEARTBEAT_FILE}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(C.HEARTBEAT_FILE), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(fields, f)
        for _ in range(10):
            try:
                os.replace(tmp, C.HEARTBEAT_FILE)
                return
            except PermissionError:
                time.sleep(0.02)
        os.remove(tmp)
    except OSError:
        pass


def port_open(port=C.PORT, host=C.HOST):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def server_responds(timeout=10.0):
    if not port_open():
        return False
    try:
        c = carla.Client(C.HOST, C.PORT)
        c.set_timeout(timeout)
        c.get_server_version()
        return True
    except RuntimeError:
        return False


def processes_running():
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines()
            if any(p.lower() in line.lower() for p in C.CARLA_PROCESSES)]


def kill():
    for name in C.CARLA_PROCESSES:
        subprocess.run(["taskkill", "/F", "/T", "/IM", name], capture_output=True)
    for _ in range(30):                                 # wait for the port to be released
        if not processes_running() and not port_open():
            return
        time.sleep(1)


def next_training_town():
    """Town the teacher-training rotation will load next (runs/train_town_state.json), else None."""
    try:
        with open(C.TOWN_STATE_FILE, encoding="utf-8") as f:
            return json.load(f).get("town")
    except (OSError, ValueError, AttributeError):
        return None


def start():
    town = next_training_town()
    extra = C.CARLA_TOWN_ARGS.get(town)                  # e.g. Town03 never loads at -quality-level=Low
    args = [C.CARLA_EXE, *(extra or C.CARLA_ARGS), f"-carla-rpc-port={C.PORT}"]
    if extra:
        log_event(f"{town} next: starting CARLA with {' '.join(extra)}")
    subprocess.Popen(args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    t0 = time.time()
    while time.time() - t0 < C.BOOT_TIMEOUT_S:
        if server_responds(5.0):
            time.sleep(5)                                # let it finish loading the default map
            log_event(f"CARLA up after {time.time() - t0:.0f}s")
            return True
        time.sleep(3)
    log_event("CARLA did not come up within the boot timeout")
    return False


def ensure_running():
    """Start CARLA if it isn't already serving."""
    if server_responds():
        return True
    if processes_running():
        log_event("CARLA process exists but does not respond - killing it")
        kill()
    log_event("starting CARLA")
    return start()


def shutdown_if_configured(keep=False):
    """End-of-job cleanup: close CARLA so it doesn't keep loading the GPU."""
    if keep or not C.SHUTDOWN_CARLA_WHEN_DONE:
        return
    if processes_running():
        kill()
        log_event("CARLA shut down (job finished)")


def restart(reason):
    log_event(f"RESTART: {reason}")
    kill()
    time.sleep(3)
    if not start():
        kill()
        time.sleep(5)
        if not start():
            raise RuntimeError("CARLA failed to restart twice - stopping")
