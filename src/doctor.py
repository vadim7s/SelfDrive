"""Setup checker: run this first, and paste its output into any GitHub issue.
    python doctor.py            (CARLA may be running or not; this never starts or stops it)
Each line is OK / WARN / FAIL, and every problem prints the fix in one line."""
import importlib
import importlib.metadata as md
import os
import shutil
import socket
import sys

import config as C

NEED = {"carla": "0.9.16", "torch": "2.14", "torchvision": "0.29", "numpy": "2.", "opencv-python": "5.",
        "stable-baselines3": "2.9", "gymnasium": "1.4", "transformers": "5.", "safetensors": "0.8",
        "pillow": "13.", "scipy": "1.18", "shapely": "2.", "pandas": "3.", "tensorboard": "2."}
fails = 0


def say(level, what, fix=""):
    global fails
    fails += level == "FAIL"
    print(f"[{level:4s}] {what}" + (f"\n         fix: {fix}" if fix and level != "OK" else ""))


def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 12)
    say("OK" if ok else "FAIL", f"Python {v.major}.{v.minor}.{v.micro}",
        "use Python 3.12: the carla 0.9.16 wheel is built for it")


def check_packages():
    for p, want in NEED.items():
        try:
            have = md.version(p)
        except md.PackageNotFoundError:
            say("FAIL", f"{p} not installed", "pip install -r requirements.txt")
            continue
        say("OK" if have.startswith(want) else "WARN", f"{p} {have}",
            f"tested with {want}*; if something breaks, pip install {p}=={want}...")


def check_gpu():
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        say("FAIL", f"torch {torch.__version__} cannot see a CUDA GPU",
            "install the CUDA build of torch from https://pytorch.org (pip's default may be CPU-only)")
        return
    p = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 2**30
    say("OK" if gb >= 7.5 else "WARN", f"GPU {p.name}, {gb:.1f} GB",
        "8 GB is what we used; with less, lower the batch size in bev_train.py (--batch)")


def check_carla():
    root = C.CARLA_ROOT
    exe = os.path.join(root, "CarlaUE4.exe")
    say("OK" if os.path.isfile(exe) else "FAIL", f"CARLA at {root}",
        "install CARLA 0.9.16 and set the CARLA_ROOT environment variable to its folder")
    api = os.path.join(root, "PythonAPI", "carla")
    sys.path.append(api)
    try:
        importlib.import_module("agents.navigation.global_route_planner")
        say("OK", "CARLA route planner (agents.navigation) imports")
    except Exception as e:
        say("FAIL", f"agents.navigation does not import: {e}",
            f"check that {api} exists; run scripts from src/ so config.py sets the path")
    try:
        with socket.create_connection((C.HOST, C.PORT), timeout=2):
            pass
    except OSError:
        say("WARN", f"no CARLA server on {C.HOST}:{C.PORT} (fine if it isn't started yet)",
            "start it with scripts\\start_carla.bat, wait ~1 minute, run doctor again")
        return
    try:
        import carla
        cl = carla.Client(C.HOST, C.PORT)
        cl.set_timeout(10.0)
        sv, cv = cl.get_server_version(), cl.get_client_version()
        say("OK" if sv == cv else "FAIL", f"CARLA server {sv}, python client {cv}",
            "server and pip 'carla' package must be the same version (0.9.16)")
    except Exception as e:
        say("FAIL", f"CARLA port is open but the server does not answer: {e}",
            "CARLA may still be loading; wait a minute. If it persists, close CarlaUE4 and restart it")


def check_disk_and_models():
    for d in {C.RUNS_DIR, C.VISION_RUNS_DIR}:
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free / 2**30
        say("OK" if free >= 50 else "WARN", f"{free:.0f} GB free at {d}",
            "a full data collection is ~100 GB; point FSD_DATA at a bigger drive")
    pts = [os.path.join(r, f) for r, _, fs in os.walk(C.VISION_RUNS_DIR) for f in fs if f.endswith(".pt")]
    say("OK" if pts else "WARN", f"{len(pts)} trained student(s) under {C.VISION_RUNS_DIR}",
        "download the pretrained models from the GitHub Release, or train your own (see README)")


if __name__ == "__main__":
    print(f"SelfDrive doctor | {sys.platform} | {sys.executable}")
    for f in (check_python, check_packages, check_gpu, check_carla, check_disk_and_models):
        try:
            f()
        except Exception as e:
            say("FAIL", f"{f.__name__} crashed: {e!r}", "paste this output into a GitHub issue")
    print("\nAll good." if not fails else f"\n{fails} problem(s). Fix the FAIL lines first; WARN lines are advice.")
    sys.exit(1 if fails else 0)
