"""Milestone clip across ALL towns: N routes per town (default 2, each up to 15 s, routes with an early
junction turn preferred), one CARLA session per town (record_clip.py), joined into one .mp4.
    python record_tour.py --steady --label "Teacher v5"
    ... --student <best.pt> --label "Student H"
Output: data\\milestone_clips\\<stamp>_<label>_all_towns.mp4 (+ .json). CARLA is shut down at the end.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import config as C
import carla_server as srv

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(C.DATA_ROOT, "milestone_clips")
TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=C.VISION_TEACHER)
    ap.add_argument("--student", default=None)
    ap.add_argument("--steady", action="store_true")
    ap.add_argument("--routes", type=int, default=2, help="routes per town")
    ap.add_argument("--route-seconds", type=float, default=15.0)
    ap.add_argument("--towns", nargs="+", default=TOWNS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg not found on PATH")
    stamp = time.strftime("%Y%m%d_%H%M")
    slug = "".join(ch if ch.isalnum() else "_" for ch in args.label).strip("_").lower()
    out = os.path.join(CLIPS_DIR, f"{stamp}_{slug}_all_towns.mp4")
    tmp = os.path.join(CLIPS_DIR, f"_tmp_{stamp}")
    os.makedirs(tmp, exist_ok=True)
    parts, meta = [], dict(label=args.label, model=args.student or args.model, towns={})
    try:
        for town in args.towns:
            part = os.path.join(tmp, f"{town}.mp4")
            cmd = [sys.executable, "-u", os.path.join(HERE, "record_clip.py"), "--label", args.label,
                   "--town", town, "--routes", str(args.routes), "--route-seconds", str(args.route_seconds),
                   "--prefer-turns", "--seed", str(args.seed), "--out", part]
            cmd += ["--student", args.student] if args.student else ["--model", args.model]
            if args.steady and not args.student:
                cmd.append("--steady")
            print(f"== {town}  {time.strftime('%H:%M')}", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
            line = [l for l in r.stdout.splitlines() if l.startswith("clip:")]
            print("  ", line[-1] if line else f"FAILED (code {r.returncode}): {r.stderr.strip().splitlines()[-1:] }", flush=True)
            if os.path.exists(part) and line:
                parts.append(part)
                with open(part.replace(".mp4", ".json"), encoding="utf-8") as f:
                    meta["towns"][town] = json.load(f)
        if not parts:
            raise SystemExit("no town recorded")
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
        with open(out.replace(".mp4", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("tour clip:", out, flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        srv.shutdown_if_configured()
        print("CARLA running:", bool(srv.processes_running()), flush=True)


if __name__ == "__main__":
    main()
