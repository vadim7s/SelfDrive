"""Print a compact table from tesla_test.py result JSONs.
    python tesla_report.py student_v2 [student_v3 ...]
"""
import config as C
import json
import os
import sys

OUT_DIR = os.path.join(C.DATA_ROOT, "tesla_test")
for name in sys.argv[1:]:
    j = json.load(open(os.path.join(OUT_DIR, name + ".json"), encoding="utf-8"))
    print(f"== {name}  ({j['target']}, {j['turn_steps']} turn steps, geometry {j['geometry']['source']})")
    print(f"   {'mode':<12} {'corr':>5} {'MAE':>7} {'sign%':>6} {'turnCorr':>8} {'bias':>7} {'jitter':>7}")
    for m in ("oracle", "follow", "camera_only", "mirrored"):
        if m not in j:
            continue
        r = j[m]
        print(f"   {m:<12} {r['all']['corr']:5.2f} {r['all']['mae']:7.4f} {100 * (r['all']['sign_agree'] or 0):6.0f} "
              f"{r['turns'].get('corr', float('nan')):8.2f} {r.get('straight_bias', 0):+7.4f} {r.get('jitter', 0):7.4f}")
    print(f"   baselines: always-straight MAE {j['baseline_zero']['all']['mae']:.4f} | "
          f"hold-current-wheel corr {j['baseline_hold_wheel']['all']['corr']:.2f} MAE {j['baseline_hold_wheel']['all']['mae']:.4f}")
