"""Summarise where a student's test drives end badly, from vision_collect --trace CSVs.
Args: trace.csv=log.log ...   For every off_route / collision ending, looks at the last 4 s before it:
junction or not, the sat-nav command, how far the car turned, student vs teacher steering."""
import csv
import math
import re
import sys
from collections import Counter, defaultdict

CMD = {1: "left", 2: "right", 3: "straight", 4: "follow"}
WIN = 80                     # steps before the ending (20 Hz -> 4 s)


def endings(log):
    out = {}
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"episode (\d+): (\w+)", line)
        if m and m.group(2) in ("off_route", "collision"):
            out[int(m.group(1))] = m.group(2)
    return out


def heading_change(rows):
    pts = [(float(r["x"]), float(r["y"])) for r in rows]
    if len(pts) < 12:
        return 0.0
    a = math.atan2(pts[5][1] - pts[0][1], pts[5][0] - pts[0][0])
    b = math.atan2(pts[-1][1] - pts[-6][1], pts[-1][0] - pts[-6][0])
    return math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)


def main():
    cats = Counter()
    for arg in sys.argv[1:]:
        tr, log = arg.split("=")
        ends = endings(log)
        eps = defaultdict(list)
        for r in csv.DictReader(open(tr, encoding="utf-8")):
            eps[int(r["episode"])].append(r)
        name = tr.split("trace_student_")[-1].replace(".csv", "")
        print(f"\n{name}: {len(eps)} episodes, {len(ends)} bad endings")
        for ep, reason in sorted(ends.items()):
            rows = eps.get(ep, [])[-WIN:]
            if not rows:
                print(f"  ep {ep}: {reason} (no trace rows)")
                continue
            last = rows[-1]
            cmds = [int(r["cmd"]) for r in rows]
            turn_cmd = next((c for c in reversed(cmds) if c in (1, 2, 3)), 4)
            in_j = any(int(r["junction"]) for r in rows[-30:])
            st = [float(r["steer_cmd"]) for r in rows[-40:]]
            te = [float(r["teacher_steer"]) for r in rows[-40:]]
            ms, mt = sum(st) / len(st), sum(te) / len(te)
            dh = heading_change(rows)
            if in_j and turn_cmd in (1, 2):
                kind = "junction turn " + CMD[turn_cmd]
                if abs(mt) > 0.1 and abs(ms) < 0.4 * abs(mt):
                    kind += ": UNDER-steered"
                elif mt * ms < 0 and abs(ms) > 0.05:
                    kind += ": WRONG way"
                elif abs(ms) > 1.6 * abs(mt) and abs(ms) > 0.1:
                    kind += ": OVER-steered"
            elif in_j:
                kind = "junction " + CMD[turn_cmd]
            else:
                kind = "outside junction (lane / bend)"
            if reason == "collision":
                kind = "collision, " + kind
            cats[kind] += 1
            print(f"  ep {ep:2d}: {reason:9s} {kind:40s} cmd {CMD[turn_cmd]:8s} turned {dh:+5.0f} deg in 4 s | "
                  f"steer student {ms:+.2f} teacher {mt:+.2f} | lat {float(last['lat']):+.2f} | "
                  f"{float(last['speed']) * 3.6:4.0f} km/h | route {int(last['route_idx'])}/{int(last['route_len'])}")
    print("\nby kind:")
    for k, n in cats.most_common():
        print(f"  {n:3d}  {k}")


if __name__ == "__main__":
    main()
