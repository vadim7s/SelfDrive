# SelfDrive: from a cheating teacher to a bird's-eye-view student

A self-driving model built in CARLA 0.9.16, with one goal: **prove that what a model learns in the simulator carries
over to a real car.** A privileged "teacher" learns to drive with reinforcement learning. A "student" that sees only
the camera learns to copy it. The student is then tested on real dashcam footage.

> Looking for the 2025 tutorials (RL tutorial, map projection, first lane model)? They are kept, unchanged, at tag
> [`legacy-2025`](../../tree/legacy-2025).

## The pipeline

```
 camera image ──► segmentation ──► projected onto the ground ──► fused over time ──► bird's-eye map
 (Mask2Former,      (road, lines,     (flat-ground camera          (memory, moved      + satnav route curve
  real-photo         kerbs, cars)      model, real-car camera)      by our own motion)   (GPS-grade)
  trained)                                                                                   │
                                                                                             ▼
          speed, cruise speed, satnav command ──────────────────────────────► CNN ─► GRU (3.2 s memory)
                                                                                             │
                                                     steer + speed  ◄─────────────────────────┤
                                                     read-outs: situation, curvature, distance to junction / stop
```

1. **Teacher (RL).** PPO on a compact privileged state: exact lane position, route and traffic. Plus a
   pure-pursuit lane tracker for steering. Only the teacher ever uses rewards.
2. **Students (imitation learning).** Behaviour cloning on the teacher's drives, then DAgger: the student drives
   while the teacher labels every frame.
3. **Camera-only students (baseline).** ResNet-18 on the windscreen image. They plateaued at ~56% route completion
   on the unseen town: the network learned to copy the steering-wheel input instead of the road (the "copycat").
4. **Bird's-eye-view student (current).** Segmentation, then a top-down map with memory, then a recurrent policy.
   No steering-wheel or yaw-rate input. The satnav route is drawn on the map as a curve with deliberate GPS-like
   errors, so the student uses it for *where to go* and the camera for *where the lane is*.

## Results (Town02, never seen in training, traffic on, 20 routes)

| Student | Input | Route completion | Routes finished |
|---|---|---|---|
| I / J | camera image + wheel angle + command | 56% | 4 / 20 |
| K | bird's-eye map + memory + command | 67% | 9 / 20 |
| K2 | K + DAgger | 70% | 10 / 20 |
| K3 | K2 + turn-weighted loss | 73% | 11 / 20 |
| K4 | K3 + satnav route curve as a map layer | 74.5% | 13 / 20 |
| **K5** | route curve twice: map layer + 8 route points into memory, trained from scratch | **88.5%** | **17 / 20** |
| K6 | K5 recipe on "voting" map memory | 89% | 17 / 20 |

- With the satnav route moved 2 m off to the side (GPS error), K5 and K6 still finish 19 / 20. That's traffic luck,
  not "better": read it as "no worse".
- K5's remaining misses are all side collisions with cars crossing at junctions. Next weak spot: giving way.

## Real road (open loop)

The K5 student, trained in CARLA only, watched one 60 s real suburban dashcam drive and predicted the steering at
every frame (it never drives the car). The footage is mirrored, because CARLA drives on the right.

| Setup | Correlation with the real path | Junction turns |
|---|---|---|
| Camera-only students (given the real wheel angle) | 0.51 - 0.63 | |
| K5 | 0.79 | 0.97, every turn the right way |
| **K5 + voting map memory at test time** | **0.85** | 0.97 |

Honest weak spot: on straights it still weaves a little. One clip is not proof; more real clips are next.

Design notes, experiments and the reasoning behind each step are in [docs/design_notes.md](docs/design_notes.md).

## Layout

```
src/       all Python code (run scripts from here; settings in src/config.py)
scripts/   the PowerShell recipes we actually ran (data collection, DAgger, training, tests)
docs/      design notes
data/      datasets, trained models, clips (git-ignored; created on first run)
```

Main modules:

| Stage | Files |
|---|---|
| Simulator + environment | `carla_env.py`, `carla_server.py`, `traffic.py`, `route_smooth.py`, `config.py` |
| Teacher (PPO) | `train.py`, `run_training.py` (supervisor), `teacher_steady.py`, `evaluate.py` |
| Data collection + DAgger | `vision_collect.py` |
| Segmentation | `seg_check.py` (segmenter), `seg_capture.py`, `seg_dataset.py` |
| Bird's-eye map | `bev_project.py`, `bev_map.py` (map truth from OpenDRIVE), `bev_dataset.py`, `bev_route.py` |
| Student | `bev_model.py`, `bev_train.py`, `situation.py` (situation labels) |
| Camera-only baseline | `vision_model.py`, `vision_train.py` |
| Clips | `record_clip.py`, `record_tour.py` |
| Real footage | `tesla_bev_test.py` (open-loop test + path video), `tesla_test.py`, `bev_tesla.py`, `tesla_calib.py` |
| Diagnostics | `doctor.py` (setup check), `route_reliance.py`, `map_quality.py`, `diag_student_report.py`, `student_offline_check.py`, `bev_check.py`, `turn_report.py`, `steady_report.py` |

## Setup

- Windows, an NVIDIA GPU (developed on an 8 GB RTX 3060 Ti), [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases)
- Python 3.12: `pip install -r requirements.txt` (install the CUDA build of PyTorch for your GPU first)
- Environment variables, all optional:
  - `CARLA_ROOT`: CARLA install folder (default `C:\CARLA`)
  - `FSD_DATA`: where data, models and clips go (default `data\` in the repo)
  - `FSD_TESLA_CLIP`: your own dashcam clip, for the real-footage test

First, check your setup (paste its output into any GitHub issue):

```
cd src
python doctor.py
```

Then a quick check that CARLA and the environment work:

```
scripts\start_carla.bat
cd src
python smoke_test.py
```

Typical order: train the teacher (`scripts/teacher_traffic.ps1`), collect data (`scripts/phaseC.ps1`), build the
map cache (`python bev_dataset.py ..\data\vision_data_v2`), train the student (K5 recipe: `scripts/phaseD_route2.ps1`, or
`python bev_train.py --data ..\data\vision_data_v2 --epochs 20 --lr 0.0003 --turn-weight 4 --route`), then test it (`scripts/phaseD_test.ps1 -Student <best.pt>`).

The scripts in `scripts/` are the exact recipes from our runs. Some point at specific checkpoints from earlier rounds;
change those to your own.

## Notes

- **Segmenter licence.** The segmenter weights (`facebook/mask2former-swin-large-mapillary-vistas-semantic`) are
  downloaded on first use, not included here. They were trained on Mapillary Vistas, whose licence is
  **non-commercial**.
- **Sim-to-real honesty.** The segmenter learned from real street photos, not from the simulator. The driving
  policy is trained on CARLA data only. On real footage the student *predicts* steering; it does not drive a car.
- **No real-world data is included.** Dashcam clips and telemetry stay private.

## Licence

MIT, see [LICENSE](LICENSE). Third-party models and CARLA keep their own licences.
