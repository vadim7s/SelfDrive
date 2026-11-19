# Segmentation + top-down middle layer (option 3) — design note

Status: AGREED 16 Nov 2026, not built.
Goal: a student that drives from a segmented TOP-DOWN view of the scene, which the same
pipeline produces from CARLA and from real Tesla footage. That makes sim-to-real transfer
visually provable, and the car narrates what it thinks it is doing.

## Decisions
- View: TOP-DOWN (bird's-eye) masks, not front-view masks.
- Situation list: as in section 5 (agreed).
- Narration: SHOWN on screen in clips (and logged).
- 17 Nov: camera-only students stopped here (I, J plateau at ~55% Town02). Phase A starts now.
- NO wheel-angle input for the top-down student. Rounds 7-8 showed copycat steering: the net copies the wheel input,
  so in closed loop it copies its own weak steering. "Sticking to a turn" must come from the situation head + GRU
  memory instead, not from the current wheel.
- Segmenter throughput: ~600k CARLA frames must be segmented. If the best (large) model is too slow, it labels a
  subset of CARLA + Tesla frames and a small fast segmenter is trained on its output (distillation) to run on everything.

## 1. Classes in the middle layer
| Class | Notes |
|---|---|
| Drivable road | |
| Lane markings | needs a Mapillary Vistas-trained segmenter (Cityscapes has none) |
| Kerb / sidewalk | |
| Vehicles | |
| Pedestrians / cyclists | |
| Traffic lights + state | the mask gives position; red/amber/green comes from a small colour check inside the mask |
| Background | everything else |

## 2. Where the masks come from
- ONE pretrained real-world segmenter runs on BOTH CARLA RGB and Tesla frames, and the student trains on its output.
  Its mistakes then look the same in sim and real.
- CARLA's ground-truth semantic camera is used ONLY to score the segmenter (IoU per class).
- Consequence: existing RGB datasets (vision_data_t5, round 7) stay usable. No re-collection for the masks.

## 3. Top-down projection
- Flat-ground homography from camera height, pitch and focal length.
- CARLA: the Tesla-matched camera settings. Tesla: the same values, checked on footage
  (lane width ~3.5 m, parallel markings stay parallel).
- Presentation (decided 17 Nov): car-centred, HEADING-UP. Our car is fixed in the grid pointing "north" (up)
  and the world rotates and slides around it. The same frame is used for the student input and for the on-screen view.
  Past frames in the time stack are rotated and shifted into the CURRENT heading-up frame.
- Grid: about 40 m ahead, 10 m behind, ±12 m sideways at 0.2 m/cell. Final size TBD in Phase B.
- Known limit: anything above the ground is smeared. Only footprints matter, and vehicles are drawn from their ground contact.

## 4. Time context
- Short memory: a stack of the last 4–6 top-down frames (~1–1.5 s), each shifted by our own
  speed and yaw rate so they line up with the current frame.
- Long memory: a small GRU carrying state through a manoeuvre ("mid-turn", "waiting to give way").
- Ego motion inputs: CARLA = vehicle state. Tesla = telemetry speed + yaw rate (or wheel angle).

## 5. Outputs (driving + explainability)
Driving: steer, target speed (as now; inputs include the satnav command and road_speed).

Extra outputs, all labelled for free in CARLA:
- Situation (one class per frame):
  - lane follow: straight / left bend / right bend
  - approaching junction
  - junction: turning left / turning right / straight through
  - lane change (left / right)
  - following a car
  - stopping for red
  - giving way
- Lane geometry: lateral offset, heading error, curvature ahead.
- Hazards: gap to lead car, distance to stop line / red light.

"Left bend" vs "turning left at a junction" is separated by the junction label (route plan + map),
not by steering, so the student must learn it from the junction layout in the masks plus the command.

Situation labelling rules (CARLA, per frame):
- junction turn = the route segment is inside a junction with route option left/right/straight
- approaching junction = within a speed-scaled distance of the next junction entry
- bend = curvature over a threshold outside a junction, sign gives left/right
- following = lead_gap within sense range and the lead speed caps ours
- red / give way = the red_dist / yield_dist cap is the active one
- Priority when several apply: red > give way > junction turn > approaching > lane change > following > bend > straight

## 6. Narration in clips
- On-screen line, for example: "turning left at junction · giving way · stop line 12 m".
- Built from the situation and hazard outputs, smoothed over ~0.5 s so it doesn't flicker.
- Clip layout: camera view | segmented top-down view with ego box + planned path | narration bar.
- In sim, show the true situation small next to the predicted one, so mismatches are visible.

## 7. Validation (what proves transfer)
1. Segmenter on CARLA vs ground truth: IoU per class, by town and weather.
2. Tesla footage:
   - mask + top-down overlays on the video
   - predicted situation timeline vs telemetry (wheel angle, speed, turn events)
   - predicted steering vs the real wheel
3. Closed-loop replay on real roads (option 2): offset the ego in the top-down view and check that the student recovers.
4. Town02 with traffic, seeds 123 + 456: compare with the camera-only student (round-7 I/H). The bridge must not cost route completion or collisions.

## 8. Left-hand traffic (Australia)
CARLA drives on the right, the Tesla footage on the left.
Mirror the top-down grid left-right and swap left/right in commands, situations and the signs of lateral offset, heading and curvature.
Apply at test time on real footage. Alternatively, mirror-augment in training.

## 9. Phases (go / no-go)
| Phase | Work | Needs CARLA | Go if |
|---|---|---|---|
| A | pick the segmenter; run on Tesla clips + CARLA frames | no | road, markings and kerb are clean on Tesla footage; IoU on CARLA is acceptable |
| B | top-down calibration both sides; ego-aligned stack | no (uses recorded frames) | markings stay parallel at ~3.5 m spacing on both |
| C | situation + hazard labels in collected data | maybe (if a field is missing from labels.csv) | label timeline looks right on sampled clips |
| D | train the top-down student + extra outputs; Town02 test; Tesla eval; narrated milestone tour clip | yes | Town02 no worse than camera-only; Tesla overlays + narration convincing |

## Phase A result (17 Nov) = GO
Segmenter: Mask2Former Swin-L, Mapillary Vistas (data\models\m2f_vistas); scripts seg_capture.py + seg_check.py.
- CARLA, 216 frames (6 towns x 3 weathers), IoU vs CARLA truth:

  | Input | Road | Lane marking | Kerb/sidewalk | Vehicle | Traffic light | s/frame |
  |---|---|---|---|---|---|---|
  | 1280x960 | 0.94 | 0.57 | 0.82 | 0.69 | 0.74 | 0.23 |
  | 640x480 | 0.94 | 0.54 | 0.82 | 0.69 | 0.65 | 0.11 |
  | 320x240 | 0.91 | 0.45 | 0.70 | 0.67 | 0.54 | 0.10 |

  Notes:
  - Lane-marking IoU is low partly because thin lines lose a lot per pixel of offset, and partly because the segmenter
    marks painted arrows that CARLA's truth calls road.
  - Vehicle IoU is lowered by our car's bonnet, which the truth counts as a vehicle.
  - Pedestrians weren't spawned, so they aren't scored.
- Tesla clip (60 s): road, kerbs, lane lines, bike symbols, arrows and the roundabout give-way line all come out
  clean by eye. Weak spots: faint worn lines, and dashes sometimes merge into one line.
  Overlays: data\seg_phaseA\tesla_PRIVATE (real footage, private).
- Consequences:
  - Our stored 320x240 frames are too small (markings 0.45, kerb 0.70).
  - Collect new data at >= 640x480 with ego pose (x, y, z, yaw) per frame. That is needed anyway for the map-based
    top-down truth (bev_map.py) and the situation labels.
  - ~0.11 s/frame at 640x480 = ~2 h per 60k frames; distil to a fast segmenter if the set gets large.
- Roundabouts appear on real footage (Australia) and in CARLA Town03: candidate extra situation class.
- Tesla-style map truth from CARLA road files: bev_map.py (heading-up; lane lines, road edges, bridge levels handled).
  Demos: data\bev_maps\demo_*.mp4

## Phase B result (17 Nov) = GO
Code: bev_project.py (projection, footprints, Fuser), bev_map.py (truth map), bev_check.py, bev_stack_check.py, bev_tesla.py.
- Projection checked against the map truth at the true pose (144 CARLA frames, 3-30 m ahead):

  | Source | Road IoU | Projected lines within 0.3 m of truth | Truth lines found |
  |---|---|---|---|
  | CARLA perfect mask | 0.87 | 0.47 | 0.57 |
  | Segmenter | 0.85 | 0.44 | 0.67 |

  - Pitch / height / fore-aft / FOV searches are all optimal at the configured camera, so the geometry is right.
  - Line scores are limited by far dashes smearing lengthwise and by a strict 0.3 m tolerance.
  - The truth map had to include parking / shoulder / bike lanes up to the kerb (Town05 bridges are filtered by road height).
- Single front camera = a narrow wedge (52 deg; the bonnet hides the first ~3.5 m).
  Time stack (Fuser): road / lines / kerbs remembered and moved by our own motion; vehicles only from the current frame,
  drawn as ground footprints; only 4.5-25 m ahead updates the memory.
  CARLA 10 Hz sequence, 20 s in Town03: coverage of 40x40 m around the car goes from 15% to 47%;
  road IoU vs truth goes from 0.95 to 0.92.
  The fused view shows the whole bend being driven (the "stick to it" context).
  Video: data\seg_phaseA\bev_check\stack_Town03.mp4
- Tesla calibration (real-footage fit, a DISCLOSED caveat):
  - pitch -2.0 deg: lane lines parallel on 92 straight frames; the config guess was +1.0
  - height 1.56 m: from a 3.3 m lane; config 1.30
  - saved to data\seg_phaseA\tesla_PRIVATE\tesla_cam_fit.json
  - Open: the actual Tesla model / camera height would replace the lane-width assumption.
  - Top-down video (PRIVATE): data\seg_phaseA\tesla_PRIVATE\tesla_topdown_PRIVATE.mp4
- Before Phase C: CARLA collection should use the same camera geometry as the real car (pitch -2, height ~1.56)
  so the sim and real top-down views match.

## Phase C result (17 Nov) = GO
Data: data\vision_data_v2. 80 episodes, 62,071 frames: Town01 15.1k, Town03 16.1k, Town04/05/10HD ~10k each.
The camera matches the real car (pitch -2, height 1.45, 640x480), with pose, traffic state and route per episode. Town02 unseen.
- Every frame segmented: <ep>\seg\*.png = class numbers 0-6, so they look black in a viewer.
  Colour previews: vision_data_v2\_seg_preview.
- Situation counts:

  | Situation | % of frames |
  |---|---|
  | stopping for red | 34.1 |
  | straight | 19.5 |
  | approaching junction | 13.7 |
  | junction straight | 8.2 |
  | junction right | 6.4 |
  | following car | 5.3 |
  | junction left | 5.1 |
  | right bend | 3.1 |
  | left bend | 2.4 |
  | giving way | 1.3 |
  | lane change right | 0.6 |
  | lane change left | 0.4 |

- Timelines (vision_data_v2\_timelines) look right:
  - junction left/right line up with the steering dips/bumps
  - red covers the slow-down and the stop
  - approaching comes just before each junction
- For Phase D:
  - Down-weight standing still at a red light (34%).
  - Lane changes and giving way are rare: up-weight them, or merge lane change into one class.

## Sim-to-real honesty (channel goal = prove sim -> real)
- The driving policy (teacher + student) trains ONLY on CARLA.
- Real-data components, disclosed in videos:
  - the segmenter (Mask2Former trained on Mapillary Vistas real photos; perception only, no driving labels)
  - the camera pitch / height fit from real footage
- Control experiment to keep: a segmenter trained ONLY on CARLA truth masks, run on the Tesla footage, so the
  videos can show what pure-sim perception achieves vs the real-pretrained one.

## Constraints
- Raw Tesla clips and telemetry.csv contain GPS: they stay local. Overlay videos are private until you approve them.
- Every CARLA job shuts CARLA down. Don't touch CARLA while another job is running.
- Milestone clip at the end of Phase D = record_tour.py all-towns tour, 2 routes per town, traffic, narration on.

## Open points
- Segmenter choice (Phase A): Mapillary Vistas model candidates, speed on our GPU.
- Tesla camera intrinsics/pitch: from the footage or known values.
- Whether labels.csv already carries the junction flag and route option per frame (decides whether Phase C needs re-collection or only re-derivation).

## Phase D first result (17 Nov): student K beats the camera-only students on route completion
Student K: data\vision_runs\bev_20261117_120600\best.pt (code: bev_dataset.py, bev_model.py, bev_train.py, phaseD_test.ps1).
- Input: fused top-down map + speed + cruise speed + command. No wheel, no yaw rate. GRU over 3.2 s.
- Offline (held-out routes, live code path): steer slope 0.82, situation 75% correct.
- Town02 (unseen), traffic, 10 routes per seed:

  | Student | Seed 123 | Seed 456 | Mean | Routes completed | Collisions | Off route |
  |---|---|---|---|---|---|---|
  | I (camera) | 56% | 55% | 56% | 4 | 2 | 14 |
  | J (camera) | 65% | 46% | 56% | 4 | 1 | 15 |
  | K (top-down) | 75% | 59% | 67% | 9 | 5 | 6 |

- Weak spot: collisions. 2 with cars at junctions; 3 with guardrails / static objects, which are drifts off the road that ended in something.
- Some early off-route endings have zero wiggles: the car held a constant steer and missed the turn. Diagnose next.

## Phase D round 2: DAgger with K -> K2 (17 Nov)
- K diagnosis (Town02 x2 + Town03): mostly junctions: 5 under-steered turns, 2 wrong-way, 4 junction follow, 4 lane/bend, 4 collisions.
- DAgger: 53,822 frames (Town01/03/04/05/10HD, beta 0.2, takeover 1.0) -> K2 = K + 12 epochs (lr 1.5e-4), best ep 9 val steer 0.0584 (val set now includes DAgger frames; not comparable to K).
- Town02: K2 83/58 = 70% mean, 10/20 done, 3 collisions, 7 off-route  (K 75/59 = 67%, 9/20, 5 coll, 6 off).
- Modest gain: collisions down, seed 456 still mostly off-route (missed turns). Situation read-out fell 71 -> 62%.
- Tour: milestone_clips\20261117_2229_student_k2_all_towns.mp4 (Town01 collision, Town03/04 off-route; 02/05/10HD clean).

## Phase D round 3: turn-weighted loss -> K3 (18 Nov)
- Offline check (K, K2): on hard-steer frames (|teacher| >= 0.3) the student steers at 0.5-0.7 of the teacher's strength, even on teacher drives -> under-steer is learned, not live-only.
- K3 = K2 + 6 epochs, steer loss x (1 + 4|teacher steer|), last.pt. Val turn strength stayed 0.55-0.59: the weighting did NOT fix the shrink.
- Town02: K3 83/63 = 73% mean, 11/20 done, 5 collisions, 4 off-route (K2 70%, 10/20, 3, 7). Fewer off-route, more collisions: within noise.
- Tour: milestone_clips\20261118_0829_student_k3_all_towns.mp4 (first tour with red-light waits cut: 162 s vs 246 s). Town03 2 off-route, Town04 1; rest clean.
- Next: map quality (evidence voting + distance weighting in Fuser; vehicles as near-edge "worms" ~1 m deep), scored vs CARLA truth before any retrain.

## Phase D round 4: route curve as a map layer -> K4 (18 Nov)
- K4 = K3 + 8 epochs with the GPS-grade route curve as a 9th map layer (zero-init weights), last.pt.
- Town02: 83/66 = 74.5%, 13/20 done, 4 collisions, 3 off-route. GPS 2 m test: 83/61, mean |lat| 0.20 -> 0.30 m on seed 456.
- BUT reliance check (route_reliance.py idea): curve on/off changes steering by 0.0006 -> K4 IGNORES the curve.
  Route-channel conv weights |w| 0.003 vs ~0.04 for classes. The early "turn strength 0.77" was a 20-batch fluke.
  Causes: a thin 3 px line is a weak signal after 4 stride-2 convs; fine-tuned from K3, which had memorised its data
  (train steer 0.010 vs val 0.056), so the curve brought no training-loss gain.
- Seed 123: the same 2 failures for K2/K3/K4 (ep 4 collision at 11%, ep 10 off-route at 34%) = persistent cases to inspect.
- Tour: milestone_clips/20261118_1112_student_k4_all_towns.mp4 (Town03 2 off-route, Town04 1).
- Next = K5: the curve ALSO as 8 route points (fwd, right) + flag into the GRU and the steering heads; trained from scratch.
  route_reliance.py is now a tool; after 20 batches the curve effect is already 10x K4's.


## K5 = route curve twice: map layer + 8 route points into memory and steering, from scratch (18 Nov)
- Training: 20 epochs, lr 3e-4, turn weight 4.  Turn strength 0.67 in training (K4: 0.57).
- Reliance check (held-out drives): curve on vs off changes steer by 0.0155 (K4 0.0006, ~25x); most at junctions
  (right 0.029, left 0.022, straight-through 0.025) vs straights 0.010.  1 m curve shift: 0.0043, about a quarter of
  the curve effect -> vision still holds the lane.  Turn strength 0.62 with the curve, 0.58 without.
- Town02 (seeds 123/456): 99% / 78% = 88.5%, 17/20 finished (K4 74.5%, 13/20).  Seed 123 all 10 done, incl. the two
  routes K2-K4 always failed.  NO off-route endings.  All 3 misses = collisions with CROSSING traffic at junctions
  (ego 3-4 m/s, other car arriving from the side) -> next weak spot is giving way, not steering.
- GPS 2 m to the right: 99% / 93% (19/20).  No harm.  The seed 456 difference vs clean is traffic luck: collisions vary
  run to run, so read the 2 m test as 'not worse', not 'better'.
- Tour: milestone_clips\20261118_1550_student_k5_all_towns.mp4

## K5 on REAL TeslaCam footage (18 Nov, open loop, tesla_bev_test.py)
One 60 s suburban clip, 10 Hz, 598 moving steps, 43 in junction turns.  Target = the car's real path curvature 0.2 s
later.  Mirrored (Australia drives on the left).  Route curve = the driven path (perfect satnav, from speed + heading).
- K5: corr 0.79, sign agreement 98%; TURNS corr 0.97, MAE 0.009, every turn the right way.
- Without the route curve: corr 0.74, turns corr 0.98 but MAE 0.020 (over-steers: -0.095 vs -0.045 at 47 s).
  The curve halves the turn error on real roads too.
- Not mirrored: turns corr 0.35 -> left- vs right-hand traffic matters; mirroring is required.
- Camera students for comparison (same clip): 0.51-0.63, and they were GIVEN the real wheel angle.
- HONEST weak spot: on straights it weaves (+-0.015, phantom bends at 35-37 s and 40-41 s); overall MAE 0.008 is
  WORSE than 'always straight' (0.0057), because the clip is mostly straight.  Strength = turns, weakness = straights.
- Only one clip.  More clips needed before claiming anything general.

## Map memory v2 = evidence voting (18 Nov) -> K6
- Real-road maps flickered 3x more than CARLA maps (3.3% vs 1.0% of cells changing class per frame) = suspected cause
  of the weave on real straights.  VoteFuser (bev_project.py): per-cell class evidence moved with our motion,
  near pixels weigh more, fades 5%/frame, a shown class sticks unless clearly beaten.
- map_quality.py (CARLA truth + real flicker): real flicker 3.3% -> 0.9%, coverage 42% -> 51%, CARLA lane-line recall
  0.78 -> 0.84, road IoU unchanged.  Car 'worms' (1 m deep): no measurable effect on these scores; not used yet.
- K5 on vote maps WITHOUT retraining: real-clip corr 0.79 -> 0.85 (straights 0.69 -> 0.81) but MAE 0.008 -> 0.010
  (trained on the old maps) -> K6 = K5 recipe on bev_vote.npz caches (phaseD_vote.ps1).
- Known gap (Vadim): the real clip has roundabouts; training has few (only Town03).  More real clips coming 19 Nov.
- Real-footage visual for the video: path overlay (white = where the driver went next 2.5 s, blue = the AI's plan) +
  its commentary line; tesla_bev_test.py path_video -> seg_phaseA	esla_PRIVATE\<name>_paths_PRIVATE.mp4.

## K6 result (19 Nov 2026) - voting map memory in training: NO GAIN
K6 = K5 recipe trained on vote maps (vision_runs\bev_20261118_195233\best.pt).
- Town02: seed 123 9/10 (94%), seed 456 8/10 (84%) = 17/20, 89% - same as K5 (17/20, 88.5%). Misses = side hits at junctions again.
- GPS 2 m: 19/20 (99/93) - same as K5.
- Route reliance: curve effect 0.0179 (K5 0.0155), 1 m shift 0.0054 - still uses the route.
- Real Tesla clip (mirrored): corr 0.71 (turns 0.97, straights 0.62), MAE 0.0097 vs always-straight 0.0057.
  Worse than K5 on vote maps with no retrain (corr 0.85, straights 0.81, MAE 0.010) and K5 on its own maps (0.79).
- Conclusion: steadier maps help at TEST time; training on them doesn't. Weave on straights (MAE above the
  always-straight baseline) is still unsolved. Best real-clip setup for the video = K5 + vote maps.
- Tour clip: milestone_clips\20261118_2356_student_k6_all_towns.mp4 (1 junction side hit in Town01).
