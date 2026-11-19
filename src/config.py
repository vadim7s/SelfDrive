"""All tunable settings in one place. Edit here, not in the code."""
import os

# ---- paths (override with environment variables; nothing personal is hard-coded) ----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_ROOT = os.environ.get("FSD_DATA", os.path.join(REPO_ROOT, "data"))       # datasets, students, clips, maps
CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\CARLA")                         # CARLA 0.9.16 install
TESLA_CLIP = os.environ.get("FSD_TESLA_CLIP", os.path.join(DATA_ROOT, "tesla_cam", "front.mp4"))   # your own dashcam clip
RUNS_DIR = os.path.join(SRC_DIR, "runs")                                    # logs, teacher checkpoints (git-ignored)
_SEG_LOCAL = os.path.join(DATA_ROOT, "models", "m2f_vistas")
SEG_MODEL = _SEG_LOCAL if os.path.isdir(_SEG_LOCAL) else "facebook/mask2former-swin-large-mapillary-vistas-semantic"   # downloaded on first use


# --- CARLA -----------------------------------------------------------------
HOST, PORT, TM_PORT = "localhost", 2000, 8000
TIMEOUT_S = 30.0
LOAD_MAP_TIMEOUT_S = 300.0       # Town03's first load took > 120 s
TRAIN_TOWN = "Town01"
# teacher training rotates through every town (train.py); evaluate.py / the student keep their own town
TRAIN_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
TOWN_SWITCH_EPISODES = 100               # episodes per town; a switch = planned CARLA restart (~1-2 min)
TOWN_MAX_CONNECT_FAILS = 3                # a town that fails to load this many times in a row is skipped
TOWN_STATE_FILE = os.path.join(RUNS_DIR, "train_town_state.json")   # survives CARLA restarts
TEST_TOWN = "Town02"                     # unseen town for the generalisation test
FIXED_DELTA = 0.05                       # 20 Hz simulation, synchronous mode
FRAME_SKIP = 2                           # each action held for 2 sim ticks -> 10 Hz control
NO_RENDERING = True                      # training: skip rendering for speed (eval turns it on)
VEHICLE = "vehicle.tesla.model3"

# --- camera matched to the Model 3 TeslaCam front clip (vision stage) --------
# Estimated from data\tesla_cam (telemetry yaw vs image shift, lane vanishing point).
TESLA_CAM = dict(
    width=1280, height=960,      # TeslaCam front resolution (4:3)
    fov=52.0,                    # horizontal FOV, deg (estimate; likely 50-55)
    x=0.6, y=0.0, z=1.30,        # metres from the vehicle origin: windscreen top, ~1.3 m above ground
    pitch=1.0,                   # deg, + = up (horizon slightly above image centre)
)
CAM_JITTER = dict(fov=5.0, z=0.10, pitch=2.0, x=0.2)   # +/- randomisation around TESLA_CAM for training
# v2 (17 Nov, segmentation / top-down stage): pitch from the Tesla clip (lane lines parallel at -2.0 deg);
# height: lane-width fit says 1.56 m (if lanes are 3.3 m), Model 3 mount ~1.4 m -> 1.45, jitter covers both.
TESLA_CAM_V2 = dict(TESLA_CAM, z=1.45, pitch=-2.0)
VISION_IMG_V2 = (640, 480)                 # segmenter needs >= 640x480 (320x240 loses lane markings)

# --- CARLA health: hang detection + auto-restart ----------------------------
CARLA_EXE = os.path.join(CARLA_ROOT, "CarlaUE4.exe")
CARLA_ARGS = ["-RenderOffScreen", "-quality-level=Low"]   # training server
CARLA_PROCESSES = ["CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"]
TICK_TIMEOUT_S = 20.0           # a single world.tick() longer than this = CARLA is hung
BOOT_TIMEOUT_S = 180.0          # max wait for a (re)started CARLA to accept connections
AUTO_START = True               # start CARLA if it isn't running when an env is created
MAX_RESTARTS = 50               # run_training.py gives up after this many (something is badly wrong)
RESTART_EVERY_EPISODES = 300    # preventive restart (CARLA slows / leaks memory over time); 0 = off
HEARTBEAT_EVERY_STEPS = 50
HEARTBEAT_FILE = os.path.join(RUNS_DIR, "heartbeat.json")
EVENTS_LOG = os.path.join(RUNS_DIR, "carla_events.log")
STALE_S = 330                  # no heartbeat for this long = frozen -> kill CARLA + training, resume
SHUTDOWN_CARLA_WHEN_DONE = True # training / eval / smoke test close CARLA at the end (frees the GPU)
EXIT_RESTART_CARLA = 3         # train.py exit code meaning "CARLA failed, restart it and resume"

# --- episodes / routes ------------------------------------------------------
ROUTE_SMOOTH = True                     # route_smooth.py: drop duplicate points, smooth lane changes, even 2 m spacing
LC_JUMP_M = 5.0                         # a route step longer than this = planner's one-step lane change
LC_PRE_M = 10.0                         # smooth lane change starts this far before the jump ...
LC_POST_M = 10.0                        # ... and ends this far after it (~30 m S-curve, ~2 m/s2 at 30 km/h)
KINK_K_MAX = 0.22                       # 1/m: sharper than a ~4.5 m radius = planner kink, smoothed locally
KINK_ITERS = 60
ROUTE_START_CHECK = True                # reject routes that don't start at the car (planner snaps to another lane)
ROUTE_START_MAX_M = 2.0
ROUTE_START_MAX_DEG = 35.0
ROUTE_RESOLUTION = 2.0                   # metres between route points
ROUTE_MIN_M, ROUTE_MAX_M = 200.0, 600.0  # random route length range
MAX_EPISODE_STEPS = 3000                 # at 10 Hz = 300 s (red lights + traffic)
STUCK_SPEED = 0.5                        # m/s
STUCK_STEPS = 50                         # 5 s below STUCK_SPEED (after warm-up) -> end
WARMUP_STEPS = 30
OFF_ROUTE_M = 3.0                        # lateral distance from route centre that ends the episode
NUM_NPC_VEHICLES = 0                     # towns not in NPC_PER_TOWN (or TRAFFIC = False)
# --- traffic (traffic.py): in ALL training and testing. Counts scaled to town size: busy, never jammed ---
TRAFFIC = True
NPC_PER_TOWN = {"Town01": 30, "Town02": 25, "Town03": 50, "Town04": 60, "Town05": 50, "Town10HD": 35}
TM_GAP_M = 2.5                           # NPC distance to their leader
TM_SPEED_DIFF_PCT = 10.0                 # NPCs drive 10% under the speed limit
TM_HYBRID_RADIUS_M = 70.0                # NPCs further away: cheap physics (0 = off)
TRAFFIC_SPEED_CAPS = True                # ego: stop for red/yellow lights, keep a gap to the car ahead
LIGHT_LOOKAHEAD_M = 40.0
LIGHT_DECEL = 3.0                        # m/s2 comfortable stop for a light
STOP_MARGIN_M = 1.0                      # stop this far before the stop line
STOP_MATCH_M = 2.5                       # stop line within this of the route = on our route
CAR_LENGTH_M = 4.8                       # centre-to-centre -> bumper gap
MIN_GAP_M = 3.0                          # standstill gap
TIME_GAP_S = 1.5                         # following time gap
FOLLOW_DECEL = 4.0
YIELD_LOOKAHEAD_M = 30.0                 # give way: our path this far ahead ...
YIELD_RANGE_M = 45.0                     # ... against moving NPCs within this box
YIELD_HORIZON_S = 4.0                    # NPC constant-velocity prediction
YIELD_TIME_S = 2.0                       # conflict = both at the same path point within this time
YIELD_CLEAR_M = 2.0                      # path point counts as taken within this distance
YIELD_MARGIN_M = 0.6                     # body-to-body clearance on top of both half widths
YIELD_DT_S = 0.25                        # prediction step
YIELD_MAX_TURN = 0.8                     # rad/s cap on the NPC turn rate used for prediction
LEAD_SIDE_MARGIN_M = 0.2                 # lead = NPC body within our half width + its half width + this
TM_AUTO_LANE_CHANGE = False              # NPC overtaking lane changes (TM side-swipes the ego car)
NPC_MIN_DIST_M = 30.0                    # NPCs spawn at least this far from the ego car (no merge into us at the start)
YIELD_MIN_EGO_SPEED = 3.0                # m/s used for our arrival time when (nearly) stopped
YIELD_DECEL = 4.0

# --- observation ------------------------------------------------------------
N_LOOKAHEAD = 10                         # route points fed to the policy
LOOKAHEAD_STEP = 3                       # use every 3rd route point (6 m apart -> 60 m ahead)
TARGET_SPEED = 30.0 / 3.6                # m/s (30 km/h)
ROAD_SPEED_TARGET = True                 # cruise = road speed limit - TM_SPEED_DIFF_PCT (>= TARGET_SPEED)
ROAD_SPEED_MAX_KMH = 80.0                # cap on that cruise speed
SENSE_EXTRA_M = 20.0                     # lights / lead / yield look-ahead >= stopping distance + this
LEAD_RANGE_M = 40.0                      # lead-vehicle sensing range

# --- reward ---------------------------------------------------------------
R_PROGRESS = 1.0       # per metre of route completed
R_SPEED = 0.2          # x min(speed, target) / target: moving always beats standing still
R_LATERAL = 0.3        # x (lateral offset / half lane width)^2
R_HEADING = 0.3        # x |heading error| / pi
R_STEER_RATE = 1.0     # x |steer - previous steer|  (was 0.2: too weak, the teacher weaved)
R_LAT_ACC = 0.3        # x (lateral acceleration / LAT_ACC_COMFORT)^2 -> slow down for corners
                       # (round 1 used 0.1: still took corners at 13 m/s^2)
LAT_ACC_COMFORT = 3.0  # m/s^2, roughly where passengers start to notice
R_ACCEL_RATE = 0.2     # x |speed command - previous|  (no surging)
R_TOWARD_LINE = 1.0    # x (decrease of |lateral offset|) / half lane width: closer to the ideal line = reward (potential-based)
R_LAT_VEL = 0.5        # x (sideways speed above LAT_VEL_OK, m/s)^2: corrections must be smooth
LAT_VEL_OK = 0.5       # m/s sideways approach speed that is free
R_LANE_OUT = 0.5       # per step with a wheel over the lane line ...
R_LANE_OUT_M = 2.0     # ... plus this per metre over
CAR_HALF_WIDTH_M = 0.95
R_OVERSPEED = 0.5      # x relative overspeed above target
R_COLLISION = -50.0
R_OFF_ROUTE = -30.0
R_STUCK = -30.0        # was -10: parking was cheaper than the risk of driving (run ppo_20261112_154333)
R_ROUTE_DONE = 50.0
BRAKE_DEADZONE = 0.3   # (raw-pedal mode only) accel in [-0.3, 0] = coast

# --- action mode -----------------------------------------------------------
# True: action[1] = target speed, a speed controller drives throttle/brake (stalling by accident is
#       impossible; stopping is still possible). False: action[1] = raw throttle(+)/brake(-).
SPEED_CONTROL = True
# curve speed limit (like a curve advisory sign): caps the speed command from the route's bends ahead
CURVE_LIMIT = True
CURVE_LAT_ACC = 2.5    # m/s^2 allowed in a bend -> 90 deg town corner (~7 m radius) ~ 15 km/h
CURVE_DECEL = 2.0      # m/s^2 gentle braking used to reach the bend speed in time
CURVE_LOOKAHEAD_M = 40.0
CURVE_MIN_SPEED = 2.0  # m/s floor so it never stops for a bend
# steady-hand teacher (teacher_steady.py): pure pursuit on straight road, PPO in bends
STEADY_AHEAD_M = 25.0          # road must be straight this far ahead ...
STEADY_BEHIND_M = 4.0          # ... and this far behind
STEADY_K_STRAIGHT = 0.004      # 1/m: radius > 250 m = fully steady
STEADY_K_BEND = 0.015          # 1/m: radius < 67 m = PPO fully in charge
STEADY_MAX_SIDE_STEP_M = 0.3   # route point stepping sideways more than this = lane change ahead (PPO in charge)
STEADY_BENDS = "track"         # who steers bends/turns/corrections: "track" = lane tracker (teacher_steady.lane_tracker), "ppo" = PPO teacher
TRACK_PREVIEW_S = 0.15         # curvature feed-forward taken this far ahead (wheel lag ~0.3 s) ...
TRACK_PREVIEW_M = 0.0          # ... plus this
TRACK_K_HEAD = 1.0             # road-wheel rad per rad of heading error
TRACK_K_LAT = 1.5              # Stanley gain on lateral offset
TRACK_K_MAX = 0.2             # 1/m curvature cap (5 m radius; lane-change steps would spike)
TRACK_V0 = 2.0                 # m/s softening at low speed
STEADY_LAT_OK_M = 0.3          # off the route centre less than this: steady hand ...
STEADY_LAT_PPO_M = 0.8         # ... more than this: PPO corrects (pure pursuit is too slow after a push: 6/8 crashes)
STEADY_HEAD_OK_DEG = 3.0       # same for heading error
STEADY_HEAD_PPO_DEG = 8.0
STEADY_LOOKAHEAD_S = 1.5      # pure-pursuit aim point, seconds ahead (human drivers ~1-2 s)
STEADY_MIN_LOOKAHEAD_M = 8.0
STEADY_WHEELBASE_M = 3.0       # CARLA Model 3 (tesla_test\carla_steer_geometry.json)
STEADY_MAX_STEER_DEG = 60.5    # effective road-wheel angle at steer = 1 (same file)
STEER_SMOOTH = 0.3    # wheel moves 30% of the way to the command per step (~0.3 s lag). 1.0 = off
V_MAX_FACTOR = 1.2     # action[1] = -1 -> 0 km/h, 0 -> 0.6 x target, +1 -> 1.2 x target
SPEED_KP = 0.25        # throttle per m/s of speed error
THROTTLE_FF = 0.15     # throttle feed-forward while below target
THROTTLE_MAX = 0.8
BRAKE_KP = 0.3         # brake per m/s over target (only beyond 1 m/s over)

# --- PPO (Stable-Baselines3) ---------------------------------------------
TOTAL_TIMESTEPS = 1_000_000
PPO = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)
# Fine-tune: start a NEW run from an existing model's weights (run_training.py --finetune)
# round 1 (smoothness rewards) started from ppo_20261112_172317\final_model.zip; its best checkpoint:
# round 3 (curve-limited speeds) started from ppo_20261113_073505\checkpoints\ppo_251875_steps.zip -> teacher v3.
# round 4 (all towns): teacher v3 -> every installed town (v3 knew only Town01: Town05 55%, Town10HD 70%)
FINETUNE_FROM = os.path.join(RUNS_DIR, "ppo_20261114_093825", "final_model.zip")   # teacher v4 (all towns) -> v5: traffic + ideal-line reward
FINETUNE_STEPS = 600_000        # ~7 h: ~100k steps per town
FINETUNE_LR = 1e-4              # lower than 3e-4: round 1 collapsed after ~270k steps
CHECKPOINT_EVERY = 10_000       # ~5 min of training; worst-case loss after a hard crash
SEED = 42

# --- stage 3: vision student (imitates the teacher from the windscreen camera) ----------------
VISION_TEACHER = os.path.join(RUNS_DIR, "ppo_20261114_093825", "final_model.zip")  # teacher v4 (all towns); v3 = runs\ppo_20261113_134320
VISION_DATA_DIR = os.path.join(DATA_ROOT, "vision_data")     # episodes of camera frames + teacher labels
VISION_RUNS_DIR = os.path.join(DATA_ROOT, "vision_runs")     # trained students
VISION_TOWN = "Town01"                    # collect in Town01, test the student in unseen Town02
VISION_TARGET_FRAMES = 100_000            # ~175 routes, ~3 h of collection
VISION_IMG = (320, 240)                   # stored frame size (4:3 like the Tesla; same FOV)
VISION_CARLA_ARGS = ["-RenderOffScreen", "-quality-level=Epic"]   # cameras need real rendering
# towns that need other server args. Town03 never finishes loading at -quality-level=Low (5 h of
# 300 s time-outs, 14 Nov); at Epic it loads in 6 s. Used when the training rotation's next town matches.
CARLA_TOWN_ARGS = {"Town03": VISION_CARLA_ARGS}
VISION_TICK_TIMEOUT_S = 60.0              # first rendered frames compile shaders: slow
# weather per route, picked with these weights (day ~half: the real Tesla clips are mostly daytime)
VISION_DAY_WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon",
                       "MidRainyNoon", "ClearSunset", "CloudySunset", "WetSunset"]
VISION_EXTRA_WEATHERS = {               # name: weight (each day weather has weight 1)
    "ClearNight": 0.7, "CloudyNight": 0.5, "WetNight": 0.5, "WetCloudyNight": 0.5,
    "SoftRainNight": 0.5, "HardRainNight": 0.4,
    "HardRainNoon": 0.6, "HardRainSunset": 0.5,
    "FogNoon": 0.8, "FogSunset": 0.6, "DawnClear": 0.8, "DawnCloudy": 0.6,
}
# not CARLA presets: built from ClearNoon with these values changed
VISION_CUSTOM_WEATHERS = {
    "FogNoon": dict(cloudiness=70.0, fog_density=45.0, fog_distance=8.0, fog_falloff=1.0, wetness=20.0),
    "FogSunset": dict(cloudiness=60.0, fog_density=35.0, fog_distance=8.0, fog_falloff=1.0,
                      sun_altitude_angle=8.0, sun_azimuth_angle=270.0),
    "DawnClear": dict(cloudiness=10.0, sun_altitude_angle=6.0, sun_azimuth_angle=90.0),
    "DawnCloudy": dict(cloudiness=65.0, sun_altitude_angle=8.0, sun_azimuth_angle=90.0),
}
HEADLIGHTS_IN = ("Night", "Fog", "HardRain", "Dawn")   # weather names that switch the car's headlights on
CMD_LOOKAHEAD_M = 20.0                    # announce a junction turn this far ahead
VISION_SKIP_STEPS = 5                     # frames dropped after each spawn (camera warm-up)
# recovery data: now and then the executed steering is pushed off for a moment, while the label
# stays the teacher's correct action -> the student learns how to get back to the lane centre
NOISE_PROB = 0.03                         # chance per step to start a push
NOISE_STEPS = (4, 10)                     # push length, steps (0.4-1 s)
TRIM_FAILED_STEPS = 30                    # route ended in a crash / off road: drop its last 3 s (teacher's mistake)
VISION_EXTRA_TOWNS = ["Town03", "Town04", "Town05", "Town10HD"]   # extra variety; Town02 stays unseen
VISION_EXTRA_FRAMES = 30_000              # per extra town
NOISE_STEER = (0.05, 0.15)              # push size added to the steering command (0.15-0.5 crashed)
