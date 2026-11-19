"""Vision student: windscreen image + speed + wheel angle + nav command (+ optional metres to the
announced manoeuvre, like a sat-nav "in 12 m turn left") -> [steer, speed command].

Architecture (conditional imitation, as in "Learning by Cheating" / CIL):
  ResNet-18 (ImageNet-pretrained) image features
  + small MLP on [speed / target, current wheel angle]
  -> one output head per nav command (follow / left / right / straight); the command picks the head.
The command is the only route information the student gets (like a sat-nav instruction);
everything else it must read from the image.
"""
import numpy as np
import torch
import torch.nn as nn

import config as C

IN_W, IN_H = 256, 192                          # network input (4:3, resized from VISION_IMG)
N_CMD = 4
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Student(nn.Module):
    def __init__(self, pretrained=True, n_meas=2):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        self.backbone = nn.Sequential(*list(net.children())[:-1])        # -> 512 x 1 x 1
        self.meas = nn.Sequential(nn.Linear(n_meas, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.join = nn.Sequential(nn.Linear(512 + 64, 256), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
                                    for _ in range(N_CMD)])

    def forward(self, img, meas, cmd):
        """img: B x 3 x H x W, float 0..1. meas: B x 2. cmd: B (long). Returns B x 2 in [-1, 1]."""
        x = (img - MEAN.to(img.device)) / STD.to(img.device)
        f = self.backbone(x).flatten(1)
        h = self.join(torch.cat([f, self.meas(meas)], 1))
        out = torch.stack([head(h) for head in self.heads], 1)           # B x N_CMD x 2
        return torch.tanh(out[torch.arange(len(cmd), device=cmd.device), cmd])


def measurements(speed, wheel, cmd_dist=None, v_target=None):
    m = [speed / C.TARGET_SPEED, wheel]
    if cmd_dist is not None:                   # metres to the manoeuvre, 1.0 = nothing announced
        m.append(min(cmd_dist, C.CMD_LOOKAHEAD_M) / C.CMD_LOOKAHEAD_M)
    if v_target is not None:                   # cruise speed from the road's speed limit (like reading the signs)
        m.append(v_target / C.TARGET_SPEED)
    return m


def n_meas(settings):
    return 2 + bool(settings.get("cmd_dist", False)) + bool(settings.get("road_speed", False))


CMD_ONSET_M = C.CMD_LOOKAHEAD_M - C.ROUTE_RESOLUTION / 2   # a junction enters the look-ahead window ~19 m out
STEP_S = C.FIXED_DELTA * C.FRAME_SKIP


class CmdDistance:
    """Metres to the announced manoeuvre, from the command sequence and speed alone (dead reckoning).
    Same logic for the saved training data, live CARLA driving and the real Tesla clip, so the
    student always sees the same signal. A command that changes counts as a new announcement at
    CMD_ONSET_M; a block already running at the first step is unknown (None here, the caller decides)."""
    def __init__(self, unknown=None):
        self.unknown = unknown
        self.reset()

    def reset(self):
        self.prev_cmd, self.d, self.prev_speed = None, None, 0.0

    def step(self, cmd, speed):
        if cmd == 0:
            d = C.CMD_LOOKAHEAD_M
        elif cmd != self.prev_cmd:
            d = CMD_ONSET_M if self.prev_cmd is not None else self.unknown
        else:
            d = None if self.d is None else max(0.0, self.d - self.prev_speed * STEP_S)
        self.prev_cmd, self.d, self.prev_speed = cmd, d, speed
        return d


def prep_image(rgb):
    """HxWx3 uint8 -> 3 x IN_H x IN_W float tensor."""
    from PIL import Image
    im = Image.fromarray(rgb)
    if im.size != (IN_W, IN_H):
        im = im.resize((IN_W, IN_H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0


class StudentDriver:
    """Loads a trained student for driving (vision_collect.py --driver student / mix)."""
    def __init__(self, path):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        settings = ck.get("settings", {})
        self.no_wheel = bool(settings.get("no_wheel", False))          # trained without wheel input
        self.cmd_dist = bool(settings.get("cmd_dist", False))          # trained with metres-to-manoeuvre
        self.road_speed = bool(settings.get("road_speed", False))      # trained with the road's cruise speed
        self.net = Student(pretrained=False, n_meas=n_meas(settings)).to(self.dev)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.tracker = CmdDistance(unknown=CMD_ONSET_M / 2)          # episode starts mid-announcement: guess

    def reset(self):
        """Call at every episode start (the distance tracker is dead reckoning)."""
        self.tracker.reset()

    @torch.no_grad()
    def act(self, rgb, speed, wheel, cmd, v_target=None):
        """Call EVERY control step (also when another driver's action is used): it updates the tracker."""
        d = self.tracker.step(cmd, speed)
        if self.no_wheel:
            wheel = 0.0
        img = prep_image(rgb).unsqueeze(0).to(self.dev)
        vt = (C.TARGET_SPEED if v_target is None else v_target) if self.road_speed else None
        meas = torch.tensor([measurements(speed, wheel, d if self.cmd_dist else None, vt)],
                            dtype=torch.float32, device=self.dev)
        out = self.net(img, meas, torch.tensor([cmd], device=self.dev))
        return out[0].cpu().numpy().astype(np.float32)
