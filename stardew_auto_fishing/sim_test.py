"""sim_test.py - offline self-test (works on any OS, no game needed).

Simulates a bobber-bar mini-game with physics as I remember them from the game
(symmetric +/-0.25 px/tick^2 acceleration at 60 ticks/s, 568 px track, bounce at
the bottom), renders synthetic frames, and runs the REAL vision + controller
code on them with input latency. It validates the control logic and pipeline,
NOT the exact colours/geometry of the real game - for that, run calibration.

    python sim_test.py
"""
from __future__ import annotations

import random
import time
from collections import deque

import cv2
import numpy as np

import vision
from config import Config
from fishing_controller import BarController
from vision import FishingVision

TRACK, TRACK_Y0, BAR_X, BAR_W = 568, 40, 100, 40
GREEN_BGR, TRACK_BGR = (70, 200, 120), (140, 110, 50)


class Game:
    def __init__(self, bar_h: int, difficulty: float, rng: random.Random) -> None:
        self.H, self.rng, self.diff = bar_h, rng, difficulty
        self.pos, self.vel = TRACK - bar_h, 0.0
        self.fish, self.target, self.fv = TRACK - 20.0, TRACK - 20.0, 0.0

    def tick(self, hold: bool) -> None:
        r = self.rng
        a = -0.25 if hold else 0.25
        if hold and (self.pos <= 0 or self.pos >= TRACK - self.H):
            self.vel = 0.0
        self.vel += a
        self.pos += self.vel
        if self.pos < 0:
            self.pos, self.vel = 0.0, 0.0
        elif self.pos > TRACK - self.H:
            self.pos, self.vel = TRACK - self.H, -self.vel * 2 / 3
        if r.random() < 0.010 + 0.01 * self.diff:             # new wander target
            self.target = r.uniform(0, TRACK - 20)
        if r.random() < 0.004 * self.diff:                    # "dart" movement
            self.target = min(TRACK - 20, max(0, self.fish + r.choice((-1, 1)) * r.uniform(120, 300)))
        vmax = 1.0 + 3.5 * self.diff
        self.fish += float(np.clip((self.target - self.fish) * 0.04, -vmax, vmax))
        self.fish = min(TRACK - 20, max(0.0, self.fish))

    def in_bar(self) -> bool:
        c = self.fish + 10
        return self.pos <= c <= self.pos + self.H


def make_fish_template() -> np.ndarray:
    tpl = np.full((20, 20, 3), (200, 170, 90), np.uint8)
    cv2.ellipse(tpl, (10, 10), (8, 5), 0, 0, 360, (240, 240, 250), -1)
    cv2.circle(tpl, (14, 9), 2, (20, 20, 20), -1)
    cv2.line(tpl, (1, 4), (5, 10), (40, 40, 200), 2)
    return tpl


def render(g: Game, tpl: np.ndarray) -> np.ndarray:
    img = np.full((700, 320, 3), (30, 60, 40), np.uint8)
    cv2.rectangle(img, (BAR_X - 6, TRACK_Y0), (BAR_X + BAR_W + 6, TRACK_Y0 + TRACK), TRACK_BGR, -1)
    top = TRACK_Y0 + int(g.pos)
    cv2.rectangle(img, (BAR_X, top), (BAR_X + BAR_W - 1, top + g.H - 1), GREEN_BGR, -1)
    cv2.rectangle(img, (BAR_X + 70, TRACK_Y0 + 300), (BAR_X + 86, TRACK_Y0 + TRACK), GREEN_BGR, -1)  # progress bar
    fy = TRACK_Y0 + int(g.fish)
    img[fy:fy + 20, BAR_X + BAR_W // 2 - 10: BAR_X + BAR_W // 2 + 10] = tpl
    return img


def run(bar_h: int, difficulty: float, seed: int, perfect: bool, seconds: float = 45.0, latency=0.045, trace=None, overrides=None):
    rng = random.Random(seed)
    cfg = Config(perfect_mode=perfect, latency_ms=latency * 1000)
    for k, v in (overrides or {}).items():       # e.g. deliberately wrong controller settings
        setattr(cfg, k, v)
    tpl = make_fish_template()
    patch = np.full((8, 8, 3), GREEN_BGR, np.uint8)
    cfg.bar_color = vision._hue_stats(patch)
    cfg.ref_bar_w = BAR_W
    vis = FishingVision(cfg)
    vis.fish_tpl = tpl

    g = Game(bar_h, difficulty, rng)
    frame = render(g, tpl)
    layout = vis.find_bar(frame)
    assert layout is not None, "bar not found in synthetic frame"
    left, width, off = vis.plan_band(layout, frame.shape[1])
    ctrl = BarController(cfg, layout.w)

    dt_c, dt_t = 1 / 120, 1 / 60
    t = next_tick = 0.0
    cmds: deque = deque([(-9.0, False)])
    holding, last_fish = False, None
    ticks = out_ticks = 0
    cpu = 0.0
    while t < seconds:
        while next_tick <= t:
            eff = False
            for ct, ch in cmds:                                # latest command older than `latency`
                if ct <= next_tick - latency:
                    eff = ch
            g.tick(eff)
            ticks += 1
            out_ticks += 0 if g.in_bar() else 1
            if trace is not None and not g.in_bar():
                trace.append((round(t, 2), round(g.pos), round(g.fish + 10), round(g.vel, 1)))
            next_tick += dt_t
        frame = render(g, tpl)
        c0 = time.perf_counter()
        obs = vis.measure(frame[:, left:left + width], off, layout.w, last_fish)
        if obs:
            last_fish = obs.fish_y
            holding = ctrl.update(t, obs.top, obs.bottom, obs.fish_y, holding).hold
            cmds.append((t, holding))
            while len(cmds) > 40:
                cmds.popleft()
        cpu += time.perf_counter() - c0
        t += dt_c
    return 100.0 * (1 - out_ticks / ticks), out_ticks, cpu / (seconds / dt_c) * 1000


if __name__ == "__main__":
    print(f"{'bar':>4} {'diff':>5} {'mode':>8} {'in-bar %':>9} {'ticks out':>10} {'ms/frame':>9}")
    for bar_h in (96, 136):
        for diff in (0.3, 0.7, 1.0):
            for perfect in (True, False):
                res = [run(bar_h, diff, sd, perfect, seconds=15.0) for sd in range(2)]
                pct = np.mean([r[0] for r in res])
                out = np.mean([r[1] for r in res])
                ms = np.mean([r[2] for r in res])
                print(f"{bar_h:>4} {diff:>5} {'perfect' if perfect else 'normal':>8} {pct:>9.2f} {out:>10.1f} {ms:>9.2f}")
