"""fishing_controller.py - control algorithm + the cast / bite / mini-game state machine.

Control algorithm (see README for the reasoning)
-------------------------------------------------
The bobber bar behaves like a point mass with constant, symmetric acceleration:
holding the button pushes it up, releasing lets "gravity" pull it down. The
only actuator is binary (hold / release), so the textbook solution is a
*time-optimal switching law* on a double integrator:

    y     = bar_position - fish_position            (+ = bar below the fish)
    y_dot = bar_velocity - fish_velocity
    sigma = y + y_dot * |y_dot| / (2 * a)
    hold the button  <=>  sigma > 0

It is combined with
  * alpha-beta filters that give clean position/velocity estimates,
  * a Smith-predictor style look-ahead that compensates capture->input latency,
  * linear fish extrapolation (prediction_strength),
  * a soft boundary layer realised with sigma-delta (PWM) dithering instead of
    hard chattering, and
  * an online estimate of the bar acceleration ``a`` (resolution/UI-scale proof).
"""
from __future__ import annotations

import random
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

import automation
from config import Config
from vision import BarLayout, FishingVision, ScreenGrabber


# --------------------------------------------------------------- estimation
class AlphaBeta:
    """Constant-velocity alpha-beta tracker with variable time step."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.beta = alpha * alpha / (2.0 - alpha)   # Benedict-Bordner optimum
        self.x: Optional[float] = None
        self.v = 0.0
        self.t = 0.0

    def update(self, z: float, t: float) -> None:
        if self.x is None:
            self.x, self.v, self.t = z, 0.0, t
            return
        dt = max(t - self.t, 1e-3)
        xp = self.x + self.v * dt
        r = z - xp
        self.x = xp + self.alpha * r
        self.v += self.beta * r / dt
        self.t = t

    def coast(self, t: float) -> None:
        """No measurement this frame: extrapolate and let the velocity decay."""
        if self.x is None:
            return
        dt = max(t - self.t, 0.0)
        self.x += self.v * dt
        self.v *= float(np.exp(-dt / 0.15))
        self.t = t


@dataclass
class Decision:
    hold: bool
    bar_center: float
    bar_h: float
    fish_y: Optional[float]
    in_bar: Optional[bool]


class BarController:
    """Real-time controller that keeps the fish centred in the bobber bar."""

    def __init__(self, cfg: Config, bar_w: float) -> None:
        self.cfg = cfg
        self.bar_w = bar_w
        self.reset()

    def reset(self) -> None:
        self.bar = AlphaBeta(self.cfg.responsiveness)
        self.fish = AlphaBeta(min(0.95, self.cfg.responsiveness + 0.05))
        self.heights: deque = deque(maxlen=600)
        self.hist: deque = deque(maxlen=90)          # (time, hold) of every command, for learning `a`
        self.a0 = self.cfg.accel_per_bar_w * self.bar_w
        self.a_est = self.a0
        self.t_start: Optional[float] = None
        self.learn_t0: Optional[float] = None
        self.learn_v0 = 0.0
        self.last_key = None
        self.last_center, self.last_H = 0.0, 1.0
        self.acc = 0.0

    # ----------------------------------------------------------------- helpers
    def _bar_center(self, top: float, bottom: float, t: float):
        """Bar centre + (constant) bar height. Handles the fish icon hiding one bar end."""
        self.heights.append(bottom - top)
        # Bar height is constant during a mini-game and occlusion can only make it look smaller,
        # so use the largest plausible value seen (outliers above 1.4x the median are ignored).
        hs = np.asarray(self.heights)
        H = float(hs[hs <= 1.4 * np.median(hs)].max())
        h = bottom - top
        if h >= 0.93 * H or self.bar.x is None:
            return (top + bottom) / 2.0, H
        pred = self.bar.x + self.bar.v * (t - self.bar.t)
        c_top, c_bot = top + H / 2.0, bottom - H / 2.0
        return (c_top if abs(c_top - pred) <= abs(c_bot - pred) else c_bot), H

    def _mean_sign(self, t_a: float, t_b: float) -> float:
        """Time-weighted mean of the commanded acceleration sign (+1 released/down, -1 held/up)."""
        h = list(self.hist)
        acc = total = 0.0
        for i, (ti, hi) in enumerate(h):
            tj = h[i + 1][0] if i + 1 < len(h) else t_b
            lo, up = max(ti, t_a), min(tj, t_b)
            if up > lo:
                acc += (-1.0 if hi else 1.0) * (up - lo)
                total += up - lo
        return acc / total if total > 0 else 0.0

    def _learn_acceleration(self, t: float) -> None:
        """Slowly adapt ``a`` from velocity change over >=60 ms windows of (mostly) constant input."""
        if self.learn_t0 is None:
            self.learn_t0, self.learn_v0 = t, self.bar.v
            return
        span = t - self.learn_t0
        if span < 0.06:
            return
        lat = self.cfg.latency_ms / 1000.0
        s = self._mean_sign(self.learn_t0 - lat, t - lat)
        if abs(s) >= 0.8:
            a_obs = (self.bar.v - self.learn_v0) / (span * s)
            if 0.3 * self.a_est < a_obs < 3.0 * self.a_est:
                w = 0.15 if (t - (self.t_start or t)) < 1.5 else 0.04
                self.a_est = float(np.clip(self.a_est + w * (a_obs - self.a_est), 0.4 * self.a0, 2.5 * self.a0))
        self.learn_t0, self.learn_v0 = t, self.bar.v

    # ------------------------------------------------------------------ update
    def update(self, t: float, top: float, bottom: float,
               fish_y: Optional[float], holding: bool) -> Decision:
        cfg = self.cfg
        if self.t_start is None:
            self.t_start = t
        # The game renders at ~60 fps; we poll faster, so many frames are duplicates. Feeding
        # duplicates to the filters would make the velocity estimates jump between 0 and 2x.
        key = (round(top, 1), round(bottom, 1), None if fish_y is None else round(fish_y, 1))
        if key != self.last_key:
            self.last_key = key
            center, H = self._bar_center(top, bottom, t)
            self.last_center, self.last_H = center, H
            self.bar.update(center, t)
            if fish_y is not None:
                self.fish.update(fish_y, t)
            else:
                self.fish.coast(t)
            self._learn_acceleration(t)
        center, H = self.last_center, self.last_H

        a = self.a_est
        lat = cfg.latency_ms / 1000.0
        u_cur = -a if holding else a                 # current bar acceleration (screen y points down)
        hb = lat + (t - self.bar.t)                  # look-ahead = latency + age of the estimate
        bar_x = self.bar.x + self.bar.v * hb + 0.5 * u_cur * lat * lat
        bar_v = self.bar.v + u_cur * lat
        k = cfg.prediction_strength
        if self.fish.x is not None:
            fish_x = self.fish.x + k * self.fish.v * (lat + (t - self.fish.t))
        else:
            fish_x = bar_x
        fish_v = k * self.fish.v

        y = bar_x - fish_x
        y_dot = bar_v - fish_v
        sigma = y + y_dot * abs(y_dot) / (2.0 * a)

        # Soft boundary layer: duty cycle of "hold" grows linearly with sigma, realised by dithering.
        phi = max(1.5, cfg.deadband_frac * H * (0.6 if cfg.perfect_mode else 1.3))
        duty = float(np.clip(0.5 + sigma / (2.0 * phi), 0.0, 1.0))
        self.acc += duty
        hold = self.acc >= 1.0
        if hold:
            self.acc -= 1.0
        self.hist.append((t, hold))

        in_bar = None if fish_y is None else (center - H / 2.0 <= fish_y <= center + H / 2.0)
        return Decision(hold, center, H, fish_y, in_bar)


# ------------------------------------------------------------ state machine
class State(Enum):
    IDLE = "Idle"
    CASTING = "Casting"
    WAITING_BITE = "Waiting for bite"
    HOOKING = "Hooking"
    MINIGAME = "Mini-game"
    RESULT = "Catch result"
    STOPPED = "Stopped"
    ERROR = "Error"


@dataclass
class Stats:
    attempts: int = 0
    successes: int = 0
    perfects: int = 0
    failures: int = 0
    errors: int = 0


@dataclass
class Status:
    state: State = State.IDLE
    message: str = ""
    bar_found: bool = False
    fish_found: bool = False
    fish_conf: float = 0.0
    in_bar: Optional[bool] = None
    progress_est: float = 0.0
    stats: Stats = field(default_factory=Stats)


class FishingBot:
    """Runs the whole cast -> bite -> mini-game -> recast loop in a worker thread."""

    def __init__(self, cfg: Config, status_cb: Optional[Callable[[Status], None]] = None,
                 log_cb: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg
        self.status_cb = status_cb
        self.log_cb = log_cb or (lambda m: None)
        self.status = Status()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_emit = 0.0

    # ------------------------------------------------------------- public api
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fishing-bot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---------------------------------------------------------------- helpers
    def _log(self, msg: str) -> None:
        self.log_cb(time.strftime("%H:%M:%S ") + msg)

    def _set(self, state: State, message: str = "", force: bool = True) -> None:
        self.status.state, self.status.message = state, message or state.value
        self._emit(force)

    def _emit(self, force: bool = False) -> None:
        now = time.perf_counter()
        if self.status_cb and (force or now - self._last_emit > 0.1):
            self._last_emit = now
            self.status_cb(self.status)

    def _alive(self) -> bool:
        if self._stop.is_set():
            return False
        if automation.key_pressed(self._estop_vk):
            self._log("Emergency stop key pressed.")
            self._stop.set()
            return False
        return True

    def _sleep(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while self._alive() and time.perf_counter() < end:
            time.sleep(max(0.0, min(0.01, end - time.perf_counter())))

    def _wait_focus(self) -> None:
        while self._alive() and self.cfg.require_focus and self.grab.hwnd \
                and not automation.is_foreground(self.grab.hwnd):
            self._set(State.IDLE, "Waiting for the game window to be focused...", force=False)
            self._sleep(0.25)
            self.grab.refresh_window()

    # --------------------------------------------------------------- main loop
    def _run(self) -> None:
        self.inp = None
        try:
            self._estop_vk = automation.vk_from_name(self.cfg.emergency_stop_key)
            self.grab = ScreenGrabber(self.cfg)
            self.vision = FishingVision(self.cfg)
            self.vision.require_ready()
            self.inp = automation.InputController(self.cfg.input_mode, self.cfg.use_tool_key)
            self._log(f"Started. Capture area {self.grab.size[0]}x{self.grab.size[1]}. "
                      f"Emergency stop: {self.cfg.emergency_stop_key}")
            self._loop()
        except Exception as exc:  # noqa: BLE001 - surface everything to the UI
            self.status.stats.errors += 1
            self._log(f"ERROR: {exc}")
            self._log(traceback.format_exc(limit=3).strip())
            self._set(State.ERROR, str(exc))
        finally:
            if self.inp:
                self.inp.release_all()
            if self.status.state is not State.ERROR:
                self._set(State.STOPPED, "Stopped")
            self._emit(True)

    def _loop(self) -> None:
        cfg, failures = self.cfg, 0
        while self._alive():
            self.grab.refresh_window()
            self._wait_focus()
            if not self._alive():
                break
            layout = self._confirm_layout()                  # mini-game already running?
            if layout is None:
                if not cfg.auto_fishing:                     # assist mode: wait for the player
                    self._set(State.WAITING_BITE, "Assist mode: waiting for the mini-game...", force=False)
                    self._sleep(0.05)
                    continue
                if self._cast_and_wait_for_bite():
                    layout = self._hook()
                    if layout is None:
                        self.status.stats.errors += 1
                        self._log("Hooked but the mini-game was not detected (check calibration).")
                if layout is None:
                    failures += 1
                    if failures >= cfg.max_consecutive_errors:
                        self._log(f"{failures} failed cycles in a row - stopping. Check: water in front of "
                                  "you, energy left, no menu open, calibration.")
                        self._set(State.ERROR, "Too many consecutive failures")
                        return
                    continue
            failures = 0
            self._play_minigame(layout)
            self._set(State.RESULT, "Waiting for catch animation")
            self._sleep(cfg.post_catch_delay_s)
            if not cfg.auto_recast and cfg.auto_fishing:
                self._log("Auto-recast is off: stopping after one catch.")
                break

    # -------------------------------------------------------- mini-game probe
    def _confirm_layout(self) -> Optional[BarLayout]:
        """Bar must be seen twice, in the same place, to rule out grass/flower false positives."""
        a = self.vision.find_bar(self.grab.grab())
        if a is None:
            return None
        self._sleep(0.03)
        b = self.vision.find_bar(self.grab.grab())
        if b is not None and abs(a.x - b.x) < 6 and abs(a.w - b.w) < 6:
            return b
        return None

    # ------------------------------------------------------------ cast & bite
    def _cast_and_wait_for_bite(self) -> bool:
        cfg = self.cfg
        self._set(State.CASTING, "Casting")
        self.inp.set_hold(True)
        self._sleep(cfg.cast_hold_ms / 1000.0)
        self.inp.set_hold(False)
        cast_time = time.perf_counter()
        self._set(State.WAITING_BITE, "Waiting for bite")
        baseline: list = []                                   # red blobs that were there before any bite
        period = 1.0 / max(5, cfg.bite_poll_hz)
        while self._alive():
            t0 = time.perf_counter()
            det = self.vision.detect_bite(self.grab.grab())
            elapsed = t0 - cast_time
            if elapsed < 1.5:
                if det:
                    baseline.append(det)
            elif det and not any(abs(det[0] - b[0]) < 25 and abs(det[1] - b[1]) < 25 for b in baseline):
                self._log("Bite detected.")
                return True
            if elapsed > cfg.bite_timeout_s:
                self._log("No bite - reeling in and recasting.")
                self.inp.click(60)
                self._sleep(2.0)
                return False
            self._sleep(max(0.0, period - (time.perf_counter() - t0)))
        return False

    def _hook(self) -> Optional[BarLayout]:
        cfg = self.cfg
        self._set(State.HOOKING, "Hooking")
        delay = (cfg.reaction_delay_ms + random.uniform(-cfg.reaction_jitter_ms, cfg.reaction_jitter_ms)) / 1000.0
        self._sleep(max(0.0, delay))
        self.inp.click(60)
        deadline = time.perf_counter() + cfg.minigame_start_timeout_s
        while self._alive() and time.perf_counter() < deadline:
            layout = self._confirm_layout()
            if layout:
                return layout
            self._sleep(0.03)
        return None

    # -------------------------------------------------------------- mini-game
    def _play_minigame(self, layout: BarLayout) -> None:
        cfg, st = self.cfg, self.status
        self._set(State.MINIGAME, "Playing mini-game")
        st.stats.attempts += 1
        fw, fh = self.grab.size
        left, width, offset = self.vision.plan_band(layout, fw)
        box = (left, 0, width, fh)
        ctrl = BarController(cfg, layout.w)
        period = 1.0 / max(30, cfg.control_hz)
        progress, out_time, lost_since = cfg.progress_start, 0.0, None
        last_fish, t_begin = None, time.perf_counter()
        t_prev = t_begin
        next_tick = t_begin
        try:
            while self._alive():
                t = time.perf_counter()
                band = self.grab.grab(box)
                obs = self.vision.measure(band, offset, layout.w, last_fish)
                dt, t_prev = t - t_prev, t
                if obs is None:                                # panel vanished (finished) or glitch
                    lost_since = lost_since or t
                    st.bar_found = False
                    if t - lost_since > 0.4:
                        break
                else:
                    lost_since = None
                    st.bar_found = True
                    last_fish = obs.fish_y
                    d = ctrl.update(t, obs.top, obs.bottom, obs.fish_y, self.inp.held)
                    self.inp.set_hold(d.hold)
                    st.fish_found, st.fish_conf, st.in_bar = obs.fish_y is not None, obs.fish_conf, d.in_bar
                    if d.in_bar is True:
                        progress += cfg.progress_gain_per_s * dt
                    elif d.in_bar is False:
                        progress -= cfg.progress_loss_per_s * dt
                        out_time += dt
                    progress = min(1.0, max(0.0, progress))
                    st.progress_est = progress
                self._emit()
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        finally:
            self.inp.set_hold(False)                           # never leave the button pressed
        st.bar_found = False
        duration = time.perf_counter() - t_begin
        if duration < 1.5:                                     # too short to be a real mini-game
            st.stats.errors += 1
            self._log("Mini-game ended almost immediately - treated as a detection error.")
        elif progress >= 0.5:
            st.stats.successes += 1
            perfect = out_time <= cfg.perfect_tolerance_s
            st.stats.perfects += int(perfect)
            self._log(f"Catch (estimated) in {duration:.1f}s{' - PERFECT' if perfect else ''}; "
                      f"time outside bar {out_time:.2f}s")
        else:
            st.stats.failures += 1
            self._log(f"Fish probably escaped after {duration:.1f}s (estimated progress {progress:.2f}).")
        self._emit(True)
