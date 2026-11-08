"""vision.py - screen capture and computer vision for the fishing mini-game.

Detection strategy (no hard-coded screen coordinates):

* Green bobber bar  : HSV colour mask -> tall, solid, rectangular blob. Its
                      position gives the column band to watch afterwards.
* Fish icon         : OpenCV template matching inside that narrow band only,
                      scaled by (current bar width / bar width at calibration).
* Bite "!" marker   : red HSV mask -> vertical blob with the aspect ratio of
                      an exclamation mark, ignoring anything that was already
                      on screen right after the cast.

Colours and the fish template are captured by ``run_calibration`` because the
exact palette depends on the game version, mods and your display settings.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mss
import numpy as np

import automation
from config import TEMPLATE_DIR, Config


# ------------------------------------------------------------------ helpers
def color_mask(bgr: np.ndarray, spec) -> np.ndarray:
    """Binary (0/255) mask of pixels matching [hue, hue_tol, min_sat, min_val]."""
    h0, tol, s_min, v_min = spec
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    d = np.abs(hsv[:, :, 0].astype(np.int16) - int(h0))
    d = np.minimum(d, 180 - d)  # hue is circular (red wraps around 0/179)
    m = (d <= tol) & (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)
    return m.astype(np.uint8) * 255


class ScreenGrabber:
    """mss wrapper that captures the game's client area (or the primary monitor).

    mss handles are thread-bound: create and use this object in one thread.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sct = mss.mss()
        self.hwnd = 0
        self.origin = (0, 0)
        self.size = (0, 0)
        self.refresh_window()

    def refresh_window(self) -> None:
        region = None
        if automation.IS_WINDOWS:
            self.hwnd = automation.find_window(self.cfg.game_window_title)
            region = automation.client_region(self.hwnd) if self.hwnd else None
        if region is None or region[2] < 200 or region[3] < 200:
            mon = self.sct.monitors[1]
            region = (mon["left"], mon["top"], mon["width"], mon["height"])
        self.origin = (region[0], region[1])
        self.size = (region[2], region[3])

    def grab(self, box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture ``box`` = (x, y, w, h) in game-window coordinates -> BGR array."""
        gw, gh = self.size
        x, y, w, h = box if box else (0, 0, gw, gh)
        x, y = max(0, int(x)), max(0, int(y))
        w, h = max(1, min(int(w), gw - x)), max(1, min(int(h), gh - y))
        mon = {"left": self.origin[0] + x, "top": self.origin[1] + y, "width": w, "height": h}
        return np.ascontiguousarray(np.asarray(self.sct.grab(mon))[:, :, :3])


@dataclass
class BarLayout:
    x: float        # left edge of the bobber bar (game-window px)
    w: float        # bar width
    top: float
    bottom: float


@dataclass
class Observation:
    top: float
    bottom: float
    fish_y: Optional[float]   # None = fish icon not found this frame
    fish_conf: float


# ------------------------------------------------------------------- vision
class FishingVision:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fish_tpl: Optional[np.ndarray] = None
        self._tpl_cache: dict = {}
        self.load_templates()

    # -------------------------------------------------------------- setup
    def load_templates(self) -> None:
        path = TEMPLATE_DIR / self.cfg.fish_template
        self.fish_tpl = cv2.imread(str(path)) if path.exists() else None
        self._tpl_cache.clear()

    def require_ready(self) -> None:
        if self.fish_tpl is None:
            raise RuntimeError("Fish template missing - run the calibration wizard first "
                               "(python main.py --calibrate).")

    def scale_for(self, bar_w: float) -> float:
        ref = self.cfg.ref_bar_w
        return float(bar_w) / ref if ref > 0 else 1.0

    def _tpl(self, scale: float) -> np.ndarray:
        key = round(scale, 2)
        if key not in self._tpl_cache:
            interp = cv2.INTER_NEAREST if key >= 1 else cv2.INTER_AREA
            self._tpl_cache[key] = cv2.resize(self.fish_tpl, None, fx=key, fy=key, interpolation=interp)
        return self._tpl_cache[key]

    # --------------------------------------------------------- bobber bar
    def find_bar(self, frame: np.ndarray) -> Optional[BarLayout]:
        """Locate the green bobber bar anywhere on screen (used to detect the mini-game)."""
        s = 0.5
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = color_mask(small, self.cfg.bar_color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        fh = mask.shape[0]
        best = None
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not (0.03 * fh <= h <= 0.45 * fh) or not (0.01 * fh <= w <= 0.10 * fh):
                continue                      # wrong size for a bobber bar
            if h < 1.4 * w or area < 0.55 * w * h:
                continue                      # not a tall, solid rectangle
            # the progress bar is also tall/green but narrower: prefer the widest candidate
            if best is None or (w, area) > (best[2], best[4]):
                best = (x, y, w, h, area)
        if best is None:
            return None
        x, y, w, h, _ = best
        return BarLayout(x / s, w / s, y / s, (y + h) / s)

    def plan_band(self, layout: BarLayout, frame_w: int) -> Tuple[int, int, int]:
        """Narrow full-height column around the bar -> (left, width, bar_offset_in_band)."""
        scale = self.scale_for(layout.w)
        tpl_w = self.fish_tpl.shape[1] * scale if self.fish_tpl is not None else layout.w
        pad = int(max(tpl_w, 0.5 * layout.w)) + 4
        left = max(0, int(layout.x) - pad)
        offset = int(layout.x) - left
        width = min(frame_w - left, offset + int(layout.w) + pad)
        return left, width, offset

    def measure(self, band: np.ndarray, offset: int, bar_w: float,
                last_fish_y: Optional[float]) -> Optional[Observation]:
        """Bar extent + fish position inside the narrow band (called every control tick)."""
        scale = self.scale_for(bar_w)
        mask = color_mask(band, self.cfg.bar_color)
        c0, c1 = int(offset + 0.1 * bar_w), int(offset + 0.9 * bar_w)
        if c1 - c0 < 2:
            return None
        counts = (mask[:, c0:c1] > 0).sum(axis=1)
        ok = counts >= 0.6 * (c1 - c0)
        # The fish icon often hides the middle of the bar but not its outer columns: count a row as
        # "bar" if either outer strip is green too, so the bar's true extent survives the occlusion.
        k = max(2, int(0.15 * bar_w))
        edge = ((mask[:, c0:c0 + k] > 0).sum(axis=1) >= 0.8 * k) | ((mask[:, c1 - k:c1] > 0).sum(axis=1) >= 0.8 * k)
        rows = np.flatnonzero(ok | edge)
        if rows.size < 5:
            return None
        # The fish icon can cover part of the bar and split it; bridge gaps up to one icon height.
        fish_h = self._tpl(scale).shape[0] if self.fish_tpl is not None else bar_w
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > int(fish_h * 1.2)) + 1)
        g = max(groups, key=lambda a: a[-1] - a[0])
        top, bottom = float(g[0]), float(g[-1] + 1)
        fish_y, conf = self.find_fish(band, scale, last_fish_y)
        return Observation(top, bottom, fish_y, conf)

    # --------------------------------------------------------------- fish
    def find_fish(self, band: np.ndarray, scale: float,
                  last_y: Optional[float]) -> Tuple[Optional[float], float]:
        tpl = self._tpl(scale)
        th, tw = tpl.shape[:2]
        bh, bw = band.shape[:2]
        if bh < th or bw < tw:
            return None, 0.0
        thr = self.cfg.match_threshold

        def search(y0: int, y1: int):
            sub = band[y0:y1]
            if sub.shape[0] < th:
                return None, 0.0
            res = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            return (y0 + loc[1] + th / 2.0, float(mx)) if np.isfinite(mx) else (None, 0.0)

        if last_y is not None:                       # fast path: small window around last position
            r = int(0.25 * bh)
            y, conf = search(max(0, int(last_y) - r - th // 2), min(bh, int(last_y) + r + th))
            if y is not None and conf >= thr:
                return y, conf
        y, conf = search(0, bh)                       # slow path: whole band
        return (y, conf) if y is not None and conf >= thr else (None, conf)

    # ---------------------------------------------------------- bite mark
    def detect_bite(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Centre (x, y) of a red exclamation-mark-shaped blob, or None."""
        fh, fw = frame.shape[:2]
        x0, x1, y0, y1 = int(0.10 * fw), int(0.90 * fw), int(0.05 * fh), int(0.80 * fh)
        s = 0.5
        sub = cv2.resize(frame[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = color_mask(sub, self.cfg.bite_color)
        kh = max(3, int(fh * s / 110))                # bridges the gap between the "!" stick and dot
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        hs = fh * s
        best = None
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if w < 2 or not (0.015 * hs <= h <= 0.12 * hs):
                continue
            if not (1.7 <= h / w <= 4.2) or area < 0.5 * w * h:
                continue
            if best is None or area > best[0]:
                best = (area, x0 + (x + w / 2.0) / s, y0 + (y + h / 2.0) / s)
        return (best[1], best[2]) if best else None


# -------------------------------------------------------------- calibration
def _hue_stats(roi: np.ndarray, circular: bool = False) -> List[int]:
    """Turn a user-selected patch into a [hue, tol, min_sat, min_val] colour spec."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = hsv[(hsv[:, 1] > 90) & (hsv[:, 2] > 90)]
    if len(sel) < 6:
        raise RuntimeError("The selection contains too few saturated pixels - select a solid coloured patch.")
    h = sel[:, 0].astype(int)
    if circular:
        h = (h + 90) % 180
    h0 = int(np.median(h))
    tol = int(min(25, np.percentile(np.abs(h - h0), 95) + 5))
    if circular:
        h0 = (h0 - 90) % 180
    return [h0, tol, int(max(40, np.percentile(sel[:, 1], 5) - 25)), int(max(40, np.percentile(sel[:, 2], 5) - 25))]


def _select_roi(img: np.ndarray, title: str) -> np.ndarray:
    """cv2.selectROI on a possibly down-scaled copy; returns the crop at full resolution."""
    k = min(1.0, 1500.0 / img.shape[1])
    view = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    x, y, w, h = cv2.selectROI(title, view, showCrosshair=True)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        raise RuntimeError("No region selected.")
    x, y, w, h = (int(v / k) for v in (x, y, w, h))
    return img[y:y + h, x:x + w].copy()


def _pick_frame(frames: List[np.ndarray], title: str) -> np.ndarray:
    win = title + "  (slide, then press ENTER)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", win, len(frames) - 1, max(1, len(frames) - 1), lambda v: None)
    while True:
        i = min(cv2.getTrackbarPos("frame", win), len(frames) - 1)
        cv2.imshow(win, frames[i])
        if cv2.waitKey(30) & 0xFF in (13, 32):
            break
    cv2.destroyWindow(win)
    return frames[i]


def run_calibration(cfg: Config) -> None:
    """Interactive wizard: learns the bar colour, bite colour and fish icon from YOUR screen."""
    automation.require_windows()
    automation.enable_dpi_awareness()
    grab = ScreenGrabber(cfg)
    vk_mark = automation.vk_from_name(cfg.mark_key)
    frames: deque = deque(maxlen=24)

    def wait_mark(prompt: str) -> List[np.ndarray]:
        print(prompt)
        while automation.key_pressed(vk_mark):
            time.sleep(0.05)
        while not automation.key_pressed(vk_mark):
            frames.append(grab.grab())
            time.sleep(0.05)                      # ~20 fps ring buffer (about 1.2 s of history)
        return list(frames)

    print("=== Stardew auto-fishing calibration ===")
    print(f"Keep the game visible (windowed or borderless). Mark key: {cfg.mark_key}\n")
    bite_frames = wait_mark(f"STEP 1: cast your line. When the red '!' appears, press {cfg.mark_key}.")
    frames.clear()
    mini = wait_mark(f"STEP 2: hook the fish. While the mini-game is visible and the fish icon is not "
                     f"hidden, press {cfg.mark_key}.")[-1]

    # ---- bite colour
    img = _pick_frame(bite_frames, "Pick the frame where the '!' is visible")
    roi = _select_roi(img, "Draw a TIGHT box around the '!' then ENTER")
    cfg.bite_color = _hue_stats(roi, circular=True)

    # ---- mini-game colours / templates
    fish = _select_roi(mini, "Draw a tight box around the FISH ICON then ENTER")
    patch = _select_roi(mini, "Draw a box fully INSIDE the green bar then ENTER")
    cfg.bar_color = _hue_stats(patch)
    TEMPLATE_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(TEMPLATE_DIR / cfg.fish_template), fish)

    vision = FishingVision(cfg)
    layout = vision.find_bar(mini)
    if layout is None:
        raise RuntimeError("Could not find the bar with the learned colour. Try again with a cleaner selection.")
    cfg.ref_bar_w, cfg.ref_frame_h = float(layout.w), float(mini.shape[0])
    vision.cfg = cfg

    # ---- verification overlay
    left, width, off = vision.plan_band(layout, mini.shape[1])
    obs = vision.measure(mini[:, left:left + width], off, layout.w, None)
    vis = mini.copy()
    cv2.rectangle(vis, (int(layout.x), int(layout.top)), (int(layout.x + layout.w), int(layout.bottom)), (0, 0, 255), 2)
    if obs and obs.fish_y is not None:
        cv2.circle(vis, (int(layout.x + layout.w / 2), int(obs.fish_y)), 12, (255, 0, 0), 2)
    print(f"Bar width {layout.w:.0f}px, bar {'found'}, fish {'found' if obs and obs.fish_y else 'NOT found'}.")
    cv2.namedWindow("Verification - press any key", cv2.WINDOW_NORMAL)
    cv2.imshow("Verification - press any key", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cfg.save()
    print("Calibration saved to config.json and templates/. You can start the bot now.")
