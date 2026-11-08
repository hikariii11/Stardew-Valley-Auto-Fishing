"""config.py - persistent settings for the Stardew Valley auto-fishing tool.

All tunables live in one dataclass that is saved to ``config.json`` next to
this file. Unknown keys in the JSON are ignored, missing keys fall back to the
defaults below, so upgrading never breaks an old config.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
TEMPLATE_DIR = BASE_DIR / "templates"

# Keys written by the calibration wizard (see vision.run_calibration).
CALIBRATION_KEYS = ("bar_color", "bite_color", "ref_bar_w", "ref_frame_h", "fish_template")


@dataclass
class Config:
    # ------------------------------------------------------------ behaviour
    auto_fishing: bool = True          # False = "assist" mode: you cast/hook, bot plays the mini-game
    perfect_mode: bool = True          # tighter control + stricter targeting
    auto_recast: bool = True           # keep fishing after every catch
    input_mode: str = "mouse"          # "mouse" (hold left button) or "keyboard" (hold use-tool key)
    use_tool_key: str = "C"            # Stardew default "Use Tool" key (only for keyboard mode)
    emergency_stop_key: str = "F10"    # global hotkey, works even when the GUI is not focused
    mark_key: str = "F8"               # used only by the calibration wizard
    game_window_title: str = "Stardew Valley"
    require_focus: bool = True         # never send input unless the game window is in the foreground

    # --------------------------------------------------------------- timing
    reaction_delay_ms: int = 120       # delay between seeing the bite "!" and clicking
    reaction_jitter_ms: int = 40       # +/- random jitter added to the reaction delay
    cast_hold_ms: int = 800            # how long the cast button is held (cast distance)
    post_catch_delay_s: float = 3.5    # wait for the catch animation before recasting
    bite_timeout_s: float = 45.0       # give up waiting for a bite after this long
    minigame_start_timeout_s: float = 3.0
    max_consecutive_errors: int = 5    # stop after this many failed cycles in a row
    bite_poll_hz: int = 40             # screen polls per second while waiting for a bite
    control_hz: int = 120              # control loop rate during the mini-game

    # ------------------------------------------------------------ detection
    detection_sensitivity: float = 0.5  # 0..1, higher = accept weaker fish matches
    # Colour specs are [hue(0-179), hue_tolerance, min_saturation, min_value].
    # The defaults are only a starting guess: run the calibration wizard.
    bar_color: list = field(default_factory=lambda: [38, 16, 90, 90])
    bite_color: list = field(default_factory=lambda: [3, 8, 140, 140])
    fish_template: str = "fish.png"
    ref_bar_w: float = 0.0             # bar width (px) when the fish template was captured
    ref_frame_h: float = 0.0           # game-window height (px) at calibration time

    # -------------------------------------------------------------- control
    responsiveness: float = 0.8        # 0.3..1.0  tracking-filter gain (higher = trusts the newest frame)
    prediction_strength: float = 1.0   # 0..1.5    how much of the fish velocity is extrapolated
    latency_ms: float = 45.0           # capture -> input -> game-tick delay compensated by the predictor
    deadband_frac: float = 0.10        # width of the soft switching zone, as a fraction of bar height
    accel_per_bar_w: float = 22.0      # initial guess of bar acceleration (px/s^2 per px of bar width)

    # ---------------------------------------------- statistics model (estimates)
    perfect_tolerance_s: float = 0.03  # fish may be seen outside the bar this long and still count "perfect"
    progress_start: float = 0.30
    progress_gain_per_s: float = 0.12  # ~0.002 per 60 Hz tick (from memory of the game code - unverified)
    progress_loss_per_s: float = 0.18  # ~0.003 per 60 Hz tick (unverified)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return cfg
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def reload_calibration(self, path: Path = CONFIG_PATH) -> None:
        """Pull only the calibration values from disk (used after running the wizard)."""
        fresh = Config.load(path)
        for key in CALIBRATION_KEYS:
            setattr(self, key, getattr(fresh, key))

    @property
    def match_threshold(self) -> float:
        """Template-match score needed to accept a fish detection."""
        return 0.85 - 0.5 * max(0.0, min(1.0, self.detection_sensitivity))
