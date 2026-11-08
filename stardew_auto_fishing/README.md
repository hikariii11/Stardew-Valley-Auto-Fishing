# Stardew Valley Auto Fishing (screen-reading bot, Windows)

Reads the screen with OpenCV, plays the fishing mini-game with a predictive
controller, and loops cast → bite → mini-game → recast. It never touches game
memory or files, so it does not need a mod loader.

> Intended for your own single-player (or co-op with your friends' consent) game.
> **Read "Honest limitations" before expecting a 100 % perfect rate.**

---

## 1. How the fishing mini-game works

What follows is the mechanic as I know it; items marked **(unverified)** come from
memory of the game's code / observation and are exactly why a calibration step exists.

1. **Cast** – hold the use-tool button to charge, release to throw. Hold time sets distance.
2. **Bite** – after a random wait a **red "!"** appears above the player and a sound plays.
   You have a short window (well under a second) to click.
3. **Mini-game** – a vertical panel appears. It contains:
   * a **green bobber bar** you control: **holding** the button accelerates it **up**,
     releasing lets "gravity" pull it **down** (roughly symmetric acceleration,
     about 0.25 px/tick² at 60 ticks/s **(unverified)**; the bar bounces off the bottom),
   * a **fish icon** that wanders, sometimes darting (behaviour depends on the fish's difficulty),
   * a thin **progress bar** that fills while the fish is inside the bar and drains otherwise,
   * sometimes a **treasure chest** icon.
4. **Result** – catch when the progress bar fills, escape when it empties.
   A **perfect** catch means the fish never left the bar (**exact rule unverified**; this
   tool estimates it from its own observations).

### Technical challenges
| Challenge | How it is handled |
|---|---|
| No API for game state | Screen capture (`mss`) of the game client area only |
| Resolution / UI scale differences | No fixed coordinates: bar found by colour+shape, fish template scaled by *bar width ÷ bar width at calibration* |
| Exact colours/sprites vary | Calibration wizard learns bar colour, "!" colour and fish icon from **your** screen |
| Game ticks at 60 Hz, we poll at 120 Hz | Duplicate frames are detected and not fed to the filters |
| Capture→input→tick latency (tens of ms) | Smith-predictor-style look-ahead (`latency_ms`) |
| Fish hides part of the bar | Edge-column measurement, gap bridging, constant bar height |
| Only a binary input (hold/release) | Time-optimal switching law + sigma-delta dithering |

## 2. Architecture

```
main.py                 entry point (GUI / --calibrate / --nogui)
gui.py                  Tkinter panel: start/stop, toggles, status, counters, config, log
fishing_controller.py   BarController (algorithm) + FishingBot (state machine, thread)
vision.py               capture, bar/fish/bite detection, calibration wizard
automation.py           SendInput (mouse/keyboard) + window helpers, ctypes only
config.py               settings dataclass <-> config.json
sim_test.py             offline simulation test of vision + controller (any OS)
```

State machine: `CASTING → WAITING_BITE → HOOKING → MINIGAME → RESULT → CASTING …`

## 3. Control algorithm and why

The bar is a point mass with constant symmetric acceleration `a` and a **binary** actuator.
For that system the classic optimum is a **time-optimal switching law** (bang-bang on a switching parabola):

```
y     = bar_position - fish_position        y_dot = bar_velocity - fish_velocity
sigma = y + y_dot*|y_dot| / (2a)            hold the button  <=>  sigma > 0
```

Why not plain PID? A PID output is continuous, but the input is on/off, and PID ignores the
hard acceleration limit, so it overshoots or reacts late. The switching law already encodes
"how early must I brake". Around it I added:

* **alpha-beta filters** for position/velocity of bar and fish (noise-robust, variable dt),
* **latency look-ahead** (`latency_ms`) and **fish extrapolation** (`prediction_strength`),
* a **soft boundary layer** with sigma-delta dithering instead of hard chattering,
* **online estimation of `a`** (bounded), so it adapts to UI scale / game version.

"Adaptive to difficulty" happens implicitly: faster fish → larger estimated velocity → earlier,
stronger corrections. No controller can beat the physics, see limitations.

## 4. Installation

Requirements: Windows 10/11, Python 3.11+, Stardew Valley in **windowed or borderless** mode.

```bat
cd stardew_auto_fishing
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Run

1. Start Stardew, stand at the water facing it (rod equipped, energy available).
2. **Calibrate once** (and again if you change resolution, zoom, UI scale or mods):
   ```bat
   python main.py --calibrate
   ```
   * Step 1 – cast; when the red "!" appears press **F8**. Pick the frame showing it
     (slider + ENTER) and draw a tight box around the "!".
   * Step 2 – hook a fish; while the mini-game is visible (fish icon not hidden) press **F8**.
     Draw a box around the **fish icon**, then a box fully **inside the green bar**.
   * A verification window shows the detected bar (red) and fish (blue). If wrong, re-run.
3. Start the tool:
   ```bat
   python main.py            # GUI
   python main.py --nogui    # headless
   ```
4. Click the game window, press **START**. **F10** is the global emergency stop.
   The bot refuses to send input while the game is not the foreground window.

Modes: *Auto fishing* off = **assist mode** (you cast and hook, the bot plays the mini-game).

## 6. Configuration (`config.json`, or the GUI)

| Setting | Meaning |
|---|---|
| `detection_sensitivity` | Higher accepts weaker fish-template matches (threshold = 0.85 − 0.5·value) |
| `responsiveness` | Tracking filter gain; raise if the bar lags, lower if it jitters |
| `prediction_strength` | Share of fish velocity extrapolated (0 = none) |
| `latency_ms` | Estimated delay capture→game reacting; tune 20–80 |
| `reaction_delay_ms` / jitter | Delay between seeing "!" and clicking |
| `input_mode` | `mouse` (hold left button) or `keyboard` (hold `use_tool_key`) |
| `perfect_mode` | Tighter switching layer |
| `auto_recast`, `cast_hold_ms`, `bite_timeout_s`, `post_catch_delay_s` | Loop behaviour |
| `emergency_stop_key` | Global hotkey |

`input_mode=mouse`: clicking makes the character face the cursor, so keep the cursor over the water.
`keyboard` uses the character's current facing instead.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "Fish template missing" | Run `--calibrate` |
| Bar never detected | Re-calibrate with a cleaner patch inside the bar; don't use exclusive fullscreen; check the verification window |
| Bar detected but wrong thing (grass etc.) | Re-calibrate; tight patch; disable shaders/mods that recolour the UI |
| Fish "NOT found" often | Raise `detection_sensitivity`; re-capture the template tighter with the fish fully visible |
| Never reacts to bites | Re-calibrate step 1 (the "!" colour); make sure the character isn't off-centre near a map edge with red objects in view |
| Clicks but mini-game doesn't start | Lower `reaction_delay_ms`; the bite window is short (also check the cast actually reached water) |
| Bar oscillates | Lower `responsiveness` to ~0.6, lower `prediction_strength` |
| Bar lags the fish | Raise `latency_ms`/`prediction_strength`, check CPU load |
| Stops with "Too many consecutive failures" | No water in front, out of energy, or a menu is open (e.g. a treasure chest menu) |
| No input at all | Game must be foreground; try `keyboard` mode; run the terminal as the same privilege level as the game |

## 8. Performance tips

* Capture is limited to a **narrow column** around the bar during the mini-game; the whole screen is only grabbed while waiting.
* Lower `control_hz` to 90 if CPU use is high (game ticks at 60 Hz, so ≥ 90 is plenty).
* Play at 1080p or lower; keep the game window unobstructed.
* Close overlays (Discord, GPU overlays) that draw over the game.
* Measured in the offline simulation: ≈3 ms of vision+control per frame on a laptop-class CPU (capture adds ~1–3 ms).

## 9. Honest limitations (please read)

* **Not tested against the real game.** I could not run Stardew here. The pipeline was
  tested in `sim_test.py` with synthetic frames and physics I reproduced from memory
  (symmetric ±0.25 px/tick², 568 px track). Real colours, sprite sizes, the real physics
  constants, the bite "!" look, and `SendInput` behaviour inside the game are
  **unverified**; calibration and the troubleshooting table exist for that reason.
* **"Perfect" cannot be guaranteed.** The bar's acceleration is limited. When a fish
  reverses direction faster than the bar can turn around, no controller can keep up.
  Simulation (bar 96 px, my own fish model), share of time the fish was inside the bar:

  | fish difficulty (sim) | in-bar time |
  |---|---|
  | easy (0.3) | ≈ 97–98 % |
  | hard (0.7–1.0) | ≈ 84–91 % (bigger bars from higher fishing level help) |

  A "perfect" (zero seconds outside) is therefore likely on easy fish and *sometimes* on harder ones.
* The **success/perfect counters are estimates** (progress-bar model + own observations).
  The progress gain/loss constants are from memory (unverified).
* **Treasure chests**: the bot never aims for them. If the bar happens to collect one, a menu opens
  and the bot will stop after repeated failures; close it and press START again.
* Only single-resolution calibration is stored; recalibrate after changing UI scale.
* Stamina/energy and a full inventory are not detected.
* More reliable alternative: a SMAPI mod that reads the game's fishing state directly. If a
  screen-reading bot proves too fragile on your setup, say so and I can write that instead.
