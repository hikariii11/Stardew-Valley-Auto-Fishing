"""main.py - entry point.

    python main.py                 # open the GUI
    python main.py --calibrate     # learn colours / fish icon from your screen (do this first)
    python main.py --nogui         # headless: run until the emergency-stop key (or Ctrl+C)
"""
from __future__ import annotations

import argparse
import sys
import time

import automation
from config import CONFIG_PATH, Config


def main() -> int:
    parser = argparse.ArgumentParser(description="Stardew Valley auto fishing (screen-reading bot)")
    parser.add_argument("--calibrate", action="store_true", help="run the calibration wizard")
    parser.add_argument("--nogui", action="store_true", help="run without the GUI")
    args = parser.parse_args()

    if not automation.IS_WINDOWS:
        print("This tool controls the Windows version of Stardew Valley (SendInput). "
              "Use sim_test.py to exercise the vision/control code on other systems.")
        return 1

    automation.enable_dpi_awareness()
    cfg = Config.load()
    if not CONFIG_PATH.exists():
        cfg.save()

    if args.calibrate:
        from vision import run_calibration
        run_calibration(cfg)
        return 0

    if args.nogui:
        from fishing_controller import FishingBot
        bot = FishingBot(cfg, log_cb=print)
        bot.start()
        print(f"Running. Press {cfg.emergency_stop_key} (or Ctrl+C) to stop.")
        try:
            while bot.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            bot.stop()
        bot.join()
        return 0

    from gui import run_gui
    run_gui(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
