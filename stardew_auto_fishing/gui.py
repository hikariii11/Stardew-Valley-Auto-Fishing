"""gui.py - small Tkinter control panel (Tkinter ships with Python, no extra dependency)."""
from __future__ import annotations

import queue
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from config import BASE_DIR, TEMPLATE_DIR, Config
from fishing_controller import FishingBot, State, Status

# (label, attribute, type, min, max, step)
NUMERIC_SETTINGS = [
    ("Detection sensitivity (0-1)", "detection_sensitivity", float, 0.0, 1.0, 0.05),
    ("Control responsiveness (0.3-1)", "responsiveness", float, 0.3, 1.0, 0.05),
    ("Prediction strength (0-1.5)", "prediction_strength", float, 0.0, 1.5, 0.1),
    ("Latency compensation (ms)", "latency_ms", float, 0, 200, 5),
    ("Reaction delay (ms)", "reaction_delay_ms", int, 0, 1000, 10),
    ("Cast hold (ms)", "cast_hold_ms", int, 100, 2500, 50),
    ("Bite timeout (s)", "bite_timeout_s", float, 5, 300, 5),
    ("Post-catch wait (s)", "post_catch_delay_s", float, 0.5, 10, 0.5),
]


class App(tk.Tk):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.title("Stardew Valley Auto Fishing")
        self.resizable(False, False)
        self.cfg = cfg
        self.bot: FishingBot | None = None
        self.q: queue.Queue = queue.Queue()
        self.vars: dict = {}
        self._build()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # --------------------------------------------------------------- layout
    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", **pad)
        self.btn_start = ttk.Button(top, text="START", command=self.start, width=12)
        self.btn_stop = ttk.Button(top, text="STOP", command=self.stop, width=12, state="disabled")
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(top, text="Calibrate...", command=self.calibrate).pack(side="left", padx=4)
        ttk.Button(top, text="Save config", command=self.save).pack(side="left", padx=4)

        toggles = ttk.LabelFrame(self, text="Modes")
        toggles.grid(row=1, column=0, sticky="ew", **pad)
        for text, attr in (("Auto fishing (cast + hook)", "auto_fishing"),
                           ("Perfect catch mode", "perfect_mode"),
                           ("Auto recast", "auto_recast")):
            v = tk.BooleanVar(value=getattr(self.cfg, attr))
            v.trace_add("write", lambda *_a, a=attr, var=v: setattr(self.cfg, a, bool(var.get())))
            ttk.Checkbutton(toggles, text=text, variable=v).pack(anchor="w", padx=6)
            self.vars[attr] = v

        status = ttk.LabelFrame(self, text="Status")
        status.grid(row=2, column=0, sticky="ew", **pad)
        self.lbl_state = ttk.Label(status, text="Idle", font=("Segoe UI", 11, "bold"))
        self.lbl_state.grid(row=0, column=0, columnspan=4, sticky="w", padx=6)
        self.lbl_detect = ttk.Label(status, text="Detection: -")
        self.lbl_detect.grid(row=1, column=0, columnspan=4, sticky="w", padx=6)
        self.progress = ttk.Progressbar(status, maximum=100, length=300)
        self.progress.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=2)
        self.counters = {}
        for i, (name, key) in enumerate((("Attempts", "attempts"), ("Successes", "successes"),
                                         ("Perfect", "perfects"), ("Failed", "failures"), ("Errors", "errors"))):
            ttk.Label(status, text=name + ":").grid(row=3 + i // 3, column=(i % 3) * 2, sticky="e", padx=(6, 0))
            lbl = ttk.Label(status, text="0", width=5)
            lbl.grid(row=3 + i // 3, column=(i % 3) * 2 + 1, sticky="w")
            self.counters[key] = lbl

        conf = ttk.LabelFrame(self, text="Configuration")
        conf.grid(row=3, column=0, sticky="ew", **pad)
        row = 0
        for label, attr, typ, lo, hi, step in NUMERIC_SETTINGS:
            ttk.Label(conf, text=label).grid(row=row, column=0, sticky="w", padx=6)
            v = tk.StringVar(value=str(getattr(self.cfg, attr)))
            ttk.Spinbox(conf, from_=lo, to=hi, increment=step, textvariable=v, width=9).grid(row=row, column=1, pady=1)
            self.vars[attr] = (v, typ)
            row += 1
        for label, attr, values in (("Input mode", "input_mode", ("mouse", "keyboard")),):
            ttk.Label(conf, text=label).grid(row=row, column=0, sticky="w", padx=6)
            v = tk.StringVar(value=getattr(self.cfg, attr))
            ttk.Combobox(conf, textvariable=v, values=values, width=8, state="readonly").grid(row=row, column=1, pady=1)
            self.vars[attr] = (v, str)
            row += 1
        ttk.Label(conf, text="Emergency stop key").grid(row=row, column=0, sticky="w", padx=6)
        v = tk.StringVar(value=self.cfg.emergency_stop_key)
        ttk.Entry(conf, textvariable=v, width=10).grid(row=row, column=1, pady=1)
        self.vars["emergency_stop_key"] = (v, str)

        logf = ttk.LabelFrame(self, text="Log / errors")
        logf.grid(row=4, column=0, sticky="ew", **pad)
        self.log = tk.Text(logf, height=8, width=62, state="disabled", wrap="word")
        self.log.pack(padx=4, pady=4)

    # -------------------------------------------------------------- actions
    def _apply(self) -> bool:
        try:
            for attr, entry in self.vars.items():
                if isinstance(entry, tuple):
                    var, typ = entry
                    setattr(self.cfg, attr, typ(var.get()))
            return True
        except ValueError as exc:
            messagebox.showerror("Invalid setting", str(exc))
            return False

    def save(self) -> None:
        if self._apply():
            self.cfg.save()
            self._log("Configuration saved.")

    def calibrate(self) -> None:
        if not self._apply():
            return
        self.cfg.save()
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen([sys.executable, str(BASE_DIR / "main.py"), "--calibrate"], creationflags=flags)
        self._log("Calibration wizard opened in a new console window. Press START afterwards.")

    def start(self) -> None:
        if not self._apply():
            return
        self.cfg.reload_calibration()                      # pick up a freshly run wizard
        if not (TEMPLATE_DIR / self.cfg.fish_template).exists():
            messagebox.showwarning("Calibration needed",
                                   "No fish template found. Click 'Calibrate...' first.")
            return
        self.cfg.save()
        self.bot = FishingBot(self.cfg, status_cb=lambda s: self.q.put(("status", s)),
                              log_cb=lambda m: self.q.put(("log", m)))
        self.bot.start()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

    def stop(self) -> None:
        if self.bot:
            self.bot.stop()

    def _close(self) -> None:
        self.stop()
        if self.bot:
            self.bot.join(2.0)
        self.destroy()

    # ---------------------------------------------------------------- updates
    def _log(self, msg: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _show(self, s: Status) -> None:
        self.lbl_state.config(text=f"{s.state.value}: {s.message}" if s.message != s.state.value else s.state.value)
        if s.state is State.MINIGAME:
            fish = f"fish {s.fish_conf:.2f}" if s.fish_found else "fish NOT found"
            inb = {True: "in bar", False: "OUT of bar", None: "?"}[s.in_bar]
            self.lbl_detect.config(text=f"Detection: bar {'OK' if s.bar_found else 'lost'} | {fish} | {inb}")
            self.progress["value"] = s.progress_est * 100
        else:
            self.lbl_detect.config(text="Detection: -")
            self.progress["value"] = 0
        for key, lbl in self.counters.items():
            lbl.config(text=str(getattr(s.stats, key)))
        if s.state in (State.STOPPED, State.ERROR):
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self._show(payload) if kind == "status" else self._log(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll)


def run_gui(cfg: Config) -> None:
    App(cfg).mainloop()
