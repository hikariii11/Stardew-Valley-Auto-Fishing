"""automation.py - Windows input simulation and window helpers.

Pure ``ctypes`` (no extra dependency). Uses ``SendInput`` so the events look
like real hardware input to the game. Everything that touches the Win32 API is
guarded so the other modules can still be imported (and unit-tested) on other
platforms.
"""
from __future__ import annotations

import ctypes
import sys
import time
from typing import Optional, Tuple

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    ULONG_PTR = ctypes.c_size_t

    INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
    MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
    KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0002, 0x0008

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _InputUnion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _InputUnion)]

    user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    user32.SendInput.restype = wintypes.UINT
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.FindWindowW.restype = wintypes.HWND
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
    user32.GetAsyncKeyState.restype = ctypes.c_short


def require_windows() -> None:
    if not IS_WINDOWS:
        raise RuntimeError("Input simulation is only implemented for Windows (SendInput).")


# ------------------------------------------------------------------ system
def enable_dpi_awareness() -> None:
    """Make coordinates physical pixels so they match what mss captures."""
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    try:  # 1 ms timer resolution -> accurate sleeps in the control loop
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


_SPECIAL_KEYS = {"ESC": 0x1B, "SPACE": 0x20, "ENTER": 0x0D, "TAB": 0x09, "SHIFT": 0x10,
                 "CTRL": 0x11, "ALT": 0x12, "PAUSE": 0x13, "END": 0x23, "HOME": 0x24}


def vk_from_name(name: str) -> int:
    """'F10' -> 0x79, 'c' -> 0x43, 'ESC' -> 0x1B ..."""
    n = name.strip().upper()
    if n in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[n]
    if len(n) == 1 and n.isalnum():
        return ord(n)
    if n.startswith("F") and n[1:].isdigit() and 1 <= int(n[1:]) <= 24:
        return 0x70 + int(n[1:]) - 1
    raise ValueError(f"Unsupported key name: {name!r}")


def key_pressed(vk: int) -> bool:
    """Global (focus independent) key state."""
    return IS_WINDOWS and bool(user32.GetAsyncKeyState(vk) & 0x8000)


# ----------------------------------------------------------------- windows
def find_window(title: str) -> int:
    return int(user32.FindWindowW(None, title) or 0) if IS_WINDOWS else 0


def is_foreground(hwnd: int) -> bool:
    return bool(hwnd) and int(user32.GetForegroundWindow() or 0) == hwnd


def client_region(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """(left, top, width, height) of the window's client area in screen pixels."""
    if not (IS_WINDOWS and hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top


# ------------------------------------------------------------------- input
class InputController:
    """Holds / releases the 'use tool' action either as left mouse button or key."""

    def __init__(self, mode: str = "mouse", key: str = "C") -> None:
        require_windows()
        if mode not in ("mouse", "keyboard"):
            raise ValueError("input_mode must be 'mouse' or 'keyboard'")
        self.mode = mode
        self.vk = vk_from_name(key)
        self.scan = user32.MapVirtualKeyW(self.vk, 0)
        self.held = False

    @staticmethod
    def _send(inp: "INPUT") -> None:
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _event(self, down: bool) -> None:
        if self.mode == "mouse":
            flag = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
            inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(0, 0, 0, flag, 0, 0))
        else:
            flags = KEYEVENTF_SCANCODE | (0 if down else KEYEVENTF_KEYUP)
            inp = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, self.scan, flags, 0, 0))
        self._send(inp)

    def set_hold(self, hold: bool) -> None:
        """Idempotent: only sends an event when the state actually changes."""
        if hold and not self.held:
            self._event(True)
            self.held = True
        elif not hold and self.held:
            self._event(False)
            self.held = False

    def click(self, ms: float = 60.0) -> None:
        self.set_hold(True)
        time.sleep(ms / 1000.0)
        self.set_hold(False)

    def release_all(self) -> None:
        try:
            self.set_hold(False)
        except Exception:
            pass
