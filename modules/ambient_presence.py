"""Opt-in Ambient Presence (strictly OFF by default — no webcam, no background screenshots).

Uses only local Win32 foreground window category + input idle duration to give Hikari subtle, silent
living presence while you work or hang out (e.g. curious head-tilt when you switch to code, warm smile
when you return from a break). Raw window titles are categorized locally in RAM and never logged, saved,
or sent to any cloud model.
"""
from __future__ import annotations

import ctypes
import re
import sys
import time
from typing import Any

DEFAULT_AMBIENT_CFG: dict[str, Any] = {
    "enabled": False,            # strictly opt-in!
    "windowReactions": True,     # silent rig expression / mood reactions on context shifts
    "proactiveCheckIn": False,   # optional gentle spoken check-in after long quiet / return (off by default)
    "minIntervalSec": 300,       # >= 5 min between any spoken check-ins
}

REACTION_COOLDOWN_SEC = 45.0
AFK_RETURN_SEC = 300.0           # 5+ min away before a silent welcome-back reaction

_CATEGORY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("studio",   re.compile(r"hikari\s*studio|127\.0\.0\.1:8800", re.I)),
    ("coding",   re.compile(r"visual studio|vscode|pycharm|intellij|cursor|windsurf|antigravity|terminal|powershell|cmd\.exe|github|gitlab|neovim|sublime", re.I)),
    ("gaming",   re.compile(r"steam|elden ring|valorant|minecraft|league of legends|fortnite|genshin|zenless|honkai|cyberpunk|baldur|overwatch|apex|counter-strike|hollow knight|osu!", re.I)),
    ("music",    re.compile(r"spotify|apple music|tidal|soundcloud|foobar2000|musicbee|winamp", re.I)),
    ("video",    re.compile(r"youtube|twitch|netflix|crunchyroll|vlc|mpv|plex|disney\+", re.I)),
    ("writing",  re.compile(r"notion|obsidian|google docs|microsoft word|notepad|typora|scrivener", re.I)),
    ("chat",     re.compile(r"discord|slack|telegram|whatsapp|messenger|signal", re.I)),
    ("browsing", re.compile(r"chrome|firefox|edge|brave|arc|opera|safari", re.I)),
)

# Silent rig-only reactions per context bucket (expression, valence nudge, arousal nudge)
CATEGORY_REACTIONS: dict[str, tuple[str, float, float]] = {
    "coding":   ("curious",  0.15,  0.10),
    "gaming":   ("excited",  0.40,  0.45),
    "music":    ("happy",    0.35,  0.25),
    "video":    ("happy",    0.20,  0.10),
    "writing":  ("curious",  0.15, -0.05),
    "return":   ("happy",    0.45,  0.30),
}

PROACTIVE_LINES: dict[str, str] = {
    "return":  "[happy] Oh, welcome back! Ready to jump back in or just chill for a bit?",
    "coding":  "[curious] Still in deep focus mode over there—want a second pair of eyes or just quiet company?",
    "gaming":  "[excited] Ooh, looks like game time! Good luck, I'm rooting for you!",
}


def classify_window_title(title: str) -> str:
    """Map a raw window title into a coarse privacy-safe category (raw title is immediately discarded)."""
    t = (title or "").strip()
    if not t:
        return "other"
    for cat, pat in _CATEGORY_RULES:
        if pat.search(t):
            return cat
    return "other"


def _win32_foreground_title() -> str:
    if sys.platform != "win32":
        return ""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0 or length > 512:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def _win32_idle_seconds() -> float:
    if sys.platform != "win32":
        return 0.0
    try:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 0.0
        tick = ctypes.windll.kernel32.GetTickCount() & 0xFFFFFFFF
        elapsed_ms = (tick - (lii.dwTime & 0xFFFFFFFF)) & 0xFFFFFFFF
        return elapsed_ms / 1000.0
    except Exception:
        return 0.0


class AmbientPresenceMonitor:
    """Stateful, cooldown-gated local presence tracker. Completely inert unless cfg['enabled'] is True."""

    def __init__(self, cfg: dict[str, Any] | None = None, *, clock=time.monotonic) -> None:
        self.cfg: dict[str, Any] = dict(DEFAULT_AMBIENT_CFG)
        if cfg:
            self.update_config(cfg)
        self._clock = clock
        self._last_category: str = ""
        self._was_afk: bool = False
        self._last_reaction_at: float = -1e9
        self._last_checkin_at: float = -1e9
        self.reactions_count: int = 0
        self.checkins_count: int = 0
        self.last_event: dict[str, Any] = {}

    def update_config(self, patch: dict[str, Any]) -> None:
        for k in DEFAULT_AMBIENT_CFG:
            if k in patch:
                if k == "minIntervalSec":
                    try:
                        self.cfg[k] = max(60, min(3600, int(patch[k])))
                    except (TypeError, ValueError):
                        pass
                else:
                    self.cfg[k] = bool(patch[k])

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.get("enabled", False)),
            "windowReactions": bool(self.cfg.get("windowReactions", True)),
            "proactiveCheckIn": bool(self.cfg.get("proactiveCheckIn", False)),
            "minIntervalSec": int(self.cfg.get("minIntervalSec", 300)),
            "category": self._last_category or "idle",
            "afk": self._was_afk,
            "reactions": self.reactions_count,
            "checkIns": self.checkins_count,
            "lastEvent": dict(self.last_event),
        }

    def poll(self, *, title: str | None = None, idle_sec: float | None = None,
             busy: bool = False) -> dict[str, Any] | None:
        """Sample local foreground category & idle time. Returns None or an action dict:
        {'kind': 'reaction' | 'checkin', 'category': str, 'expression': str, 'valence': float,
         'arousal': float, 'text': str}."""
        if not self.cfg.get("enabled", False) or busy:
            return None
        now = self._clock()
        idle = _win32_idle_seconds() if idle_sec is None else max(0.0, float(idle_sec))
        cat = classify_window_title(_win32_foreground_title() if title is None else title)

        # Detect AFK -> active return
        event_cat = ""
        if idle >= AFK_RETURN_SEC:
            self._was_afk = True
            self._last_category = cat
            return None
        if self._was_afk and idle < 20.0:
            self._was_afk = False
            event_cat = "return"
        elif cat != self._last_category and cat in CATEGORY_REACTIONS:
            if self._last_category:            # skip initial baseline sample on enable
                event_cat = cat
        self._last_category = cat
        if not event_cat:
            return None

        expr, val, aro = CATEGORY_REACTIONS[event_cat]
        min_gap = max(60.0, float(self.cfg.get("minIntervalSec", 300)))
        if (self.cfg.get("proactiveCheckIn", False) and event_cat in PROACTIVE_LINES
                and (now - self._last_checkin_at) >= min_gap):
            self._last_checkin_at = now
            self._last_reaction_at = now
            self.checkins_count += 1
            ev = {"kind": "checkin", "category": event_cat, "expression": expr,
                  "valence": val, "arousal": aro, "text": PROACTIVE_LINES[event_cat], "at": round(now, 2)}
            self.last_event = ev
            return ev

        if self.cfg.get("windowReactions", True) and (now - self._last_reaction_at) >= REACTION_COOLDOWN_SEC:
            self._last_reaction_at = now
            self.reactions_count += 1
            ev = {"kind": "reaction", "category": event_cat, "expression": expr,
                  "valence": val, "arousal": aro, "text": "", "at": round(now, 2)}
            self.last_event = ev
            return ev
        return None
