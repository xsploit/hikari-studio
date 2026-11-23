"""Plugin: Interactive Head-Pat Physics
Click or rub the top of Hikari's head on the live stage to trigger a cute head-pat reaction:
happy squint, rosy blush lines, gentle head nuzzle, and sparkle burst!
Writes only parameters the rig really has (the real brow names are `Left Brow Y` / `Right Brow Y`), and
records the pat / poke as a stage event so Hikari can react to it in her next reply.
"""
from __future__ import annotations

import math
import time

from core.plugins import StudioPlugin

LEFT_BROW_Y = "Left Brow Y"
RIGHT_BROW_Y = "Right Brow Y"


def _get(p: dict, name: str) -> float:
    v = p.get(name)
    return float(v[0]) if isinstance(v, (list, tuple)) and v else 0.0


def _set(p: dict, name: str, value: float, y: float | None = None) -> None:
    """Write a parameter only if the rig defines it (unknown names would be silently dropped anyway)."""
    if name in p:
        cur = p[name]
        p[name] = (float(value), float(y) if y is not None else (float(cur[1]) if isinstance(cur, (list, tuple)) and len(cur) > 1 else 0.0))


def _kick(rig, spring: str, dv: float) -> None:
    sp = getattr(rig, spring, None)
    if sp is not None and hasattr(sp, "v"):
        sp.v += dv


class HeadPatPlugin(StudioPlugin):
    id = "head_pat"
    name = "Interactive Head-Pat Reaction"
    version = "1.1.0"
    description = "Click or rub Hikari's head on stage for happy squint, rosy blush lines, head nuzzle & sparkle burst."
    default_enabled = True
    default_settings = {
        "intensity": 1.0,
        "decaySec": 2.6,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pat_energy = 0.0
        self._poke_energy = 0.0
        self._poke_side = 1.0
        self._last_pat = 0.0
        self._last_event_note = 0.0

    def register(self, hooks, ctx) -> None:
        hooks.register(self.id, "on_stage_interaction", self._on_interact, priority=40)
        hooks.register(self.id, "on_frame_params", self._on_frame, priority=60)

    def _note(self, ctx, text: str) -> None:
        """At most one stage event per 4 s (rubbing fires many events)."""
        now = time.monotonic()
        note = getattr(ctx, "note_stage_event", None)
        if callable(note) and now - self._last_event_note > 4.0:
            self._last_event_note = now
            try:
                note(text, kind="head_pat")
            except Exception:
                pass

    def _on_interact(self, ctx, event: dict) -> None:
        rig_cfg = ctx.config.data.get("rig", {}) if hasattr(ctx, "config") else {}
        if not rig_cfg.get("headPatEnabled", True):
            return
        act = str(event.get("action") or event.get("kind") or "").lower()
        try:
            speed = float(event.get("speed", 0.8))
        except (TypeError, ValueError):
            speed = 0.8
        if not math.isfinite(speed):
            speed = 0.8
        if act in ("head_pat", "pat"):
            spd = min(2.0, max(0.35, speed))
            self._pat_energy = min(1.0, self._pat_energy + 0.32 * spd)
            self._last_pat = time.perf_counter()
            rig = getattr(ctx, "rig", None)
            if rig is not None:
                _kick(rig, "sp_chest_y", -4.2 * spd)
                _kick(rig, "sp_eye_jelly", 5.5)
            self._note(ctx, "the viewer just patted your head")
        elif act == "poke":
            self._poke_energy = min(1.0, self._poke_energy + 0.85)
            try:
                x = float(event.get("x", 0.0))
            except (TypeError, ValueError):
                x = 0.0
            self._poke_side = -1.0 if x >= 0.0 else 1.0
            rig = getattr(ctx, "rig", None)
            if rig is not None:
                _kick(rig, "sp_chest_y", 9.5)
                _kick(rig, "sp_chest_x", 8.0 * self._poke_side)
                _kick(rig, "sp_eye_jelly", 11.0)
            self._note(ctx, "the viewer just poked you")

    def _on_frame(self, ctx, frame: dict) -> None:
        dt = float(frame.get("dt", 1 / 60))
        t = float(frame.get("time", 0.0))
        p = frame["params"]

        if self._pat_energy > 0.005:
            decay = max(0.5, float(self.settings.get("decaySec", 2.6)))
            self._pat_energy = max(0.0, self._pat_energy - dt / decay)
            k = self._pat_energy * float(self.settings.get("intensity", 1.0))
            if "Angle" in p:
                ax, ay = p["Angle"]
                p["Angle"] = (ax + 5.0 * k * math.sin(t * 7.2), ay - 5.2 * k)
            _set(p, "Angle Z", _get(p, "Angle Z") + 7.5 * k * math.sin(t * 5.4), 0.0)
            _set(p, "Body Angle Y", _get(p, "Body Angle Y") - 2.6 * k, 0.0)
            for name, amt in (("Eye L Smile", 0.92), ("Eye R Smile", 0.92), ("Eye L Squint", 0.78), ("Eye R Squint", 0.78),
                              ("Anime Blush Lines", 0.98), ("Sparkle Eyes", 0.88)):
                _set(p, name, min(1.0, max(_get(p, name), amt * k)), 0.0)
            _set(p, "Ahoge Bounce", -0.95 * k, 0.0)
            _set(p, "Ahoge Sway", 0.85 * k * math.sin(t * 9.0), 0.0)

        if self._poke_energy > 0.005:
            self._poke_energy = max(0.0, self._poke_energy - dt / 1.15)
            kp = self._poke_energy
            if "Angle" in p:
                ax, ay = p["Angle"]
                p["Angle"] = (ax + 9.0 * kp * self._poke_side, ay + 6.5 * kp)
            _set(p, "Angle Z", _get(p, "Angle Z") + 8.0 * kp * self._poke_side, 0.0)
            _set(p, LEFT_BROW_Y, min(1.0, _get(p, LEFT_BROW_Y) + 0.75 * kp), 0.0)
            _set(p, RIGHT_BROW_Y, min(1.0, _get(p, RIGHT_BROW_Y) + 0.75 * kp), 0.0)
            _set(p, "Anime Blush Lines", min(1.0, max(_get(p, "Anime Blush Lines"), 0.72 * kp)), 0.0)
