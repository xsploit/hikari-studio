"""Plugin: Interactive Head-Pat Physics
Click or rub the top of Hikari's head on the live stage to trigger a cute head-pat reaction:
happy squint, rosy blush lines, gentle head nuzzle, and sparkle burst!
"""
from __future__ import annotations

import math
import time

from core.plugins import StudioPlugin


class HeadPatPlugin(StudioPlugin):
    id = "head_pat"
    name = "Interactive Head-Pat Reaction"
    version = "1.0.0"
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

    def register(self, hooks, ctx) -> None:
        hooks.register(self.id, "on_stage_interaction", self._on_interact, priority=40)
        hooks.register(self.id, "on_frame_params", self._on_frame, priority=60)

    def _on_interact(self, ctx, event: dict) -> None:
        rig_cfg = ctx.config.data.get("rig", {}) if hasattr(ctx, "config") else {}
        if not rig_cfg.get("headPatEnabled", True):
            return
        act = str(event.get("action") or event.get("kind") or "").lower()
        if act in ("head_pat", "pat"):
            spd = min(2.0, max(0.35, float(event.get("speed", 0.8))))
            self._pat_energy = min(1.0, self._pat_energy + 0.32 * spd)
            self._last_pat = time.perf_counter()
            if hasattr(ctx, "rig"):
                ctx.rig.sp_chest_y.v -= 4.2 * spd
                ctx.rig.sp_eye_jelly.v += 5.5
        elif act == "poke":
            self._poke_energy = min(1.0, self._poke_energy + 0.85)
            self._poke_side = -1.0 if float(event.get("x", 0.0)) >= 0.0 else 1.0
            if hasattr(ctx, "rig"):
                ctx.rig.sp_chest_y.v += 9.5
                ctx.rig.sp_chest_x.v += 8.0 * self._poke_side
                ctx.rig.sp_eye_jelly.v += 11.0

    def _on_frame(self, ctx, frame: dict) -> None:
        dt = float(frame.get("dt", 1 / 60))
        t = float(frame.get("time", 0.0))
        p = frame["params"]

        if self._pat_energy > 0.005:
            decay = max(0.5, float(self.settings.get("decaySec", 2.6)))
            self._pat_energy = max(0.0, self._pat_energy - dt / decay)
            k = self._pat_energy * float(self.settings.get("intensity", 1.0))
            ax, ay = p.get("Angle", (0.0, 0.0))
            p["Angle"] = (ax + 5.0 * k * math.sin(t * 7.2), ay - 5.2 * k)
            p["Angle Z"] = (p.get("Angle Z", (0.0, 0.0))[0] + 7.5 * k * math.sin(t * 5.4), 0.0)
            p["Body Angle Y"] = (p.get("Body Angle Y", (0.0, 0.0))[0] - 2.6 * k, 0.0)
            p["Eye L Smile"] = (min(1.0, max(p.get("Eye L Smile", (0.0, 0.0))[0], 0.92 * k)), 0.0)
            p["Eye R Smile"] = (min(1.0, max(p.get("Eye R Smile", (0.0, 0.0))[0], 0.92 * k)), 0.0)
            p["Eye L Squint"] = (min(1.0, max(p.get("Eye L Squint", (0.0, 0.0))[0], 0.78 * k)), 0.0)
            p["Eye R Squint"] = (min(1.0, max(p.get("Eye R Squint", (0.0, 0.0))[0], 0.78 * k)), 0.0)
            p["Anime Blush Lines"] = (min(1.0, max(p.get("Anime Blush Lines", (0.0, 0.0))[0], 0.98 * k)), 0.0)
            p["Sparkle Eyes"] = (min(1.0, max(p.get("Sparkle Eyes", (0.0, 0.0))[0], 0.88 * k)), 0.0)
            if "Ahoge Bounce" in p:
                p["Ahoge Bounce"] = (-0.95 * k, 0.0)
            if "Ahoge Sway" in p:
                p["Ahoge Sway"] = (0.85 * k * math.sin(t * 9.0), 0.0)

        if self._poke_energy > 0.005:
            self._poke_energy = max(0.0, self._poke_energy - dt / 1.15)
            kp = self._poke_energy
            ax, ay = p.get("Angle", (0.0, 0.0))
            p["Angle"] = (ax + 9.0 * kp * self._poke_side, ay + 6.5 * kp)
            p["Angle Z"] = (p.get("Angle Z", (0.0, 0.0))[0] + 8.0 * kp * self._poke_side, 0.0)
            p["Brow L Y"] = (min(1.0, p.get("Brow L Y", (0.0, 0.0))[0] + 0.75 * kp), 0.0)
            p["Brow R Y"] = (min(1.0, p.get("Brow R Y", (0.0, 0.0))[0] + 0.75 * kp), 0.0)
            p["Anime Blush Lines"] = (min(1.0, max(p.get("Anime Blush Lines", (0.0, 0.0))[0], 0.72 * kp)), 0.0)
