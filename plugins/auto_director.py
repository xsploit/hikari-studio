"""Plugin: Auto Director
Adds subtle broadcast camera micro-zoom and dynamic eye-sparkle / rim glow pulse
whenever Hikari hits high vocal excitement during speech.
"""
from __future__ import annotations

from core.plugins import StudioPlugin


class AutoDirectorPlugin(StudioPlugin):
    id = "auto_director"
    name = "Auto Director & Reactive Lighting"
    version = "1.0.0"
    description = "Subtle anime broadcast camera push-in and rim-light pulse synchronized to vocal excitement."
    default_enabled = True
    default_settings = {
        "zoomBoost": 0.045,
        "rimPulse": 0.20,
        "sparkleBoost": 0.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self._smooth_e = 0.0

    def register(self, hooks, ctx) -> None:
        hooks.register(self.id, "on_frame_params", self._on_frame, priority=50)
        hooks.register(self.id, "on_shader_uniforms", self._on_shader, priority=50)

    def _on_frame(self, ctx, frame: dict) -> None:
        e = float(frame.get("energy", 0.0))
        self._smooth_e = self._smooth_e * 0.92 + e * 0.08
        frame["camera"]["scale"] = float(frame["camera"].get("scale", 1.62)) + self._smooth_e * float(self.settings.get("zoomBoost", 0.045))
        params = frame["params"]
        if "Sparkle Eyes" in params:
            cur = params["Sparkle Eyes"][0]
            params["Sparkle Eyes"] = (min(1.0, cur + self._smooth_e * float(self.settings.get("sparkleBoost", 0.25))), 0.0)

    def _on_shader(self, ctx, shader: dict) -> None:
        if shader.get("enabled", True):
            shader["rimStrength"] = float(shader.get("rimStrength", 0.95)) + self._smooth_e * float(self.settings.get("rimPulse", 0.22))
