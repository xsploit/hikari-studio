"""Plugin: External Automation & Webhook Bridge
Exposes clean external triggers via POST /api/hook/trigger so Stream Deck, OBS scripts,
Twitch bots, or external agent workflows (Codex / Claude / T3 Code) can drive expressions,
arm waves, shader presets, or live speech lines!
"""
from __future__ import annotations

from core.config import SHADER_PRESETS
from core.plugins import StudioPlugin


class WebhookBridgePlugin(StudioPlugin):
    id = "webhook_bridge"
    name = "External Hook & Automation Bridge"
    version = "1.0.0"
    description = "Allows external tools (Stream Deck, OBS, T3 Code, bots) to trigger expressions, waves & looks via POST /api/hook/trigger."
    default_enabled = True
    default_settings = {
        "allowSpeechTrigger": True,
        "allowShaderTrigger": True,
    }

    def register(self, hooks, ctx) -> None:
        hooks.register(self.id, "on_external_trigger", self._on_trigger, priority=20)

    async def _on_trigger(self, ctx, payload: dict) -> dict:
        action = str(payload.get("action") or "").lower()
        if action == "wave":
            ctx.rig.trigger_temp_action({"expression": "happy", "armR": 1.0, "armL": 0.0, "sparkle": 0.7, "blush": 0.35}, duration=float(payload.get("duration", 3.5)))
            return {"ok": True, "triggered": "wave"}
        if action == "cheer":
            ctx.rig.trigger_temp_action({"expression": "excited", "armR": 1.0, "armL": 1.0, "sparkle": 1.0, "blush": 0.65}, duration=float(payload.get("duration", 4.0)))
            return {"ok": True, "triggered": "cheer"}
        if action == "expression":
            expr = str(payload.get("value", "happy"))
            ctx.config.patch("rig", {"expression": expr})
            return {"ok": True, "expression": expr}
        if action == "shader_preset" and self.settings.get("allowShaderTrigger", True):
            preset = str(payload.get("value", "studio_pop"))
            if preset in SHADER_PRESETS:
                ctx.config.patch("shader", SHADER_PRESETS[preset])
                return {"ok": True, "shaderPreset": preset}
        if action == "say" and self.settings.get("allowSpeechTrigger", True):
            text = str(payload.get("text") or "").strip()
            if text:
                await ctx.speak_text(text)
                return {"ok": True, "spoke": text}
        return {"ok": False, "error": f"Unknown action: {action}"}
