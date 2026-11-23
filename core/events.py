"""Typed Hook & Event Registry for Hikari Studio plugins and external integrations.
Supports priority-ordered sync hooks (hot 60 FPS path) and async hooks (chat/actions),
with per-hook error isolation and microsecond execution metrics.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Callable


HOOK_NAMES = (
    "on_frame_params",        # (ctx, params: dict[str, tuple[float, float]]) -> None
    "on_shader_uniforms",     # (ctx, shader: dict[str, Any]) -> None
    "on_lipsync_frame",       # (ctx, axes: dict[str, float]) -> None
    "on_stage_interaction",   # (ctx, event: dict[str, Any]) -> None
    "on_user_message",        # (ctx, payload: dict[str, Any]) -> None
    "on_ai_reply",            # (ctx, payload: dict[str, Any]) -> None
    "on_external_trigger",    # (ctx, payload: dict[str, Any]) -> dict | None
)


@dataclass
class HookEntry:
    plugin_id: str
    hook_name: str
    callback: Callable[..., Any]
    priority: int = 100
    calls: int = 0
    last_ms: float = 0.0
    last_error: str = ""
    enabled: bool = True


class HookRegistry:
    def __init__(self) -> None:
        self._hooks: dict[str, list[HookEntry]] = {name: [] for name in HOOK_NAMES}

    def register(self, plugin_id: str, hook_name: str, callback: Callable[..., Any], priority: int = 100) -> HookEntry:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        entry = HookEntry(plugin_id=plugin_id, hook_name=hook_name, callback=callback, priority=priority)
        self._hooks[hook_name].append(entry)
        self._hooks[hook_name].sort(key=lambda e: e.priority)
        return entry

    def unregister_plugin(self, plugin_id: str) -> None:
        for lst in self._hooks.values():
            lst[:] = [e for e in lst if e.plugin_id != plugin_id]

    def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> None:
        for lst in self._hooks.values():
            for e in lst:
                if e.plugin_id == plugin_id:
                    e.enabled = enabled

    def emit_sync(self, hook_name: str, ctx: Any, data: Any) -> None:
        """Zero-allocation hot-path dispatcher for 60 FPS frame & shader hooks."""
        entries = self._hooks.get(hook_name)
        if not entries:
            return
        for entry in entries:
            if not entry.enabled:
                continue
            t0 = time.perf_counter()
            try:
                entry.callback(ctx, data)
                entry.last_error = ""
            except Exception as exc:
                entry.last_error = f"{exc.__class__.__name__}: {exc}"
            finally:
                entry.calls += 1
                entry.last_ms = (time.perf_counter() - t0) * 1000.0

    async def emit_async(self, hook_name: str, ctx: Any, data: Any) -> list[Any]:
        """Async dispatcher for chat, AI reply, and external webhook events."""
        results = []
        for entry in self._hooks.get(hook_name, ()):
            if not entry.enabled:
                continue
            t0 = time.perf_counter()
            try:
                res = entry.callback(ctx, data)
                if asyncio.iscoroutine(res):
                    res = await res
                if res is not None:
                    results.append(res)
                entry.last_error = ""
            except Exception as exc:
                entry.last_error = f"{exc.__class__.__name__}: {exc}"
            finally:
                entry.calls += 1
                entry.last_ms = (time.perf_counter() - t0) * 1000.0
        return results

    def stats(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for name, entries in self._hooks.items():
            if not entries:
                continue
            out[name] = [
                {
                    "plugin": e.plugin_id,
                    "priority": e.priority,
                    "enabled": e.enabled,
                    "calls": e.calls,
                    "lastMs": round(e.last_ms, 3),
                    "lastError": e.last_error,
                }
                for e in entries
            ]
        return out
