"""Auto-discovering Plugin Manager for Hikari Studio.
Drop any .py file defining a StudioPlugin subclass into plugins/ and it registers cleanly.
Supports live enable/disable, per-plugin configurable settings, and hook introspection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .events import HookRegistry


class StudioPlugin:
    """Base class for all Hikari Studio plugins."""
    id: str = "unnamed"
    name: str = "Unnamed Plugin"
    version: str = "1.0.0"
    description: str = ""
    default_enabled: bool = True
    default_settings: dict[str, Any] = {}

    def __init__(self) -> None:
        self.enabled: bool = bool(self.default_enabled)
        self.settings: dict[str, Any] = dict(self.default_settings)

    def register(self, hooks: HookRegistry, ctx: Any) -> None:
        """Override to bind hooks via hooks.register(self.id, '<hook>', self.handler)."""
        raise NotImplementedError

    def on_settings_changed(self, patch: dict[str, Any]) -> None:
        self.settings.update(patch)


class PluginManager:
    def __init__(self, plugins_dir: Path, hooks: HookRegistry) -> None:
        self.plugins_dir = plugins_dir
        self.hooks = hooks
        self.plugins: dict[str, StudioPlugin] = {}
        self.load_errors: dict[str, str] = {}

    def discover_and_load(self, ctx: Any, saved_states: dict[str, dict] | None = None) -> None:
        saved_states = saved_states or {}
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        for py_file in sorted(self.plugins_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            self._load_file(py_file, ctx, saved_states)

    def _load_file(self, py_file: Path, ctx: Any, saved_states: dict[str, dict]) -> None:
        mod_name = f"hikari_studio_plugin_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            if spec is None or spec.loader is None:
                return
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for attr in vars(mod).values():
                if isinstance(attr, type) and issubclass(attr, StudioPlugin) and attr is not StudioPlugin:
                    inst = attr()
                    st = saved_states.get(inst.id, {})
                    if "enabled" in st:
                        inst.enabled = bool(st["enabled"])
                    if isinstance(st.get("settings"), dict):
                        inst.settings.update(st["settings"])
                    inst.register(self.hooks, ctx)
                    self.hooks.set_plugin_enabled(inst.id, inst.enabled)
                    self.plugins[inst.id] = inst
        except Exception as exc:
            self.load_errors[py_file.name] = f"{exc.__class__.__name__}: {exc}"

    def set_enabled(self, plugin_id: str, enabled: bool) -> bool:
        p = self.plugins.get(plugin_id)
        if not p:
            return False
        p.enabled = bool(enabled)
        self.hooks.set_plugin_enabled(plugin_id, p.enabled)
        return True

    def update_plugin_settings(self, plugin_id: str, patch: dict[str, Any]) -> bool:
        p = self.plugins.get(plugin_id)
        if not p:
            return False
        p.on_settings_changed(patch)
        return True

    def serialize_states(self) -> dict[str, dict]:
        return {
            pid: {"enabled": p.enabled, "settings": dict(p.settings)}
            for pid, p in self.plugins.items()
        }

    def list_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.id,
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "enabled": p.enabled,
                "settings": dict(p.settings),
            }
            for p in self.plugins.values()
        ]
