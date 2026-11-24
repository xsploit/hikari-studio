"""1-click Character Profiles (data/profiles.json).

A profile bundles everything that makes a character feel like *that* character, for any puppet (Hikari, Hiyori,
Mao, any models/library puppet): puppet, customizer look, Fish voice + model, personality, idle clip, camera.
apply_to_config writes ONLY those fields into a full studio config dict: API keys, provider URLs and every
other setting are never read, stored or overwritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any

from core.customizer import DEFAULT_CUSTOMIZER_CFG
from core.puppet_adapter import valid_model_id

MAX_PROFILES = 200
MAX_NAME = 60
MAX_PERSONA = 32_000
DEFAULT_PROFILE_ID = "hikari-default"
DEFAULT_PROFILE_NAME = "Hikari (Default)"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{0,64}$")
_LOOK_FIELDS = {*DEFAULT_CUSTOMIZER_CFG, "outfit"}
_ASSET_DIR = Path(__file__).resolve().parents[1] / "data" / "custom_assets"
_CLIP_RE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}")


def slugify(text: str, fallback: str = "profile") -> str:
    s = _SLUG_RE.sub("-", str(text or "").lower()).strip("-")[:48].strip("-")
    if not s and text and fallback:
        return f"{fallback}-{hashlib.sha256(str(text).encode('utf-8')).hexdigest()[:12]}"
    return s or fallback


def _now() -> float:
    return round(time.time(), 3)


def _num(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return default if not math.isfinite(f) else max(lo, min(hi, f))


def _clean_look(values: Any) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    look = {}
    for key in _LOOK_FIELDS.intersection(values):
        value = values[key]
        if key in ("customEyePath", "customFrontHairPath") and value:
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.parent != _ASSET_DIR or path.suffix.lower() not in (".jpg", ".png", ".webp"):
                continue
            try:
                if path.resolve().parent != _ASSET_DIR.resolve():
                    continue
            except (ValueError, OSError, RuntimeError):
                continue
        if isinstance(DEFAULT_CUSTOMIZER_CFG.get(key), float):
            look[key] = _num(value, DEFAULT_CUSTOMIZER_CFG[key], -4000.0, 4000.0)
        elif isinstance(value, str):
            look[key] = value[:4000]
    return look


def _sanitize(p: Any) -> dict[str, Any] | None:
    if not isinstance(p, dict):
        return None
    pid = slugify(p.get("id") or p.get("name") or "")
    name = (p["name"] if isinstance(p.get("name"), str) else pid).strip()[:MAX_NAME] or pid
    puppet = str(p.get("puppetId") or "hikari").strip()
    voice = str(p.get("fishReferenceId") or "").strip()
    clip = str(p.get("idleClip") or "").strip()
    cam = p.get("camera") if isinstance(p.get("camera"), dict) else {}
    camera = {k: _num(cam[k], 0.0, -4000.0, 4000.0) for k in ("x", "y") if k in cam}
    if "scale" in cam:
        camera["scale"] = _num(cam["scale"], 1.0, 0.2, 6.0)
    try:
        updated = float(p.get("updatedAt") or 0.0)
    except (TypeError, ValueError, OverflowError):
        updated = 0.0
    return {
        "id": pid,
        "name": name,
        "puppetId": puppet if valid_model_id(puppet) else "hikari",
        "customizer": _clean_look(p.get("customizer")),
        "fishReferenceId": voice if _VOICE_RE.match(voice) else "",
        "fishModel": (p["fishModel"] if isinstance(p.get("fishModel"), str) else "").strip()[:40] or "drama-3",
        "personality": p["personality"][:MAX_PERSONA] if isinstance(p.get("personality"), str) else "",
        "idleClip": clip if _CLIP_RE.fullmatch(clip) and ".." not in clip else "",
        "idleClipMix": _num(p.get("idleClipMix", 1.0), 1.0, 0.0, 2.0),
        "camera": camera,
        "updatedAt": updated if math.isfinite(updated) else 0.0,
    }


def _section(cfg: dict, name: str) -> dict:
    sec = cfg.get(name)
    return sec if isinstance(sec, dict) else {}


def profile_from_config(name: str, cfg: dict, *, profile_id: str = "", puppet_id: str = "") -> dict[str, Any]:
    """Snapshot the character-defining fields of a full studio config (never keys / URLs)."""
    ai, rig, cam = _section(cfg, "ai"), _section(cfg, "rig"), _section(cfg, "camera")
    prof = _sanitize({
        "id": profile_id or name,
        "name": name,
        "puppetId": puppet_id or _section(cfg, "stage").get("activeModel") or "hikari",
        "customizer": _section(cfg, "customizer"),
        "fishReferenceId": ai.get("fishReferenceId", ""),
        "fishModel": ai.get("fishModel", "drama-3"),
        "personality": ai.get("persona", ""),
        "idleClip": rig.get("idleClip", ""),
        "idleClipMix": rig.get("idleClipMix", 1.0),
        "camera": {k: cam[k] for k in ("scale", "x", "y") if k in cam},
        "updatedAt": _now(),
    })
    assert prof is not None
    return prof


def _default_profile() -> dict[str, Any]:
    try:
        from core.config import default_config
        cfg = default_config()
    except Exception:                         # standalone use: minimal Hikari defaults
        from modules.ai_companion import DEFAULT_AI_CFG
        cfg = {"ai": dict(DEFAULT_AI_CFG), "rig": {"idleClip": "", "idleClipMix": 1.0},
               "camera": {"scale": 1.82, "x": 0.0, "y": 355.0}, "customizer": {}}
    return profile_from_config(DEFAULT_PROFILE_NAME, cfg, profile_id=DEFAULT_PROFILE_ID, puppet_id="hikari")


class ProfileStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._profiles: dict[str, dict[str, Any]] = {}
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self) -> None:
        with self._lock:
            self._profiles = {}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                raw = None
            except ValueError:
                # Keep the original recoverable when seeding after corruption.
                self.path.replace(self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.corrupt"))
                raw = None
            items = raw.get("profiles") if isinstance(raw, dict) else raw
            for item in items if isinstance(items, list) else []:
                prof = _sanitize(item)
                if prof and len(self._profiles) < MAX_PROFILES:
                    self._profiles[prof["id"]] = prof
            if not self._profiles:
                d = _default_profile()
                self._profiles[d["id"]] = d
                self._save(self._profiles)

    def _save(self, profiles: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        data = {"version": 1, "profiles": list(profiles.values())}
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
            self._profiles = profiles
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ API
    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((copy.deepcopy(p) for p in self._profiles.values()),
                          key=lambda p: (p["id"] != DEFAULT_PROFILE_ID, p["name"].lower()))

    def get(self, profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            p = self._profiles.get(profile_id)
            return copy.deepcopy(p) if p else None

    def save_from_config(self, name: str, cfg: dict, *, profile_id: str = "", puppet_id: str = "") -> dict[str, Any]:
        """Create or update (same id / same name slug) a profile from the current studio config."""
        name = str(name or "").strip()[:MAX_NAME]
        if not name and not profile_id:
            raise ValueError("profile name required")
        prof = profile_from_config(name or profile_id, cfg if isinstance(cfg, dict) else {},
                                   profile_id=profile_id, puppet_id=puppet_id)
        with self._lock:
            if prof["id"] not in self._profiles and len(self._profiles) >= MAX_PROFILES:
                raise ValueError("too many profiles")
            self._save({**self._profiles, prof["id"]: prof})
        return copy.deepcopy(prof)

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            pid = profile_id
            if pid not in self._profiles:
                return False
            remaining = self._profiles.copy()
            del remaining[pid]
            if not remaining:                  # never leave the store empty
                d = _default_profile()
                remaining[d["id"]] = d
            self._save(remaining)
            return True

    def apply_to_config(self, profile_id: str, cfg: dict) -> dict[str, Any]:
        """Write the profile's character fields into `cfg` (a full studio config, updated in place) and return
        the profile. Keys, URLs, providers and all unrelated settings are left untouched."""
        prof = self.get(profile_id)
        if prof is None:
            raise KeyError(f"unknown profile: {profile_id}")
        stage = cfg.setdefault("stage", {})
        stage["activeModel"] = prof["puppetId"]
        cust = cfg.setdefault("customizer", {})
        for key in _LOOK_FIELDS:
            cust.pop(key, None)
        cust.update(prof["customizer"])
        ai = cfg.setdefault("ai", {})
        ai["fishReferenceId"] = prof["fishReferenceId"]
        ai["fishModel"] = prof["fishModel"]
        from modules.ai_companion import DEFAULT_PERSONA, HIKARI_PERSONALITY
        ai["persona"] = prof["personality"] if prof["personality"].strip() else DEFAULT_PERSONA
        ai["personaId"] = HIKARI_PERSONALITY["id"] if ai["persona"] == DEFAULT_PERSONA else "custom"
        rig = cfg.setdefault("rig", {})
        rig["idleClip"] = prof["idleClip"]
        rig["idleClipMix"] = prof["idleClipMix"]
        if prof["camera"]:
            cfg.setdefault("camera", {}).update(prof["camera"])
        return prof
