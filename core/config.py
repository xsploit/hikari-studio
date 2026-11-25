"""Central Studio Configuration, Shader Presets, Expression Presets & Persistence.
Zero borrowed external assets: all default stage scenes are 100% original procedural
architectural/anime studio environments + custom user room uploads.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import threading
from pathlib import Path
from typing import Any

from core.customizer import DEFAULT_CUSTOMIZER_CFG, EYE_STYLE_PRESETS
from core.puppet_adapter import sanitize_model_cameras, valid_model_id
from modules.ai_companion import BRAIN_MODES, DEFAULT_AI_CFG, LEGACY_AI_KEYS, LEGACY_BRAIN_MODES, upgrade_persona
from modules.vercel_gateway import is_vision_model_id
from modules.fish_tts_stream import FISH_TRANSPORTS
from modules.memory_store import DEFAULT_MEMORY_CFG, LEGACY_MEMORY_KEYS
from modules.ambient_presence import DEFAULT_AMBIENT_CFG
from modules.lipsync import DEFAULT_LIPSYNC_CFG, DEFAULT_PARAM_MAP, DEFAULT_VISEMES, LIPSYNC_ENGINES, MFCC_SCORING

PARAM_MAP_FIELDS = ("inMin", "inMax", "outMin", "outMax", "smooth", "curve")
SECRET_MASK = "********"
# Every saved credential override. Masked in EVERY browser payload; never echoed raw.
SECRET_AI_FIELDS = ("apiKey", "fishApiKey", "vercelApiKey", "cloudflareAccountId", "cloudflareApiToken", "perplexityApiKey")
# Internal test / proxy hooks: never settable from a client patch.
INTERNAL_AI_FIELDS = ("jevEndpoint", "clefEndpoint", "pplxEndpoint", "lunaEndpoint")


def _id_list(value: Any, limit: int = 12) -> list[str]:
    import re
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value) if v]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for v in value:
        t = str(v).strip()
        if t and len(t) <= 96 and t not in out:
            out.append(t)
    return out[:limit]


def sanitize_ai_brain_patch(values: dict[str, Any]) -> dict[str, Any]:
    """Validate brain / voice / gateway fields of an `ai` patch; the masked secret is ignored."""
    out = dict(values)
    if "brainMode" in out:
        m = str(out["brainMode"] or "").lower()
        m = LEGACY_BRAIN_MODES.get(m, m)
        out["brainMode"] = m if m in BRAIN_MODES else "fish_duplex"
    if "fishTtsTransport" in out:
        t = str(out["fishTtsTransport"] or "").lower()
        out["fishTtsTransport"] = t if t in FISH_TRANSPORTS else "sse"
    if "fishLatency" in out and out["fishLatency"] not in ("normal", "balanced", "low"):
        out["fishLatency"] = "balanced"
    for k in ("vercelProviderOrder", "vercelFallbackModels"):
        if k in out:
            out[k] = _id_list(out[k], 6 if k == "vercelFallbackModels" else 12)
    if "vercelProviderOnly" in out:
        # True = restrict strictly to vercelProviderOrder (UI checkbox); a list/string = explicit providers
        only = out["vercelProviderOnly"]
        out["vercelProviderOnly"] = True if only is True else ([] if only in (False, None) else _id_list(only, 12))
    if "gatewayEvalModel" in out:
        from modules.decision_engine import DEFAULT_GATEWAY_EVAL_MODEL, is_eval_model_id
        gm = str(out["gatewayEvalModel"] or "").strip()
        out["gatewayEvalModel"] = gm if is_eval_model_id(gm) else DEFAULT_GATEWAY_EVAL_MODEL
    if "deciderRoute" in out:
        r = str(out["deciderRoute"] or "").lower()
        out["deciderRoute"] = r if r in ("auto", "gateway", "direct") else "auto"
    if "vercelHeaders" in out:
        out["vercelHeaders"] = sanitize_headers(out["vercelHeaders"])
    if "gatewayEvalModel" in out:
        out["gatewayModelSource"] = "gatewayEvalModel"          # the dropdown was the last explicit choice
    elif "decisionModel" in out:
        from modules.decision_engine import is_eval_model_id as _is_eval
        if _is_eval(str(out.get("decisionModel") or "").strip()):
            out["gatewayModelSource"] = "decisionModel"
    if "gatewayModelSource" in out and out["gatewayModelSource"] not in ("gatewayEvalModel", "decisionModel"):
        out["gatewayModelSource"] = "gatewayEvalModel"
    for k in LEGACY_AI_KEYS:
        out.pop(k, None)                  # the local 4B brain is gone
    for k in ("vercelModel", "vercelEmbeddingModel",
              "clefModel", "pplxModel", "jevModel", "lunaModel", "decisionModel"):
        if k in out:
            out[k] = str(out[k] or "").strip()[:300]
    clear = out.pop("clearKeys", None)
    clear = {k for k in clear if k in SECRET_AI_FIELDS} if isinstance(clear, (list, tuple)) else set()
    for k in SECRET_AI_FIELDS:
        if k in clear:
            out[k] = ""                   # explicit clear
        elif k in out:
            v = out[k]
            if v is None:
                out[k] = ""               # explicit clear (JSON null)
            elif not isinstance(v, str) or v.strip() in ("", SECRET_MASK):
                out.pop(k)                # empty / echoed mask: keep the stored key
            else:
                out[k] = v.strip()
    for k in INTERNAL_AI_FIELDS:
        out.pop(k, None)
    return out


_HEADER_NAME_RE = __import__("re").compile(r"^[a-z0-9-]{1,64}$")


def sanitize_headers(value: Any) -> dict[str, str]:
    """Extra gateway headers: lowercase token names, no CR / LF / NUL anywhere (header injection), <= 8."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        name = str(k).strip().lower()
        val = str(v) if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""
        if not _HEADER_NAME_RE.match(name) or any(c in name + val for c in "\r\n\x00"):
            continue
        out[name] = val.strip()[:256]
        if len(out) >= 8:
            break
    return out


# ----------------------------------------------------------------------------- frame-safe section patches
# rig / camera / shader / stage / visemes feed the 60 FPS loop. Every value is type-checked against the default
# config; numbers must be finite and are clamped, so a bad patch (NaN, Inf, "abc", huge values) can never break
# a frame or be persisted by "Save".
import math as _math

_NUM_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("camera", "scale"): (0.2, 6.0), ("camera", "x"): (-4000.0, 4000.0), ("camera", "y"): (-4000.0, 4000.0),
    ("stage", "avatarScale"): (0.2, 4.0), ("stage", "avatarX"): (-2.0, 3.0), ("stage", "avatarY"): (-2.0, 2.0),
    ("stage", "roomBlur"): (0.0, 40.0), ("stage", "roomDim"): (0.0, 1.0), ("stage", "parallax"): (0.0, 4.0),
    ("shader", "keyAngle"): (-720.0, 720.0),
    ("rig", "jiggleScale"): (0.0, 2.5), ("rig", "emissiveGlow"): (0.0, 2.0), ("rig", "chibiMode"): (0.0, 1.0),
    ("rig", "gloom"): (0.0, 1.0), ("rig", "blinkRate"): (0.05, 6.0), ("rig", "breathAmp"): (0.0, 3.0),
    ("rig", "idleSway"): (0.0, 3.0), ("rig", "idleClipMix"): (0.0, 2.0), ("rig", "mouseLookStrength"): (0.0, 2.0), ("rig", "autoGazeLiveliness"): (0.0, 3.0),
    ("rig", "idlePhraseSec"): (3.0, 6.0), ("rig", "prosodyGain"): (0.0, 2.5), ("rig", "backchannelGain"): (0.0, 2.5),
}
_SECTION_DEFAULT_LIMIT: dict[str, tuple[float, float]] = {
    "shader": (-10.0, 10.0), "rig": (-100.0, 100.0), "camera": (-4000.0, 4000.0), "stage": (-100.0, 100.0),
}
_COLOR_KEYS = {"keyColor", "fillColor", "tint", "ambientColor"}
_HEX_RE = __import__("re").compile(r"^#[0-9a-fA-F]{6}$")


def _finite_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            return None
    else:
        return None
    return f if _math.isfinite(f) else None


def _scalar(sec: str, key: str, v: Any) -> tuple[bool, Any]:
    """Accept an unknown key's value only if it is a safe scalar (finite number / bool / short string)."""
    if isinstance(v, bool):
        return True, v
    if isinstance(v, (int, float)):
        f = _finite_number(v)
        if f is None:
            return False, None
        lo, hi = _SECTION_DEFAULT_LIMIT.get(sec, (-1e4, 1e4))
        return True, max(lo, min(hi, f))
    if isinstance(v, str) and len(v) <= 500:
        return True, v
    if v is None:
        return True, None
    return False, None


def sanitize_overrides(value: Any) -> dict[str, Any]:
    """rig.overrides {param: [x, y] | null}: finite numbers only, clamped, <= 128 params."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for name, ov in list(value.items())[:128]:
        if not isinstance(name, str) or not name or len(name) > 64:
            continue
        if ov is None:
            out[name] = None
            continue
        if isinstance(ov, (list, tuple)) and len(ov) == 2:
            a, b = _finite_number(ov[0]), _finite_number(ov[1])
            if a is not None and b is not None:
                out[name] = [max(-1000.0, min(1000.0, a)), max(-1000.0, min(1000.0, b))]
    return out


def sanitize_section(sec: str, values: Any, template: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a rig / camera / shader / stage / visemes patch against the default config's types."""
    if not isinstance(values, dict):
        return {}
    if sec == "visemes":
        out_v: dict[str, Any] = {}
        for code, axes in values.items():
            if isinstance(code, str) and len(code) <= 4 and isinstance(axes, dict):
                clean = {}
                for ax, v in axes.items():
                    if ax == "label" and isinstance(v, str):
                        clean[ax] = v[:60]
                        continue
                    f = _finite_number(v)
                    if f is not None and isinstance(ax, str) and len(ax) <= 16:
                        clean[ax] = max(-2.0, min(2.0, f))
                if clean:
                    out_v[code] = clean
        return out_v
    tpl = template if template is not None else (default_config().get(sec) or {})
    out: dict[str, Any] = {}
    for k, v in values.items():
        if not isinstance(k, str) or not k or len(k) > 64:
            continue
        if sec == "rig" and k == "overrides":
            out[k] = sanitize_overrides(v)
            continue
        if sec == "stage" and k == "savedModelCameras":
            out[k] = sanitize_model_cameras(v)
            continue
        if sec == "stage" and k == "activeModel":
            if isinstance(v, str) and valid_model_id(v):
                out[k] = v
            continue
        if k not in tpl:
            if isinstance(v, dict):                       # forward-compat: new nested settings, scalars only
                sub = {str(a): _scalar(sec, str(a), b) for a, b in list(v.items())[:64] if isinstance(a, str) and len(a) <= 64}
                out[k] = {a: r[1] for a, r in sub.items() if r[0]}
                continue
            ok, val = _scalar(sec, k, v)
            if ok:
                out[k] = val
            continue
        d = tpl[k]
        if isinstance(d, bool):
            f = _finite_number(v) if not isinstance(v, bool) else None
            if isinstance(v, bool):
                out[k] = v
            elif f is not None:
                out[k] = bool(f)
        elif isinstance(d, (int, float)):
            f = _finite_number(v)
            if f is None:
                continue                                   # NaN / Inf / "abc" / null -> dropped
            lo, hi = _NUM_LIMITS.get((sec, k), _SECTION_DEFAULT_LIMIT.get(sec, (-1e4, 1e4)))
            out[k] = max(lo, min(hi, f))
        elif isinstance(d, str):
            if not isinstance(v, str):
                continue
            if k in _COLOR_KEYS and not _HEX_RE.match(v):
                continue
            if sec == "rig" and k == "expression" and v not in EXPRESSION_PRESETS:
                continue
            out[k] = v[:500]
        elif isinstance(d, dict):
            if isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in ((str(a), _scalar(sec, str(a), b)) for a, b in v.items()) if vv[0]}
                out[k] = {kk: vv[1] for kk, vv in out[k].items()}
        else:
            ok, val = _scalar(sec, k, v)
            if ok:
                out[k] = val
    return out


FRAME_SECTIONS = ("stage", "camera", "shader", "rig", "visemes")

# Screen Vision. model "" = Auto (openai/gpt-5-mini Vision Eye); providerOrder = optional comma-separated upstream pin
# for the vision model; mode "describe" = Vision Eye summarizes the screen into the turn context so the main chat
# model (even a text-only one such as DeepSeek) replies in character; "direct" = the vision model answers the turn.
VISION_MODES = ("describe", "direct")
DEFAULT_VISION_CFG: dict[str, Any] = {
    "enabled": True,
    "model": "",
    "providerOrder": "",
    "mode": "describe",
    "intervalSec": 10,
    "source": "desktop",
    "sourceLabel": "Primary Monitor",
}
_VISION_PROVIDER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_VISION_SOURCE_RE = re.compile(r"^(?:desktop|browser|monitor:[1-8]|window:[0-9]{1,18})$")


def sanitize_vision_patch(values: Any) -> dict[str, Any]:
    """`vision` section: bool enabled, bounded routed model id ("" = Auto), comma-separated provider ids,
    mode in VISION_MODES (anything else -> "describe"), intervalSec (3..120), source, sourceLabel."""
    if not isinstance(values, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(values.get("enabled"), bool):
        out["enabled"] = values["enabled"]
    m = values.get("model")
    if isinstance(m, str):                              # non-strings are dropped; "" = Auto
        m = m.strip()
        if not m or is_vision_model_id(m):
            out["model"] = m
    if "providerOrder" in values:
        raw = values.get("providerOrder")
        items = raw if isinstance(raw, (list, tuple)) else re.split(r"[,\s]+", raw if isinstance(raw, str) else "")
        provs: list[str] = []
        for p in items:
            if not isinstance(p, str):
                continue
            p = p.strip().lower()
            if p and _VISION_PROVIDER_RE.fullmatch(p) and p not in provs:
                provs.append(p)
                if len(provs) == 8:
                    break
        out["providerOrder"] = ",".join(provs)
    if "mode" in values:
        mode = values.get("mode")
        mode = mode.strip().lower() if isinstance(mode, str) else ""
        out["mode"] = mode if mode in VISION_MODES else "describe"
    if "intervalSec" in values and not isinstance(values.get("intervalSec"), bool):
        try:
            iv = int(round(float(values["intervalSec"])))
            out["intervalSec"] = max(3, min(120, iv))
        except (TypeError, ValueError, OverflowError):
            pass
    if "source" in values and isinstance(values.get("source"), str):
        src = values["source"].strip()
        if _VISION_SOURCE_RE.fullmatch(src):
            out["source"] = src
    if "sourceLabel" in values and isinstance(values.get("sourceLabel"), str):
        lbl = " ".join(values["sourceLabel"].split())[:80]
        if lbl:
            out["sourceLabel"] = lbl
    return out


def sanitize_ambient_patch(values: Any) -> dict[str, Any]:
    """`ambient` section (off by default): bool toggles + clamped minIntervalSec."""
    if not isinstance(values, dict):
        return {}
    out: dict[str, Any] = {}
    for k in ("enabled", "windowReactions", "proactiveCheckIn"):
        if isinstance(values.get(k), bool):
            out[k] = values[k]
    if "minIntervalSec" in values:
        try:
            out["minIntervalSec"] = max(60, min(3600, int(values["minIntervalSec"])))
        except (TypeError, ValueError):
            pass
    return out


def sanitize_memory_patch(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "enabled" in values:
        out["enabled"] = bool(values["enabled"])
    if "embeddingProvider" in values:
        out["embeddingProvider"] = "local" if values["embeddingProvider"] in ("local", "local_hash") else "vercel"
    for k in ("persistHistory", "autoExtract"):
        if k in values:
            out[k] = bool(values[k])
    if "historyTurns" in values:
        try:
            out["historyTurns"] = max(2, min(200, int(values["historyTurns"])))
        except (TypeError, ValueError):
            out["historyTurns"] = DEFAULT_MEMORY_CFG["historyTurns"]
    if "triage" in values:
        t = str(values["triage"] or "").lower()
        out["triage"] = "off" if t == "off" else "auto"
    if "triageModel" in values:
        out["triageModel"] = str(values["triageModel"] or "").strip()[:200]
    if "recallTopK" in values:
        try:
            out["recallTopK"] = max(1, min(12, int(values["recallTopK"])))
        except (TypeError, ValueError):
            out["recallTopK"] = DEFAULT_MEMORY_CFG["recallTopK"]
    return out


def migrate_legacy(data: dict[str, Any]) -> bool:
    """SLEEP-era settings -> current ones, in place. True when something changed (then saved once):
    brainMode local_4b_fish -> vercel_fish, local 4B keys and autoSleepIdleMin / colabBaseModel dropped."""
    changed = False
    ai = data.get("ai")
    if isinstance(ai, dict):
        m = str(ai.get("brainMode") or "").lower()
        if m in LEGACY_BRAIN_MODES:
            ai["brainMode"] = LEGACY_BRAIN_MODES[m]
            changed = True
        for k in LEGACY_AI_KEYS:
            if k in ai:
                ai.pop(k)
                changed = True
    mem = data.get("memory")
    if isinstance(mem, dict):
        for k in LEGACY_MEMORY_KEYS:
            if k in mem:
                mem.pop(k)
                changed = True
        if mem.get("embeddingProvider") == "local_hash":
            mem["embeddingProvider"] = "local"
            changed = True
    elif mem is not None:
        data["memory"] = copy.deepcopy(DEFAULT_MEMORY_CFG)
        changed = True
    return changed


def mask_secrets(section: dict[str, Any]) -> dict[str, Any]:
    """Copy of an `ai`-like dict with every saved credential override masked (set -> mask, unset -> "")."""
    out = {k: v for k, v in section.items() if k not in INTERNAL_AI_FIELDS}
    for k in SECRET_AI_FIELDS:
        if k in out:
            out[k] = SECRET_MASK if out.get(k) else ""
    return out


def secrets_set(section: dict[str, Any]) -> dict[str, bool]:
    return {k: bool(section.get(k)) for k in SECRET_AI_FIELDS}


def public_config(data: dict[str, Any]) -> dict[str, Any]:
    """Config for the browser: every saved key override masked (server env keys are never in config)."""
    out = dict(data)
    out["ai"] = mask_secrets(data.get("ai") or {})
    if isinstance(out.get("customizer"), dict):
        out["customizer"] = {key: value for key, value in out["customizer"].items()
                             if key in DEFAULT_CUSTOMIZER_CFG and isinstance(value, (str, int, float, bool))}
    return out


_LIPSYNC_SCALAR_BOUNDS: dict[str, tuple[float, float]] = {
    "gain": (0.1, 6.0), "noiseGate": (0.0, 0.25), "attackOmega": (5.0, 80.0), "dampingZeta": (0.2, 2.5),
    "smileBias": (-1.0, 1.0), "roundness": (0.0, 2.5), "jawBounce": (0.0, 2.5),
    "lipSealSharpness": (0.0, 2.5), "microFlutter": (0.0, 1.5), "coarticulation": (0.08, 0.60),
    "speechToHeadNod": (0.0, 3.0), "speechToBrows": (0.0, 3.0), "speechToArmGesture": (0.0, 3.0),
}


def sanitize_lipsync_patch(values: dict[str, Any], *, keep_calibration: bool = False) -> dict[str, Any]:
    """Validate every lip-sync scalar, engine, scoring, and paramMap field so NaN/bad types never freeze frames."""
    if not isinstance(values, dict):
        return {}
    import math
    out: dict[str, Any] = {}
    for k, v in values.items():
        if k == "engine":
            e = str(v or "").lower()
            out[k] = e if e in LIPSYNC_ENGINES else "hybrid"
        elif k == "mfccScoring":
            s = str(v or "").lower()
            out[k] = s if s in MFCC_SCORING else "cosine"
        elif k in _LIPSYNC_SCALAR_BOUNDS:
            lo, hi = _LIPSYNC_SCALAR_BOUNDS[k]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv) or isinstance(v, bool):
                continue
            out[k] = max(lo, min(hi, fv))
        elif k == "micEnabled":
            out[k] = bool(v)
        elif k == "holdViseme":
            hv = str(v or "").strip().upper()
            out[k] = hv if hv in ("", "X", "A", "B", "C", "D", "E", "F", "G", "H") else ""
        elif k == "mfccCalibration" and keep_calibration and isinstance(v, dict):
            out[k] = v
        elif k == "paramMap":
            pm_in = v if isinstance(v, dict) else {}
            pm: dict[str, dict[str, Any]] = {}
            for ch, vals in pm_in.items():
                if (ch not in DEFAULT_PARAM_MAP and ch != "mouthSpring") or not isinstance(vals, dict):
                    continue
                clean: dict[str, Any] = {}
                if "enabled" in vals:
                    clean["enabled"] = bool(vals["enabled"])
                for f in PARAM_MAP_FIELDS:
                    if f in vals:
                        try:
                            fv = float(vals[f])
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(fv) or isinstance(vals[f], bool):
                            continue
                        if f == "smooth":
                            fv = max(0.0, min(0.98, fv))
                        elif f == "curve":
                            fv = max(0.2, min(5.0, fv))
                        else:
                            fv = max(-2.0, min(2.0, fv))
                        clean[f] = fv
                pm[ch] = clean
            out[k] = pm
    return out


SHADER_PRESETS: dict[str, dict[str, Any]] = {
    "hikari_live_hd_v2": {
        "label": "Hikari Live HD v2 (Flagship)",
        "enabled": True, "sharpness": 0.35, "rimStrength": 0.72, "rimWidth": 1.00,
        "keyColor": "#ffb8e0", "fillColor": "#8ce0ff", "keyAngle": 138.0,
        "bloomStrength": 0.20, "bloomThresh": 0.94, "vibrance": 1.08,
        "contrast": 1.04, "brightness": 1.01, "warmth": 0.08, "tint": "#fffbfc",
        "ambientColor": "#f4b8da", "ambientWrap": 0.24, "chromaAb": 0.0,
    },
    "studio_pop": {
        "label": "Studio Broadcast Pop",
        "enabled": True, "sharpness": 0.75, "rimStrength": 0.95, "rimWidth": 1.0,
        "keyColor": "#ffb8e0", "fillColor": "#8ce0ff", "keyAngle": 135.0,
        "bloomStrength": 0.65, "bloomThresh": 0.90, "vibrance": 1.06,
        "contrast": 1.04, "brightness": 1.01, "warmth": 0.04, "tint": "#ffffff",
        "ambientColor": "#f4b8da", "ambientWrap": 0.22, "chromaAb": 0.0,
    },
    "twilight_atelier": {
        "label": "Twilight Atelier Warm",
        "enabled": True, "sharpness": 0.95, "rimStrength": 1.00, "rimWidth": 1.00,
        "keyColor": "#ffbe7a", "fillColor": "#a892ff", "keyAngle": 148.0,
        "bloomStrength": 0.24, "bloomThresh": 0.94, "vibrance": 1.06,
        "contrast": 1.05, "brightness": 1.01, "warmth": 0.14, "tint": "#fffaf6",
        "ambientColor": "#ffb470", "ambientWrap": 0.30, "chromaAb": 0.0,
    },
    "midnight_neon": {
        "label": "Cyber Neon Balcony",
        "enabled": True, "sharpness": 0.85, "rimStrength": 1.35, "rimWidth": 1.25,
        "keyColor": "#ff5cc8", "fillColor": "#3ae8ff", "keyAngle": 122.0,
        "bloomStrength": 0.95, "bloomThresh": 0.84, "vibrance": 1.14,
        "contrast": 1.10, "brightness": 0.98, "warmth": -0.14, "tint": "#f5f4ff",
        "ambientColor": "#8258ff", "ambientWrap": 0.48, "chromaAb": 0.22,
    },
    "sakura_dawn": {
        "label": "Sakura Pastel Dream",
        "enabled": True, "sharpness": 0.60, "rimStrength": 0.88, "rimWidth": 1.05,
        "keyColor": "#ffd1e6", "fillColor": "#fff0d6", "keyAngle": 115.0,
        "bloomStrength": 0.85, "bloomThresh": 0.87, "vibrance": 1.05,
        "contrast": 1.00, "brightness": 1.04, "warmth": 0.14, "tint": "#fff8fb",
        "ambientColor": "#ffc2dc", "ambientWrap": 0.34, "chromaAb": 0.05,
    },
    "crisp_cel": {
        "label": "Ultra-Sharp Anime Cel",
        "enabled": True, "sharpness": 1.18, "rimStrength": 0.60, "rimWidth": 0.80,
        "keyColor": "#ffffff", "fillColor": "#c4e4ff", "keyAngle": 135.0,
        "bloomStrength": 0.28, "bloomThresh": 0.93, "vibrance": 1.10,
        "contrast": 1.08, "brightness": 1.02, "warmth": 0.0, "tint": "#ffffff",
        "ambientColor": "#ffffff", "ambientWrap": 0.08, "chromaAb": 0.0,
    },
    "raw_unshaded": {
        "label": "Raw Unshaded Rig",
        "enabled": False, "sharpness": 0.0, "rimStrength": 0.0, "rimWidth": 1.0,
        "keyColor": "#ffffff", "fillColor": "#ffffff", "keyAngle": 135.0,
        "bloomStrength": 0.0, "bloomThresh": 0.95, "vibrance": 1.0,
        "contrast": 1.0, "brightness": 1.0, "warmth": 0.0, "tint": "#ffffff",
        "ambientColor": "#ffffff", "ambientWrap": 0.0, "chromaAb": 0.0,
    },
}

ROOM_AMBIENT_MAP: dict[str, dict[str, Any]] = {
    "twilight_atelier": {"ambientColor": "#ffb470", "keyColor": "#ffbe7a", "fillColor": "#9d8cff"},
    "midnight_loft":    {"ambientColor": "#7f5cff", "keyColor": "#ff5cc8", "fillColor": "#3ae8ff"},
    "sakura_pavilion":  {"ambientColor": "#ffb8d8", "keyColor": "#ffd1e6", "fillColor": "#b8d6ff"},
    "obsidian_studio":  {"ambientColor": "#f4b8da", "keyColor": "#ffb8e0", "fillColor": "#8ce0ff"},
    "chroma_green":     {"ambientColor": "#ffffff", "keyColor": "#ffffff", "fillColor": "#d6ecff"},
    "transparent":      {"ambientColor": "#ffffff", "keyColor": "#ffb8e0", "fillColor": "#8ce0ff"},
}

EXPRESSION_PRESETS: dict[str, dict[str, float]] = {
    "neutral":   {"eyeSmile": 0.0, "squint": 0.0, "browY": 0.0,  "browForm":  0.0,  "mouthForm":  0.08, "cheekPuff": 0.0, "sparkle": 0.0,  "blush": 0.0},
    "happy":     {"eyeSmile": 0.40,"squint": 0.20,"browY": 0.25, "browForm":  0.40, "mouthForm":  0.60, "cheekPuff": 0.0, "sparkle": 0.00, "blush": 0.20},
    "excited":   {"eyeSmile": 0.85,"squint": 0.50,"browY": 0.55, "browForm":  0.65, "mouthForm":  0.80, "cheekPuff": 0.0, "sparkle": 0.95, "blush": 0.55},
    "shy":       {"eyeSmile": 0.25,"squint": 0.20,"browY": -0.15,"browForm": -0.35, "mouthForm":  0.20, "cheekPuff": 0.15,"sparkle": 0.10, "blush": 0.92},
    "smug":      {"eyeSmile": 0.20,"squint": 0.55,"browY": 0.15, "browForm":  0.20, "mouthForm":  0.55, "cheekPuff": 0.0, "sparkle": 0.15, "blush": 0.10},
    "pout":      {"eyeSmile": 0.0, "squint": 0.25,"browY": -0.45,"browForm": -0.75, "mouthForm": -0.55, "cheekPuff": 0.92,"sparkle": 0.0,  "blush": 0.45},
    "surprised": {"eyeSmile": 0.0, "squint": 0.0, "browY": 0.85, "browForm":  0.30, "mouthForm": -0.35, "cheekPuff": 0.0, "sparkle": 0.60, "blush": 0.15},
    "sleepy":    {"eyeSmile": 0.15,"squint": 0.82,"browY": -0.30,"browForm": -0.25, "mouthForm": -0.10, "cheekPuff": 0.0, "sparkle": 0.0,  "blush": 0.15},
    "curious":   {"eyeSmile": 0.10,"squint": 0.05,"browY": 0.50, "browForm":  0.15, "mouthForm":  0.25, "cheekPuff": 0.0, "sparkle": 0.30, "blush": 0.10},
    "flustered": {"eyeSmile": 0.35,"squint": 0.45,"browY": 0.20, "browForm": -0.45, "mouthForm":  0.10, "cheekPuff": 0.30,"sparkle": 0.05, "blush": 1.00},
    "sad":       {"eyeSmile": 0.0, "squint": 0.30,"browY": -0.20,"browForm": -0.80, "mouthForm": -0.45, "cheekPuff": 0.0, "sparkle": 0.0,  "blush": 0.05},
}


def default_config() -> dict[str, Any]:
    return {
        "stage": {
            "room": "twilight_atelier",   # procedural original rooms: twilight_atelier | midnight_loft | sakura_pavilion | obsidian_studio | chroma_green | transparent | custom
            "customRoomUrl": "",
            "roomBlur": 1.5,
            "roomDim": 0.14,
            "parallax": 0.55,
            "particles": "auto",          # auto | embers | rain | sakura | dust | none
            "autoMatchLighting": True,
            "avatarX": 0.50,
            "avatarY": 0.00,
            "avatarScale": 1.00,
            "activeModel": "hikari",      # "hikari" (customizable) or a models/library/<id>.inp / .inx puppet
            "savedModelCameras": {},      # {model_id: {scale, x, y}}: each puppet's own framing, restored on switch
        },
        "camera": {
            "scale": 1.82,
            "x": 0.0,
            "y": 355.0,
        },
        "shader": copy.deepcopy(SHADER_PRESETS["hikari_live_hd_v2"]),
        "rig": {
            "physicsEnabled": True,
            "jiggleScale": 1.25,
            "emissiveGlow": 1.0,
            "headPatEnabled": True,
            "chibiMode": 0.0,
            "gloom": 0.0,
            "autoBlink": True,
            "blinkRate": 1.0,
            "autoBreath": True,
            "breathAmp": 1.0,
            "idleSway": 1.0,
            "idleClip": "",              # Motion Lab clip id looped as her idle (whole-body sway / torso / drift)
            "idleClipMix": 1.0,
            "idleMotionGraph": True,     # walk the idle clip as a pose-matched motion graph (False = classic linear loop)
            "idlePhraseSec": 4.5,        # target motion-phrase length for the idle motion graph (3..6 s)
            "causalProsody": True,       # stressed syllables -> micro-nods, rising pitch -> brow lift, emphasis -> lean-in
            "prosodyGain": 1.0,
            "visualBackchannel": True,   # listening: attentive nods / brow / lean-in on user micro-pauses
            "backchannelGain": 1.0,
            "autoGaze": True,
            "autoGazeLiveliness": 1.0,
            "mouseLookAt": True,
            "mouseLookStrength": 0.72,
            "expression": "happy",
            "armRPose": 0.0,
            "armLPose": 0.0,
            "armRGesture": 0.0,
            "armLGesture": 0.0,
            "armRLift": 0.0,
            "armLLift": 0.0,
            "autoGestureOnSpeech": True,
            "sparkleEyes": 0.0,
            "blushLines": 0.15,
            "cheekPuff": 0.0,
            "tongueOut": 0.0,
            "eyeSquint": 0.0,
            "autoVfx": True,              # emotion-driven anime VFX + micro-expressions (sparkles, hearts, :3 ...)
            "overrides": {},
        },
        "lipsync": copy.deepcopy(DEFAULT_LIPSYNC_CFG),
        "visemes": copy.deepcopy(DEFAULT_VISEMES),
        "ai": {**copy.deepcopy(DEFAULT_AI_CFG), "fishVocalTags": DEFAULT_AI_CFG.get("fishVocalTags", True)},
        "vision": copy.deepcopy(DEFAULT_VISION_CFG),
        "ambient": copy.deepcopy(DEFAULT_AMBIENT_CFG),
        "customizer": copy.deepcopy(DEFAULT_CUSTOMIZER_CFG),
        "memory": copy.deepcopy(DEFAULT_MEMORY_CFG),
        "plugins": {},
        "userPresets": {},
    }


def _deep_merge(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._save_lock = threading.Lock()
        self.data: dict[str, Any] = default_config()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    for sec in FRAME_SECTIONS:                # a bad value already on disk can't break frames
                        if isinstance(saved.get(sec), dict):
                            saved[sec] = sanitize_section(sec, saved[sec], self.data.get(sec) if sec != "visemes" else None)
                    if "lipsync" in saved:
                        saved["lipsync"] = sanitize_lipsync_patch(saved["lipsync"], keep_calibration=True)
                    if "vision" in saved:              # bounded model / provider ids, mode enum
                        saved["vision"] = sanitize_vision_patch(saved["vision"])
                    if "ambient" in saved:
                        saved["ambient"] = sanitize_ambient_patch(saved["ambient"])
                    _deep_merge(self.data, saved)
            except Exception:
                pass
        migrated = migrate_legacy(self.data)
        # Upgrade a saved legacy generic persona to Hikari-chan's verbatim personality
        # (custom user personas are left untouched) and persist it once.
        upgraded = isinstance(self.data.get("ai"), dict) and upgrade_persona(self.data["ai"])
        if (migrated or upgraded) and self.path.exists():
            try:
                self.save()
            except Exception:
                pass

    def _write(self, text: str) -> None:
        with self._save_lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.path)

    def save(self) -> None:
        self._write(json.dumps(self.data, indent=2))

    async def save_async(self) -> None:
        """Snapshot on the loop (consistent, ~1 ms), write on a worker thread (no disk I/O on the loop)."""
        text = json.dumps(self.data, indent=2)
        await asyncio.to_thread(self._write, text)

    def patch(self, section: str, values: dict[str, Any]) -> dict[str, Any]:
        if section not in self.data:
            self.data[section] = {}
        if isinstance(self.data[section], dict) and isinstance(values, dict):
            _deep_merge(self.data[section], values)
        else:
            self.data[section] = copy.deepcopy(values)
        return self.data[section]
