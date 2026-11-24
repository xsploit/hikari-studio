"""On-demand screen vision ("look at my screen").

Only explicit user cues trigger a capture (wants_screen_vision); the screenshot is downscaled, JPEG-encoded and
returned as a data URL for AICompanion.generate_reply(..., image_data_url=...). Nothing is stored on disk and the
image never enters chat history. Works for any character / persona.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any

_CUE_PATTERNS: tuple[str, ...] = (
    r"\b(?:look|looking|peek|glance)\s+(?:at|on)\s+(?:my|the|this|our)\s+(?:screen|monitor|display|desktop)\b",
    r"\bwhat'?s\s+(?:on|up\s+on)\s+(?:my|the)\s+(?:screen|monitor|display)\b",
    r"\bwhat\s+is\s+on\s+(?:my|the)\s+(?:screen|monitor|display)\b",
    r"\b(?:see|check|read|watch|view)\s+(?:my|the)\s+(?:screen|monitor|display|desktop)\b",
    r"\bcan\s+you\s+see\s+(?:my|the|this)\s+(?:screen|monitor|display|window)\b",
    r"\bwhat\s+am\s+i\s+(?:playing|looking\s+at|watching|doing\s+on\s+(?:my|the)\s+screen)\b",
    r"\bscreen\s*(?:shot|cap)\s+(?:my|the)\s+screen\b",
)
# Require a direct request. A quoted command, reported speech or a noun like
# "screen protector" does not authorize attaching the desktop to an LLM request.
_CUE_RE = re.compile(
    r"(?:(?:hey(?: hikari)?|hikari|please|can you|could you|would you)[, ]+)*"
    r"(?:" + "|".join(_CUE_PATTERNS) + r")(?:[, ]+please)?(?:[, ]+and\s+.+)?[?!. ]*",
    re.IGNORECASE,
)
# "don't look at my screen" / "no need to look at this"
_NEG_RE = re.compile(r"\b(?:don'?t|do\s+not|never|stop|no\s+need\s+to)\s+(?:\w+\s+){0,2}"
                     r"(?:look|looking|peek|glance|see|check|read|watch|view|screen\s*(?:shot|cap))\b", re.IGNORECASE)


def wants_screen_vision(text: str) -> bool:
    """True only for explicit cues such as "look at my screen", "what's on my screen", "what am I playing"."""
    t = re.sub(r"\s+", " ", str(text or "").replace("’", "'")).strip()
    if not t or not _CUE_RE.fullmatch(t):
        return False
    return not _NEG_RE.search(t)


def _grab_pil() -> Any:
    from PIL import ImageGrab  # lazy: Pillow is optional at import time
    return ImageGrab.grab()     # primary monitor


def _grab_mss() -> Any:
    import mss                  # optional fallback
    from PIL import Image
    with mss.mss() as sct:
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        shot = sct.grab(mon)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def capture_screen_jpeg(*, max_dim: int = 1024, quality: int = 76) -> dict[str, Any]:
    """Capture the primary monitor -> proportional downscale to <= max_dim -> JPEG data URL.
    {"ok": True, "width", "height", "mime": "image/jpeg", "dataUrl", "bytes"} or {"ok": False, "error"};
    never raises (headless session, locked desktop, missing Pillow...)."""
    try:
        max_dim = max(64, min(4096, int(max_dim)))
        quality = max(30, min(95, int(quality)))
    except (TypeError, ValueError, OverflowError):
        max_dim, quality = 1024, 76
    img, errors = None, []
    for label, grab in (("PIL.ImageGrab", _grab_pil), ("mss", _grab_mss)):
        try:
            img = grab()
            if img is not None and img.size[0] > 0 and img.size[1] > 0:
                break
            img = None
        except Exception as exc:          # ImportError, OSError (no display), ...
            errors.append(f"{label}: {exc.__class__.__name__}")
            img = None
    if img is None:
        return {"ok": False, "error": "screen capture unavailable (" + ("; ".join(errors) or "no image") + ")"}
    try:
        from PIL import Image
        w, h = img.size
        scale = min(1.0, max_dim / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        while buf.tell() > 4 * 1024 * 1024:
            img = img.resize((max(1, img.width // 2), max(1, img.height // 2)), Image.LANCZOS)
            buf.seek(0)
            buf.truncate()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
    except Exception as exc:
        return {"ok": False, "error": f"screen encode failed ({exc.__class__.__name__})"}
    return {"ok": True, "width": img.size[0], "height": img.size[1], "mime": "image/jpeg",
            "dataUrl": "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}
