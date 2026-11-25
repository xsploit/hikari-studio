"""Server-side API key resolution. Keys never enter config, telemetry or any browser payload.

Lookup order per name: process environment, then %USERPROFILE%/Documents/treblo/.env.local
(or $HIKARI_KEY_FILE / $FISH_KEY_FILE). The file is read once and cached; call `reload()` to refresh.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

from core.paths import studio_root

KNOWN_KEYS = (
    "AI_GATEWAY_API_KEY", "AI_GATEWAY_MODEL", "VERCEL_OIDC_TOKEN",
    "FISH_SPEECH_API_KEY", "FISHSPEECH_API_KEY", "FISH_AUDIO_API_KEY", "FISH_API_KEY",
    "OPENAI_API_KEY", "OPENROUTER_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "PERPLEXITY_API_KEY",
)

_lock = threading.Lock()
_file_cache: dict[str, str] | None = None


def _key_files() -> list[Path]:
    out: list[Path] = []
    for var in ("HIKARI_KEY_FILE", "FISH_KEY_FILE"):
        if os.environ.get(var):
            out.append(Path(os.environ[var]))
    repo_root = studio_root()
    out.append(repo_root / ".env")
    out.append(repo_root / ".env.local")
    if os.environ.get("USERPROFILE"):
        out.append(Path(os.environ["USERPROFILE"]) / "Documents" / "treblo" / ".env.local")
    return out


def _load_files() -> dict[str, str]:
    global _file_cache
    with _lock:
        if _file_cache is not None:
            return _file_cache
        found: dict[str, str] = {}
        for p in _key_files():
            try:
                if not p.exists():
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().removeprefix("export ").strip()
                    v = v.strip().strip("\"'")
                    if k in KNOWN_KEYS and v and k not in found:
                        found[k] = v
            except Exception:
                continue
        _file_cache = found
        return found


def reload() -> None:
    global _file_cache
    with _lock:
        _file_cache = None


def get(name: str) -> str:
    val = (os.environ.get(name) or "").strip()
    return val or _load_files().get(name, "")


def first(*names: str) -> str:
    for n in names:
        v = get(n)
        if v:
            return v
    return ""


def fish_key() -> str:
    return first("FISH_SPEECH_API_KEY", "FISHSPEECH_API_KEY", "FISH_AUDIO_API_KEY", "FISH_API_KEY")


def gateway_key() -> str:
    return first("AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN")


def availability() -> dict[str, bool]:
    """Which keys are present (booleans only — safe to show in a UI)."""
    return {
        "aiGateway": bool(gateway_key()),
        "fishSpeech": bool(fish_key()),
        "openai": bool(get("OPENAI_API_KEY")),
        "openrouter": bool(get("OPENROUTER_API_KEY")),
        "cloudflare": bool(get("CLOUDFLARE_ACCOUNT_ID") and get("CLOUDFLARE_API_TOKEN")),
        "perplexity": bool(get("PERPLEXITY_API_KEY")),
    }


# ----------------------------------------------------------------------------- credential policy
# A key (server-managed OR a saved override) is attached ONLY to its provider's official HTTPS host.
# Client-configurable endpoints (fishUrl, openaiTtsUrl, baseUrl, test hooks...) never receive a key
# unless they are exactly the official host; localhost targets get no key at all.
OFFICIAL_HOSTS: dict[str, tuple[str, ...]] = {
    "vercel": ("ai-gateway.vercel.sh",),
    "cloudflare": ("api.cloudflare.com",),
    "perplexity": ("api.perplexity.ai",),
    "openai": ("api.openai.com",),
    "fish": ("api.fish.audio",),
    "openrouter": ("openrouter.ai",),
}
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def is_official_url(url: str, provider: str) -> bool:
    try:
        u = urlsplit(str(url or ""))
        port = u.port
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    return (u.scheme in ("https", "wss") and host in OFFICIAL_HOSTS.get(provider, ())
            and not u.username and not u.password and port in (None, 443))


def is_local_url(url: str) -> bool:
    try:
        u = urlsplit(str(url or ""))
    except ValueError:
        return False
    return u.scheme in ("http", "https", "ws", "wss") and (u.hostname or "").lower() in _LOCAL_HOSTS


def key_for_url(url: str, provider: str, key: str) -> str:
    """`key` if `url` is the provider's official host, else "" (never send a credential elsewhere)."""
    return key if key and is_official_url(url, provider) else ""
