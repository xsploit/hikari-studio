"""Fish Audio Voice Library + Voice Design (port of Waifu Studio voice_library.rs + voice_design.rs).

* list_voices        GET  https://api.fish.audio/model   (self=true for the account's own models; public
                     library sorted by score), normalized + cached 60 s per (scope, query, page, limit, key hash).
* design_voice       POST https://api.fish.audio/v1/voice-design  (header `model: voice-design-1`, JSON body
                     {instruction, reference_text?, language?, n, speed}); candidates are decoded and kept in a
                     bounded TTL cache (24 entries, 30 min) so they can be saved later by id.
* save_designed_voice POST https://api.fish.audio/model  multipart/form-data (type=tts, train_mode=fast, title,
                     visibility, description, enhance_audio_quality, voices=<file>, texts, voice_design_signatures)
                     -> a permanent voice model on the user's account (documented in api.fish.audio/openapi.json).
* delete_voice       DELETE https://api.fish.audio/model/{id}

Errors never include the API key or the remote response body (only the HTTP status + a short Fish reason code).
All network I/O uses the shared pooled client (modules.http_pool); tests swap `_client`.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
import struct
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

import httpx

from modules import http_pool

FISH_API_BASE = "https://api.fish.audio"
MODEL_URL = f"{FISH_API_BASE}/model"
VOICE_DESIGN_URL = f"{FISH_API_BASE}/v1/voice-design"
VOICE_DESIGN_MODEL = "voice-design-1"

LIST_TTL_SEC = 60.0
LIST_CACHE_MAX = 64
CANDIDATE_TTL_SEC = 30 * 60.0
CANDIDATE_MAX = 24
MAX_LIST_BYTES = 4 * 1024 * 1024          # voice_library.rs MAX_BYTES
MAX_DESIGN_BYTES = 32 * 1024 * 1024       # voice_design.rs MAX_BYTES
MAX_PAGE_SIZE = 100
MAX_PAGE_NUMBER = 1000
SCOPES = ("self", "public", "all")
VISIBILITIES = ("private", "unlist", "public")

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_BIDI_RE = re.compile("[‪-‮⁦-⁩]")


class FishVoiceError(RuntimeError):
    """User-safe error (no key, no remote body). `status` is the HTTP status to return (400/404/502...)."""

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


def _client() -> httpx.AsyncClient:          # tests replace this with an httpx.MockTransport client
    return http_pool.client()


# --------------------------------------------------------------------------- helpers
def _clean(value: Any, limit: int) -> str:
    """voice_library.rs `clean`: drop control / bidi-override characters, collapse whitespace, bound length."""
    if not isinstance(value, str):
        return ""
    text = "".join(" " if c.isspace() else c for c in value)
    text = "".join(c for c in text if c.isprintable())
    text = " ".join(_BIDI_RE.sub("", text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def valid_voice_id(voice_id: Any) -> bool:
    return isinstance(voice_id, str) and bool(_ID_RE.fullmatch(voice_id))


def _check_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise FishVoiceError("Fish Audio API key is not configured", 400)
    if len(key) > 8192 or any(ord(c) < 32 or ord(c) == 127 for c in key):
        raise FishVoiceError("Fish Audio API key is invalid", 400)
    return key


def _key_hash(api_key: str) -> str:
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _fail(what: str, r: httpx.Response) -> FishVoiceError:
    reason = ""
    try:
        data = r.json()
        if isinstance(data, dict):
            val = data.get("code") or data.get("reason") or data.get("status")
            if isinstance(val, (str, int)) and str(val).strip():
                reason = f" ({_clean(str(val), 60)})"
    except Exception:
        pass
    status = 400 if r.status_code in (400, 402, 422) else (404 if r.status_code == 404 else 502)
    if r.status_code in (401, 403):
        status = 400                       # bad / unauthorized key: a config problem, not a server crash
    return FishVoiceError(f"{what} returned HTTP {r.status_code}{reason}", status)


async def _read_limited(resp: httpx.Response, limit: int, what: str) -> bytes:
    length = resp.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        raise FishVoiceError(f"{what} response exceeds size limit")
    buf = bytearray()
    async for chunk in resp.aiter_bytes():
        buf += chunk
        if len(buf) > limit:
            raise FishVoiceError(f"{what} response exceeds size limit")
    return bytes(buf)


async def _request_json(method: str, url: str, what: str, limit: int, *, timeout: float = 20.0,
                        **kw: Any) -> Any:
    import json
    try:
        async with _client().stream(method, url, timeout=httpx.Timeout(timeout, connect=10.0), **kw) as r:
            if r.status_code >= 400:
                await r.aread()
                raise _fail(what, r)
            if r.status_code >= 300:
                raise FishVoiceError(f"{what} returned HTTP {r.status_code}")
            raw = await _read_limited(r, limit, what)
    except FishVoiceError:
        raise
    except httpx.TimeoutException:
        raise FishVoiceError(f"{what} timed out", 504) from None
    except httpx.HTTPError:
        raise FishVoiceError(f"{what} request failed", 502) from None
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        raise FishVoiceError(f"{what} returned invalid JSON") from None


def sniff_mime(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    return "audio/wav" if data[:4] == b"RIFF" else "application/octet-stream"


_EXT = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/ogg": "ogg", "audio/flac": "flac"}


def wav_duration(data: bytes) -> float:
    """Seconds of PCM in a RIFF/WAVE blob (0.0 if not parseable)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0.0
    pos, byte_rate, data_len = 12, 0, 0
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt " and size >= 16 and pos + 24 <= len(data):
            byte_rate = struct.unpack("<I", data[pos + 16:pos + 20])[0]
        elif cid == b"data":
            data_len = min(size, len(data) - pos - 8)
            break
        pos += 8 + size + (size & 1)
    return round(data_len / byte_rate, 3) if byte_rate else 0.0


# --------------------------------------------------------------------------- voice library
_list_cache: "OrderedDict[tuple, tuple[float, list[dict[str, Any]]]]" = OrderedDict()
_list_lock = threading.Lock()


def clear_list_cache() -> None:
    with _list_lock:
        _list_cache.clear()


def normalize_voice(model: Any, *, mine: bool = False) -> dict[str, Any] | None:
    """One Fish `/model` item -> the frontend schema; None if it is not a usable TTS voice."""
    if not isinstance(model, dict):
        return None
    vid = model.get("_id") or model.get("id")
    if not valid_voice_id(vid):
        return None
    kind = str(model.get("type") or "tts")
    if kind != "tts":
        return None
    title = _clean(model.get("title"), 120) or vid
    langs = [_clean(v, 32) for v in (model.get("languages") or []) if isinstance(v, str)]
    samples = []
    for s in model.get("samples") or []:
        audio = s.get("audio") if isinstance(s, dict) else s
        if isinstance(audio, str) and audio.startswith("https://") and len(audio) <= 2048:
            samples.append(audio)
    author = model.get("author")
    return {
        "id": vid,
        "title": title,
        "description": _clean(model.get("description"), 300),
        "visibility": _clean(model.get("visibility"), 16) or "private",
        "languages": [v for v in langs if v][:12],
        "samples": samples[:6],
        "mine": bool(mine),
        "author": _clean(author.get("nickname"), 64) if isinstance(author, dict) else "",
        "state": _clean(model.get("state"), 16),
        "tags": [_clean(v, 32) for v in (model.get("tags") or []) if isinstance(v, str)][:12],
    }


async def _fetch_page(key: str, *, own: bool, query: str, page: int, limit: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"page_size": limit, "page_number": page}
    if own:
        params["self"] = "true"
    else:
        params["sort_by"] = "score"
    if query:
        params["title"] = query
    data = await _request_json("GET", MODEL_URL, "Fish voice library", MAX_LIST_BYTES,
                               headers=_auth(key), params=params)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise FishVoiceError("Fish voice library returned an unexpected schema")
    out: list[dict[str, Any]] = []
    for item in items:
        v = normalize_voice(item, mine=own)
        if v is None or any(x["id"] == v["id"] for x in out):
            continue
        # The public library only lists ready voices; your own models also show while training.
        if not own and v["state"] not in ("", "trained"):
            continue
        if own and v["state"] == "failed":
            continue
        out.append(v)
    return out[:limit]


async def list_voices(api_key: str, scope: str = "self", query: str = "", page: int = 1, limit: int = 30,
                      *, force: bool = False) -> list[dict[str, Any]]:
    key = _check_key(api_key)
    if scope == "community":
        scope = "public"
    scope = scope if scope in SCOPES else "self"
    query = _clean(query, 128)
    try:
        page = min(max(int(page), 1), MAX_PAGE_NUMBER)
        limit = min(max(int(limit), 1), MAX_PAGE_SIZE)
    except (TypeError, ValueError):
        page, limit = 1, 30
    ck = (scope, query, page, limit, _key_hash(key))
    now = time.monotonic()
    if not force:
        with _list_lock:
            hit = _list_cache.get(ck)
            if hit and now - hit[0] < LIST_TTL_SEC:
                return [dict(v) for v in hit[1]]
    if scope == "all":
        mine = await list_voices(key, "self", query, page, limit, force=force)
        public = await list_voices(key, "public", query, page, limit, force=force)
        seen = {v["id"] for v in mine}
        voices = mine + [v for v in public if v["id"] not in seen]
    else:
        voices = await _fetch_page(key, own=scope == "self", query=query, page=page, limit=limit)
    with _list_lock:
        _list_cache[ck] = (time.monotonic(), voices)
        _list_cache.move_to_end(ck)
        while len(_list_cache) > LIST_CACHE_MAX:
            _list_cache.popitem(last=False)
    return [dict(v) for v in voices]


# --------------------------------------------------------------------------- voice design
_candidates: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_cand_lock = threading.Lock()


def _prune_candidates(now: float) -> None:
    for cid in [c for c, v in _candidates.items() if now - v["created_at"] > CANDIDATE_TTL_SEC]:
        _candidates.pop(cid, None)
    while len(_candidates) > CANDIDATE_MAX:
        _candidates.popitem(last=False)


def store_candidate(audio: bytes, mime: str, instruction: str, reference_text: str = "",
                    *, signature: str = "", text: str = "") -> str:
    cid = f"cand_{uuid.uuid4().hex[:20]}"
    now = time.monotonic()
    with _cand_lock:
        _candidates[cid] = {"bytes": audio, "mime": mime, "instruction": instruction,
                            "reference_text": reference_text, "signature": signature, "text": text,
                            "created_at": now}
        _prune_candidates(now)
    return cid


def get_candidate(cid: str) -> dict[str, Any] | None:
    with _cand_lock:
        _prune_candidates(time.monotonic())
        return _candidates.get(cid)


async def design_voice(api_key: str, instruction: str, reference_text: str = "", language: str = "en",
                       n: int = 2, *, speed: float = 1.0, seed: int | None = None) -> dict[str, Any]:
    """Paid Voice Design previews. Returns {"ok": True, "candidates": [{id, audioBase64, mime, durationSec}]}."""
    key = _check_key(api_key)
    instruction = (instruction or "").strip()
    reference_text = (reference_text or "").strip()
    if not instruction or len(instruction) > 2000:
        raise FishVoiceError("Describe a voice in 1-2000 characters", 400)
    if len(reference_text) > 150:
        raise FishVoiceError("Preview text exceeds 150 characters", 400)
    try:
        n = int(n)
        speed = float(speed)
    except (TypeError, ValueError):
        raise FishVoiceError("Use 1-4 candidates and speed greater than 0, at most 3", 400) from None
    if not 1 <= n <= 4 or not (0.0 < speed <= 3.0):
        raise FishVoiceError("Use 1-4 candidates and speed greater than 0, at most 3", 400)
    body: dict[str, Any] = {"instruction": instruction, "n": n, "speed": speed}
    if reference_text:
        body["reference_text"] = reference_text
    lang = _clean(language, 16)
    if lang and re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*", lang):
        body["language"] = lang
    if isinstance(seed, int) and not isinstance(seed, bool):
        body["seed"] = seed
    headers = dict(_auth(key), model=VOICE_DESIGN_MODEL)
    data = await _request_json("POST", VOICE_DESIGN_URL, "Fish Voice Design", MAX_DESIGN_BYTES,
                               timeout=90.0, headers=headers, json=body)
    wire = data.get("candidates") if isinstance(data, dict) else None
    if not isinstance(wire, list) or not wire or len(wire) > n:
        raise FishVoiceError("Fish Voice Design returned an unexpected candidate count")
    decoded: list[tuple[int, dict[str, Any], bytes]] = []
    total = 0
    for i, c in enumerate(wire):
        if not isinstance(c, dict) or not isinstance(c.get("audio_base64"), str):
            raise FishVoiceError("Fish Voice Design returned invalid candidate metadata")
        try:
            audio = base64.b64decode(c["audio_base64"], validate=True)
        except (binascii.Error, ValueError):
            raise FishVoiceError("Fish Voice Design returned invalid preview audio") from None
        total += len(audio)
        if not audio or total > MAX_DESIGN_BYTES:
            raise FishVoiceError("Fish Voice Design preview audio exceeds limits")
        idx = c.get("index") if isinstance(c.get("index"), int) else i
        decoded.append((idx, c, audio))
    decoded.sort(key=lambda t: t[0])
    out = []
    for _, c, audio in decoded:
        mime = sniff_mime(audio)
        dur_ms = c.get("duration_ms")
        dur = round(dur_ms / 1000.0, 3) if isinstance(dur_ms, (int, float)) and dur_ms > 0 else wav_duration(audio)
        sig = c.get("signature") or c.get("voice_design_signature") or ""
        cid = store_candidate(audio, mime, instruction, reference_text,
                              signature=sig if isinstance(sig, str) and len(sig) <= 4096 else "",
                              text=_clean(c.get("text"), 300))
        out.append({"id": cid, "audioBase64": base64.b64encode(audio).decode("ascii"), "mime": mime,
                    "durationSec": dur, "sampleRate": c.get("sample_rate") if isinstance(c.get("sample_rate"), int) else None})
    return {"ok": True, "candidates": out}


# --------------------------------------------------------------------------- save / delete
async def save_designed_voice(api_key: str, title: str, candidate_id: str = "", audio_base64: str = "",
                              description: str = "", visibility: str = "private",
                              enhance: bool = True) -> dict[str, Any]:
    """Create a permanent TTS voice model on the user's Fish account from a designed candidate."""
    key = _check_key(api_key)
    title = _clean(title, 120)
    if not title:
        raise FishVoiceError("Give the voice a title", 400)
    if visibility == "unlisted":
        visibility = "unlist"
    visibility = visibility if visibility in VISIBILITIES else "private"
    description = _clean(description, 500)
    text, signature = "", ""
    if candidate_id:
        cand = get_candidate(str(candidate_id))
        if cand is None:
            raise FishVoiceError("That preview expired; generate it again", 404)
        audio, mime = cand["bytes"], cand["mime"]
        text = cand.get("text") or cand.get("reference_text") or ""
        signature = cand.get("signature") or ""
        if not description:
            description = _clean(cand.get("instruction"), 500)
    elif audio_base64:
        raw = audio_base64.split(",", 1)[1] if audio_base64.startswith("data:") else audio_base64
        try:
            audio = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise FishVoiceError("audioBase64 is not valid base64", 400) from None
        mime = sniff_mime(audio)
    else:
        raise FishVoiceError("Pick a designed candidate (candidateId) or send audioBase64", 400)
    if not audio or len(audio) > MAX_DESIGN_BYTES:
        raise FishVoiceError("Voice audio is empty or too large", 400)
    form: dict[str, Any] = {"type": "tts", "train_mode": "fast", "title": title, "visibility": visibility,
                            "enhance_audio_quality": "true" if enhance else "false"}
    if description:
        form["description"] = description
    if text:
        form["texts"] = text
    if signature:
        form["voice_design_signatures"] = signature
    files = {"voices": (f"voice.{_EXT.get(mime, 'wav')}", audio, mime if mime in _EXT else "audio/wav")}
    data = await _request_json("POST", MODEL_URL, "Fish create voice", MAX_LIST_BYTES, timeout=120.0,
                               headers=_auth(key), data=form, files=files)
    voice = normalize_voice(data, mine=True)
    if voice is None:
        raise FishVoiceError("Fish create voice returned an unexpected schema")
    clear_list_cache()
    return voice


async def delete_voice(api_key: str, voice_id: str) -> str:
    key = _check_key(api_key)
    if not valid_voice_id(voice_id):
        raise FishVoiceError("Invalid voice id", 400)
    await _request_json("DELETE", f"{MODEL_URL}/{voice_id}", "Fish delete voice", MAX_LIST_BYTES,
                        headers=_auth(key))
    clear_list_cache()
    return voice_id
