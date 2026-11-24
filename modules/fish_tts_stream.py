"""Fish Speech multi-transport TTS: "sse" | "ws" | "http" (config ai.fishTtsTransport).

  sse   POST https://api.fish.audio/v1/tts/stream/with-timestamp (Accept: text/event-stream; events
        carry base64 WAV fragments) — sentence-chunked as the LLM streams, so Fish synthesizes early
        sentences while later ones are still being written (Waifu Studio fish_live.rs).
  ws    wss://api.fish.audio/v1/tts/live, MessagePack frames: start{request} -> text/flush ... -> stop,
        receiving audio{bytes} and finish{reason} (Waifu Studio fish_ws.rs / WebWaifu4
        RemoteTtsProvider convertRealtime). LLM text is forwarded phrase by phrase on one socket.
  http  POST https://api.fish.audio/v1/tts (whole reply) — simplest fallback.

All transports produce one 16-bit mono WAV per reply, assembled in order (no ffmpeg, writes off the
event loop). HTTP uses the shared pooled httpx client; WS uses one `websockets` connection per reply.
"""
from __future__ import annotations

import asyncio
import copy
import base64
import json
from pathlib import Path
import re
import struct
import time
from typing import Any, AsyncIterator, Awaitable, Callable
import uuid
import wave

import httpx

from modules import http_pool, server_keys

FISH_TTS_URL = "https://api.fish.audio/v1/tts"
FISH_SSE_URL = "https://api.fish.audio/v1/tts/stream/with-timestamp"
FISH_WS_URL = "wss://api.fish.audio/v1/tts/live"
FISH_TRANSPORTS = ("sse", "ws", "http")
REMOTE_TTS_TIMEOUT = 45.0
MAX_AUDIO_BYTES = 32 * 1024 * 1024
DEFAULT_PCM_RATE = 44100

ANY_TAG_RE = re.compile(r"\[([^\[\]\n]*)\]")
_SENT_END_RE = re.compile(r"([.!?…~。！？]+[\"')\]]*)(\s+|$)")


def _strip_tags(text: str) -> str:
    from modules.ai_companion import strip_vocal_tags
    return strip_vocal_tags(text)


def fish_model_supports_tags(model: str) -> bool:
    """Fish Drama-3 (inline [directions]) and S2 / S2.1 ([official tags]) voice bracket cues; S1 does not."""
    m = str(model or "").strip().lower()
    return m.startswith("drama") or m.startswith("s2")


def clean_for_speech(text: str, *, preserve_vocal_tags: bool = False, fish_model: str = "drama-3") -> str:
    """Text for the TTS engine. Default: every [...] cue + markdown stripped (backwards compatible).
    preserve_vocal_tags=True (Fish Drama-3 / S2 with ai.fishVocalTags): cues are kept / normalized through
    ai_companion.prepare_tts_text (Drama-3 directions sanitized, S2 cues mapped to official tags)."""
    if not preserve_vocal_tags:
        return _strip_tags(text)
    from modules.ai_companion import prepare_tts_text   # lazy: ai_companion imports this module
    return prepare_tts_text(text or "", tts_provider="fish", fish_model=fish_model or "drama-3",
                            vocal_tags_enabled=True).strip()


class SentenceChunker:
    """Incremental phrase splitter for streamed LLM text (WebWaifu4 splitTextForStreaming
    thresholds: min ~20, soft ~90, hard ~180 chars). Holds back an unclosed [tag].
    With preserve_vocal_tags=True, phrases keep their (normalized) Fish vocal cues; length thresholds are
    always measured on the tag-free text."""

    def __init__(self, min_chars: int = 20, soft_chars: int = 90, max_chars: int = 180, *,
                 preserve_vocal_tags: bool = False, fish_model: str = "drama-3") -> None:
        self.buf = ""
        self.min_chars, self.soft_chars, self.max_chars = min_chars, soft_chars, max_chars
        self.preserve_vocal_tags, self.fish_model = bool(preserve_vocal_tags), fish_model

    def _speech(self, chunk: str) -> str:
        c = clean_for_speech(chunk, preserve_vocal_tags=self.preserve_vocal_tags, fish_model=self.fish_model)
        return c if _strip_tags(c) else ""          # a cue-only fragment has nothing to say

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                break
            c = self._speech(chunk)
            if c:
                out.append(c)
        return out

    def _take(self) -> str | None:
        b = self.buf
        if b.rfind("[") > b.rfind("]"):
            b_scan = b[: b.rfind("[")]                  # don't split inside a pending [tag]
        else:
            b_scan = b
        # Each cue occupies a single space while choosing a boundary. Then map the cut
        # back to the original reply, keeping directions attached to the following phrase.
        plain = ANY_TAG_RE.sub(" ", b_scan)
        cut = None
        for match in _SENT_END_RE.finditer(plain):
            if len(plain[:match.end(1)].strip()) >= self.min_chars:
                cut = match.end(1)
                break
        if cut is None and len(plain) >= self.max_chars:
            window = plain[:self.max_chars + 1]
            space = window.rfind(" ")
            cut = space if space > self.min_chars else self.max_chars
        if cut is None:
            return None
        cut = len(plain[:cut].rstrip())
        if not cut:
            return None
        for cue in ANY_TAG_RE.finditer(b_scan):
            if cue.start() >= cut:
                break
            cut += len(cue[0]) - 1
        self.buf = b[cut:].lstrip()
        return b[:cut]

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        c = self._speech(rest)
        return [c] if c else []


# --------------------------------------------------------------------------- audio helpers
def parse_wav_stream(data: bytes, default_rate: int = DEFAULT_PCM_RATE) -> tuple[bytes, int, int]:
    """Tolerant WAV parse for streamed audio (placeholder sizes allowed). Raw PCM passes through.
    Returns (pcm_s16le_mono, sample_rate, channels)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data[: len(data) & ~1], default_rate, 1
    rate, channels, bits = default_rate, 1, 16
    off = 12
    while off + 8 <= len(data):
        cid = data[off:off + 4]
        size = struct.unpack_from("<I", data, off + 4)[0]
        start = off + 8
        if cid == b"fmt " and start + 16 <= len(data):
            _fmt, channels, rate, _br, _ba, bits = struct.unpack_from("<HHIIHH", data, start)
        elif cid == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or start + size > len(data) else start + size
            pcm = data[start:end]
            if bits != 16:
                return b"", rate, channels
            if channels > 1:
                import numpy as np
                arr = np.frombuffer(pcm[: len(pcm) // (2 * channels) * 2 * channels], dtype="<i2")
                pcm = arr.reshape(-1, channels).mean(axis=1).astype("<i2").tobytes()
                channels = 1
            return pcm[: len(pcm) & ~1], rate, channels
        if size in (0xFFFFFFFF,) or size > len(data):
            break
        off = start + size + (size % 2)
    return b"", rate, channels


def _resample_pcm(pcm: bytes, src: int, dst: int) -> bytes:
    if src == dst or not pcm:
        return pcm
    import numpy as np
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = int(round(len(x) * dst / src))
    if n <= 1:
        return b""
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


def write_wav(path: Path, pcm: bytes, rate: int) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return path


# --------------------------------------------------------------------------- minimal-dep msgpack
try:
    import msgpack as _msgpack
except Exception:  # pragma: no cover
    _msgpack = None


def mp_pack(obj: Any) -> bytes:
    if _msgpack is None:
        raise RuntimeError("msgpack is required for the Fish WebSocket transport")
    return _msgpack.packb(obj, use_bin_type=True)


def mp_unpack(data: bytes) -> Any:
    if _msgpack is None:
        raise RuntimeError("msgpack is required for the Fish WebSocket transport")
    return _msgpack.unpackb(data, raw=False, max_bin_len=MAX_AUDIO_BYTES, max_str_len=1 << 20)


async def _cancel_all(tasks: list[asyncio.Task]) -> None:
    pending = [t for t in tasks if not t.done()]
    for t in pending:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------------- streamer
class FishTtsStreamer:
    def __init__(self, cfg: dict[str, Any] | None = None, audio_dir: Path | None = None) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.audio_dir = audio_dir
        self.urls = {"http": FISH_TTS_URL, "sse": FISH_SSE_URL, "ws": FISH_WS_URL}
        self.last_stats: dict[str, Any] = {}

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg.update(cfg or {})

    @property
    def transport(self) -> str:
        t = str(self.cfg.get("fishTtsTransport") or "sse").lower()
        return t if t in FISH_TRANSPORTS else "sse"

    def key(self) -> str:
        return str(self.cfg.get("fishApiKey") or "").strip() or server_keys.fish_key()

    @property
    def configured(self) -> bool:
        url = self.urls["sse"]
        return (bool(self.key()) and server_keys.is_official_url(url, "fish")) or server_keys.is_local_url(url)

    def request_body(self, text: str, fmt: str = "wav") -> dict[str, Any]:
        """Shared Fish request body (fish_live.rs / fish_ws.rs / RemoteTtsProvider)."""
        latency = str(self.cfg.get("fishLatency") or "balanced")
        if latency not in ("normal", "balanced", "low"):
            latency = "balanced"
        try:
            chunk_length = int(max(100, min(300, int(self.cfg.get("fishChunkLength", 160)))))
        except (TypeError, ValueError):
            chunk_length = 160
        body: dict[str, Any] = {
            "text": text,
            "format": fmt,
            "normalize": True,
            "latency": latency,
            "chunk_length": chunk_length,
            "min_chunk_length": 20,
            "condition_on_previous_chunks": bool(self.cfg.get("fishConditionOnPrevious", True)),
            "prosody": {"speed": max(0.5, min(2.0, float(self.cfg.get("fishSpeed", 1.0) or 1.0)))},
        }
        ref = str(self.cfg.get("fishReferenceId") or "").strip()
        if ref:
            body["reference_id"] = ref
        return body

    @property
    def fish_model(self) -> str:
        return str(self.cfg.get("fishModel") or "drama-3")

    @property
    def preserve_vocal_tags(self) -> bool:
        """Fish Drama-3 / S2 with ai.fishVocalTags on: streamed phrases keep their [vocal cues]."""
        return (str(self.cfg.get("ttsProvider", "fish")).lower() == "fish"
                and bool(self.cfg.get("fishVocalTags", True)) and fish_model_supports_tags(self.fish_model))

    def _headers(self, url: str, accept: str | None = None) -> dict[str, str]:
        """Fish key only for the official api.fish.audio host; localhost gets none; others are refused."""
        key = server_keys.key_for_url(url, "fish", self.key())
        if not key and not server_keys.is_local_url(url):
            raise RuntimeError("Fish Speech TTS requires FISH_SPEECH_API_KEY / FISH_AUDIO_API_KEY "
                               "and the official api.fish.audio endpoint")
        h = {"Content-Type": "application/json", "model": str(self.cfg.get("fishModel") or "drama-3")}
        if key:
            h["Authorization"] = f"Bearer {key}"
        if accept:
            h["Accept"] = accept
        return h

    # ---------------- single-text transports -> (pcm, rate)
    async def synth_http(self, text: str) -> tuple[bytes, int]:
        r = await http_pool.client().post(self.urls["http"], headers=self._headers(self.urls["http"]),
                                          json=self.request_body(text, "wav"),
                                          timeout=httpx.Timeout(REMOTE_TTS_TIMEOUT, connect=10.0))
        if r.status_code >= 400:
            raise RuntimeError(f"Fish TTS HTTP {r.status_code}")
        pcm, rate, _ = parse_wav_stream(r.content[:MAX_AUDIO_BYTES])
        return pcm, rate

    async def synth_sse(self, text: str) -> tuple[bytes, int]:
        audio = bytearray()
        async with http_pool.client().stream(
                "POST", self.urls["sse"], headers=self._headers(self.urls["sse"], "text/event-stream"),
                json=self.request_body(text, "wav"),
                timeout=httpx.Timeout(REMOTE_TTS_TIMEOUT, connect=10.0)) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise RuntimeError(f"Fish SSE HTTP {resp.status_code}")
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" not in ctype.lower():
                raise RuntimeError("Fish SSE returned an unsupported content type")
            event = "message"
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip() or "message"
                    if event == "error":
                        raise RuntimeError("Fish provider reported a stream error")
                    continue
                if not line.startswith("data:"):
                    if not line:
                        event = "message"
                    continue
                try:
                    frame = json.loads(line[5:].strip())
                except Exception:
                    continue
                b64 = frame.get("audio_base64") if isinstance(frame, dict) else None
                if b64:
                    audio += base64.b64decode(b64)
                    if len(audio) > MAX_AUDIO_BYTES:
                        raise RuntimeError("Fish SSE audio exceeds its size limit")
        pcm, rate, _ = parse_wav_stream(bytes(audio))
        return pcm, rate

    async def stream_ws(self, phrases: AsyncIterator[str]) -> tuple[bytes, int]:
        """One Fish live socket; forward phrases as they arrive; collect audio until finish."""
        import websockets
        headers = self._headers(self.urls["ws"])
        hdrs = {k: v for k, v in headers.items() if k in ("Authorization", "model")}
        audio = bytearray()
        async with websockets.connect(self.urls["ws"], additional_headers=hdrs, max_size=2 * 1024 * 1024,
                                      open_timeout=10, close_timeout=2) as ws:
            await ws.send(mp_pack({"event": "start", "request": self.request_body("", "wav")}))
            stop_sent = asyncio.Event()

            async def sender() -> None:
                async for phrase in phrases:
                    await ws.send(mp_pack({"event": "text", "text": phrase + " "}))
                    await ws.send(mp_pack({"event": "flush"}))
                await ws.send(mp_pack({"event": "stop"}))
                stop_sent.set()

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    if isinstance(raw, str):
                        continue
                    msg = mp_unpack(raw)
                    ev = msg.get("event") if isinstance(msg, dict) else None
                    if ev == "audio":
                        chunk = msg.get("audio")
                        if isinstance(chunk, (bytes, bytearray)):
                            audio += chunk
                            if len(audio) > MAX_AUDIO_BYTES:
                                raise RuntimeError("Fish WebSocket audio exceeds its size limit")
                    elif ev == "finish":
                        if msg.get("reason") == "stop" and stop_sent.is_set():
                            break
                        raise RuntimeError("Fish WebSocket finished unexpectedly")
                    elif ev == "error":
                        raise RuntimeError("Fish WebSocket reported a synthesis error")
                if send_task.done() and send_task.exception():
                    raise send_task.exception()  # type: ignore[misc]
            finally:
                if not send_task.done():
                    send_task.cancel()
        pcm, rate, _ = parse_wav_stream(bytes(audio))
        return pcm, rate

    # ---------------- orchestration: LLM delta stream -> one WAV
    async def speak_stream(self, deltas: AsyncIterator[str], *, transport: str | None = None,
                           on_text: Callable[[str], Awaitable[None] | None] | None = None,
                           on_segment: Callable[[bytes, int, str, int], Awaitable[None]] | None = None,
                           preserve_vocal_tags: bool | None = None,
                           ) -> tuple[str, bytes, int]:
        """Consume LLM deltas, synthesize while streaming, return (full_text, pcm, rate).
        `on_segment(pcm, rate, text, index)` (optional) receives each phrase's audio IN ORDER as soon as it is
        ready, so playback of sentence 1 starts while later sentences are still being written / synthesized.
        (The live WebSocket transport yields one continuous stream: it is delivered as a single segment.)
        preserve_vocal_tags (None = from cfg: Fish Drama-3 / S2 + fishVocalTags) keeps inline [cues] in the text
        sent to Fish; `on_segment` always receives the tag-free display text."""
        # Profile changes affect the next reply. All requests for this one use the same voice/model
        # and transport settings that were used to format its cues.
        voice = copy.copy(self)
        voice.cfg = self.cfg.copy()
        transport = transport or voice.transport
        if preserve_vocal_tags is None:
            preserve_vocal_tags = voice.preserve_vocal_tags
        model = voice.fish_model
        t0 = time.perf_counter()
        full: list[str] = []
        chunker = SentenceChunker(preserve_vocal_tags=preserve_vocal_tags, fish_model=model)

        def speech(text: str) -> str:
            return clean_for_speech(text, preserve_vocal_tags=preserve_vocal_tags, fish_model=model)
        first_audio_at: list[float] = []

        if transport == "ws":
            q: asyncio.Queue[str | None] = asyncio.Queue()   # unbounded: a dead socket never stalls the LLM reader

            async def phrase_iter() -> AsyncIterator[str]:
                while True:
                    item = await q.get()
                    if item is None:
                        return
                    yield item

            ws_task = asyncio.create_task(voice.stream_ws(phrase_iter()))
            try:
                try:
                    async for d in deltas:
                        full.append(d)
                        for ph in chunker.feed(d):
                            q.put_nowait(ph)
                    for ph in chunker.flush():
                        q.put_nowait(ph)
                finally:
                    q.put_nowait(None)
                pcm, rate = await asyncio.wait_for(ws_task, REMOTE_TTS_TIMEOUT)
            except BaseException:
                await _cancel_all([ws_task])         # LLM failed / timeout / caller cancelled: stop the socket
                raise
            if on_segment is not None and pcm:
                await on_segment(pcm, rate, _strip_tags("".join(full)), 0)
        else:
            synth = voice.synth_sse if transport == "sse" else voice.synth_http
            sem = asyncio.Semaphore(2)               # bounded concurrent Fish requests per reply
            tasks: list[asyncio.Task] = []
            emit_q: asyncio.Queue = asyncio.Queue()
            emitter: asyncio.Task | None = None

            async def emit_in_order() -> None:
                idx = 0
                while True:
                    item = await emit_q.get()
                    if item is None:
                        return
                    task, phrase = item
                    seg_pcm, seg_rate = await task
                    if seg_pcm:
                        await on_segment(seg_pcm, seg_rate, _strip_tags(phrase), idx)
                        idx += 1

            def spawn(phrase: str) -> None:
                task = asyncio.create_task(run(phrase))
                tasks.append(task)
                if on_segment is not None:
                    emit_q.put_nowait((task, phrase))

            async def run(phrase: str) -> tuple[bytes, int]:
                async with sem:
                    res = await synth(phrase)
                    if not res[0]:
                        raise RuntimeError("Fish returned an empty audio segment")
                    if not first_audio_at:
                        first_audio_at.append(time.perf_counter() - t0)
                    return res

            try:
                if on_segment is not None:
                    emitter = asyncio.create_task(emit_in_order())
                if transport == "http" and not voice.cfg.get("fishHttpChunked", False):
                    async for d in deltas:
                        full.append(d)
                    text = speech("".join(full))
                    if text and _strip_tags(text):
                        spawn(text)
                else:
                    async for d in deltas:
                        full.append(d)
                        for ph in chunker.feed(d):
                            spawn(ph)
                    for ph in chunker.flush():
                        spawn(ph)
                results = await asyncio.wait_for(asyncio.gather(*tasks), REMOTE_TTS_TIMEOUT) if tasks else []
                if emitter is not None:
                    emit_q.put_nowait(None)
                    await asyncio.wait_for(emitter, REMOTE_TTS_TIMEOUT + 120.0)   # playback of queued segments
            except BaseException:
                await _cancel_all(tasks + ([emitter] if emitter is not None else []))   # never leave work running
                raise
            rate = max((r for _, r in results), default=DEFAULT_PCM_RATE)
            gap = b"\x00\x00" * int(rate * 0.06)     # 60 ms breath between phrases
            pcm = gap.join(_resample_pcm(p, r, rate) for p, r in results if p)
        self.last_stats = {"transport": transport, "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                           "firstAudioMs": round(first_audio_at[0] * 1000.0, 1) if first_audio_at else None,
                           "audioBytes": len(pcm)}
        return "".join(full), pcm, rate

    async def save_wav(self, pcm: bytes, rate: int, prefix: str = "tts") -> Path | None:
        if not pcm or self.audio_dir is None:
            return None
        path = self.audio_dir / f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        return await asyncio.to_thread(write_wav, path, pcm, rate)
