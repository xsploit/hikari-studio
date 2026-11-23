"""Hikari Studio — Modular Real-Time 60 FPS AI VTuber Server.
Thin orchestration layer connecting:
  - core.config (ConfigStore & Presets)
  - core.events / core.plugins (HookRegistry & Auto-Discovered Plugins)
  - core.texture_upscaler (2x Anime Lineart HD Model Builder)
  - core.engine_bridge (Native OpenGL 3.2 + GLSL 330 Live Renderer)
  - core.rig_controller (60 FPS Rig & Physics Compositor)
  - modules.lipsync / modules.video_retarget / modules.ai_companion
"""
from __future__ import annotations

import asyncio
import colorsys
import copy
import ctypes
import json
import math
from pathlib import Path
import re
import struct
import sys
import threading
import time
from typing import Any
import uuid

from aiohttp import WSMsgType, web
from aiohttp.web_runner import GracefulExit

from core.config import (
    FRAME_SECTIONS, ConfigStore, EXPRESSION_PRESETS, EYE_STYLE_PRESETS, ROOM_AMBIENT_MAP, SHADER_PRESETS, default_config,
    mask_secrets, public_config, sanitize_ai_brain_patch, sanitize_lipsync_patch, sanitize_memory_patch,
    sanitize_section, secrets_set,
)
from core.customizer import CharacterCustomizer
from core.engine_bridge import NativeEngineBridge
from core.events import HookRegistry
from core.plugins import PluginManager
from core.rig_controller import EMOTE_CATALOG, RigController
from core.wardrobe import WARDROBE_STYLES
from modules import http_pool, server_keys
from modules.ai_companion import (
    BRAIN_MODES, MAX_CHAT_CHARS, AICompanion, HIKARI_PERSONALITY, TAG_RE, TAG_TO_STATE, parse_emotion_and_clean,
)
from modules.decision_engine import (
    DECISION_DEFAULTS, DECISION_MODES, EMOTIONS, Decision, DecisionEngine, classify_text,
)
from modules.fish_duplex import FishDuplexController, compose_session_prompt
from modules.fish_tts_stream import FISH_TRANSPORTS, clean_for_speech
from modules.lipsync import LiveLipSyncEngine
from modules.sleep_memory import COLAB_BASE_MODELS, MEMORY_KINDS, SleepMemory
from modules.vercel_gateway import (
    catalog_entries, complete_openai_compat, evaluation_catalog, local_backend_hint, local_chat_request,
)
from modules.video_retarget import extract_motion_from_video

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB_DIR = HERE / "web"
MODELS_DIR = HERE / "models"
MOTIONS_DIR = HERE / "motions"
UPLOADS_DIR = HERE / "uploads"
AUDIO_DIR = UPLOADS_DIR / "audio"
MEMORY_DIR = HERE / "data" / "memory"
COLAB_NOTEBOOK = HERE / "tools" / "hikari_sleep_colab_lora.ipynb"
CONFIG_SECTIONS = ("stage", "camera", "shader", "rig", "lipsync", "visemes", "ai", "memory")
TMP_DIR = HERE / "data" / "tmp"                      # private (never served)
CUSTOM_ASSETS_DIR = HERE / "data" / "custom_assets"  # validated eye / hair images (never served)
MAX_VIDEO_BYTES = 120 * 1024 * 1024
MAX_MODEL_BYTES = 120 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
IMAGE_EXT = {"PNG": ".png", "WEBP": ".webp", "JPEG": ".jpg"}
MOTION_ID_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_.\-]{0,127}$")
# Browser origins allowed to drive the control surface (WebSocket + state-changing requests).
ALLOWED_ORIGIN_RE = re.compile(
    r"^(?:http://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?|tauri://localhost|https?://tauri\.localhost)$")
# DNS-rebinding guard: the Host header must name loopback (optionally with the port we are listening on).
HOST_RE = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(?::(\d{1,5}))?$")
DEFAULT_PORT = 8800

# Request guards
EMOTE_MIN_SEC, EMOTE_MAX_SEC = 0.3, 15.0
HOOK_TEXT_MAX = 4000
WS_ERROR_REPLY_GAP_SEC = 1.0                       # at most one malformed-message reply per second per socket
SLOW_SEND_SEC = 0.080                              # a viewer is "fast" again only after 3 sends under 80 ms
SLOW_RECOVER_STREAK = 3
FRAME_TEMPLATES = {sec: copy.deepcopy(default_config().get(sec) or {}) for sec in FRAME_SECTIONS if sec != "visemes"}

ENGINE_EXE = HERE / "engine" / "hikari-live-engine.exe"
SDL_DLL = Path("D:/Hikari-Rig-Tools/Inochi-Creator-0.8.6/SDL2.dll")
SRC_MODEL = ROOT / "v28-inochi" / "Hikari-v28.inp"
HD_MODEL = MODELS_DIR / "Hikari-v28-HD.inp"
SAMPLE_WAV = ROOT / "v23" / "watch" / "dialogue.wav"
SAMPLE_CUES = ROOT / "v23" / "watch" / "dialogue.cues.json"
DEMO_VIDEO = ROOT / "v28-inochi" / "Hikari-v28-Flagship-HD.mp4"


_WIN_TIMER_ON = False


def win_timer_begin() -> bool:
    """timeBeginPeriod(1): 1 ms OS scheduler granularity for the 60 FPS pacer (Windows only)."""
    global _WIN_TIMER_ON
    if sys.platform == "win32" and not _WIN_TIMER_ON:
        try:
            _WIN_TIMER_ON = ctypes.windll.winmm.timeBeginPeriod(1) == 0   # TIMERR_NOERROR
        except Exception:
            _WIN_TIMER_ON = False
    return _WIN_TIMER_ON


def win_timer_end() -> None:
    global _WIN_TIMER_ON
    if sys.platform == "win32" and _WIN_TIMER_ON:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        _WIN_TIMER_ON = False


SPIN_TAIL_SEC = 0.0004


def pace_until(deadline: float, spin_tail: float = SPIN_TAIL_SEC) -> None:
    """Hybrid frame pacer: coarse sleep until ~1.5 ms before the deadline, then a short
    GIL-yielding spin (sleep(0)) so the frame lands on time without burning a core."""
    remaining = deadline - time.perf_counter()
    if remaining > spin_tail:
        time.sleep(remaining - spin_tail)
    while time.perf_counter() < deadline:
        time.sleep(0)


class UploadTooLarge(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f"Upload exceeds the {limit // (1024 * 1024)} MiB limit")
        self.limit = limit


class BadUpload(Exception):
    pass


async def _save_upload(field: Any, dest: Path, max_bytes: int, chunk_size: int = 256 * 1024) -> int:
    """Stream a multipart field to disk off the event loop, enforcing `max_bytes` while streaming.
    On overflow (or any failure) the partial file is deleted and UploadTooLarge -> HTTP 413."""
    total = 0
    f = await asyncio.to_thread(dest.open, "wb")
    ok = False
    try:
        while True:
            chunk = await field.read_chunk(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge(max_bytes)
            await asyncio.to_thread(f.write, chunk)
        ok = True
        return total
    finally:
        await asyncio.to_thread(f.close)
        if not ok:
            await asyncio.to_thread(dest.unlink, True)


def _validate_image(path: Path) -> str:
    """Decode-check an uploaded image with Pillow; returns the safe extension for its REAL format."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            fmt = (im.format or "").upper()
            im.verify()
        with Image.open(path) as im:            # verify() invalidates; reopen to force a full decode
            im.load()
            if im.width * im.height > 64_000_000:
                raise BadUpload("Image is too large")
    except BadUpload:
        raise
    except Exception:
        raise BadUpload("Not a valid PNG, WebP or JPEG image") from None
    if fmt not in IMAGE_EXT:
        raise BadUpload("Only PNG, WebP or JPEG images are accepted")
    return IMAGE_EXT[fmt]


async def _receive_image(field: Any, final_dir: Path, stem: str) -> Path:
    """Private temp file -> Pillow validation -> final file with a whitelisted extension."""
    await asyncio.to_thread(TMP_DIR.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(final_dir.mkdir, parents=True, exist_ok=True)
    tmp = TMP_DIR / f"up_{uuid.uuid4().hex}.part"
    await _save_upload(field, tmp, MAX_IMAGE_BYTES)
    try:
        ext = await asyncio.to_thread(_validate_image, tmp)
        dest = final_dir / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        await asyncio.to_thread(tmp.replace, dest)
        return dest
    finally:
        await asyncio.to_thread(tmp.unlink, True)


def _motion_path(cid: str, suffix: str) -> Path | None:
    """Motion clip path strictly contained in MOTIONS_DIR (no separators, no '..')."""
    if not MOTION_ID_RE.fullmatch(cid) or ".." in cid or "/" in cid or "\\" in cid:
        return None
    root = MOTIONS_DIR.resolve()
    path = (MOTIONS_DIR / f"{cid}{suffix}").resolve()
    return path if path.is_relative_to(root) and path.parent == root else None


def _origin_allowed(origin: str | None) -> bool:
    """Missing Origin = non-browser / local tool; otherwise only loopback + the Tauri shell."""
    return origin is None or origin == "" or bool(ALLOWED_ORIGIN_RE.fullmatch(origin))


def _local_port(request: web.Request) -> int | None:
    """The port this request actually arrived on (the listening socket), not what the client claims."""
    try:
        sock = request.transport.get_extra_info("sockname") if request.transport is not None else None
        return int(sock[1]) if sock else None
    except Exception:
        return None


def _host_allowed(host: str | None, port: int | None) -> bool:
    """DNS-rebinding guard. Accept only loopback names: `127.0.0.1`, `localhost`, `[::1]`, either bare or with
    the port we are listening on. A rebound page always sends its own name (`evil.example:8800`) -> rejected."""
    if not host:
        return False
    m = HOST_RE.fullmatch(host.strip().lower())
    if m is None:
        return False
    if m.group(2) is None:
        return True
    want = port if port is not None else DEFAULT_PORT
    return int(m.group(2)) == want


@web.middleware
async def security_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Host check (DNS rebinding) on EVERY request incl. reads and the WebSocket; Origin check for the
    WebSocket and every state-changing request; upload errors -> JSON."""
    if not _host_allowed(request.headers.get("Host"), _local_port(request)):
        return web.json_response({"ok": False, "error": "Host not allowed"}, status=403)
    is_ws = request.headers.get("Upgrade", "").lower() == "websocket"
    if (is_ws or request.method not in ("GET", "HEAD", "OPTIONS")) and not _origin_allowed(request.headers.get("Origin")):
        return web.json_response({"ok": False, "error": "Origin not allowed"}, status=403)
    try:
        return await handler(request)
    except UploadTooLarge as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=413)
    except BadUpload as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=415)


def _prune_audio_cache(keep: int = 20) -> None:
    """Keep only the newest TTS audio files so uploads/audio never grows unbounded."""
    try:
        files = sorted(
            (p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".ogg", ".webm")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _read_motion_index() -> list[dict[str, Any]]:
    motions = []
    for jf in sorted(MOTIONS_DIR.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            motions.append({k: v for k, v in d.items() if k != "tracks"})
        except Exception:
            pass
    return motions


def _read_param_meta(inp_path: Path) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    raw = inp_path.read_bytes()
    jlen = int.from_bytes(raw[8:12], "big")
    model = json.loads(raw[12 : 12 + jlen].decode("utf-8"))
    defaults: dict[str, tuple[float, float]] = {}
    meta: list[dict[str, Any]] = []
    for p in model["param"]:
        d = p.get("defaults", [0.0, 0.0])
        defaults[p["name"]] = (float(d[0]), float(d[1]))
        if p.get("bindings"):
            meta.append({
                "name": p["name"],
                "isVec2": bool(p.get("is_vec2", False)),
                "min": [float(p["min"][0]), float(p["min"][1])],
                "max": [float(p["max"][0]), float(p["max"][1])],
                "defaults": [float(d[0]), float(d[1])],
            })
    return defaults, meta


class SegmentPlayer:
    """Plays one reply's sentences back to back as Fish TTS finishes each one, so her voice starts with
    sentence 1 instead of after the whole reply is synthesized. Each segment is a normal `speech` event
    (the stage swaps the <audio> source); the next one is sent when the previous one has finished."""

    GAP_SEC = 0.04

    def __init__(self, runtime: "StudioRuntime") -> None:
        self.rt = runtime
        self.q: asyncio.Queue = asyncio.Queue()
        self.started = asyncio.Event()           # first segment is playing
        self.segments = 0
        self.end_at = 0.0                        # monotonic time the last queued-and-started segment ends
        self.action: dict[str, Any] | None = None
        self.tag = "hikari"
        self._closed = False
        self.task = runtime.spawn(self._run())

    async def push_pcm(self, pcm: bytes, rate: int, text: str, _idx: int) -> None:
        """`on_segment` callback for AICompanion.generate_reply_with_voice (called in order)."""
        if self._closed:
            return
        wav = await self.rt.ai.fish_tts.save_wav(pcm, rate)
        if wav is not None:
            self.q.put_nowait((wav, clean_for_speech(text)))

    def push_wav(self, wav: Path | None, text: str) -> None:
        if not self._closed:
            self.q.put_nowait((wav, clean_for_speech(text)))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.q.put_nowait(None)

    def cancel(self) -> None:
        self._closed = True
        if not self.task.done():
            self.task.cancel()

    def remaining(self) -> float:
        return max(0.0, self.end_at - time.monotonic())

    async def _run(self) -> None:
        while True:
            item = await self.q.get()
            if item is None:
                return
            wav, text = item
            wait = self.end_at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            dur, audio_url = max(1.2, len(text) / 14.0), ""
            if wav is not None and await asyncio.to_thread(wav.exists):
                dur = await asyncio.to_thread(self.rt.lipsync.load_wav_and_play, wav, None, wav.name)
                audio_url = f"/uploads/audio/{wav.name}"
            self.end_at = time.monotonic() + dur + self.GAP_SEC
            self.rt._subtitle = text
            if self.action is not None:
                self.rt.rig.trigger_temp_action(self.action, duration=max(2.6, dur + 0.6))
            await self.rt.broadcast_json({"type": "speech", "text": text, "tag": self.tag, "audioUrl": audio_url,
                                          "duration": round(dur, 2), "segment": self.segments})
            self.segments += 1
            self.started.set()
            if self.segments == 1:
                self.rt.set_interaction("idle")
        # (no audio-cache pruning mid-reply: the stage may still be fetching a queued segment)


class StudioRuntime:
    def __init__(self) -> None:
        for d in (MODELS_DIR, MOTIONS_DIR, UPLOADS_DIR, AUDIO_DIR):
            d.mkdir(parents=True, exist_ok=True)

        self.config = ConfigStore(HERE / "settings.json")
        print("[Hikari Studio] Building HD puppet with VRM Refined eyes & customizer...", flush=True)
        self.customizer = CharacterCustomizer(SRC_MODEL, MODELS_DIR, HERE / "assets" / "vrm_eyes")
        self._customizer_lock = asyncio.Lock()
        self.model_path = self.customizer.build_custom_model(self.config.data["customizer"], MODELS_DIR / "Hikari-v28-Custom.inp")
        self.param_defaults, self.param_meta = _read_param_meta(self.model_path)

        self.hooks = HookRegistry()
        self.lipsync = LiveLipSyncEngine()
        self.lipsync.update_config(self.config.data["lipsync"])
        self.lipsync.update_visemes(self.config.data["visemes"])

        self.ai = AICompanion(AUDIO_DIR)
        self.ai.update_config(self.config.data["ai"])
        self.fish_duplex = FishDuplexController()
        self.fish_duplex.set_key_override(str(self.config.data["ai"].get("fishApiKey") or ""))
        self.fish_duplex.set_persona(self.ai.persona)
        self.decisions = DecisionEngine(self.config.data["ai"])
        self._idle_task: asyncio.Task | None = None
        self.memory = SleepMemory(MEMORY_DIR, self.config.data.get("memory") or {}, gateway=self.ai.gateway,
                                  persona=self.ai.persona, decision_cfg=lambda: self.config.data["ai"])
        self._sleep_task: asyncio.Task | None = None
        self._last_user_activity = time.monotonic()
        self._fish_pending_user = ""
        self._bg_tasks: set[asyncio.Task] = set()

        self.rig = RigController(self.param_defaults, self.lipsync, self.hooks)
        self.engine = NativeEngineBridge(ENGINE_EXE, SDL_DLL, self.model_path, width=960, height=1080)
        self.engine.start()

        self.plugins = PluginManager(HERE / "plugins", self.hooks)
        self.plugins.discover_and_load(self, self.config.data.get("plugins", {}))

        self.ws_clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._latest_frame: bytes | None = None
        self._fps_measured: float = 60.0
        self._subtitle: str = ""
        self._ai_busy: bool = False
        self._chat_inflight = 0
        self._frame_in_flight = False
        self._telem_future: Any = None
        self._client_sends: dict[Any, Any] = {}
        self._slow_clients: set[Any] = set()
        self._fast_streak: dict[Any, int] = {}           # consecutive fast sends of a viewer marked slow
        self._segment_player: SegmentPlayer | None = None
        self._interaction_state = "idle"
        self._shutting_down = False
        self._shutdown_task: asyncio.Task | None = None
        self._fish_ended_on_shutdown = False
        self._loop_stopped = False
        # Mood Momentum -> body baseline (the rig API is optional: guarded so an older rig keeps working)
        self.decisions.on_mood = self._push_mood

    # ------------------------------------------------------------------ liveliness glue (rig API is optional)
    def _push_mood(self, valence: float, arousal: float) -> None:
        fn = getattr(self.rig, "set_mood_state", None) if hasattr(self, "rig") else None
        if callable(fn):
            try:
                fn(float(valence), float(arousal))
            except Exception as exc:
                print(f"[Hikari Studio] set_mood_state error: {exc.__class__.__name__}", flush=True)

    def set_interaction(self, state: str) -> None:
        """thinking (model busy) / listening (viewer talking) / idle. No-op on a rig without the API."""
        self._interaction_state = state
        fn = getattr(self.rig, "set_interaction_state", None)
        if callable(fn):
            try:
                fn(state)
            except Exception as exc:
                print(f"[Hikari Studio] set_interaction_state error: {exc.__class__.__name__}", flush=True)

    def note_stage_event(self, text: str, kind: str = "stage") -> bool:
        """Plugins / endpoints: remember a notable stage event for her next reply (memory's recent ring)."""
        try:
            return self.memory.note_stage_event(text, kind=kind)
        except Exception:
            return False

    async def apply_customizer(self, values: dict[str, Any], vrm: Path | None = None) -> dict[str, Any]:
        async with self._customizer_lock:
            config = {**self.config.data["customizer"], **values}
            if vrm is not None:
                await asyncio.to_thread(self.customizer.import_vrm, vrm)
            await asyncio.to_thread(self._reload_customizer, config)
            self.config.patch("customizer", config)
            await self.config.save_async()
            return config

    def _reload_customizer(self, config: dict[str, Any]) -> None:
        candidate = self.model_path.with_suffix(".pending.inp")
        try:
            self.customizer.build_custom_model(config, candidate)
            self.engine.reload_model(candidate)
        finally:
            candidate.unlink(missing_ok=True)

    def start_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._stop_evt.clear()
        win_timer_begin()
        self._render_thread = threading.Thread(target=self._render_worker, daemon=True, name="hikari-render")
        self._render_thread.start()
        self._idle_task = loop.create_task(self._idle_decision_loop())
        self.spawn(self.memory.init())

    def spawn(self, coro: Any) -> asyncio.Task:
        """Fire-and-forget background work that is kept alive and never raises into the loop."""
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._bg_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                print(f"[Hikari Studio] Background task error: {t.exception().__class__.__name__}", flush=True)
        task.add_done_callback(_done)
        return task

    def apply_ai_config(self, values: dict[str, Any]) -> None:
        """Push an `ai` config patch into the companion, Fish duplex and decision engine."""
        self.ai.update_config(values)
        # upgrade_persona may have restored Hikari's verbatim persona; keep the store in sync
        self.config.data["ai"]["persona"] = self.ai.cfg["persona"]
        self.config.data["ai"]["personaId"] = self.ai.cfg.get("personaId", "")
        self.fish_duplex.set_key_override(str(self.config.data["ai"].get("fishApiKey") or ""))
        self.fish_duplex.set_persona(self.ai.persona)
        self.decisions.update_config(self.config.data["ai"])
        self.memory.persona = self.ai.persona

    async def apply_decision(self, d: Decision, *, temp_action: bool = False, hold: float = 4.2) -> Decision:
        """Play a decided emote (already cooldown-gated) and tell connected clients."""
        if d.emote:
            res = self.rig.trigger_emote(d.emote)
            if not res.get("ok"):
                d.skipped, d.emote = "unknown emote", None
        if temp_action:
            self.rig.trigger_temp_action(TAG_TO_STATE.get(d.tag, TAG_TO_STATE["happy"]), duration=hold)
        await self.broadcast_json({"type": "decision", "decision": d.to_dict()})
        return d

    async def decide_and_apply(self, text: str, source: str, *, temp_action: bool = False, hold: float = 4.2) -> Decision:
        try:
            d = await self.decisions.decide(text, source)
        except Exception as exc:  # the decision layer must never break chat / calls
            print(f"[Hikari Studio] Decision error: {exc}", flush=True)
            d = Decision(source=source, skipped="error", note=exc.__class__.__name__)
        return await self.apply_decision(d, temp_action=temp_action, hold=hold)

    def _auto_sleep_check(self, busy: bool) -> bool:
        """SLEEP auto-consolidation after a long quiet stretch (background task). Runs whether or not a viewer
        tab is connected: the desktop app can sit minimised / closed-to-tray for hours."""
        if busy or self._shutting_down:
            return False
        if not self.memory.should_auto_sleep(time.monotonic() - self._last_user_activity):
            return False
        if self._sleep_task is not None and not self._sleep_task.done():
            return False
        self._sleep_task = self.spawn(self.run_sleep(reason="idle"))
        return True

    async def _idle_decision_loop(self) -> None:
        """Gentle autonomous idle moments; postponed while she is speaking, thinking or in a call turn.
        Lightweight 4 s timer: idle emotes only while someone is watching; mood push + auto-sleep always."""
        while not self._stop_evt.is_set():
            await asyncio.sleep(4.0)
            try:
                st = self.decisions.get_mood_state()          # decays toward baseline even with nobody chatting
                self._push_mood(st["valence"], st["arousal"])
                if not self.ws_clients:
                    self.decisions.note_activity()   # nobody watching: don't bank idle-emote time
                    busy = self._ai_busy or self.lipsync.is_speaking() or bool(self.lipsync.fish_duplex_active)
                    self._auto_sleep_check(busy)
                    continue
                busy = self._ai_busy or self.lipsync.is_speaking() or bool(self.lipsync.fish_duplex_active) \
                    or bool(self.rig.emote_status()["active"]) or bool(self.rig.motion_status().get("playing"))
                d = self.decisions.idle_tick(busy=busy)
                if d is not None:
                    await self.apply_decision(d, temp_action=bool(d.emote), hold=3.4)
                self._auto_sleep_check(busy)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Hikari Studio] Idle decision error: {exc}", flush=True)

    def stop_loop(self) -> None:
        """Idempotent: /api/shutdown runs it before exit, the aiohttp cleanup hook may run it again."""
        if self._loop_stopped:
            return
        self._loop_stopped = True
        self._stop_evt.set()
        if self._idle_task is not None:
            self._idle_task.cancel()
        if self._render_thread:
            self._render_thread.join(timeout=2.0)
        win_timer_end()
        self.lipsync.set_mic_enabled(False)
        self.lipsync.stop_fish_pcm_bridge()
        self.engine.stop()
        self.config.data["plugins"] = self.plugins.serialize_states()
        self.config.save()

    async def graceful_shutdown(self, exit_cb: Any = None) -> dict[str, Any]:
        """POST /api/shutdown (hikari-desktop.exe on quit). Best effort, bounded (~2 s total), never raises:
        end a live Fish call, flush the call's last viewer line, save config + plugin settings, stop the
        render thread and engine cleanly, close memory, then ask aiohttp for a clean exit."""
        report: dict[str, Any] = {}
        self._shutting_down = True
        try:
            if self._segment_player is not None:
                self._segment_player.cancel()
            self.lipsync.stop_playback()
        except Exception:
            pass
        try:
            pending = self._fish_pending_user
            self._fish_pending_user = ""
            if pending:
                await asyncio.wait_for(self.memory.wake(pending, "", "fish"), 0.6)
        except Exception:
            pass
        try:
            if self.fish_duplex.active_session_id:
                await asyncio.wait_for(self.fish_duplex.end_session(), 1.2)
                report["fishCall"] = "ended"
            else:
                report["fishCall"] = "none"
        except Exception as exc:
            report["fishCall"] = f"failed: {exc.__class__.__name__}"
        self._fish_ended_on_shutdown = True
        try:
            self.config.data["plugins"] = self.plugins.serialize_states()
            await asyncio.wait_for(self.config.save_async(), 1.0)
            report["config"] = "saved"
        except Exception as exc:
            report["config"] = f"failed: {exc.__class__.__name__}"
        try:
            await asyncio.wait_for(asyncio.to_thread(self.stop_loop), 2.5)
            report["engine"] = "stopped"
        except Exception as exc:
            report["engine"] = f"failed: {exc.__class__.__name__}"
        try:
            await asyncio.wait_for(self.memory.close(), 0.8)
        except Exception:
            pass
        print(f"[Hikari Studio] Graceful shutdown: {report}", flush=True)
        if exit_cb is not None:
            exit_cb()
        return report

    def _track_send(self, ws: Any, fut: Any) -> None:
        def _done(f: Any) -> None:
            if not f.cancelled() and f.exception() is not None:
                self.ws_clients.discard(ws)          # dead socket
                self._client_sends.pop(ws, None)
                self._slow_clients.discard(ws)
                self._fast_streak.pop(ws, None)
        fut.add_done_callback(_done)

    def _slow_send_done(self, ws: Any, t0: float, fut: Any) -> None:
        """A viewer marked slow is un-marked only after SLOW_RECOVER_STREAK consecutive sends under
        SLOW_SEND_SEC. One quick send between slow ones never makes the fast viewers wait on it again."""
        if fut.cancelled() or fut.exception() is not None:
            return
        if time.perf_counter() - t0 < SLOW_SEND_SEC:
            n = self._fast_streak.get(ws, 0) + 1
            if n >= SLOW_RECOVER_STREAK:
                self._slow_clients.discard(ws)
                self._fast_streak.pop(ws, None)
            else:
                self._fast_streak[ws] = n
        else:
            self._fast_streak[ws] = 0

    async def broadcast_json(self, payload: dict[str, Any], exclude: Any = None) -> None:
        """Concurrent to every client; never waits more than 0.25 s on any of them (sends keep draining)."""
        clients = [ws for ws in list(self.ws_clients) if ws is not exclude]
        if not clients:
            return
        txt = json.dumps(payload)
        futs = []
        for ws in clients:
            fut = asyncio.ensure_future(ws.send_str(txt))
            self._track_send(ws, fut)
            futs.append(fut)
        await asyncio.wait(futs, timeout=0.25)

    async def broadcast_frame(self, pkt: bytes) -> None:
        """Frames go to all viewers concurrently. A viewer still draining its previous frame is skipped
        (no queue growth); a viewer slower than 0.25 s is marked slow and never awaited again until it
        catches up, so one stalled OBS tab can never freeze the others."""
        try:
            clients = list(self.ws_clients)
            if not clients:
                return
            awaited = []
            for ws in clients:
                prev = self._client_sends.get(ws)
                if prev is not None and not prev.done():
                    continue                         # still draining: drop this frame for this viewer
                t_send = time.perf_counter()
                fut = asyncio.ensure_future(ws.send_bytes(pkt))
                self._client_sends[ws] = fut
                self._track_send(ws, fut)
                if ws in self._slow_clients:
                    fut.add_done_callback(lambda f, w=ws, t=t_send: self._slow_send_done(w, t, f))
                else:
                    awaited.append((ws, fut))
            if awaited:
                _done, pending = await asyncio.wait([f for _, f in awaited], timeout=0.25)
                for ws, fut in awaited:
                    if fut in pending:
                        self._slow_clients.add(ws)
                        self._fast_streak[ws] = 0
            for ws in list(self._client_sends):
                if ws not in self.ws_clients:
                    self._client_sends.pop(ws, None)
                    self._slow_clients.discard(ws)
                    self._fast_streak.pop(ws, None)
        finally:
            self._frame_in_flight = False

    def _render_worker(self) -> None:
        target_dt = 1.0 / 60.0
        last_t = time.perf_counter()
        fps_t0 = last_t
        fps_n = 0
        telem_t = last_t
        next_frame = last_t
        self._frame_in_flight = False

        while not self._stop_evt.is_set():
            if not self.ws_clients:
                time.sleep(0.05)                      # idle: ~20 wakeups/s, no rendering
                last_t = next_frame = time.perf_counter()
                continue

            t0 = time.perf_counter()
            if self._frame_in_flight or self._stop_evt.is_set():
                # Previous frame still being handed to viewers (or shutting down): don't render/encode
                # a frame nobody can take. Keep pacing; dt simply covers the skipped tick.
                next_frame += target_dt
                pace_until(max(next_frame, time.perf_counter()))
                continue
            dt = max(0.002, min(0.05, t0 - last_t))
            last_t = t0

            try:
                params, shader, camera = self.rig.step(dt, self.config.data, self)
                phys_on = bool(self.config.data["rig"].get("physicsEnabled", True))
                pkt = self.engine.render_packet(dt, camera, shader, phys_on, params, jpeg_quality=95)
                self._latest_frame = pkt

                if self._loop is not None:
                    self._frame_in_flight = True
                    asyncio.run_coroutine_threadsafe(self.broadcast_frame(pkt), self._loop)

                fps_n += 1
                if t0 - fps_t0 >= 0.5:
                    self._fps_measured = fps_n / (t0 - fps_t0)
                    fps_t0 = t0
                    fps_n = 0

                telem_busy = self._telem_future is not None and not self._telem_future.done()
                if t0 - telem_t >= 0.085 and self._loop is not None and not telem_busy:   # bounded: 1 outstanding
                    telem_t = t0
                    telem = {
                        "type": "telemetry",
                        "fps": round(self._fps_measured, 1),
                        "viseme": self.lipsync.active_viseme,
                        "axes": {k: round(v, 3) for k, v in self.lipsync.live_axes.items()},
                        "bands": [round(b, 3) for b in self.lipsync.live_bands],
                        "speaking": self.lipsync.is_speaking(),
                        "fishDuplexConnected": self.lipsync.fish_duplex_connected,
                        "fishDuplexActive": self.lipsync.fish_duplex_active,
                        "motion": self.rig.motion_status(),
                        "lipsync": self.lipsync_telemetry(),
                        "hooks": self.hooks.stats(),
                    }
                    self._telem_future = asyncio.run_coroutine_threadsafe(self.broadcast_json(telem), self._loop)
            except Exception as exc:
                print(f"[Hikari Studio] Render tick error: {exc}", flush=True)
                time.sleep(0.1)

            # Absolute-deadline pacing (no drift): sleep, then spin the last ~1.5 ms.
            next_frame += target_dt
            now = time.perf_counter()
            if next_frame < now - target_dt:          # fell >1 frame behind (stall): resync
                next_frame = now
            pace_until(next_frame)

    async def speak_text(self, text: str, action: dict[str, Any] | None = None,
                         decision: Decision | None = None, tag: str | None = None,
                         wav_path: Path | None = None, synth: bool = True) -> dict[str, Any]:
        clean, parsed_tag, parsed_act = parse_emotion_and_clean(text)
        tag = tag or parsed_tag
        act = action or parsed_act
        self._subtitle = clean
        if wav_path is None and synth:
            wav_path = await self.ai.synthesize_tts(clean)
        audio_url = ""
        dur = 3.8
        if wav_path and await asyncio.to_thread(wav_path.exists):
            dur = await asyncio.to_thread(self.lipsync.load_wav_and_play, wav_path, None, wav_path.name)
            audio_url = f"/uploads/audio/{wav_path.name}"
            self.spawn(asyncio.to_thread(_prune_audio_cache, 20))
        self.rig.trigger_temp_action(act, duration=max(2.6, dur + 0.6))
        if decision is not None:
            # The reply's autonomous emote starts together with her voice, not before TTS is ready.
            await self.apply_decision(decision)
        evt = {"type": "speech", "text": clean, "tag": tag, "audioUrl": audio_url, "duration": round(dur, 2)}
        if decision is not None:
            evt["emote"] = decision.emote or ""
        await self.broadcast_json(evt)
        return evt

    # ------------------------------------------------------------------ reply delivery (voice never waits on emotion)
    DECISION_GRACE_SEC = 0.12        # a decision this fast still lands with the first syllable

    async def _await_decision(self, task: asyncio.Task, source: str, tag: str) -> Decision:
        try:
            return await task
        except Exception as exc:  # the decision layer must never break chat
            print(f"[Hikari Studio] Reply decision error: {exc}", flush=True)
            return Decision(source=source, tag=tag, skipped="error", note=exc.__class__.__name__)

    async def _late_decision(self, task: asyncio.Task, explicit_tag: bool, speech_end: float) -> None:
        """The reply's emotion decision arrived after her voice started: apply it then (emote + mood)."""
        d = await self._await_decision(task, "reply", "happy")
        if not explicit_tag:
            hold = max(1.5, speech_end - time.monotonic() + 0.6)
            self.rig.trigger_temp_action(TAG_TO_STATE.get(d.tag, TAG_TO_STATE["happy"]), duration=hold)
        await self.apply_decision(d)

    async def deliver_reply(self, reply: str, decision_task: asyncio.Task, *, wav_path: Path | None = None,
                            player: SegmentPlayer | None = None) -> tuple[dict[str, Any], Decision | None]:
        """Speak a finished reply. The emotion decision (up to decisionTimeoutMs) runs CONCURRENTLY with TTS
        synthesis / segment playback; her voice starts as soon as audio is ready and the decision's emote +
        mood land when it arrives. An explicit [tag] in the reply always wins over the decision's tag."""
        clean, parsed_tag, parsed_act = parse_emotion_and_clean(reply)
        explicit = bool(TAG_RE.search(reply))
        if player is None and wav_path is None:
            try:
                wav_path = await self.ai.synthesize_tts(clean)          # decision_task keeps running meanwhile
            except Exception as exc:
                print(f"[Hikari Studio] TTS error: {exc.__class__.__name__}", flush=True)
                wav_path = None
        if not decision_task.done():
            await asyncio.wait({decision_task}, timeout=self.DECISION_GRACE_SEC)
        decision = await self._await_decision(decision_task, "reply", parsed_tag) if decision_task.done() else None
        tag, act = parsed_tag, parsed_act
        if decision is not None and not explicit:
            tag = decision.tag
            act = TAG_TO_STATE.get(tag, TAG_TO_STATE["happy"])
        await self.hooks.emit_async("on_ai_reply", self, {"reply": clean, "tag": tag, "action": act})
        if player is not None:
            player.action, player.tag = act, tag
            unvoiced = str((self.ai.last_turn or {}).get("unvoiced") or "").strip()
            if unvoiced:                       # Fish stream broke (or never started): voice the rest classically
                try:
                    wav = await self.ai.synthesize_tts(unvoiced)
                except Exception:
                    wav = None
                player.push_wav(wav, unvoiced)
            player.close()
            est = max(2.6, player.remaining() + 0.6)
            self.rig.trigger_temp_action(act, duration=est)
            if decision is not None:
                await self.apply_decision(decision)
            evt = {"type": "speech", "text": clean, "tag": tag, "audioUrl": "", "segmented": True,
                   "duration": round(est, 2)}
            speech_end = time.monotonic() + est
        else:
            evt = await self.speak_text(reply, action=act, decision=decision, tag=tag, wav_path=wav_path, synth=False)
            speech_end = time.monotonic() + float(evt.get("duration") or 3.8)
        if decision is None:
            self.spawn(self._late_decision(decision_task, explicit, speech_end))
        return evt, decision

    def new_segment_player(self) -> SegmentPlayer:
        """One reply at a time: a new reply cuts off the previous one's queued sentences."""
        if self._segment_player is not None:
            self._segment_player.cancel()
        self._segment_player = SegmentPlayer(self)
        return self._segment_player

    async def full_state_async(self) -> dict[str, Any]:
        motions = await asyncio.to_thread(_read_motion_index)
        return self.full_state(motions)

    def full_state(self, motions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if motions is None:
            motions = _read_motion_index()
        return {
            "config": public_config(self.config.data),
            "shaderPresets": SHADER_PRESETS,
            "expressionPresets": list(EXPRESSION_PRESETS.keys()),
            "eyeStylePresets": EYE_STYLE_PRESETS,
            "roomAmbient": ROOM_AMBIENT_MAP,
            "paramsMeta": self.param_meta,
            "plugins": self.plugins.list_metadata(),
            "pluginErrors": self.plugins.load_errors,
            "hooks": self.hooks.stats(),
            "motions": motions,
            "motionStatus": self.rig.motion_status(),
            "history": self.ai.history[-24:],
            "engineSize": [self.engine.width, self.engine.height],
            "fishStatus": {
                "keyReady": self.ai.fish_key_ready,
                "activeSession": bool(self.fish_duplex.active_session_id),
                "activeMode": self.fish_duplex.active_mode,
                "duplexActive": self.lipsync.fish_duplex_active,
                "personaSynced": self.fish_duplex.persona_synced,
                "personaNote": self.fish_duplex.last_persona_note,
            },
            "persona": self.persona_info(),
            "emotes": emote_list(),
            "emotions": list(EMOTIONS.keys()),
            "decision": self.decisions.status(),
            "lipsyncStatus": self.lipsync_status(),
            "brain": self.brain_status(),
        }

    def brain_status(self) -> dict[str, Any]:
        ai = self.config.data["ai"]
        return {
            "brainMode": self.ai.brain_mode,
            "brainModes": list(BRAIN_MODES),
            "fishTtsTransport": self.ai.fish_tts.transport,
            "fishTtsTransports": list(FISH_TRANSPORTS),
            "keys": server_keys.availability(),          # booleans only, never values
            "vercelKeySet": bool(str(ai.get("vercelApiKey") or "").strip()),
            "vercelConfigured": self.ai.gateway.configured,
            "fishTtsConfigured": self.ai.fish_tts.configured,
            "lastTurn": self.ai.last_turn,
            "local": local_backend_hint(ai),
            "colabNotebook": str(COLAB_NOTEBOOK),
            "colabBaseModels": list(COLAB_BASE_MODELS),
        }

    # ------------------------------------------------------------------ SLEEP memory
    async def prompt_context(self, query: str, *, session_start: bool | None = None, greeting: bool = False) -> Any:
        """Safe dynamic context (pinned core facts + now line + recalled notes + one optional session hook).
        Building it changes NOTHING (no dream / recap is used up): commit_context() does that once the reply
        or call really happened. None when memory is off or failed (chat never breaks on memory)."""
        if not self.memory.cfg.get("enabled", True):
            return None
        try:
            return await self.memory.build_prompt_context(query, session_start=session_start,
                                                          mood=self.decisions.get_mood_state(), greeting=greeting)
        except Exception as exc:
            print(f"[Hikari Studio] Memory recall error: {exc.__class__.__name__}", flush=True)
            return None

    def commit_context(self, ctx: Any, reply_text: str = "", delivered: bool = True) -> None:
        if ctx is not None:
            self.spawn(self.memory.commit_prompt_context(ctx, reply_text, delivered=delivered))

    async def memory_context(self, query: str) -> str:
        """Compatibility for plugins: the dynamic block only (no state change)."""
        ctx = await self.prompt_context(query)
        return ctx.dynamic if ctx is not None else ""

    def _sleep_llm(self):
        """Consolidator: Vercel Gateway if configured, else local 4B in local mode, else deterministic."""
        if self.ai.gateway.configured:
            async def via_gateway(msgs):
                return await self.ai.gateway.complete(msgs, json_mode=True, max_tokens=900, temperature=0.3)
            return via_gateway, "vercel"
        if self.ai.brain_mode == "local_4b_fish":
            async def via_local(msgs):
                endpoint, headers, payload = local_chat_request({**self.config.data["ai"], "maxTokens": 900, "temperature": 0.3}, msgs)
                payload.pop("lora", None)
                return await complete_openai_compat(endpoint, headers, payload, timeout=90.0)
            return via_local, "local_4b"
        return None, "deterministic"

    async def run_sleep(self, reason: str = "manual") -> dict[str, Any]:
        self.rig.trigger_emote("sleepy_yawn")
        self.decisions.note_manual_emote("sleepy_yawn")
        await self.broadcast_json({"type": "memory_sleep", "phase": "start", "reason": reason})
        llm, label = self._sleep_llm()
        report = await self.memory.sleep(llm, label)
        report["reason"] = reason
        await self.broadcast_json({"type": "memory_sleep", "phase": "done",
                                   "report": {k: v for k, v in report.items() if k != "newCards"}})
        return report

    def flush_fish_pending(self) -> None:
        """Log an unanswered viewer line from a live call so the viewer's side of memory is never lost."""
        pending = getattr(self, "_fish_pending_user", "")
        if pending:
            self._fish_pending_user = ""
            self.spawn(self.memory.wake(pending, "", "fish"))

    def lipsync_status(self) -> dict[str, Any]:
        st = self.lipsync.status()
        st["mapped"] = {k: (round(v, 4) if v is not None else None) for k, v in self.rig.last_mapped.items()}
        st["paramMap"] = self.config.data["lipsync"].get("paramMap", {})
        return st

    def lipsync_telemetry(self) -> dict[str, Any]:
        """Lightweight per-telemetry-tick lip-sync block (~12 Hz)."""
        return {
            "engine": self.lipsync.engine,
            "source": self.lipsync.active_source,
            "scores": {k: round(v, 3) for k, v in list(self.lipsync.live_scores.items())[:4]},
            "onset": round(self.lipsync.live_onset, 3),
            "frameMs": round(self.lipsync.frame_ms, 3),
            "mapped": {k: (round(v, 3) if v is not None else None) for k, v in self.rig.last_mapped.items()},
        }

    def persona_info(self) -> dict[str, Any]:
        is_hikari = self.ai.persona == HIKARI_PERSONALITY["system_prompt"]
        return {
            "id": HIKARI_PERSONALITY["id"] if is_hikari else "custom",
            "name": HIKARI_PERSONALITY["name"] if is_hikari else "Custom",
            "description": HIKARI_PERSONALITY["description"] if is_hikari else "",
            "verbatimHikari": is_hikari,
        }




def emote_list() -> list[dict[str, Any]]:
    original = {"excited_hop", "shy_blush", "wink_wave", "gloom_shake", "curious_lean", "chibi_pop", "pout_puff", "heart_sparkle"}
    return [
        {"id": eid, "label": m["label"], "duration": m["duration"], "category": m["category"],
         "description": m["desc"], "new": eid not in original}
        for eid, m in EMOTE_CATALOG.items()
    ]


def _finite(value: Any, default: float, lo: float, hi: float) -> float:
    """float(value) clamped to [lo, hi]; NaN / Inf / non-numbers -> default."""
    try:
        f = float(value) if not isinstance(value, bool) else default
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f)) if math.isfinite(f) else default


_COLOR_NAMES = (
    (15, "red"), (40, "orange"), (65, "yellow"), (160, "green"), (195, "teal"), (250, "blue"), (290, "purple"),
    (335, "pink"), (360, "red"),
)


def color_name(hex_color: str) -> str:
    """Rough plain-word name for a #RRGGBB color, so a stage event reads 'changed your hair to pink'."""
    try:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    except (TypeError, ValueError, IndexError):
        return "a new color"
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l > 0.9:
        return "white"
    if l < 0.12:
        return "black"
    if s < 0.15:
        return "silver" if l > 0.6 else "gray"
    deg = h * 360.0
    name = next(n for limit, n in _COLOR_NAMES if deg < limit)
    if name == "red" and l > 0.7:
        name = "pink"
    return ("light " if l > 0.72 and name not in ("pink",) else "dark " if l < 0.28 else "") + name


def _sanitize_hook_payload(body: Any) -> tuple[dict[str, Any] | None, str]:
    """/api/hook/trigger body: a JSON object with bounded text / duration / value. (payload, error)."""
    if not isinstance(body, dict):
        return None, "Expected a JSON object"
    out = {str(k)[:64]: v for k, v in list(body.items())[:32]}
    text = out.get("text")
    if text is not None:
        if not isinstance(text, str):
            return None, "text must be a string"
        if len(text) > HOOK_TEXT_MAX:
            return None, f"text is longer than {HOOK_TEXT_MAX} characters"
    if "duration" in out:
        out["duration"] = _finite(out["duration"], 3.5, EMOTE_MIN_SEC, EMOTE_MAX_SEC)
    if "value" in out:
        out["value"] = str(out["value"])[:200]
    action = str(out.get("action") or "").lower()
    if action == "expression" and out.get("value", "happy") not in EXPRESSION_PRESETS:
        return None, "Unknown expression"
    return out, ""


def _sanitize_ai_patch(values: dict[str, Any]) -> dict[str, Any]:
    """Coerce decision settings to valid values so a bad UI value can't wedge the engine."""
    out = dict(values)
    if "decisionMode" in out and str(out["decisionMode"]).lower() not in DECISION_MODES:
        out["decisionMode"] = "auto"
    elif "decisionMode" in out:
        out["decisionMode"] = str(out["decisionMode"]).lower()
    for k in ("autonomousGestures", "autoWaveOnGreeting", "autoIdleEmotes", "fishPersonaOverride"):
        if k in out:
            out[k] = bool(out[k])
    for k, default in DECISION_DEFAULTS.items():
        if k in out and isinstance(default, (int, float)) and not isinstance(default, bool):
            try:
                out[k] = max(0.0, float(out[k]))
            except (TypeError, ValueError):
                out[k] = default
    return out


def build_app(runtime: StudioRuntime) -> web.Application:
    app = web.Application(client_max_size=120 * 1024 * 1024, middlewares=[security_middleware])

    chat_lock = asyncio.Lock()
    framing_sync: dict[str, Any] = {"task": None, "sender": None}

    async def _push_framing(sender: Any) -> None:
        await asyncio.sleep(0.06)                 # coalesce slider drags (<= ~16 syncs/s)
        framing_sync["task"] = None
        await runtime.broadcast_json({"type": "config_sync", "config": public_config(runtime.config.data)},
                                     exclude=framing_sync.pop("sender", None))

    def schedule_framing_sync(sender: Any) -> None:
        """Stage / camera edits reach OBS (/obs) and other tabs right away (debounced)."""
        framing_sync["sender"] = sender
        if framing_sync.get("task") is None:
            framing_sync["task"] = asyncio.ensure_future(_push_framing(sender))

    async def apply_section(sec: str, raw: dict[str, Any]) -> None:
        """Shared by POST /api/config and the WebSocket `patch` message (same allowlist + validation)."""
        if sec == "ai":
            vals = sanitize_ai_brain_patch(_sanitize_ai_patch(raw))
        elif sec == "lipsync":
            vals = sanitize_lipsync_patch(raw)
        elif sec == "memory":
            vals = sanitize_memory_patch(raw)
        elif sec in FRAME_SECTIONS:
            # rig / camera / shader / stage / visemes feed the 60 FPS loop: type-checked against the defaults,
            # NaN / Inf / non-numeric dropped, numbers clamped, rig.overrides finite [x, y] pairs only.
            vals = sanitize_section(sec, raw, FRAME_TEMPLATES.get(sec))
        else:
            return
        if not vals:
            return
        runtime.config.patch(sec, vals)
        if sec == "lipsync":
            # may open/close PortAudio (micEnabled) -> never on the event loop
            await asyncio.to_thread(runtime.lipsync.update_config, vals)
        elif sec == "visemes":
            runtime.lipsync.update_visemes(vals)
        elif sec == "ai":
            runtime.apply_ai_config(vals)
        elif sec == "memory":
            runtime.memory.update_config(runtime.config.data["memory"])

    async def index(_req):
        return web.FileResponse(WEB_DIR / "index.html")

    async def obs_page(_req):
        return web.FileResponse(WEB_DIR / "obs.html")

    async def api_state(_req):
        return web.json_response(await runtime.full_state_async())

    async def api_patch_config(req):
        body = await req.json()
        for sec in CONFIG_SECTIONS:
            if sec in body and isinstance(body[sec], dict):
                await apply_section(sec, body[sec])
        if body.get("save", True):
            await runtime.config.save_async()
        await runtime.broadcast_json({"type": "config_sync", "config": public_config(runtime.config.data)})
        return web.json_response({"ok": True, "config": public_config(runtime.config.data), "decision": runtime.decisions.status()})

    async def api_get_config(_req):
        ai = runtime.config.data["ai"]
        return web.json_response({
            "ok": True,
            "config": public_config(runtime.config.data),
            "decisionSettings": mask_secrets({k: ai.get(k, v) for k, v in DECISION_DEFAULTS.items()}),
            "secretsSet": secrets_set(ai),            # booleans only
            "decisionModes": list(DECISION_MODES),
            "decision": runtime.decisions.status(),
            "persona": runtime.persona_info(),
            "brain": runtime.brain_status(),
            "memoryKinds": list(MEMORY_KINDS),
        })

    async def api_emotes(_req):
        return web.json_response({
            "ok": True,
            "emotes": emote_list(),
            "emotions": {k: {"tag": v["tag"], "emotes": v["emotes"], "description": v["desc"]} for k, v in EMOTIONS.items()},
            "active": runtime.rig.emote_status(),
        })

    async def api_decision_status(_req):
        return web.json_response({"ok": True, **runtime.decisions.status()})

    async def api_decider_models(req):
        """Vercel AI Gateway `type: evaluation` models (catalog cached 10 min; curated list if offline)."""
        from modules.decision_engine import decider_configured, decider_model
        cat = await evaluation_catalog(runtime.config.data["ai"], force=req.query.get("refresh") == "1")
        dec = runtime.decisions
        active = dec.resolve_decider()
        return web.json_response({
            "ok": True,
            "gatewayConfigured": decider_configured(dec.cfg, "gateway"),
            "activeMode": dec.mode,
            "activeDecider": active,
            "activeModel": dec.route_model(active),
            "gatewayEvalModel": decider_model(dec.cfg, "gateway"),
            "source": cat["source"],
            "error": cat["error"],
            "models": catalog_entries(cat["models"]),
        })

    async def api_decision_test(req):
        """Ask the Decision Engine what she'd do for a line of text. Dry run unless apply=true
        (a dry run never touches cooldowns)."""
        body = await req.json()
        text = str(body.get("text") or "").strip()[:2000]
        source = str(body.get("source") or "user")
        if source not in ("user", "reply", "fish_agent", "fish_user"):
            source = "user"
        if body.get("apply"):
            d = await runtime.decide_and_apply(text, source, temp_action=True)
            return web.json_response({"ok": True, "applied": True, "decision": d.to_dict()})
        from modules.decision_engine import classify_text
        c = classify_text(text, source)
        return web.json_response({"ok": True, "applied": False, "classification": {
            "emotion": c.emotion, "intent": c.intent, "confidence": c.confidence, "scores": c.scores,
            "tag": EMOTIONS[c.emotion]["tag"], "candidateEmotes": EMOTIONS[c.emotion]["emotes"]}})

    async def api_lipsync_status(_req):
        return web.json_response({"ok": True, **runtime.lipsync_status()})

    async def api_lipsync_calibrate(req):
        """Capture the live incoming audio's MFCCs into the user's voice calibration profile.
        Body: {"phoneme": "D"|"B"|"F"|"C"|"E"|"G"|"A"} or {"reset": true}."""
        try:
            body = await req.json()
        except Exception:
            body = {}
        if body.get("reset"):
            res = runtime.lipsync.reset_calibration()
        else:
            label = str(body.get("phoneme") or body.get("vowel") or body.get("viseme") or "")
            res = await asyncio.to_thread(runtime.lipsync.calibrate, label)   # AA/EE/OO/OH/EH/FV/MBP or D/B/F/...
            if not res.get("ok"):
                return web.json_response(res, status=400)
        # Replace (not deep-merge) so a reset really clears saved samples
        runtime.config.data["lipsync"]["mfccCalibration"] = res.get("calibration", {})
        await runtime.config.save_async()
        res["status"] = runtime.lipsync_status()
        return web.json_response(res)

    async def api_reset_visemes(_req):
        runtime.lipsync.reset_visemes()
        runtime.config.data["visemes"] = json.loads(json.dumps(runtime.lipsync.visemes))
        await runtime.config.save_async()
        await runtime.broadcast_json({"type": "config_sync", "config": public_config(runtime.config.data)})
        return web.json_response({"ok": True, "visemes": runtime.lipsync.visemes})

    async def _json_body(req) -> Any:
        try:
            return await req.json()
        except Exception:
            return None

    async def api_chat(req):
        body = await _json_body(req)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "Expected a JSON object"}, status=400)
        raw = body.get("message")
        msg = (raw if isinstance(raw, str) else str(raw or "")).strip()
        if not msg:
            return web.json_response({"ok": False, "error": "Empty message"}, status=400)
        truncated = len(msg) > MAX_CHAT_CHARS
        msg = msg[:MAX_CHAT_CHARS]                      # history itself is size-bounded (~16 KB) in AICompanion
        payload = {"message": msg}
        await runtime.hooks.emit_async("on_user_message", runtime, payload)
        msg = str(payload.get("message") or msg)[:MAX_CHAT_CHARS]
        # She reacts to what you said right away (e.g. waves at "hey!") while she thinks of a reply.
        user_task = asyncio.create_task(runtime.decide_and_apply(msg, "user"))
        greeting = classify_text(msg, "user").emotion == "greeting"
        runtime._last_user_activity = time.monotonic()
        runtime._chat_inflight = getattr(runtime, "_chat_inflight", 0) + 1
        runtime._ai_busy = True
        runtime.set_interaction("thinking")
        pre_voiced: Path | None = None
        player: SegmentPlayer | None = None
        ctx = None
        try:
            async with chat_lock:                  # history: user turn + reply stay paired and in order
                ctx = await runtime.prompt_context(msg, greeting=greeting)
                if runtime.ai.voices_inline:
                    if runtime.config.data["ai"].get("segmentedPlayback", True):
                        player = runtime.new_segment_player()
                    reply, pre_voiced = await runtime.ai.generate_reply_with_voice(
                        msg, ctx or "", on_segment=player.push_pcm if player is not None else None)
                else:
                    reply = await runtime.ai.generate_reply(msg, ctx or "")
        except Exception:
            if player is not None:
                player.cancel()
            runtime.set_interaction("idle")
            raise
        finally:
            runtime._chat_inflight -= 1
            runtime._ai_busy = runtime._chat_inflight > 0
        if not reply:
            # The turn failed: nothing is spoken or remembered as her line; no dream / recap is used up.
            if player is not None:
                player.cancel()
            runtime.set_interaction("idle")
            err = runtime.ai.last_error or "No reply"
            runtime.spawn(runtime.memory.wake(msg, "", "chat"))
            await runtime.broadcast_json({"type": "chat_error", "error": err})
            await runtime.broadcast_json({"type": "speech", "text": f"({err})", "tag": "error", "audioUrl": "",
                                          "duration": 4.0})
            user_decision = await user_task
            return web.json_response({"ok": False, "error": err, "history": runtime.ai.history[-24:],
                                      "decisions": {"user": user_decision.to_dict()}, "brainMode": runtime.ai.brain_mode},
                                     status=502)
        runtime.spawn(runtime.memory.wake(msg, reply, "chat"))
        runtime.commit_context(ctx, reply, delivered=True)
        # Emotion decision runs alongside TTS / playback; it never delays her voice.
        decision_task = asyncio.create_task(runtime.decisions.decide(reply, "reply"))
        evt, reply_decision = await runtime.deliver_reply(reply, decision_task, wav_path=pre_voiced, player=player)
        if player is None:
            runtime.set_interaction("idle")
        user_decision = await user_task          # (started with the request; awaited only after her voice started)
        clean = evt.get("text", "")
        return web.json_response({
            "ok": True, "reply": clean, "tag": evt.get("tag"), "audioUrl": evt.get("audioUrl", ""),
            "history": runtime.ai.history[-24:], "truncated": truncated,
            "emote": (reply_decision.emote or "") if reply_decision else "",
            "decisions": {"user": user_decision.to_dict(), "reply": reply_decision.to_dict() if reply_decision else None},
            "decisionPending": reply_decision is None,
            "brainMode": runtime.ai.brain_mode, "memories": runtime.memory.last_recall,
            "context": ctx.debug if ctx is not None else {},
            "recallMs": round(runtime.memory.last_recall_ms, 3),
            "turn": runtime.ai.last_turn if (pre_voiced or player is not None) else {},
        })

    async def api_memory_status(_req):
        st = await runtime.memory.status(str(runtime.config.data["ai"].get("localLoraPath") or ""))
        st["colabNotebook"] = str(COLAB_NOTEBOOK)
        st["mood"] = runtime.decisions.get_mood_state()
        st["interactionState"] = runtime._interaction_state
        return web.json_response({"ok": True, **st})

    async def api_memory_fact_pin(req):
        fid = req.match_info["fact_id"]
        body = await _json_body(req)
        if not isinstance(body, dict) or not isinstance(body.get("pinned"), bool):
            return web.json_response({"ok": False, "error": 'Expected {"pinned": true|false}'}, status=400)
        ok = await runtime.memory.set_pinned(fid, body["pinned"])
        return web.json_response({"ok": ok, "id": fid, "pinned": body["pinned"]}, status=200 if ok else 404)

    async def api_memory_recall_debug(_req):
        """Recall inspector: why each card was included / excluded last time, plus the last prompt context."""
        await runtime.memory.init()
        return web.json_response({"ok": True, **runtime.memory.recall_debug(),
                                  "stageEvents": runtime.memory.recent_stage_events(include_used=True),
                                  "mood": runtime.decisions.get_mood_state()})

    async def api_memory_sleep(_req):
        if runtime.memory.sleeping:
            return web.json_response({"ok": False, "error": "Hikari is already asleep"}, status=409)
        task = runtime.spawn(runtime.run_sleep(reason="manual"))
        runtime._sleep_task = task
        report = await task
        return web.json_response(report, status=200 if report.get("ok") else 409)

    async def api_memory_facts(_req):
        return web.json_response({"ok": True, "facts": await runtime.memory.list_facts(), "kinds": list(MEMORY_KINDS)})

    async def api_memory_fact_add(req):
        body = await req.json()
        try:
            fact = await runtime.memory.add_fact(str(body.get("kind") or "fact"), str(body.get("text") or "")[:240])
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "fact": fact})

    async def api_memory_fact_delete(req):
        fid = req.match_info["fact_id"]
        ok = await runtime.memory.delete_fact(fid)
        return web.json_response({"ok": ok, "id": fid}, status=200 if ok else 404)

    async def api_memory_export_colab(_req):
        res = await runtime.memory.export_batch()
        return web.json_response({
            "ok": True, **res, "notebook": str(COLAB_NOTEBOOK),
            "baseModel": runtime.config.data["memory"].get("colabBaseModel"),
            "instructions": "Upload sleep_batch.jsonl to MyDrive/HikariSleep/, open the notebook in Colab (T4 GPU), Run all.",
        })

    def _request_exit() -> None:
        """Clean aiohttp exit: GracefulExit raised from a loop callback unwinds web.run_app, which then runs
        the cleanup hooks (runner.cleanup -> on_cleanup). Tests replace app["exit_cb"]."""
        def _raise() -> None:
            raise GracefulExit()
        asyncio.get_running_loop().call_later(0.05, _raise)

    app["exit_cb"] = _request_exit

    async def api_shutdown(_req):
        """Called by hikari-desktop.exe on quit (it waits ~2.5 s, then force-kills). Answers immediately; the
        bounded cleanup runs in the background and ends with a clean process exit. Idempotent."""
        if runtime._shutdown_task is None:
            runtime._shutdown_task = asyncio.ensure_future(runtime.graceful_shutdown(app["exit_cb"]))
            return web.json_response({"ok": True, "shuttingDown": True})
        return web.json_response({"ok": True, "shuttingDown": True, "already": True})

    async def api_interrupt(_req):
        if runtime._segment_player is not None:
            runtime._segment_player.cancel()           # queued sentences of the current reply are dropped too
        runtime.lipsync.stop_playback()
        await runtime.broadcast_json({"type": "interrupt"})
        return web.json_response({"ok": True})

    async def api_clear_memory(_req):
        runtime.ai.clear_history()
        return web.json_response({"ok": True})

    async def api_play_sample(_req):
        if not await asyncio.to_thread(SAMPLE_WAV.exists):
            return web.json_response({"ok": False, "error": "Sample dialogue.wav missing"}, status=404)
        cues = SAMPLE_CUES if await asyncio.to_thread(SAMPLE_CUES.exists) else None
        dur = await asyncio.to_thread(runtime.lipsync.load_wav_and_play, SAMPLE_WAV, cues, "sample")
        evt = {"type": "speech", "text": "Playing Hikari sample voice take (live editable Rhubarb + VBridger visemes)...", "tag": "happy", "audioUrl": "/api/sample.wav", "duration": round(dur, 2)}
        await runtime.broadcast_json(evt)
        return web.json_response({"ok": True, "duration": dur, "audioUrl": "/api/sample.wav"})

    async def api_sample_wav(_req):
        return web.FileResponse(SAMPLE_WAV)

    async def api_upload_room(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No file"}, status=400)
        dest = await _receive_image(field, UPLOADS_DIR, "room")     # validated, safe extension
        url = f"/uploads/{dest.name}"
        runtime.config.patch("stage", {"room": "custom", "customRoomUrl": url})
        await runtime.config.save_async()
        await runtime.broadcast_json({"type": "config_sync", "config": public_config(runtime.config.data)})
        return web.json_response({"ok": True, "url": url})

    async def api_motion_extract(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No video file"}, status=400)
        orig = field.filename or "clip.mp4"
        await asyncio.to_thread(TMP_DIR.mkdir, parents=True, exist_ok=True)
        dest = TMP_DIR / f"vid_{uuid.uuid4().hex}.part"             # private temp, never served
        await _save_upload(field, dest, MAX_VIDEO_BYTES)
        stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", Path(orig).stem)[:60] or "clip"
        try:
            meta = await asyncio.to_thread(extract_motion_from_video, dest, MOTIONS_DIR, stem)
        finally:
            await asyncio.to_thread(dest.unlink, True)
        return web.json_response({"ok": True, "clip": meta})

    async def api_motion_demo(_req):
        if not await asyncio.to_thread(DEMO_VIDEO.exists):
            return web.json_response({"ok": False, "error": "Demo video not found"}, status=404)
        meta = await asyncio.to_thread(extract_motion_from_video, DEMO_VIDEO, MOTIONS_DIR, "Hikari_v28_Demo")
        return web.json_response({"ok": True, "clip": meta})

    async def api_motion_play(req):
        body = await req.json()
        cid = str(body.get("id") or "")
        if not cid:
            runtime.rig.stop_motion()
            return web.json_response({"ok": True, "status": runtime.rig.motion_status()})
        clip_path = _motion_path(cid, ".json")
        if clip_path is None:
            return web.json_response({"ok": False, "error": "Invalid clip id"}, status=400)
        if not await asyncio.to_thread(clip_path.exists):
            return web.json_response({"ok": False, "error": "Clip not found"}, status=404)
        try:
            info = await asyncio.to_thread(runtime.rig.load_motion_clip, clip_path, bool(body.get("loop", True)), float(body.get("mix", 0.85)))
        except (ValueError, json.JSONDecodeError) as exc:
            return web.json_response({"ok": False, "error": f"Invalid motion clip: {exc}"}, status=400)
        audio_url = ""
        wav_path = _motion_path(cid, ".wav")
        if info.get("hasAudio") and body.get("withAudio", False) and wav_path is not None and await asyncio.to_thread(wav_path.exists):
            await asyncio.to_thread(runtime.lipsync.load_wav_and_play, wav_path, None, cid)
            audio_url = f"/motions/{cid}.wav"
            await runtime.broadcast_json({"type": "speech", "text": f"Motion Lab: {info['name']}", "tag": "happy", "audioUrl": audio_url, "duration": info["duration"]})
        return web.json_response({"ok": True, "status": runtime.rig.motion_status(), "audioUrl": audio_url})

    async def api_motion_update(req):
        body = await req.json()
        runtime.rig.update_motion_playback(
            playing=body.get("playing"), loop=body.get("loop"), mix=body.get("mix")
        )
        return web.json_response({"ok": True, "status": runtime.rig.motion_status()})

    async def api_plugins_toggle(req):
        body = await req.json()
        pid = str(body.get("id") or "")
        if "enabled" in body:
            runtime.plugins.set_enabled(pid, bool(body["enabled"]))
        if isinstance(body.get("settings"), dict):
            runtime.plugins.update_plugin_settings(pid, body["settings"])
        runtime.config.data["plugins"] = runtime.plugins.serialize_states()
        await runtime.config.save_async()
        return web.json_response({"ok": True, "plugins": runtime.plugins.list_metadata(), "hooks": runtime.hooks.stats()})

    async def api_hook_trigger(req):
        body = await _json_body(req)
        payload, err = _sanitize_hook_payload(body)
        if payload is None:
            return web.json_response({"ok": False, "error": err}, status=400)
        try:
            results = await runtime.hooks.emit_async("on_external_trigger", runtime, payload)
            return web.json_response({"ok": True, "results": json.loads(json.dumps(results, default=str))})
        except Exception as exc:
            print(f"[Hikari Studio] Hook trigger error: {exc.__class__.__name__}: {exc}", flush=True)
            return web.json_response({"ok": False, "error": f"Hook failed: {exc.__class__.__name__}"}, status=500)

    async def api_save_preset(req):
        body = await req.json()
        name = str(body.get("name") or "").strip()[:40]
        if not name:
            return web.json_response({"ok": False, "error": "Preset name required"}, status=400)
        runtime.config.data.setdefault("userPresets", {})[name] = copy.deepcopy({
            "stage": runtime.config.data["stage"],
            "shader": runtime.config.data["shader"],
            "camera": runtime.config.data["camera"],
            "lipsync": {k: v for k, v in runtime.config.data["lipsync"].items() if k != "mfccCalibration"},
            "visemes": runtime.config.data["visemes"],
        })
        await runtime.config.save_async()
        return web.json_response({"ok": True, "userPresets": runtime.config.data["userPresets"]})

    async def api_apply_preset(req):
        body = await req.json()
        name = str(body.get("name") or "")
        preset = runtime.config.data.get("userPresets", {}).get(name)
        if not preset:
            return web.json_response({"ok": False, "error": "Preset not found"}, status=404)
        for sec, val in copy.deepcopy(preset).items():
            if sec not in CONFIG_SECTIONS or not isinstance(val, dict):
                continue
            if sec == "lipsync":
                val = sanitize_lipsync_patch(val)      # voice calibration is per-voice, not per-preset
            elif sec in FRAME_SECTIONS:
                val = sanitize_section(sec, val, FRAME_TEMPLATES.get(sec))   # an old / hand-edited preset can't break frames
            else:
                continue
            runtime.config.patch(sec, val)
            if sec == "lipsync":
                await asyncio.to_thread(runtime.lipsync.update_config, val)
            elif sec == "visemes":
                runtime.lipsync.update_visemes(val)
        await runtime.config.save_async()
        await runtime.broadcast_json({"type": "config_sync", "config": public_config(runtime.config.data)})
        return web.json_response({"ok": True, "config": public_config(runtime.config.data)})

    async def api_customizer_apply(req):
        body = await req.json()
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "Expected a settings object."}, status=400)
        for field in ("hairColor", "outfitTint"):
            if field in body and (
                not isinstance(body[field], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", body[field])
            ):
                return web.json_response({"ok": False, "error": "Enter a color in #RRGGBB format."}, status=400)
        for field, styles in WARDROBE_STYLES.items():
            if field in body and body[field] not in styles:
                return web.json_response({"ok": False, "error": "Unknown wardrobe option."}, status=400)
        for field in ("customEyePath", "customFrontHairPath"):
            if field in body and body[field]:
                try:
                    pth = Path(str(body[field])).resolve()
                    allowed = pth.is_relative_to(CUSTOM_ASSETS_DIR.resolve()) and pth.suffix.lower() in IMAGE_EXT.values()
                except (OSError, ValueError):
                    allowed = False
                if not allowed:
                    return web.json_response({"ok": False, "error": "Custom images must be uploaded first."}, status=400)
        before = dict(runtime.config.data.get("customizer") or {})
        t0 = time.perf_counter()
        config = await runtime.apply_customizer(body)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        note_customizer_events(before, config)
        return web.json_response({"ok": True, "elapsedMs": elapsed_ms, "customizer": config})

    def note_customizer_events(before: dict[str, Any], after: dict[str, Any]) -> None:
        """Notable look changes become stage events ("the viewer just changed your hair to pink")."""
        changed = [k for k in after if after.get(k) != before.get(k)]
        if "hairColor" in changed:
            runtime.note_stage_event(f"the viewer just changed your hair to {color_name(str(after['hairColor']))}", "hair")
        if "outfitTint" in changed:
            runtime.note_stage_event(f"the viewer just changed your outfit color to {color_name(str(after['outfitTint']))}", "outfit")
        styles = [k for k in changed if k in WARDROBE_STYLES]
        if styles:
            k = styles[0]
            label = re.sub(r"(?<!^)(?=[A-Z])", " ", k).lower()
            runtime.note_stage_event(f"the viewer just changed your {label} to {str(after[k]).replace('_', ' ')}", "outfit")
        if "eyeStyle" in changed or "bangsMode" in changed:
            runtime.note_stage_event("the viewer just gave you a new look", "look")

    async def api_customizer_upload_vrm(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No VRM file"}, status=400)
        await asyncio.to_thread(TMP_DIR.mkdir, parents=True, exist_ok=True)
        dest = TMP_DIR / f"import_{uuid.uuid4().hex}.vrm"
        await _save_upload(field, dest, MAX_MODEL_BYTES)
        try:
            config = await runtime.apply_customizer(
                {"eyeStyle": "vrm_refined", "eyeHue": 0.0, "eyeSat": 1.05, "eyeVal": 1.08}, vrm=dest
            )
            return web.json_response({"ok": True, "customizer": config})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        finally:
            await asyncio.to_thread(dest.unlink, True)

    async def api_customizer_upload_eye(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No image file"}, status=400)
        dest = await _receive_image(field, CUSTOM_ASSETS_DIR, "custom_eye")
        config = await runtime.apply_customizer({"eyeStyle": "custom", "customEyePath": str(dest)})
        return web.json_response({"ok": True, "customizer": config})

    async def api_customizer_front_hair_template(_req):
        tpl = HERE / "assets" / "front_hair_editable.png"
        if not await asyncio.to_thread(tpl.exists):
            return web.json_response({"ok": False, "error": "Template missing"}, status=404)
        return web.FileResponse(tpl)

    async def api_customizer_upload_front_hair(req):
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No image file"}, status=400)
        dest = await _receive_image(field, CUSTOM_ASSETS_DIR, "custom_front_hair")
        config = await runtime.apply_customizer({"bangsMode": "custom", "customFrontHairPath": str(dest)})
        return web.json_response({"ok": True, "customizer": config})

    async def api_fish_session_start(req):
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        mode = str(body.get("mode") or runtime.config.data["ai"].get("fishAgentMode") or "drama-3")
        ctx = None
        try:
            # Hikari's verbatim personality rides along as the per-session system prompt override, followed
            # by pinned core facts + the dynamic context as quoted DATA blocks (compose_session_prompt).
            runtime.flush_fish_pending()
            # Recall keyed on what the viewer said last (not a fixed sentence); none yet -> pinned + hook only.
            last_user = next((m.get("content", "") for m in reversed(runtime.ai.history) if m.get("role") == "user"), "")
            ctx = await runtime.prompt_context(str(last_user)[-500:], session_start=True)
            prompt = compose_session_prompt(runtime.ai.persona, ctx.pinned if ctx else "", ctx.dynamic if ctx else "")
            try:
                session = await runtime.fish_duplex.start_session(
                    mode=mode, persona=prompt,
                    override_persona=bool(runtime.config.data["ai"].get("fishPersonaOverride", True)),
                )
            finally:
                runtime.fish_duplex.set_persona(runtime.ai.persona)   # the per-call context never sticks
            # The call really started. The dream / recap counts as offered ONLY if Fish accepted the prompt
            # override (a rejected override means she never saw it); "mentioned" is checked on her transcript.
            runtime.commit_context(ctx, delivered=bool(runtime.fish_duplex.persona_synced))
            runtime.decisions.note_activity()
            runtime.set_interaction("listening")
            return web.json_response(session)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=502)

    async def api_fish_agent_persona(req):
        """Explicit opt-in: write Hikari's persona into the private Fish agent's draft config
        (and publish it when {"publish": true}). Sessions already get it via overrides."""
        try:
            body = await req.json()
        except Exception:
            body = {}
        mode = str(body.get("mode") or runtime.config.data["ai"].get("fishAgentMode") or "drama-3")
        try:
            res = await runtime.fish_duplex.update_agent_persona(mode=mode, persona=runtime.ai.persona,
                                                                 publish=bool(body.get("publish", False)))
            return web.json_response(res)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=502)

    async def api_fish_session_end(req):
        """End the live call. Optional {"sessionId"}: a stale tab cannot end a newer call. Reports a real
        error (502) if Fish could not confirm the shutdown; the session id is kept so it can be retried."""
        try:
            body = await req.json() if req.can_read_body else {}
        except Exception:
            body = {}
        sid = str((body or {}).get("sessionId") or req.query.get("sessionId") or "").strip() or None
        runtime.flush_fish_pending()
        try:
            res = await runtime.fish_duplex.end_session(sid)
        except Exception as exc:
            return web.json_response({"ended": False, "error": f"Fish did not confirm the end: {exc}"}, status=502)
        if res.get("ended"):
            runtime.set_interaction("idle")
        return web.json_response(res, status=409 if res.get("stale") else 200)

    async def api_emote(req):
        """Trigger a smooth cubic/spring tweened Live2D emote clip (duration clamped to 0.3..15 s)."""
        body = await _json_body(req)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "Expected a JSON object"}, status=400)
        eid = str(body.get("emote") or "").strip()[:64]
        dur = _finite(body.get("duration") or 2.6, 2.6, EMOTE_MIN_SEC, EMOTE_MAX_SEC)
        res = runtime.rig.trigger_emote(eid, duration=dur)
        if res.get("ok"):
            runtime.decisions.note_manual_emote(res["emote"])   # manual choice wins; autonomy waits its cooldown
            label = str(EMOTE_CATALOG.get(res["emote"], {}).get("label") or res["emote"]).lower()
            runtime.note_stage_event(f"the viewer just had you do the {label} emote", "emote")
        return web.json_response(res, status=200 if res.get("ok") else 400)

    async def api_fish_emote(req):
        """Parse incoming live Fish Agent transcript to trigger matching anime expressions & gestures!"""
        body = await _json_body(req)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "Expected a JSON object"}, status=400)
        text = str(body.get("text") or "").strip()[:MAX_CHAT_CHARS]
        role = str(body.get("role") or "agent").lower()
        source = "fish_user" if role in ("user", "you", "human", "viewer") else "fish_agent"
        if text:
            runtime.set_interaction("listening" if source == "fish_user" else "idle")
            clean, tag, act = parse_emotion_and_clean(text)
            # Live call turns are latency-critical: a model only if chosen explicitly or auto resolves to Clef.
            d = await runtime.decisions.decide(text, source, allow_llm=runtime.decisions.fast_model_ok())
            if not TAG_RE.search(text):
                tag = d.tag
                act = TAG_TO_STATE.get(tag, TAG_TO_STATE["happy"])
            if source == "fish_agent":
                runtime.rig.trigger_temp_action(act, duration=4.2)
            await runtime.apply_decision(d)
            # SLEEP wake log: viewer lines accumulate until Hikari answers, then the pair is logged
            # (viewer triage runs inside wake). Unanswered viewer lines are flushed on call start/end.
            if source == "fish_user":
                prev = runtime._fish_pending_user
                runtime._fish_pending_user = (f"{prev} {clean}".strip() if prev else clean)[-2000:]
                runtime._last_user_activity = time.monotonic()
            else:
                runtime.spawn(runtime.memory.wake(runtime._fish_pending_user, clean, "fish"))
                runtime._fish_pending_user = ""
                # Dream lifecycle: an offered dream becomes "mentioned" only once she actually talks about it.
                runtime.spawn(runtime.memory.check_dream_mentioned(clean))
            return web.json_response({"ok": True, "tag": tag, "clean": clean, "emote": d.emote or "", "decision": d.to_dict()})
        return web.json_response({"ok": True, "tag": "happy", "clean": "", "emote": ""})

    async def api_stt(req):
        """Ultra-fast Fish Audio ASR (transcribe-1) for push-to-talk speech-to-text."""
        reader = await req.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"ok": False, "error": "No audio uploaded"}, status=400)
        buf = bytearray()
        while True:
            ch = await field.read_chunk(256 * 1024)
            if not ch:
                break
            if len(buf) + len(ch) > MAX_AUDIO_BYTES:
                raise UploadTooLarge(MAX_AUDIO_BYTES)
            buf += ch
        raw = bytes(buf)
        del buf
        fname = field.filename or "mic.webm"
        mime = field.headers.get("Content-Type", "audio/webm")
        runtime.set_interaction("listening")
        try:
            text = await runtime.fish_duplex.transcribe_audio(raw, filename=fname, mime=mime)
            if not str(text or "").strip():
                runtime.set_interaction("idle")          # nothing heard -> no chat turn will follow
            return web.json_response({"ok": True, "text": text})
        except Exception as exc:
            runtime.set_interaction("idle")
            return web.json_response({"ok": False, "error": str(exc)}, status=502)

    async def ws_fish_pcm(req):
        """Direct zero-latency 48 kHz mono s16le PCM uplink from built-in Fish WebRTC AudioWorklet."""
        ws = web.WebSocketResponse(max_msg_size=65536)
        await ws.prepare(req)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY and msg.data:
                    runtime.lipsync.ingest_fish_pcm(msg.data)
        finally:
            pass
        return ws

    async def ws_handler(req):
        ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024, compress=False)   # frames are JPEG/PNG
        await ws.prepare(req)
        runtime.ws_clients.add(ws)
        last_err_reply = 0.0

        async def ws_error(err: str) -> None:
            """Tell the client (rate-limited) instead of dropping the socket."""
            nonlocal last_err_reply
            now = time.monotonic()
            if now - last_err_reply >= WS_ERROR_REPLY_GAP_SEC:
                last_err_reply = now
                try:
                    await ws.send_str(json.dumps({"type": "error", "error": err}))
                except Exception:
                    pass

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except (ValueError, TypeError):
                    await ws_error("malformed JSON")
                    continue
                if not isinstance(data, dict):
                    await ws_error("expected a JSON object")
                    continue
                try:
                    t = data.get("type")
                    if t == "pointer":
                        runtime.rig.set_pointer(_finite(data.get("x", 0.0), 0.0, -4.0, 4.0),
                                                _finite(data.get("y", 0.0), 0.0, -4.0, 4.0))
                    elif t == "patch":
                        sec = data.get("section")
                        vals = data.get("values")
                        if sec in CONFIG_SECTIONS and isinstance(vals, dict):
                            await apply_section(sec, vals)
                            if sec in ("stage", "camera"):
                                schedule_framing_sync(ws)
                    elif t == "save":
                        await runtime.config.save_async()
                    elif t == "interact":
                        runtime.hooks.emit_sync("on_stage_interaction", runtime, data)
                    elif t == "audio_sync":
                        runtime.lipsync.sync_playback_time(_finite(data.get("time", 0.0), 0.0, 0.0, 3600.0))
                    elif t == "override_clear":
                        runtime.config.data["rig"]["overrides"] = {}
                except Exception as exc:          # one bad message never drops the viewer's socket
                    print(f"[Hikari Studio] WS message error ({str(data.get('type'))[:24]}): {exc.__class__.__name__}", flush=True)
                    await ws_error(f"could not handle message: {exc.__class__.__name__}")
        finally:
            runtime.ws_clients.discard(ws)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/obs", obs_page)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/ws/fish-pcm", ws_fish_pcm)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/fish/session", api_fish_session_start)
    app.router.add_post("/api/fish/session/end", api_fish_session_end)
    app.router.add_post("/api/fish/end", api_fish_session_end)
    app.router.add_post("/api/emote", api_emote)
    app.router.add_get("/api/emotes", api_emotes)
    app.router.add_get("/api/config", api_get_config)
    app.router.add_get("/api/decision", api_decision_status)
    app.router.add_get("/api/deciders/models", api_decider_models)
    app.router.add_post("/api/decision/test", api_decision_test)
    app.router.add_post("/api/fish/agent/persona", api_fish_agent_persona)
    app.router.add_post("/api/fish/emote", api_fish_emote)
    app.router.add_post("/api/stt", api_stt)
    app.router.add_post("/api/customizer/apply", api_customizer_apply)
    app.router.add_post("/api/customizer/upload-vrm", api_customizer_upload_vrm)
    app.router.add_post("/api/customizer/upload-eye", api_customizer_upload_eye)
    app.router.add_get("/api/customizer/front-hair-template.png", api_customizer_front_hair_template)
    app.router.add_post("/api/customizer/upload-front-hair", api_customizer_upload_front_hair)
    app.router.add_post("/api/config", api_patch_config)
    app.router.add_post("/api/visemes/reset", api_reset_visemes)
    app.router.add_get("/api/lipsync/status", api_lipsync_status)
    app.router.add_post("/api/lipsync/calibrate", api_lipsync_calibrate)
    app.router.add_post("/api/chat", api_chat)
    app.router.add_post("/api/interrupt", api_interrupt)
    app.router.add_post("/api/memory/clear", api_clear_memory)
    app.router.add_get("/api/memory/status", api_memory_status)
    app.router.add_post("/api/memory/sleep", api_memory_sleep)
    app.router.add_get("/api/memory/facts", api_memory_facts)
    app.router.add_post("/api/memory/facts", api_memory_fact_add)
    app.router.add_delete("/api/memory/facts/{fact_id}", api_memory_fact_delete)
    app.router.add_post("/api/memory/facts/{fact_id}/pin", api_memory_fact_pin)
    app.router.add_get("/api/memory/recall-debug", api_memory_recall_debug)
    app.router.add_post("/api/shutdown", api_shutdown)
    app.router.add_post("/api/memory/export-colab", api_memory_export_colab)
    app.router.add_post("/api/audio/sample", api_play_sample)
    app.router.add_get("/api/sample.wav", api_sample_wav)
    app.router.add_post("/api/upload/room", api_upload_room)
    app.router.add_post("/api/motion/extract", api_motion_extract)
    app.router.add_post("/api/motion/demo", api_motion_demo)
    app.router.add_post("/api/motion/play", api_motion_play)
    app.router.add_post("/api/motion/update", api_motion_update)
    app.router.add_post("/api/plugins", api_plugins_toggle)
    app.router.add_post("/api/hook/trigger", api_hook_trigger)
    app.router.add_post("/api/presets/save", api_save_preset)
    app.router.add_post("/api/presets/apply", api_apply_preset)

    app.router.add_static("/static/", WEB_DIR)
    app.router.add_static("/uploads/", UPLOADS_DIR)
    app.router.add_static("/motions/", MOTIONS_DIR)

    async def on_startup(a):
        runtime.start_loop(asyncio.get_running_loop())

    async def on_cleanup(a):
        if runtime._shutdown_task is not None and not runtime._shutdown_task.done():
            try:                                    # /api/shutdown already doing the work: let it finish
                await asyncio.wait_for(asyncio.shield(runtime._shutdown_task), 3.0)
            except Exception:
                pass
        if not runtime._fish_ended_on_shutdown:
            try:
                await asyncio.wait_for(runtime.fish_duplex.end_session(), 5.0)
            except Exception as exc:
                print(f"[Hikari Studio] Fish session end on shutdown failed: {exc.__class__.__name__}", flush=True)
        runtime.stop_loop()                         # idempotent
        try:
            await runtime.memory.close()
        except Exception:
            pass                                    # already closed by /api/shutdown
        await http_pool.aclose()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    runtime = StudioRuntime()
    app = build_app(runtime)
    print("[Hikari Studio] Live at http://127.0.0.1:8800/  |  OBS Source: http://127.0.0.1:8800/obs", flush=True)
    web.run_app(app, host="127.0.0.1", port=8800, access_log=None)


if __name__ == "__main__":
    main()
