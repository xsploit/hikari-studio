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
import copy
import ctypes
import json
from pathlib import Path
import re
import struct
import sys
import threading
import time
from typing import Any
import uuid

from aiohttp import WSMsgType, web

from core.config import (
    ConfigStore, EXPRESSION_PRESETS, EYE_STYLE_PRESETS, ROOM_AMBIENT_MAP, SHADER_PRESETS, mask_secrets, public_config,
    sanitize_ai_brain_patch, sanitize_lipsync_patch, sanitize_memory_patch, secrets_set,
)
from core.customizer import CharacterCustomizer
from core.engine_bridge import NativeEngineBridge
from core.events import HookRegistry
from core.plugins import PluginManager
from core.rig_controller import EMOTE_CATALOG, RigController
from core.wardrobe import WARDROBE_STYLES
from modules import http_pool, server_keys
from modules.ai_companion import BRAIN_MODES, AICompanion, HIKARI_PERSONALITY, TAG_RE, TAG_TO_STATE, parse_emotion_and_clean
from modules.decision_engine import DECISION_DEFAULTS, DECISION_MODES, EMOTIONS, Decision, DecisionEngine
from modules.fish_duplex import FishDuplexController
from modules.fish_tts_stream import FISH_TRANSPORTS
from modules.lipsync import LiveLipSyncEngine
from modules.sleep_memory import COLAB_BASE_MODELS, MEMORY_KINDS, SleepMemory
from modules.vercel_gateway import complete_openai_compat, local_backend_hint, local_chat_request
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


@web.middleware
async def security_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Origin check for the WebSocket and every state-changing request; upload errors -> JSON."""
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

    async def _idle_decision_loop(self) -> None:
        """Gentle autonomous idle moments; postponed while she is speaking, thinking or in a call turn."""
        while not self._stop_evt.is_set():
            await asyncio.sleep(4.0)
            try:
                if not self.ws_clients:
                    self.decisions.note_activity()   # nobody watching: don't bank idle time
                    continue
                busy = self._ai_busy or self.lipsync.is_speaking() or bool(self.lipsync.fish_duplex_active) \
                    or bool(self.rig.emote_status()["active"]) or bool(self.rig.motion_status().get("playing"))
                d = self.decisions.idle_tick(busy=busy)
                if d is not None:
                    await self.apply_decision(d, temp_action=bool(d.emote), hold=3.4)
                # SLEEP auto-consolidation after a long quiet stretch (runs in the background)
                if not busy and self.memory.should_auto_sleep(time.monotonic() - self._last_user_activity):
                    if self._sleep_task is None or self._sleep_task.done():
                        self._sleep_task = self.spawn(self.run_sleep(reason="idle"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Hikari Studio] Idle decision error: {exc}", flush=True)

    def stop_loop(self) -> None:
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

    def _track_send(self, ws: Any, fut: Any) -> None:
        def _done(f: Any) -> None:
            if not f.cancelled() and f.exception() is not None:
                self.ws_clients.discard(ws)          # dead socket
                self._client_sends.pop(ws, None)
                self._slow_clients.discard(ws)
        fut.add_done_callback(_done)

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
                fut = asyncio.ensure_future(ws.send_bytes(pkt))
                self._client_sends[ws] = fut
                self._track_send(ws, fut)
                if ws in self._slow_clients:
                    fut.add_done_callback(lambda f, w=ws: self._slow_clients.discard(w))
                else:
                    awaited.append((ws, fut))
            if awaited:
                _done, pending = await asyncio.wait([f for _, f in awaited], timeout=0.25)
                for ws, fut in awaited:
                    if fut in pending:
                        self._slow_clients.add(ws)
            for ws in list(self._client_sends):
                if ws not in self.ws_clients:
                    self._client_sends.pop(ws, None)
                    self._slow_clients.discard(ws)
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
                         wav_path: Path | None = None) -> dict[str, Any]:
        clean, parsed_tag, parsed_act = parse_emotion_and_clean(text)
        tag = tag or parsed_tag
        act = action or parsed_act
        self._subtitle = clean
        if wav_path is None:
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
    async def memory_context(self, query: str) -> str:
        if not self.memory.cfg.get("enabled", True):
            return ""
        try:
            mems = await self.memory.recall(query)
            dreams = await self.memory.take_dreams()
            return self.memory.context_block(mems, dreams)
        except Exception as exc:
            print(f"[Hikari Studio] Memory recall error: {exc.__class__.__name__}", flush=True)
            return ""

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
        else:
            vals = raw
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

    async def api_chat(req):
        body = await req.json()
        msg = str(body.get("message") or "").strip()
        if not msg:
            return web.json_response({"ok": False, "error": "Empty message"}, status=400)
        payload = {"message": msg}
        await runtime.hooks.emit_async("on_user_message", runtime, payload)
        # She reacts to what you said right away (e.g. waves at "hey!") while she thinks of a reply.
        user_task = asyncio.create_task(runtime.decide_and_apply(payload["message"], "user"))
        runtime._last_user_activity = time.monotonic()
        runtime._chat_inflight = getattr(runtime, "_chat_inflight", 0) + 1
        runtime._ai_busy = True
        pre_voiced: Path | None = None
        try:
            async with chat_lock:                  # history: user turn + reply stay paired and in order
                context = await runtime.memory_context(payload["message"])     # wake: top-5 recall + dreams
                if runtime.ai.voices_inline:
                    reply, pre_voiced = await runtime.ai.generate_reply_with_voice(payload["message"], context)
                else:
                    reply = await runtime.ai.generate_reply(payload["message"], context)
        finally:
            runtime._chat_inflight -= 1
            runtime._ai_busy = runtime._chat_inflight > 0
        runtime.spawn(runtime.memory.wake(payload["message"], reply, "chat"))
        user_decision = await user_task
        clean, tag, act = parse_emotion_and_clean(reply)
        try:
            reply_decision = await runtime.decisions.decide(reply, "reply")
        except Exception as exc:
            print(f"[Hikari Studio] Reply decision error: {exc}", flush=True)
            reply_decision = Decision(source="reply", tag=tag, skipped="error")
        if not TAG_RE.search(reply):
            # No explicit cue in the reply: the Decision Model owns her mood.
            tag = reply_decision.tag
            act = TAG_TO_STATE.get(tag, TAG_TO_STATE["happy"])
        await runtime.hooks.emit_async("on_ai_reply", runtime, {"reply": clean, "tag": tag, "action": act})
        evt = await runtime.speak_text(reply, action=act, decision=reply_decision, tag=tag, wav_path=pre_voiced)
        return web.json_response({
            "ok": True, "reply": clean, "tag": tag, "audioUrl": evt["audioUrl"], "history": runtime.ai.history[-24:],
            "emote": reply_decision.emote or "", "decisions": {"user": user_decision.to_dict(), "reply": reply_decision.to_dict()},
            "brainMode": runtime.ai.brain_mode, "memories": runtime.memory.last_recall,
            "recallMs": round(runtime.memory.last_recall_ms, 3), "turn": runtime.ai.last_turn if pre_voiced else {},
        })

    async def api_memory_status(_req):
        st = await runtime.memory.status(str(runtime.config.data["ai"].get("localLoraPath") or ""))
        st["colabNotebook"] = str(COLAB_NOTEBOOK)
        return web.json_response({"ok": True, **st})

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

    async def api_interrupt(_req):
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
        body = await req.json()
        results = await runtime.hooks.emit_async("on_external_trigger", runtime, body)
        return web.json_response({"ok": True, "results": results})

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
            if sec == "lipsync" and isinstance(val, dict):
                val = sanitize_lipsync_patch(val)      # voice calibration is per-voice, not per-preset
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
        t0 = time.perf_counter()
        config = await runtime.apply_customizer(body)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return web.json_response({"ok": True, "elapsedMs": elapsed_ms, "customizer": config})

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
        mode = str(body.get("mode") or runtime.config.data["ai"].get("fishAgentMode") or "drama-3")
        try:
            # Hikari's verbatim personality rides along as the per-session system prompt override.
            runtime.flush_fish_pending()
            mem_ctx = await runtime.memory_context("hello again, what do you remember about me")
            session = await runtime.fish_duplex.start_session(
                mode=mode, persona=(f"{runtime.ai.persona}\n\n{mem_ctx}" if mem_ctx else runtime.ai.persona),
                override_persona=bool(runtime.config.data["ai"].get("fishPersonaOverride", True)),
            )
            runtime.decisions.note_activity()
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
        return web.json_response(res, status=409 if res.get("stale") else 200)

    async def api_emote(req):
        """Trigger a smooth cubic/spring tweened Live2D emote clip."""
        body = await req.json()
        eid = str(body.get("emote") or "").strip()
        dur = float(body.get("duration") or 2.6)
        res = runtime.rig.trigger_emote(eid, duration=dur)
        if res.get("ok"):
            runtime.decisions.note_manual_emote(res["emote"])   # manual choice wins; autonomy waits its cooldown
        return web.json_response(res, status=200 if res.get("ok") else 400)

    async def api_fish_emote(req):
        """Parse incoming live Fish Agent transcript to trigger matching anime expressions & gestures!"""
        body = await req.json()
        text = str(body.get("text") or "").strip()
        role = str(body.get("role") or "agent").lower()
        source = "fish_user" if role in ("user", "you", "human", "viewer") else "fish_agent"
        if text:
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
        try:
            text = await runtime.fish_duplex.transcribe_audio(raw, filename=fname, mime=mime)
            return web.json_response({"ok": True, "text": text})
        except Exception as exc:
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
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                t = data.get("type")
                if t == "pointer":
                    runtime.rig.set_pointer(data.get("x", 0.0), data.get("y", 0.0))
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
                    runtime.lipsync.sync_playback_time(float(data.get("time", 0.0)))
                elif t == "override_clear":
                    runtime.config.data["rig"]["overrides"] = {}
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
        try:
            await asyncio.wait_for(runtime.fish_duplex.end_session(), 5.0)
        except Exception as exc:
            print(f"[Hikari Studio] Fish session end on shutdown failed: {exc.__class__.__name__}", flush=True)
        runtime.stop_loop()
        await runtime.memory.close()
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
