"""End-to-end wiring of Tracks 1-3 into fish_tts_stream.py, server.py and the Studio Web UI.

Fully offline: the real server.py handlers run on a random loopback port (never 8800) against TEMP settings /
profiles files; screen capture, the LLM stream, Fish voice saving and Fish synthesis are all mocked.

    python tests/test_studio_wiring.py        (or: python -m pytest tests/test_studio_wiring.py)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
from typing import Any
import unittest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import server  # noqa: E402
from core.config import ConfigStore  # noqa: E402
from core.events import HookRegistry  # noqa: E402
from core.rig_controller import RIG_FX_PARAMS, VFX_PARAMS, RigController  # noqa: E402
from modules import ai_companion, fish_voices  # noqa: E402
from modules.ai_companion import AICompanion  # noqa: E402
from modules.decision_engine import Decision, DecisionEngine  # noqa: E402
from modules.fish_duplex import FishDuplexController  # noqa: E402
from modules.fish_tts_stream import FishTtsStreamer, SentenceChunker, clean_for_speech  # noqa: E402
from modules.lipsync import PROSODY_FIELDS  # noqa: E402
from modules.profiles import ProfileStore  # noqa: E402

S = server.StudioRuntime
DRAMA = "[giggling softly, playful] Wait, you really did that? [gasps, excited] No way, that is amazing!"
FAKE_JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD"
VOICE_ID = "0123456789abcdef0123456789abcdef"
SECRET_FISH, SECRET_VERCEL = "sk-fish-SECRET-1234", "vk-vercel-SECRET-5678"


def run(coro):
    return asyncio.run(coro)


async def collect(gen) -> list:
    return [x async for x in gen]


async def deltas(text: str, step: int = 7):
    for i in range(0, len(text), step):
        yield text[i:i + step]


# ------------------------------------------------------------------------------------------ fakes
class FakeLip:
    """LiveLipSyncEngine stand-in for a real RigController (silent, no prosody)."""
    fish_duplex_active = False

    def step(self, dt: float, t_now: float) -> dict[str, float]:
        return {"open": 0.0, "form": 0.1, "funnel": 0.0, "shrug": 0.0, "x": 0.0, "tongue": 0.0, "press": 0.0,
                "energy": 0.0, "stress": 0.0, "onset": 0.0, "pitch": 0.5, "voiced": 0.0,
                **{k: 0.0 for k in PROSODY_FIELDS}}

    def is_speaking(self) -> bool:
        return False


def rig_param_defaults() -> dict[str, tuple[float, float]]:
    src = (HERE / "core" / "rig_controller.py").read_text(encoding="utf-8")
    names = set(re.findall(r'p\["([^"]+)"\]', src))
    names |= {f"Arm {s} {j}" for s in "RL" for j in ("Elbow", "Wrist")}
    names |= {"Brow L Angle", "Brow R Angle", "Cheek", "Proportion (Chibi)", "Gloom", "Emissive Glow", "Chest X",
              "Chest Y", "Eye Jelly", "Mouth Spring", "Arm R Swing", "Arm L Swing", "Lips Press / Part",
              "Shoulder R", "Shoulder L"}
    names |= set(VFX_PARAMS) | set(RIG_FX_PARAMS)
    d = {n: (0.0, 0.0) for n in names}
    d["Emissive Glow"] = (1.0, 0.0)
    d["Left Eye Open"] = d["Right Eye Open"] = (1.0, 0.0)
    return d


class FakeAI:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.history: list = []
        self.last_turn: dict = {}
        self.last_error = ""
        self.fish_key_ready = True
        self.synth_calls: list[tuple[str, str]] = []

    @property
    def persona(self) -> str:
        return str(self.cfg.get("persona") or "")

    def update_config(self, vals: dict[str, Any]) -> None:
        self.cfg.update(vals)

    async def synthesize_tts(self, text: str, *, emotion: str = "") -> None:
        self.synth_calls.append((text, emotion))
        return None


class FakeMemory:
    def __init__(self) -> None:
        self.cfg = {"enabled": True}
        self.persona = ""
        self.calls: list[dict] = []
        self.last_recall: list = []
        self.last_recall_ms = 0.0

    async def build_prompt_context(self, query: str, **kw: Any) -> None:
        self.calls.append({"query": query, **kw})
        return None

    async def commit_prompt_context(self, *a: Any, **kw: Any) -> None:
        pass

    async def close(self) -> None:
        pass


class FakeRuntime:
    """Just enough StudioRuntime for the routes under test; the wiring methods are the REAL ones."""
    apply_ai_config = S.apply_ai_config
    sync_fish_voice = S.sync_fish_voice
    fish_status = S.fish_status
    activate_profile = S.activate_profile
    _restore_idle_clip = S._restore_idle_clip
    sync_rig_flags = S.sync_rig_flags
    prompt_context = S.prompt_context
    speak_text = S.speak_text
    deliver_reply = S.deliver_reply
    _await_decision = S._await_decision
    _late_decision = S._late_decision
    library_model_path = staticmethod(S.library_model_path)
    DECISION_GRACE_SEC = S.DECISION_GRACE_SEC

    def __init__(self, tmp: Path, ai: Any = None) -> None:
        self.config = ConfigStore(tmp / "settings.json")
        a = self.config.data["ai"]
        a.update({"fishApiKey": SECRET_FISH, "vercelApiKey": SECRET_VERCEL, "segmentedPlayback": False})
        self.ai = ai if ai is not None else FakeAI(a)
        self.fish_duplex = FishDuplexController()
        self.sync_calls: list[tuple[str, bool]] = []
        real_sync = self.fish_duplex.sync_voice

        def spy(reference_id: str, cfg: dict | None = None) -> dict:
            self.sync_calls.append((reference_id, cfg is not None))
            return real_sync(reference_id, cfg)
        self.fish_duplex.sync_voice = spy                         # type: ignore[method-assign]
        self.decisions = DecisionEngine(a)
        self.memory = FakeMemory()
        self.profiles = ProfileStore(tmp / "profiles.json")
        self.rig = RigController(rig_param_defaults(), FakeLip(), HookRegistry())   # type: ignore[arg-type]
        self.lipsync = FakeLip()
        self.hooks = HookRegistry()
        self.active_model_id = "hikari"
        self.broadcasts: list = []
        self.customizer_applied: list = []
        self.selected: list = []
        self._shutdown_task = None
        self._fish_ended_on_shutdown = True
        self._chat_inflight = 0
        self._ai_busy = False
        self._last_user_activity = 0.0
        self._subtitle = ""
        self._interaction_state = "idle"
        self.voice_sync: dict = {}

    # --- runtime surface used by the routes
    def start_loop(self, loop) -> None:
        pass

    def stop_loop(self) -> None:
        pass

    async def broadcast_json(self, msg, exclude=None) -> None:
        self.broadcasts.append(msg)

    async def full_state_async(self) -> dict:
        return {"config": server.public_config(self.config.data), "profiles": self.profiles.list_profiles()}

    async def apply_customizer(self, values: dict) -> dict:
        self.customizer_applied.append(dict(values))
        self.config.patch("customizer", values)
        return values

    async def select_model(self, model_id: str, auto_frame: bool = True, *, hikari_look=None) -> dict:
        self.selected.append(model_id)
        self.active_model_id = model_id
        return {"ok": True}

    def apply_memory_config(self) -> None:
        pass

    def set_interaction(self, state: str) -> None:
        self._interaction_state = state

    async def ensure_memory(self) -> None:
        pass

    async def decide_and_apply(self, text: str, source: str, **kw: Any) -> Decision:
        return Decision(source=source, tag="happy")

    def commit_context(self, ctx, reply_text: str = "", delivered: bool = True) -> None:
        pass

    def _fire_pending_reply_emote(self, d) -> None:
        return None

    async def apply_decision(self, d, **kw) -> Any:
        return d

    def spawn(self, coro):
        return asyncio.ensure_future(coro)


async def with_client(rt: FakeRuntime, fn):
    async with TestClient(TestServer(server.build_app(rt), host="127.0.0.1", port=0)) as cl:
        assert cl.server.port != 8800
        return await fn(cl)


# ------------------------------------------------------------------------------------------ 1. streaming vocal tags
class StreamingVocalTagTests(unittest.TestCase):
    def test_clean_for_speech_default_strips(self) -> None:
        out = clean_for_speech(DRAMA)
        self.assertNotIn("[", out)
        self.assertEqual(out, "Wait, you really did that? No way, that is amazing!")

    def test_clean_for_speech_preserves_drama3_and_maps_s2(self) -> None:
        self.assertEqual(clean_for_speech(DRAMA, preserve_vocal_tags=True, fish_model="drama-3"), DRAMA)
        self.assertEqual(clean_for_speech(DRAMA, preserve_vocal_tags=True, fish_model="s2.1-pro"),
                         "[giggling] Wait, you really did that? [gasps] No way, that is amazing!")

    def test_sentence_chunker_keeps_cues_per_phrase(self) -> None:
        def chunks(**kw) -> list[str]:
            c = SentenceChunker(**kw)
            out: list[str] = []
            for ch in DRAMA:
                out += c.feed(ch)
            return out + c.flush()
        self.assertEqual(chunks(preserve_vocal_tags=True),
                         ["[giggling softly, playful] Wait, you really did that?", "[gasps, excited] No way, that is amazing!"])
        self.assertEqual(chunks(), ["Wait, you really did that?", "No way, that is amazing!"])

    def _stream(self, **cfg) -> tuple[list[str], list[str]]:
        st = FishTtsStreamer({"ttsProvider": "fish", "fishModel": "drama-3", "fishTtsTransport": "http",
                              "fishHttpChunked": True, **cfg})
        sent: list[str] = []
        shown: list[str] = []

        async def fake_synth(text: str) -> tuple[bytes, int]:
            sent.append(text)
            return b"\x01\x00" * 64, 16000

        async def on_segment(pcm, rate, text, idx) -> None:
            shown.append(text)
        st.synth_http = fake_synth                                     # type: ignore[method-assign]
        full, pcm, _ = run(st.speak_stream(deltas(DRAMA), on_segment=on_segment))
        self.assertEqual(full, DRAMA)
        self.assertTrue(pcm)
        return sent, shown

    def test_streamer_sends_cues_to_fish_but_shows_clean_text(self) -> None:
        sent, shown = self._stream(fishVocalTags=True)
        self.assertEqual(sent, ["[giggling softly, playful] Wait, you really did that?",
                                "[gasps, excited] No way, that is amazing!"])
        self.assertEqual(shown, ["Wait, you really did that?", "No way, that is amazing!"])

    def test_streamer_strips_when_disabled_or_s1(self) -> None:
        for cfg in ({"fishVocalTags": False}, {"fishVocalTags": True, "fishModel": "s1"}):
            sent, _ = self._stream(**cfg)
            self.assertTrue(sent)
            self.assertFalse(any("[" in s for s in sent), cfg)

    def test_speak_text_and_deliver_reply_pass_raw_reply(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rt = FakeRuntime(Path(td))

            async def go():
                evt = await rt.speak_text(DRAMA)
                fut = asyncio.get_running_loop().create_future()
                fut.set_result(Decision(source="reply", tag="happy"))
                evt2, _ = await rt.deliver_reply(DRAMA, fut)
                return evt, evt2
            evt, evt2 = run(go())
        self.assertEqual(len(rt.ai.synth_calls), 2)
        for text, emotion in rt.ai.synth_calls:
            self.assertEqual(text, DRAMA)                       # raw reply with inline cues reaches TTS
            self.assertTrue(emotion)                            # emotion=tag
        self.assertNotIn("[", evt["text"])                      # chat bubble / subtitle stay tag-free
        self.assertNotIn("[", evt2["text"])


# ------------------------------------------------------------------------------------------ 2. profiles
class ProfileRouteTests(unittest.TestCase):
    def test_foreign_origins_rejected_before_route_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rt = FakeRuntime(Path(td))
            async def go(cl):
                for method, path in (
                    ("GET", "/api/decisions/status"), ("GET", "/api/fish/status"),
                    ("GET", "/api/profiles"), ("POST", "/api/profiles"),
                    ("DELETE", "/api/profiles/hikari-default"), ("POST", "/api/profiles/hikari-default/activate"),
                    ("POST", "/api/vision/capture"), ("POST", "/api/rig/user-audio"),
                    ("POST", "/api/chat"),
                ):
                    response = await cl.request(method, path, headers={"Origin": "https://example.invalid"})
                    self.assertEqual(response.status, 403, path)
            run(with_client(rt, go))

    def test_failed_switch_preserves_character_and_camera(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rt = FakeRuntime(Path(td))
            rt.profiles.save_from_config("Library", rt.config.data, puppet_id="sample")
            rt.config.data["camera"]["x"] = 24
            rt.config.data["ai"]["persona"] = "Current character"
            before = json.dumps(rt.config.data, sort_keys=True)
            rt.library_model_path = lambda _: Path(td) / "sample.inp"
            async def fail(*args, **kwargs):
                raise RuntimeError("model startup failed")
            rt.select_model = fail
            with self.assertRaises(RuntimeError):
                run(rt.activate_profile("library"))
            self.assertEqual(json.dumps(rt.config.data, sort_keys=True), before)
            self.assertFalse(rt.broadcasts)

    def test_profiles_crud_and_activate_never_touch_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rt = FakeRuntime(tmp)
            cfg = rt.config.data
            cfg["ai"]["fishReferenceId"] = VOICE_ID
            cfg["ai"]["persona"] = "You are Cozy Hikari."
            cfg["customizer"]["hairHue"] = 0.33
            cfg["customizer"]["headers"] = {"Authorization": "SECRET-customizer"}

            async def go(cl):
                out = {}
                r = await cl.get("/api/profiles")
                out["list"] = (r.status, await r.json())
                r = await cl.post("/api/profiles", json={"name": "Cozy Hikari"})
                out["save"] = (r.status, await r.json(), await r.text())
                r = await cl.post("/api/profiles", json={"name": "  "})
                out["bad"] = r.status
                # drift away from the saved character, then 1-click back
                cfg["ai"]["fishReferenceId"] = "fedcba9876543210fedcba9876543210"
                cfg["ai"]["persona"] = "Someone else."
                cfg["customizer"]["hairHue"] = 0.9
                r = await cl.post("/api/profiles/cozy-hikari/activate")
                out["activate"] = (r.status, await r.json(), await r.text())
                r = await cl.post("/api/profiles/nope/activate")
                out["missing"] = r.status
                r = await cl.delete("/api/profiles/cozy-hikari")
                out["delete"] = (r.status, await r.json())
                r = await cl.delete("/api/profiles/cozy-hikari")
                out["delete2"] = r.status
                return out
            out = run(with_client(rt, go))
            disk = (tmp / "profiles.json").read_text(encoding="utf-8")

        st, j = out["list"]
        self.assertEqual(st, 200)
        self.assertTrue(j["ok"] and j["profiles"])
        st, j, raw = out["save"]
        self.assertEqual(st, 200)
        self.assertEqual(j["profile"]["id"], "cozy-hikari")
        self.assertEqual(j["profile"]["fishReferenceId"], VOICE_ID)
        self.assertIn("cozy-hikari", [p["id"] for p in j["profiles"]])
        self.assertEqual(out["bad"], 400)
        st, j, raw_act = out["activate"]
        self.assertEqual(st, 200, raw_act)
        self.assertEqual(cfg["ai"]["fishReferenceId"], VOICE_ID)
        self.assertEqual(cfg["ai"]["persona"], "You are Cozy Hikari.")
        self.assertEqual(rt.ai.cfg["persona"], "You are Cozy Hikari.")          # pushed into the companion
        self.assertEqual(cfg["customizer"]["hairHue"], 0.33)
        self.assertEqual(cfg["customizer"]["headers"]["Authorization"], "SECRET-customizer")
        self.assertTrue(rt.customizer_applied, "Hikari look changed -> customizer rebuilt")
        self.assertIn((VOICE_ID, True), rt.sync_calls)                            # Fish call voice synced
        self.assertEqual(rt.fish_duplex.voice_id, VOICE_ID)
        self.assertTrue(any(b.get("type") == "state" for b in rt.broadcasts))
        # API keys never stored, never leaked, never clobbered
        for blob in (disk, raw, raw_act):
            self.assertNotIn("SECRET", blob)
        self.assertEqual(cfg["ai"]["fishApiKey"], SECRET_FISH)
        self.assertEqual(cfg["ai"]["vercelApiKey"], SECRET_VERCEL)
        self.assertEqual(out["missing"], 404)
        st, j = out["delete"]
        self.assertEqual(st, 200)
        self.assertNotIn("cozy-hikari", [p["id"] for p in j["profiles"]])
        self.assertEqual(out["delete2"], 404)


# ------------------------------------------------------------------------------------------ 3. screen vision
class VisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.captures = 0
        self.cap_ok = True
        self.orig_cap = server.capture_screen_jpeg
        self.orig_stream = ai_companion.stream_openai_compat

        def fake_capture(**_kw) -> dict:
            self.captures += 1
            if not self.cap_ok:
                return {"ok": False, "error": "screen capture unavailable (test)"}
            return {"ok": True, "width": 64, "height": 36, "mime": "image/jpeg", "bytes": 42, "dataUrl": FAKE_JPEG}
        server.capture_screen_jpeg = fake_capture                    # type: ignore[assignment]
        self.payloads: list[dict] = []

        async def fake_stream(endpoint, headers, payload):
            self.payloads.append(payload)
            for part in ("I can see ", "your screen!"):
                yield part
        ai_companion.stream_openai_compat = fake_stream             # type: ignore[assignment]

    def tearDown(self) -> None:
        server.capture_screen_jpeg = self.orig_cap                   # type: ignore[assignment]
        ai_companion.stream_openai_compat = self.orig_stream         # type: ignore[assignment]

    def _runtime(self, tmp: Path) -> FakeRuntime:
        (tmp / "audio").mkdir()
        ai = AICompanion(tmp / "audio")
        rt = FakeRuntime(tmp, ai=ai)
        rt.config.data["ai"].update({"brainMode": "vercel_fish", "ttsEnabled": False})
        # image straight to the answering model (the Vision Eye "describe" relay is covered by test_vision_model.py)
        rt.config.data["vision"]["mode"] = "direct"
        ai.update_config(rt.config.data["ai"])
        ai.gateway.chat_request = lambda messages, stream=True, **kw: (   # type: ignore[method-assign]
            "https://ai-gateway.vercel.sh/v1/chat/completions", {}, {"model": "test", "messages": messages, "stream": stream})
        return rt

    @staticmethod
    def _image_in(payload: dict) -> str:
        last_user = [m for m in payload["messages"] if m.get("role") == "user"][-1]
        content = last_user.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return part["image_url"]["url"]
        return ""

    def test_capture_route_and_chat_vision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rt = self._runtime(Path(td))

            async def go(cl):
                out = {}
                r = await cl.post("/api/vision/capture")
                out["cap"] = (r.status, await r.json())
                r = await cl.post("/api/chat", json={"message": "what do you think?", "includeScreen": True})
                out["explicit"] = (r.status, await r.json())
                r = await cl.post("/api/chat", json={"message": "hey, look at my screen"})
                out["cue"] = (r.status, await r.json())
                r = await cl.post("/api/chat", json={"message": "tell me a joke"})
                out["plain"] = (r.status, await r.json())
                rt.config.data["vision"]["enabled"] = False
                r = await cl.post("/api/chat", json={"message": "look at my screen"})
                out["disabled"] = (r.status, await r.json())
                r = await cl.post("/api/chat", json={"message": "take a look", "includeScreen": True})
                out["explicit_disabled"] = (r.status, await r.json())
                out["capture_disabled"] = (await cl.post("/api/vision/capture")).status
                rt.config.data["vision"]["enabled"] = True
                self.cap_ok = False
                r = await cl.post("/api/vision/capture")
                out["cap_fail"] = r.status
                r = await cl.post("/api/chat", json={"message": "x", "includeScreen": True})
                out["chat_fail"] = (r.status, await r.json())
                return out
            out = run(with_client(rt, go))
            history = json.dumps(rt.ai.history)

        st, j = out["cap"]
        self.assertEqual(st, 200)
        self.assertEqual((j["ok"], j["mime"], j["dataUrl"]), (True, "image/jpeg", FAKE_JPEG))
        for key in ("explicit", "cue"):
            st, j = out[key]
            self.assertEqual(st, 200, j)
            self.assertTrue(j["usedVision"], key)
        st, j = out["plain"]
        self.assertFalse(j["usedVision"])
        st, j = out["disabled"]
        self.assertFalse(j["usedVision"])                       # vision.enabled False: voice cue ignored
        self.assertFalse(out["explicit_disabled"][1]["usedVision"])
        self.assertEqual(out["capture_disabled"], 403)
        self.assertEqual(self.captures, 5)                      # route, includeScreen, cue, failing route + chat (never plain / disabled)
        # the image really reached the companion's LLM request (and only for vision turns)
        self.assertEqual([self._image_in(p) for p in self.payloads[:4]], [FAKE_JPEG, FAKE_JPEG, "", ""])
        self.assertNotIn("base64", history)                     # screenshots never enter chat history
        self.assertEqual(out["cap_fail"], 503)
        st, j = out["chat_fail"]
        self.assertEqual(st, 200)                               # failed capture never breaks chat
        self.assertFalse(j["usedVision"])
        self.assertEqual(self._image_in(self.payloads[-1]), "")


# ------------------------------------------------------------------------------------------ 4. backchannel + motion graph
class BackchannelTests(unittest.TestCase):
    def test_user_audio_route_triggers_backchannel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rt = FakeRuntime(Path(td))

            async def go(cl):
                out = {}
                r = await cl.post("/api/rig/user-audio", json={"rms": 0.2})
                out["rms"] = (r.status, await r.json())
                r = await cl.post("/api/rig/user-audio", json={"speaking": True})
                out["speak"] = (r.status, await r.json())
                r = await cl.post("/api/rig/user-audio", json={"pause": True})
                out["pause"] = (r.status, await r.json())
                r = await cl.post("/api/rig/user-audio", json={"rms": "NaN"})
                out["nan"] = r.status
                r = await cl.post("/api/rig/user-audio", data="nope")
                out["bad"] = r.status
                r = await cl.post("/api/motion/idle", json={"id": ""})
                out["idle"] = (r.status, await r.json())
                return out
            out = run(with_client(rt, go))
            cfg = rt.config.data
            before = rt.rig.backchannel_status()["count"]
            for _ in range(30):                                  # half a second of frames
                rt.rig.step(1 / 60, cfg, None)
            after = rt.rig.backchannel_status()

        for key in ("rms", "speak", "pause"):
            st, j = out[key]
            self.assertEqual(st, 200)
            self.assertTrue(j["ok"])
            self.assertIn("count", j["backchannel"])
        self.assertTrue(out["pause"][1]["triggered"])
        self.assertEqual(out["nan"], 200)                        # non-finite rms clamped, never crashes
        self.assertEqual(out["bad"], 400)
        self.assertGreater(after["count"], before, after)        # the pause pulse played an attentive nod
        st, j = out["idle"]
        self.assertEqual(st, 200)
        self.assertIn("phraseCount", j["idleGraph"])
        self.assertIn("count", j["backchannel"])


# ------------------------------------------------------------------------------------------ 5. fish voice sync
class FishVoiceSyncTests(unittest.TestCase):
    def test_voice_save_apply_now_syncs_call_voice(self) -> None:
        orig = fish_voices.save_designed_voice

        async def fake_save(key, title, **kw):
            return {"id": VOICE_ID, "title": title or "Designed", "mine": True}
        fish_voices.save_designed_voice = fake_save              # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as td:
                rt = FakeRuntime(Path(td))

                async def go(cl):
                    out = {}
                    r = await cl.post("/api/fish/voices/save", json={"candidateId": "c1", "title": "Mine", "applyNow": True})
                    out["save"] = (r.status, await r.json(), await r.text())
                    r = await cl.get("/api/fish/status")
                    out["status"] = (r.status, await r.json())
                    rt.sync_calls.clear()
                    r = await cl.post("/api/config", json={"ai": {"fishReferenceId": "fedcba9876543210fedcba9876543210"},
                                                           "save": False})
                    out["patch"] = r.status
                    r = await cl.get("/api/decision")
                    out["decision"] = (r.status, await r.json())
                    return out
                out = run(with_client(rt, go))
                prompt = run(rt.prompt_context("did my exam go ok?"))
        finally:
            fish_voices.save_designed_voice = orig               # type: ignore[assignment]

        st, j, raw = out["save"]
        self.assertEqual(st, 200, raw)
        self.assertTrue(j["applied"])
        self.assertEqual(j["voiceSync"]["voiceId"], VOICE_ID)
        self.assertEqual(j["voiceSync"]["status"], "next-call")
        self.assertNotIn("SECRET", raw)
        self.assertEqual(rt.fish_duplex.voice_id, "fedcba9876543210fedcba9876543210")   # (after the later patch)
        st, j = out["status"]
        self.assertEqual(st, 200)
        self.assertEqual(j["voiceStatus"]["voiceId"], VOICE_ID)
        self.assertEqual(out["patch"], 200)
        self.assertEqual(rt.sync_calls, [("fedcba9876543210fedcba9876543210", True)])   # config patch -> sync
        # Reconsider gate is wired into prompt context + exposed in decision status
        self.assertIsNone(prompt)
        self.assertEqual(rt.memory.calls[-1]["reconsider"], rt.decisions.reconsider_memories)
        st, j = out["decision"]
        self.assertIn("reconsider", j)
        self.assertIn("heldBack", j["reconsider"])

    def test_startup_syncs_voice(self) -> None:
        src = (HERE / "server.py").read_text(encoding="utf-8")
        init = src[src.index("class StudioRuntime"):src.index("def _push_mood")]
        self.assertIn('self.fish_duplex.sync_voice(self.config.data["ai"].get("fishReferenceId", "")', init)
        self.assertIn('ProfileStore(DATA_DIR / "profiles.json")', init)


# ------------------------------------------------------------------------------------------ 6. web UI
class WebUiTests(unittest.TestCase):
    WEB = HERE / "web"

    def read(self, rel: str) -> str:
        return (self.WEB / rel).read_text(encoding="utf-8")

    def test_single_cheek_puff_control(self) -> None:
        motion = self.read("js/panels/motion_panel.js")
        rig = self.read("js/panels/rig_panel.js")
        pills = re.findall(r"\['Cheek Puff'", motion, re.IGNORECASE)
        sliders = re.findall(r"rs\('Cheek puff'", rig, re.IGNORECASE)
        self.assertLessEqual(len(pills), 1, "motion_panel.js has at most ONE Cheek puff control")
        self.assertEqual(len(pills) + len(sliders), 1, "exactly ONE Cheek puff control in Motion & Emotions")

    def test_new_controls_present(self) -> None:
        motion, ai, lib = self.read("js/panels/motion_panel.js"), self.read("js/panels/ai_panel.js"), self.read("js/panels/puppet_library.js")
        for s in ("Motion Graph (non-repeating phrases)", "idleMotionGraph", "Causal Prosody-to-Body", "causalProsody",
                  "Visual Backchanneling", "visualBackchannel", "phraseCount"):
            self.assertIn(s, motion)
        for s in ("Fish vocal expression tags", "fishVocalTags", "Screen Vision", "store.patch('vision'", "next Live Call"):
            self.assertIn(s, ai)
        for s in ("Character Profiles", "Save Current Profile", "/activate", "method: 'DELETE'"):
            self.assertIn(s, lib)
        self.assertIn("temporalLabel", self.read("js/panels/memory_panel.js"))
        html, app = self.read("index.html"), self.read("js/app.js")
        self.assertIn('id="btn-vision"', html)
        self.assertIn('id="vision-popover"', html)
        for s in ("/api/vision/observe", "/api/vision/sources", "/api/rig/user-audio", "startVisionWatch"):
            self.assertIn(s, app)
        ui = self.read("js/ui.js")
        for kw in ("'motion graph'", "prosody:", "backchannel:", "'vocal tags'", "'drama 3'", "'screen vision'",
                   "'character profiles'"):
            self.assertIn(kw, ui)


if __name__ == "__main__":
    unittest.main(verbosity=2)
