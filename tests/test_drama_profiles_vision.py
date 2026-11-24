"""Fish Drama-3 / S2 vocal tags, on-demand screen vision, character profiles, Fish call voice sync.
Fully offline: every HTTP call (Vercel gateway, Fish TTS, Fish Agent) is mocked; no server is started."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import ai_companion, screen_vision  # noqa: E402
from modules.ai_companion import (  # noqa: E402
    DEFAULT_PERSONA, HIKARI_PERSONALITY, S2_TAGS, TAG_TO_STATE, AICompanion, parse_emotion_and_clean,
    prepare_tts_text,
)
from modules.fish_duplex import FishAgentHTTPError, FishDuplexController  # noqa: E402
from modules.profiles import DEFAULT_PROFILE_ID, ProfileStore  # noqa: E402

BRACKET = "["


def run(coro):
    return asyncio.run(coro)


class VocalTagTests(unittest.TestCase):
    def test_drama3_keeps_inline_directions(self) -> None:
        reply = "[giggling softly, playful] Wait, you really did that? [gasps, excited] No way!"
        out = prepare_tts_text(reply, tts_provider="fish", fish_model="drama-3")
        self.assertEqual(out, "[giggling softly, playful] Wait, you really did that? [gasps, excited] No way!")

    def test_drama3_translates_stage_tags(self) -> None:
        self.assertEqual(prepare_tts_text("[wave] Hi!", fish_model="drama-3"), "[bright, warm friendly wave] Hi!")
        self.assertEqual(prepare_tts_text("[sparkle] Whoa.", fish_model="drama-3"), "[amazed, starry-eyed wonder] Whoa.")
        self.assertEqual(prepare_tts_text("[cat_smug] Told you.", fish_model="drama-3"), "[playful, teasing smirk] Told you.")
        self.assertEqual(prepare_tts_text("[smug] Told you.", fish_model="drama-3"), "[playful, teasing smirk] Told you.")
        self.assertEqual(prepare_tts_text("[blush] Thanks...", fish_model="drama-3"), "[soft, flustered, shy] Thanks...")
        self.assertEqual(prepare_tts_text("[shy] Thanks...", fish_model="drama-3"), "[soft, flustered, shy] Thanks...")
        self.assertEqual(prepare_tts_text("[neutral] Okay.", fish_model="drama-3"), "Okay.")

    def test_drama3_sanitizes_and_strips_markdown(self) -> None:
        out = prepare_tts_text("**Hey** [laughs <loudly>] you!", fish_model="drama-3")
        self.assertEqual(out, "Hey [laughs loudly] you!")

    def test_drama3_fallback_emotion_only_without_tags(self) -> None:
        self.assertEqual(prepare_tts_text("Hello there!", fish_model="drama-3", fallback_emotion="excited"),
                         "[excited, bubbly] Hello there!")
        self.assertEqual(prepare_tts_text("Hello there!", fish_model="drama-3", fallback_emotion="neutral"), "Hello there!")
        self.assertEqual(prepare_tts_text("Hello there!", fish_model="drama-3", fallback_emotion=""), "Hello there!")
        self.assertEqual(prepare_tts_text("[sighs] Fine.", fish_model="drama-3", fallback_emotion="excited"), "[sighs] Fine.")

    def test_s2_maps_to_official_tags(self) -> None:
        for model in ("s2.1-pro", "s2-pro", "s2.1-pro-free"):
            out = prepare_tts_text("[giggling softly, playful] Wait, really? [gasps, excited] No way! [wave] Bye!",
                                   fish_model=model)
            self.assertEqual(out, "[giggling] Wait, really? [gasps] No way! [happy] Bye!", model)
        self.assertEqual(prepare_tts_text("[soft sigh] Okay.", fish_model="s2.1-pro"), "[sighing] Okay.")
        self.assertEqual(prepare_tts_text("[smug] Easy.", fish_model="s2.1-pro"), "[confident] Easy.")
        self.assertEqual(prepare_tts_text("[whispering] Secret.", fish_model="s2.1-pro"), "[whisper] Secret.")
        self.assertEqual(prepare_tts_text("[zxqv] Unknown cue.", fish_model="s2.1-pro"), "Unknown cue.")
        self.assertEqual(prepare_tts_text("Plain.", fish_model="s2.1-pro", fallback_emotion="sad"), "[sad] Plain.")
        out = prepare_tts_text("[happy] [excited] [sad] [angry] [calm] [nervous] Hi.", fish_model="s2.1-pro")
        for tag in ("happy", "excited", "sad", "angry", "calm", "nervous"):
            self.assertIn(f"[{tag}]", out)
        for tag in S2_TAGS:                     # official tags pass through unchanged
            self.assertEqual(prepare_tts_text(f"[{tag}] x", fish_model="s2.1-pro"), f"[{tag}] x")

    def test_non_fish_and_disabled_strip_everything(self) -> None:
        reply = "[giggling softly, playful] Wait, you really did that? [gasps, excited] No way! [wave]"
        for kw in ({"tts_provider": "windows"}, {"tts_provider": "openai"},
                   {"tts_provider": "fish", "vocal_tags_enabled": False},
                   {"tts_provider": "fish", "fish_model": "s1"}):
            out = prepare_tts_text(reply, fallback_emotion="happy", **kw)
            self.assertNotIn(BRACKET, out, kw)
            self.assertEqual(out, "Wait, you really did that? No way!", kw)

    def test_only_cues_yields_empty(self) -> None:
        self.assertEqual(prepare_tts_text("[neutral]", fish_model="drama-3"), "")
        self.assertEqual(prepare_tts_text("[zxqv]", fish_model="s2.1-pro"), "")

    def test_parse_strips_multiword_directions_and_detects_emotion(self) -> None:
        clean, tag, state = parse_emotion_and_clean("[giggling softly, playful] Wait, you really did that? [gasps, excited] No way!")
        self.assertEqual(clean, "Wait, you really did that? No way!")
        self.assertEqual(tag, "happy")
        self.assertIs(state, TAG_TO_STATE["happy"])
        cases = {
            "[laughing] that's hilarious": "happy",
            "[excited, laughing] let's go": "excited",
            "[gasps] what?!": "surprised",
            "[whisper] it's a secret": "shy",
            "[teasing] oh really": "smug",
            "[soft sigh] okay then": "sad",
            "[tired sigh] long day": "sleepy",
            "[smug] obviously": "smug",
            "[shy] th-thanks": "shy",
            "[happy] yay": "happy",
            "[cat_smug] heh": "smug",
            "Hi there [wave] friend": "wave",
        }
        for text, want in cases.items():
            clean, tag, _ = parse_emotion_and_clean(text)
            self.assertEqual(tag, want, text)
            self.assertNotIn(BRACKET, clean, text)
        clean, _, _ = parse_emotion_and_clean("Hi there [wave] friend!")
        self.assertEqual(clean, "Hi there friend!")

    def test_parse_without_tags_still_classifies(self) -> None:
        clean, tag, state = parse_emotion_and_clean("Thank you so much, that means a lot!")
        self.assertEqual(clean, "Thank you so much, that means a lot!")
        self.assertIn(tag, TAG_TO_STATE)
        self.assertIsInstance(state, dict)

    def test_tag_to_state_only_wave_has_arms(self) -> None:
        arm = {"armR", "armL"}
        self.assertEqual({t: arm & s.keys() for t, s in TAG_TO_STATE.items() if arm & s.keys()}, {"wave": arm})

    def test_streaming_long_cue_stays_intact_across_hard_boundary(self) -> None:
        from modules.fish_tts_stream import SentenceChunker
        text = "Some words [softly. Then enthusiastically, with a deliberately extended pause] more words follow here."
        for keep in (False, True):
            chunker = SentenceChunker(min_chars=6, max_chars=18, preserve_vocal_tags=keep)
            parts = []
            for char in text:
                parts.extend(chunker.feed(char))
            parts.extend(chunker.flush())
            joined = " ".join(parts)
            self.assertEqual(ai_companion.strip_vocal_tags(joined), "Some words more words follow here.")
            self.assertEqual("[" in joined, keep)

    def test_stream_keeps_original_voice_when_config_changes(self) -> None:
        from modules.fish_tts_stream import FishTtsStreamer
        stream = FishTtsStreamer({"fishModel": "drama-3", "fishReferenceId": "first"})
        calls = []

        async def fake_http(client, phrase):
            calls.append((client.fish_model, client.request_body(phrase)["reference_id"], phrase))
            return b"\0\0", 16000

        async def deltas():
            yield "[softly laughing] "
            stream.update_config({"fishModel": "s2-pro", "fishReferenceId": "second"})
            yield "Hello!"

        with mock.patch.object(FishTtsStreamer, "synth_http", fake_http):
            run(stream.speak_stream(deltas(), transport="http"))
        self.assertEqual(calls, [("drama-3", "first", "[softly laughing] Hello!")])

    def test_inline_voice_respects_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ai = AICompanion(Path(td))
            for provider in ("openai", "windows", "fish"):
                ai.update_config({"brainMode": "vercel_fish", "ttsProvider": provider})
                self.assertEqual(ai.voices_inline, provider == "fish")

    def test_persona_verbatim_and_vocal_hint(self) -> None:
        self.assertEqual(DEFAULT_PERSONA, HIKARI_PERSONALITY["system_prompt"])
        self.assertNotIn("Voice note", DEFAULT_PERSONA)            # verbatim original untouched
        self.assertIn("[giggles]", HIKARI_PERSONALITY["vocal_cues"])
        with tempfile.TemporaryDirectory() as td:
            ai = AICompanion(Path(td))
            msgs = ai._messages("")
            self.assertEqual(msgs[0]["content"], DEFAULT_PERSONA)
            self.assertEqual(msgs[1]["content"], HIKARI_PERSONALITY["vocal_cues"])
            ai.update_config({"ttsProvider": "windows"})
            self.assertNotIn(HIKARI_PERSONALITY["vocal_cues"], [m["content"] for m in ai._messages("")])
            ai.update_config({"ttsProvider": "fish", "fishVocalTags": False})
            self.assertNotIn(HIKARI_PERSONALITY["vocal_cues"], [m["content"] for m in ai._messages("")])


class _FakeResp:
    def __init__(self, status: int = 200, data: dict | None = None, content: bytes = b"RIFFfake") -> None:
        self.status_code = status
        self._data = data or {}
        self.content = content

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, resp: _FakeResp | None = None) -> None:
        self.calls: list[dict] = []
        self.resp = resp or _FakeResp()

    async def post(self, url, headers=None, json=None, timeout=None, **kw):  # noqa: A002
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.resp


class SynthesizeTtsTests(unittest.TestCase):
    def _run_tts(self, patch: dict, text: str, emotion: str = "") -> dict:
        with tempfile.TemporaryDirectory() as td:
            ai = AICompanion(Path(td))
            ai.update_config({"fishApiKey": "test-key", **patch})
            client = _FakeClient()

            async def fake_ffmpeg(raw, raw_tmp, wav_out):
                wav_out.write_bytes(b"x" * 3000)
                return True
            with mock.patch.object(ai_companion.http_pool, "client", return_value=client), \
                 mock.patch.object(ai, "_ffmpeg_to_wav", side_effect=fake_ffmpeg):
                path = run(ai.synthesize_tts(text, emotion=emotion))
            self.assertIsNotNone(path)
            self.assertEqual(len(client.calls), 1)
            return client.calls[0]["json"]

    def test_fish_drama3_body_keeps_directions(self) -> None:
        body = self._run_tts({"ttsProvider": "fish", "fishModel": "drama-3"}, "[giggling softly] Hehe, hi!")
        self.assertEqual(body["text"], "[giggling softly] Hehe, hi!")

    def test_fish_drama3_emotion_fallback(self) -> None:
        body = self._run_tts({"ttsProvider": "fish", "fishModel": "drama-3"}, "Hehe, hi!", emotion="smug")
        self.assertEqual(body["text"], "[playful, teasing smirk] Hehe, hi!")

    def test_fish_vocal_tags_disabled(self) -> None:
        body = self._run_tts({"ttsProvider": "fish", "fishVocalTags": False}, "[giggling softly] Hehe, hi!", "happy")
        self.assertEqual(body["text"], "Hehe, hi!")

    def test_fish_s2(self) -> None:
        body = self._run_tts({"ttsProvider": "fish", "fishModel": "s2.1-pro"}, "[giggling softly] Hehe, hi!")
        self.assertEqual(body["text"], "[giggling] Hehe, hi!")

    def test_openai_input_stripped(self) -> None:
        with mock.patch.object(ai_companion.server_keys, "key_for_url", return_value="sk-test"):
            body = self._run_tts({"ttsProvider": "openai"}, "[gasps, excited] No way!")
        self.assertEqual(body["input"], "No way!")

    def test_positional_call_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ai = AICompanion(Path(td))
            ai.update_config({"ttsEnabled": False})
            self.assertIsNone(run(ai.synthesize_tts("hello")))
            ai.update_config({"ttsEnabled": True})
            self.assertIsNone(run(ai.synthesize_tts("[wave]")))      # nothing to say


# --------------------------------------------------------------------------- screen vision
class ScreenVisionTests(unittest.TestCase):
    def test_wants_screen_vision(self) -> None:
        for t in ("look at my screen", "Hey, can you look at my screen?", "what's on my screen", "What’s on my screen?",
                  "see my screen please", "check my screen", "look at this screen!", "what am I playing?",
                  "What am I looking at", "can you see my screen"):
            self.assertTrue(screen_vision.wants_screen_vision(t), t)
        for t in ("", "hello!", "I love my new screen protector", "the screen is too bright", "look at you go",
                  "don't look at my screen", "I'm playing Minecraft", "what should I play next?",
                  "look at this!", "check this out", "please do not check my screen",
                  "don't peek at my screen", "do not view my screen", "I cannot see my screen",
                  "look at my screen protector", "he told me to check my screen"):
            self.assertFalse(screen_vision.wants_screen_vision(t), t)

    def test_capture_downscales_to_jpeg_data_url(self) -> None:
        from PIL import Image
        fake = Image.new("RGB", (2560, 1440), (30, 160, 220))
        with mock.patch.object(screen_vision, "_grab_pil", return_value=fake):
            res = screen_vision.capture_screen_jpeg(max_dim=1024, quality=70)
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["width"], res["height"]), (1024, 576))
        self.assertEqual(res["mime"], "image/jpeg")
        self.assertTrue(res["dataUrl"].startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(res["dataUrl"].split(",", 1)[1])
        self.assertEqual(len(raw), res["bytes"])
        self.assertEqual(Image.open(io.BytesIO(raw)).size, (1024, 576))

    def test_capture_small_screen_not_upscaled(self) -> None:
        from PIL import Image
        with mock.patch.object(screen_vision, "_grab_pil", return_value=Image.new("RGBA", (800, 600))):
            res = screen_vision.capture_screen_jpeg(max_dim=1024)
        self.assertTrue(res["ok"])
        self.assertEqual((res["width"], res["height"]), (800, 600))

    def test_capture_headless_never_crashes(self) -> None:
        with mock.patch.object(screen_vision, "_grab_pil", side_effect=OSError("no display")), \
             mock.patch.object(screen_vision, "_grab_mss", side_effect=ImportError("no mss")):
            res = screen_vision.capture_screen_jpeg()
        self.assertFalse(res["ok"])
        self.assertIn("unavailable", res["error"])


# --------------------------------------------------------------------------- multimodal payload
IMG = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff fake jpeg").decode()


class MultimodalTests(unittest.TestCase):
    def _ai(self, td: str) -> AICompanion:
        ai = AICompanion(Path(td))
        ai.update_config({"brainMode": "vercel_fish", "vercelApiKey": "test-gw-key", "ttsEnabled": False})
        return ai

    def test_gateway_gets_image_history_stays_text(self) -> None:
        sent: list[dict] = []

        async def fake_complete(endpoint, headers, payload, *a, **kw):
            sent.append(json.loads(json.dumps(payload)))
            return "[gasps, excited] Ooh, is that a boss fight?"
        with tempfile.TemporaryDirectory() as td:
            ai = self._ai(td)
            with mock.patch.object(ai_companion, "complete_openai_compat", side_effect=fake_complete):
                reply = run(ai.generate_reply("look at my screen", "", image_data_url=IMG))
            self.assertIn("boss fight", reply)
            user = [m for m in sent[0]["messages"] if m["role"] == "user"][-1]
            self.assertEqual(user["content"], [{"type": "text", "text": "look at my screen"},
                                               {"type": "image_url", "image_url": {"url": IMG}}])
            self.assertEqual(ai.history[0], {"role": "user", "content": "look at my screen"})
            self.assertTrue(all(isinstance(m["content"], str) for m in ai.history))
            self.assertNotIn("base64", json.dumps(ai.history))
            # next turn without an image is text-only again
            with mock.patch.object(ai_companion, "complete_openai_compat", side_effect=fake_complete):
                run(ai.generate_reply("cool", ""))
            self.assertTrue(all(isinstance(m["content"], str) for m in sent[1]["messages"]))

    def test_backwards_compatible_without_image(self) -> None:
        sent: list[dict] = []

        async def fake_complete(endpoint, headers, payload, *a, **kw):
            sent.append(payload)
            return "hi"
        with tempfile.TemporaryDirectory() as td:
            ai = self._ai(td)
            with mock.patch.object(ai_companion, "complete_openai_compat", side_effect=fake_complete):
                run(ai.generate_reply("hello"))
        self.assertEqual([m for m in sent[0]["messages"] if m["role"] == "user"][-1]["content"], "hello")

    def test_streaming_voice_path_gets_image(self) -> None:
        sent: list[dict] = []

        async def fake_stream(endpoint, headers, payload, *a, **kw):
            sent.append(payload)
            for d in ("[giggles] ", "That's ", "a cute cat!"):
                yield d
        with tempfile.TemporaryDirectory() as td:
            ai = self._ai(td)
            with mock.patch.object(ai_companion, "stream_openai_compat", side_effect=fake_stream):
                reply, wav = run(ai.generate_reply_with_voice("what am I looking at", "", image_data_url=IMG))
            self.assertEqual(reply, "[giggles] That's a cute cat!")
            self.assertIsNone(wav)
            user = [m for m in sent[0]["messages"] if m["role"] == "user"][-1]
            self.assertEqual(user["content"][1]["image_url"]["url"], IMG)
            self.assertEqual(ai.history[0]["content"], "what am I looking at")

    def test_legacy_openai_compat_provider_gets_image(self) -> None:
        client = _FakeClient(_FakeResp(data={"choices": [{"message": {"content": "Nice game!"}}]}))
        with tempfile.TemporaryDirectory() as td:
            ai = AICompanion(Path(td))
            ai.update_config({"brainMode": "fish_duplex", "provider": "custom", "baseUrl": "http://127.0.0.1:9/v1",
                              "apiKey": "x"})
            with mock.patch.object(ai_companion.http_pool, "client", return_value=client):
                reply = run(ai.generate_reply("what am I playing", "", image_data_url=IMG))
        self.assertEqual(reply, "Nice game!")
        user = [m for m in client.calls[0]["json"]["messages"] if m["role"] == "user"][-1]
        self.assertEqual(user["content"][1], {"type": "image_url", "image_url": {"url": IMG}})

    def test_bad_image_url_ignored(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        self.assertIs(ai_companion.attach_image(msgs, "javascript:alert(1)"), msgs)
        self.assertIs(ai_companion.attach_image(msgs, None), msgs)


# --------------------------------------------------------------------------- profiles
def _studio_cfg() -> dict:
    return {
        "stage": {"activeModel": "mao", "theme": "night"},
        "camera": {"scale": 2.1, "x": 10.0, "y": 300.0},
        "rig": {"idleClip": "sway_soft", "idleClipMix": 0.7, "physicsEnabled": True},
        "customizer": {"hairHue": 0.3, "outfit": "hoodie", "apiKey": "LEAK"},
        "ai": {"fishReferenceId": "voiceMao123", "fishModel": "s2.1-pro", "persona": "You are Mao.",
               "fishApiKey": "fish-secret", "vercelApiKey": "gw-secret", "apiKey": "sk-secret",
               "baseUrl": "https://example.invalid/v1", "provider": "openai"},
    }


class ProfileStoreTests(unittest.TestCase):
    def test_seed_save_list_get_apply_delete_persist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "data" / "profiles.json"
            store = ProfileStore(path)
            listed = store.list_profiles()
            self.assertEqual([p["id"] for p in listed], [DEFAULT_PROFILE_ID])
            self.assertEqual(listed[0]["name"], "Hikari (Default)")
            self.assertEqual(listed[0]["puppetId"], "hikari")
            self.assertTrue(path.exists())

            prof = store.save_from_config("Mao  Night Mode!", _studio_cfg(), puppet_id="mao")
            self.assertEqual(prof["id"], "mao-night-mode")
            for key in ("id", "name", "puppetId", "customizer", "fishReferenceId", "fishModel", "personality",
                        "idleClip", "idleClipMix", "camera", "updatedAt"):
                self.assertIn(key, prof)
            self.assertEqual((prof["puppetId"], prof["fishReferenceId"], prof["fishModel"]), ("mao", "voiceMao123", "s2.1-pro"))
            self.assertEqual(prof["camera"], {"scale": 2.1, "x": 10.0, "y": 300.0})
            self.assertEqual(prof["idleClip"], "sway_soft")
            raw = path.read_text(encoding="utf-8")
            for secret in ("fish-secret", "gw-secret", "sk-secret", "LEAK", "example.invalid"):
                self.assertNotIn(secret, raw)
            self.assertEqual(store.get("mao-night-mode")["personality"], "You are Mao.")
            self.assertIsNone(store.get("nope"))
            self.assertIsNone(store.get("../../etc/passwd"))

            # update in place by id
            cfg2 = _studio_cfg()
            cfg2["ai"]["fishReferenceId"] = "voiceMao456"
            store.save_from_config("Mao Night Mode", cfg2, profile_id="mao-night-mode", puppet_id="mao")
            self.assertEqual(len(store.list_profiles()), 2)

            # persistence across reload
            store2 = ProfileStore(path)
            self.assertEqual(store2.get("mao-night-mode")["fishReferenceId"], "voiceMao456")

            # apply never clobbers keys / unrelated settings
            target = {
                "stage": {"activeModel": "hikari", "theme": "day"},
                "camera": {"scale": 1.0, "x": 0.0, "y": 0.0},
                "rig": {"idleClip": "", "idleClipMix": 1.0, "physicsEnabled": False},
                "customizer": {"hairHue": 0.9},
                "ai": {"fishApiKey": "KEEP-fish", "vercelApiKey": "KEEP-gw", "apiKey": "KEEP-sk",
                       "provider": "offline", "baseUrl": "https://api.openai.com/v1", "persona": DEFAULT_PERSONA},
            }
            applied = store2.apply_to_config("mao-night-mode", target)
            self.assertEqual(applied["id"], "mao-night-mode")
            self.assertEqual(target["stage"], {"activeModel": "mao", "theme": "day"})
            self.assertEqual(target["camera"], {"scale": 2.1, "x": 10.0, "y": 300.0})
            self.assertEqual(target["rig"], {"idleClip": "sway_soft", "idleClipMix": 0.7, "physicsEnabled": False})
            self.assertEqual(target["customizer"], {"hairHue": 0.3, "outfit": "hoodie"})
            ai = target["ai"]
            self.assertEqual((ai["fishApiKey"], ai["vercelApiKey"], ai["apiKey"]), ("KEEP-fish", "KEEP-gw", "KEEP-sk"))
            self.assertEqual((ai["provider"], ai["baseUrl"]), ("offline", "https://api.openai.com/v1"))
            self.assertEqual((ai["fishReferenceId"], ai["fishModel"], ai["persona"]), ("voiceMao456", "s2.1-pro", "You are Mao."))
            self.assertEqual(ai["personaId"], "custom")

            # default profile restores Hikari
            store2.apply_to_config(DEFAULT_PROFILE_ID, target)
            self.assertEqual(target["stage"]["activeModel"], "hikari")
            self.assertEqual(ai["persona"], DEFAULT_PERSONA)
            self.assertEqual(ai["personaId"], HIKARI_PERSONALITY["id"])
            self.assertEqual(ai["fishApiKey"], "KEEP-fish")

            with self.assertRaises(KeyError):
                store2.apply_to_config("missing", target)

            self.assertTrue(store2.delete("mao-night-mode"))
            self.assertFalse(store2.delete("mao-night-mode"))
            self.assertIsNone(ProfileStore(path).get("mao-night-mode"))
            self.assertTrue(store2.delete(DEFAULT_PROFILE_ID))         # store never left empty
            self.assertEqual([p["id"] for p in ProfileStore(path).list_profiles()], [DEFAULT_PROFILE_ID])
            self.assertEqual([p.name for p in path.parent.iterdir()], ["profiles.json"])   # no temp files left

    def test_failed_write_preserves_disk_and_in_memory_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profiles.json"
            store = ProfileStore(path)
            before = store.list_profiles()
            old_bytes = path.read_bytes()
            with mock.patch("modules.profiles.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.save_from_config("new", _studio_cfg())
                with self.assertRaises(OSError):
                    store.delete(DEFAULT_PROFILE_ID)
            self.assertEqual(store.list_profiles(), before)
            self.assertEqual(path.read_bytes(), old_bytes)

    def test_snapshot_and_restore_only_character_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = ProfileStore(Path(td) / "profiles.json")
            cfg = _studio_cfg()
            cfg["stage"]["activeModel"] = "Mao Night"
            cfg["customizer"].update(customEyePath="../private.png", headers={"Authorization": "private"})
            cfg["ai"]["fishReferenceId"] = ""
            cfg["ai"]["persona"] = ""
            p = store.save_from_config("Night", cfg)
            self.assertEqual(p["puppetId"], "Mao Night")
            self.assertNotIn("private", json.dumps(p))
            self.assertIsNone(store.get("../night"))
            store.apply_to_config("night", cfg)
            self.assertEqual(cfg["customizer"]["headers"], {"Authorization": "private"})
            self.assertEqual(cfg["ai"]["fishReferenceId"], "")
            self.assertEqual(cfg["ai"]["persona"], DEFAULT_PERSONA)

    def test_corrupt_file_reseeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profiles.json"
            path.write_text("{not json", encoding="utf-8")
            store = ProfileStore(path)
            self.assertEqual([p["id"] for p in store.list_profiles()], [DEFAULT_PROFILE_ID])
            self.assertEqual(next(path.parent.glob("*.corrupt")).read_text(), "{not json")
            ids = [store.save_from_config(name, {})["id"] for name in ("\u6708", "\u96ea")]
            self.assertNotEqual(*ids)


# --------------------------------------------------------------------------- Fish call voice sync
class FishVoiceSyncTests(unittest.TestCase):
    def test_sync_voice_offline(self) -> None:
        fd = FishDuplexController()
        cfg = {"fishReferenceId": "old"}
        res = fd.sync_voice("8a11628d5c0d4b318286735d78aad236", cfg)
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], "next-call")
        self.assertFalse(res["appliedLive"])
        self.assertEqual(cfg["fishReferenceId"], "8a11628d5c0d4b318286735d78aad236")
        self.assertEqual(fd.voice_id, "8a11628d5c0d4b318286735d78aad236")
        bad = fd.sync_voice("bad id; DROP")
        self.assertFalse(bad["ok"])
        self.assertEqual(fd.voice_id, "8a11628d5c0d4b318286735d78aad236")
        self.assertEqual(fd.sync_voice("")["status"], "agent-default")

    def _controller(self, reject_voice: bool = False) -> tuple[FishDuplexController, list]:
        fd = FishDuplexController()
        calls: list = []

        async def fake_req(method, path, body=None):
            calls.append((method, path, json.loads(json.dumps(body)) if body else None))
            if path == "/agents":
                return {"agents": [{"agent_id": "ag1", "name": "Hikari Drama 3", "public_enabled": False,
                                    "description": "Private Hikari Live2D companion with Hikari's custom voice and Drama 3"}]}
            if path == "/sessions":
                if reject_voice and "voice_id" in (body or {}).get("overrides", {}):
                    raise FishAgentHTTPError(422, "unknown override")
                return {"session_id": f"s{len(calls)}", "transport": "livekit"}
            return {}
        fd._req = fake_req  # type: ignore[method-assign]
        return fd, calls

    def test_session_passes_voice_override(self) -> None:
        fd, calls = self._controller()
        fd.sync_voice("voiceA")
        run(fd.start_session("drama-3"))
        body = [c for c in calls if c[1] == "/sessions"][0][2]
        self.assertEqual(body["overrides"], {"system_prompt": fd._persona, "voice_id": "voiceA"})
        self.assertTrue(fd.voice_synced and fd.persona_synced)
        live = fd.sync_voice("voiceA")
        self.assertEqual((live["status"], live["appliedLive"]), ("live", True))
        swap = fd.sync_voice("voiceB")                      # mid-call: applies next call, no crash
        self.assertEqual((swap["status"], swap["appliedLive"], swap["activeSession"]), ("next-call", False, True))
        self.assertTrue(fd.voice_status()["pendingNextCall"])

    def test_voice_override_rejected_keeps_persona(self) -> None:
        fd, calls = self._controller(reject_voice=True)
        fd.set_voice("voiceA")
        session = run(fd.start_session("drama-3"))
        self.assertEqual(session["transport"], "livekit")
        bodies = [c[2] for c in calls if c[1] == "/sessions"]
        self.assertEqual(bodies[-1]["overrides"], {"system_prompt": fd._persona})
        self.assertTrue(fd.persona_synced)
        self.assertFalse(fd.voice_synced)
        self.assertIn("rejected", fd.last_voice_note)

    def test_no_voice_is_unchanged_behaviour(self) -> None:
        fd, calls = self._controller()
        run(fd.start_session("drama-3"))
        body = [c for c in calls if c[1] == "/sessions"][0][2]
        self.assertEqual(body["overrides"], {"system_prompt": fd._persona})

    def test_update_agent_voice_patches_config(self) -> None:
        fd, calls = self._controller()
        res = run(fd.update_agent_voice("drama-3", "voiceZ"))
        self.assertTrue(res["ok"])
        self.assertIn(("PATCH", "/agents/ag1/config", {"voice": {"voice_id": "voiceZ"}}), calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
