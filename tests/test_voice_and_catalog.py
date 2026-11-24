"""Verification: Fish Voice Library / Voice Design / save / delete + Vercel AI Gateway chat-model + provider catalog.

Fully offline: every Fish / Vercel call goes to an httpx.MockTransport (no keys, no paid calls, nothing is
created or deleted on a real account). The real server.py handlers are served on a random loopback port
(never 8800) against a TEMP settings.json.

    python tests/test_voice_and_catalog.py        (or: python -m pytest tests/test_voice_and_catalog.py)
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import httpx  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from core.config import ConfigStore  # noqa: E402
from modules import fish_voices, server_keys, vercel_gateway  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
FAKE_KEY = "fake-fish-test-key"


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(data)) + data)


# ------------------------------------------------------------------------------------------ fake Fish API
class FakeFish:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.deleted: list[str] = []
        self.created: list[dict[str, Any]] = []
        self.list_status = 200

    def model(self, mid: str, title: str, *, state: str = "trained", kind: str = "tts", vis: str = "public") -> dict:
        return {"_id": mid, "type": kind, "state": state, "title": title, "description": f"{title} desc",
                "visibility": vis, "languages": ["en"], "tags": ["calm"], "author": {"nickname": "amy"},
                "samples": [{"title": "s", "text": "hi", "task_id": "t", "audio": f"https://cdn.fish.audio/{mid}.mp3"}]}

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        path = req.url.path
        if req.method == "GET" and path == "/model":
            if self.list_status != 200:
                return httpx.Response(self.list_status, json={"code": "unauthorized", "detail": "SECRET-BODY"})
            if req.url.params.get("self") == "true":
                items = [self.model("mine1", "My Voice", vis="private"), self.model("shared1", "Shared"),
                         self.model("training1", "Still training", state="training"),
                         self.model("failed1", "Broken", state="failed")]
            else:
                items = [self.model("shared1", "Shared"), self.model("pub1", "Warm‮ voice\n"),
                         self.model("svc1", "Old SVC", kind="svc"), self.model("bad id", "x"),
                         self.model("pubtrain", "Training", state="training")]
            return httpx.Response(200, json={"total": len(items), "items": items})
        if req.method == "POST" and path == "/v1/voice-design":
            body = json.loads(req.content)
            n = body["n"]
            cands = [{"id": f"remote{i}", "index": n - 1 - i, "audio_base64": base64.b64encode(wav_bytes(0.25 * (i + 1))).decode(),
                      "sample_rate": 16000, "duration_ms": 250 * (i + 1), "text": "Hello there", "signature": f"sig{i}"}
                     for i in range(n)]
            return httpx.Response(200, json={"candidates": cands})
        if req.method == "POST" and path == "/model":
            ctype = req.headers.get("content-type", "")
            raw = req.content
            self.created.append({"ctype": ctype, "raw": raw})
            return httpx.Response(201, json=self.model("newvoice1", "Designed Hikari", vis="private"))
        if req.method == "DELETE" and path.startswith("/model/"):
            self.deleted.append(path.rsplit("/", 1)[1])
            return httpx.Response(200)
        return httpx.Response(404, json={"code": "not_found"})


def install_fish(fake: FakeFish) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    fish_voices._client = lambda: client          # type: ignore[assignment]
    fish_voices.clear_list_cache()


# ------------------------------------------------------------------------------------------ fake Vercel API
VERCEL_MODELS = {"object": "list", "data": [
    {"id": "openai/gpt-5-nano", "owned_by": "openai", "name": "GPT-5 nano", "type": "language",
     "context_window": 400000, "max_tokens": 128000, "pricing": {"input": "0.00000005", "output": "0.0000004"}},
    {"id": "deepseek/deepseek-v3.2", "owned_by": "deepseek", "name": "DeepSeek V3.2", "type": "language",
     "context_window": 163840, "pricing": {"input": "0.00000027"}},
    {"id": "anthropic/claude-x-batch", "owned_by": "anthropic", "name": "Batch only", "type": "language"},
    {"id": "openai/text-embedding-3-small", "owned_by": "openai", "name": "Embed", "type": "embedding"},
    {"id": "typesafe-ai/jev", "owned_by": "typesafe-ai", "name": "Jev", "type": "evaluation"},
    {"id": "no-slash-model", "owned_by": "x", "name": "Unrouted", "type": "language"},
]}
VERCEL_ENDPOINTS = {
    "openai/gpt-5-nano": ["openai", "azure", "openai"],
    "deepseek/deepseek-v3.2": ["deepinfra", "bedrock"],
}


class FakeVercel:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.offline = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        if self.offline:
            raise httpx.ConnectError("offline", request=req)
        path = req.url.path
        if path == "/v1/models":
            return httpx.Response(200, json=VERCEL_MODELS)
        if path.startswith("/v1/models/") and path.endswith("/endpoints"):
            mid = path[len("/v1/models/"):-len("/endpoints")]
            if mid not in VERCEL_ENDPOINTS:
                return httpx.Response(404, json={"error": {"code": "model_not_found"}})
            return httpx.Response(200, json={"data": {"id": mid, "endpoints": [
                {"name": f"{p} | {mid}", "provider_name": p} for p in VERCEL_ENDPOINTS[mid]]}})
        return httpx.Response(404)


def install_vercel(fake: FakeVercel) -> None:
    vercel_gateway._catalog_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))  # type: ignore[assignment]
    vercel_gateway.clear_chat_catalog_cache()


# ------------------------------------------------------------------------------------------ module tests
async def test_fish_module() -> None:
    fake = FakeFish()
    install_fish(fake)
    mine = await fish_voices.list_voices(FAKE_KEY, "self", page=2, limit=10)
    req = fake.calls[-1]
    check("fish.list self: GET /model?self=true&page_size&page_number + bearer",
          req.method == "GET" and req.url.params.get("self") == "true" and req.url.params.get("page_size") == "10"
          and req.url.params.get("page_number") == "2" and req.headers["authorization"] == f"Bearer {FAKE_KEY}")
    check("fish.list self: own voices incl. training, failed + non-tts dropped",
          [v["id"] for v in mine] == ["mine1", "shared1", "training1"], str([v["id"] for v in mine]))
    check("fish.list self: every voice marked mine", all(v["mine"] for v in mine))
    keys = {"id", "title", "description", "visibility", "languages", "samples", "mine", "author", "state"}
    check("fish.list: normalized schema", keys <= set(mine[0]) and mine[0]["samples"] == ["https://cdn.fish.audio/mine1.mp3"]
          and mine[0]["author"] == "amy" and mine[0]["languages"] == ["en"])
    n_calls = len(fake.calls)
    await fish_voices.list_voices(FAKE_KEY, "self", page=2, limit=10)
    check("fish.list: 60 s cache hit (no second request)", len(fake.calls) == n_calls)
    await fish_voices.list_voices(FAKE_KEY, "self", page=2, limit=10, force=True)
    check("fish.list: force=True bypasses the cache", len(fake.calls) == n_calls + 1)
    await fish_voices.list_voices("another-key", "self", page=2, limit=10)
    check("fish.list: cache is per key", len(fake.calls) == n_calls + 2)

    pub = await fish_voices.list_voices(FAKE_KEY, "public", query="warm voice")
    req = fake.calls[-1]
    check("fish.list public: sort_by=score + title query, no self",
          req.url.params.get("sort_by") == "score" and req.url.params.get("title") == "warm voice"
          and "self" not in req.url.params)
    check("fish.list public: trained tts only, bad ids / svc / training dropped",
          [v["id"] for v in pub] == ["shared1", "pub1"], str([v["id"] for v in pub]))
    check("fish.list public: title sanitized (bidi + newline)", pub[1]["title"] == "Warm voice", repr(pub[1]["title"]))
    check("fish.list public: not mine", not any(v["mine"] for v in pub))

    allv = await fish_voices.list_voices(FAKE_KEY, "all")
    check("fish.list all: self first, then public, de-duplicated by id",
          [v["id"] for v in allv] == ["mine1", "shared1", "training1", "pub1"] and allv[1]["mine"], str([v["id"] for v in allv]))

    try:
        await fish_voices.list_voices("", "self")
        check("fish.list: empty key refused", False)
    except fish_voices.FishVoiceError as exc:
        check("fish.list: empty key refused (400)", exc.status == 400)
    fake.list_status = 401
    try:
        await fish_voices.list_voices(FAKE_KEY, "public", force=True)
        check("fish.list: HTTP 401 raised", False)
    except fish_voices.FishVoiceError as exc:
        check("fish.list: HTTP 401 -> safe error (no remote body, no key)",
              "401" in str(exc) and "SECRET-BODY" not in str(exc) and FAKE_KEY not in str(exc), str(exc))
    fake.list_status = 200

    # ---- voice design
    res = await fish_voices.design_voice(FAKE_KEY, "  Bright playful anime girl  ", "Hello there", "en", 2)
    req = fake.calls[-1]
    body = json.loads(req.content)
    check("fish.design: POST /v1/voice-design with header model: voice-design-1",
          req.method == "POST" and req.url.path == "/v1/voice-design" and req.headers.get("model") == "voice-design-1"
          and req.headers["authorization"] == f"Bearer {FAKE_KEY}")
    check("fish.design: payload {instruction, reference_text, language, n, speed}",
          body == {"instruction": "Bright playful anime girl", "reference_text": "Hello there", "language": "en",
                   "n": 2, "speed": 1.0}, str(body))
    c = res["candidates"]
    check("fish.design: 2 candidates with id / audioBase64 / mime / durationSec",
          res["ok"] and len(c) == 2 and all(x["id"].startswith("cand_") and x["mime"] == "audio/wav"
                                            and base64.b64decode(x["audioBase64"])[:4] == b"RIFF" for x in c))
    check("fish.design: sorted by Fish index, duration from duration_ms",
          [x["durationSec"] for x in c] == [0.5, 0.25], str([x["durationSec"] for x in c]))
    cached = fish_voices.get_candidate(c[0]["id"])
    check("fish.design: candidate cached server-side (bytes, mime, instruction, signature)",
          cached is not None and cached["bytes"][:4] == b"RIFF" and cached["instruction"] == "Bright playful anime girl"
          and cached["signature"] == "sig1")
    for bad, label in ((("", "x"), "empty instruction"), (("x" * 2001, ""), "2001-char instruction"),
                       (("ok", "y" * 151), "151-char preview text")):
        try:
            await fish_voices.design_voice(FAKE_KEY, bad[0], bad[1])
            check(f"fish.design: {label} refused", False)
        except fish_voices.FishVoiceError as exc:
            check(f"fish.design: {label} refused (400)", exc.status == 400)
    try:
        await fish_voices.design_voice(FAKE_KEY, "ok", n=5)
        check("fish.design: n=5 refused", False)
    except fish_voices.FishVoiceError:
        check("fish.design: n=5 refused", True)

    # bounded TTL cache
    for i in range(fish_voices.CANDIDATE_MAX + 5):
        fish_voices.store_candidate(b"RIFFxxxxWAVE", "audio/wav", f"i{i}")
    check("fish.design: candidate cache bounded to 24", len(fish_voices._candidates) == fish_voices.CANDIDATE_MAX)
    check("fish.design: oldest candidates evicted first", fish_voices.get_candidate(c[0]["id"]) is None)
    old = fish_voices.store_candidate(b"RIFFxxxxWAVE", "audio/wav", "old")
    fish_voices._candidates[old]["created_at"] -= fish_voices.CANDIDATE_TTL_SEC + 1
    check("fish.design: candidates expire after 30 min", fish_voices.get_candidate(old) is None)
    check("fish.wav_duration parses a WAV header", fish_voices.wav_duration(wav_bytes(1.5)) == 1.5)

    # ---- save
    res = await fish_voices.design_voice(FAKE_KEY, "Soft calm narrator", "Hello there", n=1)
    cid = res["candidates"][0]["id"]
    await fish_voices.list_voices(FAKE_KEY, "self")               # warm the list cache
    voice = await fish_voices.save_designed_voice(FAKE_KEY, "Designed Hikari", candidate_id=cid,
                                                  description="", visibility="private")
    req = fake.calls[-1]
    raw = fake.created[-1]["raw"]
    check("fish.save: POST /model as multipart/form-data", req.method == "POST" and req.url.path == "/model"
          and fake.created[-1]["ctype"].startswith("multipart/form-data"))
    for field, value in (("type", b"tts"), ("train_mode", b"fast"), ("title", b"Designed Hikari"),
                         ("visibility", b"private"), ("enhance_audio_quality", b"true"), ("texts", b"Hello there"),
                         ("voice_design_signatures", b"sig0"), ("description", b"Soft calm narrator")):
        check(f"fish.save: form field {field}", f'name="{field}"'.encode() in raw and value in raw)
    check("fish.save: voices file part carries the candidate WAV",
          b'name="voices"; filename="voice.wav"' in raw and b"RIFF" in raw)
    check("fish.save: returns normalized voice, mine=True",
          voice["id"] == "newvoice1" and voice["mine"] is True and voice["visibility"] == "private")
    check("fish.save: list cache cleared", not fish_voices._list_cache)
    try:
        await fish_voices.save_designed_voice(FAKE_KEY, "x", candidate_id="cand_missing")
        check("fish.save: unknown candidate -> 404", False)
    except fish_voices.FishVoiceError as exc:
        check("fish.save: unknown / expired candidate -> 404", exc.status == 404)
    voice2 = await fish_voices.save_designed_voice(FAKE_KEY, "Raw upload", audio_base64=base64.b64encode(wav_bytes()).decode(),
                                                   enhance=False)
    check("fish.save: audioBase64 path works (enhance false)", voice2["id"] == "newvoice1"
          and b"false" in fake.created[-1]["raw"])
    try:
        await fish_voices.save_designed_voice(FAKE_KEY, "  ", candidate_id=cid)
        check("fish.save: empty title refused", False)
    except fish_voices.FishVoiceError as exc:
        check("fish.save: empty title refused (400)", exc.status == 400)

    # ---- delete
    await fish_voices.list_voices(FAKE_KEY, "self")
    await fish_voices.delete_voice(FAKE_KEY, "mine1")
    check("fish.delete: DELETE /model/{id}", fake.calls[-1].method == "DELETE" and fake.deleted[-1] == "mine1")
    check("fish.delete: list cache cleared", not fish_voices._list_cache)
    try:
        await fish_voices.delete_voice(FAKE_KEY, "../etc")
        check("fish.delete: path-traversal id refused", False)
    except fish_voices.FishVoiceError as exc:
        check("fish.delete: invalid id refused (400), no request", exc.status == 400 and fake.deleted[-1] == "mine1")


async def test_vercel_module() -> None:
    fake = FakeVercel()
    install_vercel(fake)
    cat = await vercel_gateway.list_chat_models_and_providers("fake-gw-key", force=True)
    ids = [m["id"] for m in cat["models"]]
    check("vercel: live source", cat["source"] == "live", cat["source"] + cat["error"])
    check("vercel: language models only, routed ids, 404 (*-batch) dropped, sorted",
          ids == ["deepseek/deepseek-v3.2", "openai/gpt-5-nano"], str(ids))
    nano = next(m for m in cat["models"] if m["id"] == "openai/gpt-5-nano")
    check("vercel: per-model exact upstream providers (order kept, de-duplicated)", nano["providers"] == ["openai", "azure"],
          str(nano["providers"]))
    check("vercel: model fields id/name/owner/type/contextWindow/pricing",
          nano["name"] == "GPT-5 nano" and nano["owner"] == "openai" and nano["type"] == "language"
          and nano["contextWindow"] == 400000 and nano["pricing"]["input"] == "0.00000005")
    check("vercel: global providers sorted + de-duplicated", cat["providers"] == ["azure", "bedrock", "deepinfra", "openai"],
          str(cat["providers"]))
    check("vercel: key sent to the official host only", fake.calls[0].headers.get("authorization") == "Bearer fake-gw-key")
    n = len(fake.calls)
    again = await vercel_gateway.list_chat_models_and_providers("fake-gw-key")
    check("vercel: 5-minute cache hit (no requests)", again["source"] == "cache" and len(fake.calls) == n)
    vercel_gateway._chat_catalog["at"] -= vercel_gateway.CHAT_CATALOG_TTL_SEC + 1
    await vercel_gateway.list_chat_models_and_providers("fake-gw-key")
    check("vercel: stale listing re-fetched, per-model providers reused (1 request)", len(fake.calls) == n + 1,
          str(len(fake.calls) - n))
    fake.offline = True
    vercel_gateway.clear_chat_catalog_cache()
    off = await vercel_gateway.list_chat_models_and_providers("", force=True)
    check("vercel: offline -> graceful fallback list, never raises",
          off["source"] == "fallback" and any(m["id"] == "openai/gpt-5-nano" for m in off["models"]) and off["error"])
    fake.offline = False
    vercel_gateway.clear_chat_catalog_cache()


# ------------------------------------------------------------------------------------------ server handlers
async def test_http(tmp: Path) -> None:
    import server

    fake_fish, fake_vercel = FakeFish(), FakeVercel()
    install_fish(fake_fish)
    install_vercel(fake_vercel)
    env_fish = {"key": ""}
    orig_fish_key, orig_get, orig_resolve = server_keys.fish_key, server_keys.get, server.resolve_gateway_key
    server_keys.fish_key = lambda: env_fish["key"]                      # type: ignore[assignment]
    server_keys.get = lambda name: ""                                   # type: ignore[assignment]
    server.resolve_gateway_key = lambda cfg: str(cfg.get("vercelApiKey") or "")   # type: ignore[assignment]

    class FakeRuntime:
        def __init__(self) -> None:
            (tmp / "http").mkdir()
            self.config = ConfigStore(tmp / "http" / "settings.json")
            self.config.data["ai"]["fishApiKey"] = ""
            self.config.data["ai"]["vercelApiKey"] = ""
            self._shutdown_task = None
            self._fish_ended_on_shutdown = True
            self.broadcasts: list = []
            self.applied: list = []

        def apply_ai_config(self, vals) -> None:
            self.applied.append(dict(vals))

        def start_loop(self, loop) -> None:
            pass

        def stop_loop(self) -> None:
            pass

        async def broadcast_json(self, msg, exclude=None) -> None:
            self.broadcasts.append(msg)

        class memory:                                   # noqa: N801 - on_cleanup closes it
            @staticmethod
            async def close() -> None:
                pass

    rt = FakeRuntime()
    app = server.build_app(rt)
    try:
        async with TestClient(TestServer(app, host="127.0.0.1")) as cl:
            r = await cl.get("/api/fish/voices?scope=self")
            j = await r.json()
            check("http. GET /api/fish/voices unconfigured -> 200 configured:false, empty voices",
                  r.status == 200 and j == {"ok": True, "configured": False, "scope": "self",
                                            "activeVoiceId": j.get("activeVoiceId"), "voices": []}, str(j))
            r = await cl.post("/api/fish/voice-design", json={"instruction": "warm"})
            check("http. voice-design unconfigured -> 400", r.status == 400)
            r = await cl.delete("/api/fish/voices/mine1")
            check("http. delete unconfigured -> 400", r.status == 400)

            env_fish["key"] = FAKE_KEY                  # server-managed key (FISH_API_KEY / .env.local)
            r = await cl.get("/api/fish/voices?scope=all&limit=30")
            j = await r.json()
            check("http. GET /api/fish/voices scope=all configured", r.status == 200 and j["configured"] and j["scope"] == "all"
                  and [v["id"] for v in j["voices"]] == ["mine1", "shared1", "training1", "pub1"], str(j)[:300])
            check("http. activeVoiceId is ai.fishReferenceId", j["activeVoiceId"] == rt.config.data["ai"].get("fishReferenceId"))
            r = await cl.get("/api/fish/voices?scope=bogus")
            check("http. unknown scope falls back to self", (await r.json())["scope"] == "self")
            r = await cl.get("/api/fish/voices?page=abc")
            check("http. non-integer page -> 400", r.status == 400)

            r = await cl.post("/api/fish/voice-design", json={"instruction": "   "})
            check("http. voice-design empty instruction -> 400", r.status == 400)
            r = await cl.post("/api/fish/voice-design", json={"instruction": "Bright anime girl", "referenceText": "Hi!",
                                                              "language": "en", "n": 2})
            j = await r.json()
            check("http. POST /api/fish/voice-design -> candidates", r.status == 200 and j["ok"] and len(j["candidates"]) == 2
                  and {"id", "audioBase64", "mime", "durationSec"} <= set(j["candidates"][0]), str(j)[:200])
            cid = j["candidates"][0]["id"]

            r = await cl.post("/api/fish/voices/save", json={"candidateId": cid, "title": "Designed Hikari",
                                                             "description": "d", "visibility": "private", "applyNow": True})
            j = await r.json()
            check("http. POST /api/fish/voices/save -> voice + applied", r.status == 200 and j["ok"] and j["applied"]
                  and j["voice"]["id"] == "newvoice1" and j["voice"]["mine"], str(j)[:300])
            check("http. applyNow sets ai.fishReferenceId (TTS voice) and pushes it into the runtime",
                  rt.config.data["ai"]["fishReferenceId"] == "newvoice1" and j["ai"]["fishReferenceId"] == "newvoice1"
                  and any(a.get("fishReferenceId") == "newvoice1" for a in rt.applied))
            saved = json.loads((tmp / "http" / "settings.json").read_text(encoding="utf-8"))
            check("http. applyNow persisted settings.json", saved["ai"]["fishReferenceId"] == "newvoice1")
            check("http. applyNow broadcast config_sync", any(b.get("type") == "config_sync" for b in rt.broadcasts))
            rt.config.data["ai"]["fishApiKey"] = "saved-override-key"
            r = await cl.post("/api/fish/voices/save", json={"audioBase64": base64.b64encode(wav_bytes()).decode(),
                                                             "title": "Upload", "applyNow": False})
            j = await r.json()
            check("http. save without applyNow leaves the active voice", j["applied"] is False
                  and rt.config.data["ai"]["fishReferenceId"] == "newvoice1")
            check("http. saved fishApiKey override is used and masked in the response",
                  fake_fish.calls[-1].headers["authorization"] == "Bearer saved-override-key"
                  and j["ai"]["fishApiKey"] != "saved-override-key")
            r = await cl.post("/api/fish/voices/save", json={"candidateId": "cand_gone", "title": "x"})
            check("http. save expired candidate -> 404", r.status == 404)

            r = await cl.delete("/api/fish/voices/mine1")
            j = await r.json()
            check("http. DELETE /api/fish/voices/{id}", r.status == 200 and j == {"ok": True, "deleted": "mine1"})
            r = await cl.delete("/api/fish/voices/bad%20id")
            check("http. DELETE invalid id -> 400", r.status == 400)

            r = await cl.get("/api/vercel/models")
            j = await r.json()
            check("http. GET /api/vercel/models unconfigured still lists models", r.status == 200 and j["ok"]
                  and j["configured"] is False and len(j["models"]) == 2 and j["providers"], str(j)[:300])
            rt.config.data["ai"].update({"vercelApiKey": "gw-key", "vercelModel": "deepseek/deepseek-v3.2",
                                         "vercelProviderOrder": ["deepinfra", "bedrock"], "vercelProviderOnly": True})
            r = await cl.get("/api/vercel/models?force=1")
            j = await r.json()
            check("http. /api/vercel/models configured + activeModel/activeProvider",
                  j["configured"] and j["activeModel"] == "deepseek/deepseek-v3.2" and j["activeProvider"] == "deepinfra"
                  and j["providerOnly"] == ["deepinfra", "bedrock"], str({k: j[k] for k in ("configured", "activeModel", "activeProvider")}))
            m = next(x for x in j["models"] if x["id"] == "deepseek/deepseek-v3.2")
            check("http. model rows carry providers[]", m["providers"] == ["deepinfra", "bedrock"] and m["type"] == "language")

            evil = {"Origin": "https://evil.example"}
            for method, path, kw in (("post", "/api/fish/voice-design", {"json": {"instruction": "x"}}),
                                     ("post", "/api/fish/voices/save", {"json": {"candidateId": "x", "title": "x"}}),
                                     ("delete", "/api/fish/voices/mine1", {}),
                                     ("get", "/api/fish/voices", {}), ("get", "/api/vercel/models", {})):
                n = len(fake_fish.calls)
                r = await getattr(cl, method)(path, headers=evil, **kw)
                check(f"http. foreign Origin -> 403 on {method.upper()} {path}", r.status == 403 and len(fake_fish.calls) == n)
    finally:
        server_keys.fish_key, server_keys.get, server.resolve_gateway_key = orig_fish_key, orig_get, orig_resolve


async def amain() -> int:
    await test_fish_module()
    await test_vercel_module()
    with tempfile.TemporaryDirectory(prefix="hikari-voice-test-") as td:
        await test_http(Path(td))
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed", flush=True)
    return 0 if passed == len(RESULTS) else 1


def test_all() -> None:                      # pytest entry point
    assert asyncio.run(amain()) == 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
