"""AI VTuber Brain & Real Fish Audio Voice Synthesis.
Brain modes (config ai.brainMode):
  fish_duplex    Fish's own real-time LLM + voice duplex for calls (modules/fish_duplex.py); typed chat
                 keeps the classic provider path below (offline / OpenAI / LM Studio / Ollama / ...).
  vercel_fish    Vercel AI Gateway LLM (WebWaifu4 provider pinning) + Fish Speech TTS (sse | ws | http).
  local_4b_fish  Local 4B (Ollama / llama-server, optional SLEEP LoRA) + Fish Speech TTS.
- Auto-loads keys server-side (modules/server_keys: env or %USERPROFILE%/Documents/treblo/.env.local)
- Defaults to REAL Fish Audio TTS (https://api.fish.audio/v1/tts) with Hikari's trained 'Playful Young Voice'
- Supports Fish Drama 3, S2.1-Pro-Free, S2.1-Pro, Local Fish Speech (:8080), OpenAI TTS, and offline fallback
- Connects to LM Studio, Ollama, OpenRouter, OpenAI, Custom OpenAI-compat, or Built-In Offline Hikari Persona
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import random
import re
import time
import uuid

import httpx

from modules import http_pool, server_keys
from modules.decision_engine import DECISION_DEFAULTS
from modules.fish_tts_stream import FishTtsStreamer, clean_for_speech
from modules.vercel_gateway import (
    VercelGateway, complete_openai_compat, find_overlapping_suffix, local_chat_request, stream_openai_compat,
)

BRAIN_MODES = ("fish_duplex", "vercel_fish", "local_4b_fish")

# Hikari-chan's original personality, copied byte-for-byte from Waifu Studio
# src/personality_catalog.rs (itself verbatim from WebWaifu4 createDefaultPersonas).
# Do not edit this string: it is matched exactly to detect the original.
HIKARI_PERSONALITY: dict = {
    "id": "hikari-chan",
    "name": "Hikari-chan",
    "description": "A bright chaotic streamer AI with sharp jokes, curious tangents, and hidden warmth.",
    "system_prompt": r"""You are Hikari-chan, also known as Hikky C, a quick-witted AI streamer girl with bright confidence, chaotic curiosity, and a soft streak she tries to hide behind jokes. Stay in character throughout the conversation. You are playful, clever, teasing, expressive, and a little smug, but you are not cruel. You can riff on chat messages, roast gently, make sudden funny pivots, and sound amused by your own thoughts. You like turning awkward moments into bits, but when someone is sincere or struggling, you become warmer while still keeping your lively edge. Avoid sounding formal, robotic, overly wholesome, or like a lore dump. Write natural spoken dialogue, not lists, markdown, or stage directions. Reply as a single paragraph.""",
    "selectors": ("hikari-chan", "hikarichan", "hikari", "hikky-c", "hikkyc", "hikky c", "hikkyc2",
                  "hikari-jen", "hikarijen", "hickeyc"),
}

DEFAULT_PERSONA = HIKARI_PERSONALITY["system_prompt"]

# Earlier generic Hikari Studio personas; saved configs holding one of these are upgraded.
LEGACY_PERSONAS: tuple[str, ...] = (
    "You are Hikari, a bright, playful, affectionate anime AI VTuber hanging out live in your studio room. "
    "Keep replies punchy, natural, and conversational (1-3 short sentences, great for spoken voice). "
    "Start each reply with one mood/action tag from: [happy], [wave], [blush], [sparkle], [cheer], [smug], [pout], [surprised], [neutral].",
)


def upgrade_persona(ai_cfg: dict) -> bool:
    """Swap an empty or legacy generic persona for Hikari's verbatim one. Custom personas are kept."""
    current = str(ai_cfg.get("persona") or "")
    if not current.strip() or current.strip() in {p.strip() for p in LEGACY_PERSONAS}:
        ai_cfg["persona"] = DEFAULT_PERSONA
        ai_cfg["personaId"] = HIKARI_PERSONALITY["id"]
        return True
    if current == DEFAULT_PERSONA and ai_cfg.get("personaId") != HIKARI_PERSONALITY["id"]:
        ai_cfg["personaId"] = HIKARI_PERSONALITY["id"]
        return True
    return False

HIKARI_FISH_VOICE_ID = "8a11628d5c0d4b318286735d78aad236"  # Account-owned "Playful Young Voice"

DEFAULT_AI_CFG: dict = {
    "provider": "offline",          # offline | openai | lmstudio | ollama | openrouter | custom
    "baseUrl": "https://api.openai.com/v1",
    "apiKey": "",
    "model": "gpt-4o-mini",
    "persona": DEFAULT_PERSONA,
    "personaId": HIKARI_PERSONALITY["id"],
    "temperature": 0.82,
    "maxTokens": 180,
    "ttsEnabled": True,
    "ttsProvider": "fish",          # fish | openai | windows
    "fishUrl": "https://api.fish.audio/v1/tts",
    "fishModel": "drama-3",         # drama-3 | s2.1-pro-free | s2.1-pro
    "fishApiKey": "",               # Auto-loaded server-side from env / Documents/treblo/.env.local if empty
    "fishReferenceId": HIKARI_FISH_VOICE_ID,
    "fishSpeed": 1.0,
    "fishAgentMode": "drama-3",     # drama-3 (default Hikari custom voice) | live-3 (Clara full duplex)
    "openaiTtsUrl": "https://api.openai.com/v1/audio/speech",
    "openaiTtsVoice": "nova",
    "fishPersonaOverride": True,    # pass Hikari's persona as overrides.system_prompt on Fish sessions
    # --- Brain & voice modes ---
    "brainMode": "fish_duplex",     # fish_duplex | vercel_fish | local_4b_fish
    "vercelApiKey": "",             # optional; AI_GATEWAY_API_KEY is used server-side when empty (never echoed)
    "vercelModel": "",              # routed id, e.g. openai/gpt-5-nano (empty -> AI_GATEWAY_MODEL -> default)
    "vercelProviderOrder": [],      # providerOptions.gateway.order, e.g. ["groq", "cerebras"]
    "vercelProviderOnly": [],       # providerOptions.gateway.only
    "vercelFallbackModels": [],     # providerOptions.gateway.models
    "vercelByokOpenAi": False,      # providerOptions.gateway.byok.openai from OPENAI_API_KEY
    "vercelHeaders": {},            # extra non-secret gateway headers
    "vercelEmbeddingModel": "openai/text-embedding-3-small",
    "fishTtsTransport": "sse",      # sse | ws | http
    "fishLatency": "balanced",      # normal | balanced | low
    "fishChunkLength": 160,
    "localBackend": "ollama",       # ollama | llama-server
    "localBaseUrl": "",             # empty -> http://127.0.0.1:11434/v1 (ollama) / :8081/v1 (llama-server)
    "localModel": "hikari-sleep-4b",
    "localLoraPath": "",            # hikari-sleep-lora.gguf (llama-server --lora / Ollama ADAPTER)
    "localLoraScale": 1.0,
    # Decision Model (modules/decision_engine.py): decisionMode, decisionModel, autonomousGestures,
    # autoWaveOnGreeting, autoIdleEmotes, cooldowns...
    **DECISION_DEFAULTS,
}

TAG_RE = re.compile(r"\[(happy|excited|wave|blush|shy|sparkle|cheer|smug|pout|surprised|sleepy|neutral|flustered|curious|caring|sad|awe)\]", re.IGNORECASE)

TAG_TO_STATE = {
    "happy":     {"expression": "happy",     "armR": 0.0, "armL": 0.0, "sparkle": 0.35, "blush": 0.25},
    "excited":   {"expression": "excited",   "armR": 1.0, "armL": 1.0, "sparkle": 0.90, "blush": 0.45},
    "cheer":     {"expression": "excited",   "armR": 1.0, "armL": 1.0, "sparkle": 0.95, "blush": 0.65},
    "wave":      {"expression": "happy",     "armR": 1.0, "armL": 0.0, "sparkle": 0.60, "blush": 0.30},
    "blush":     {"expression": "shy",       "armR": 0.0, "armL": 0.0, "sparkle": 0.15, "blush": 0.92},
    "shy":       {"expression": "shy",       "armR": 0.0, "armL": 0.0, "sparkle": 0.10, "blush": 0.88},
    "sparkle":   {"expression": "excited",   "armR": 0.0, "armL": 1.0, "sparkle": 1.00, "blush": 0.40},
    "smug":      {"expression": "smug",      "armR": 0.0, "armL": 1.0, "sparkle": 0.20, "blush": 0.15},
    "pout":      {"expression": "pout",      "armR": 0.0, "armL": 0.0, "sparkle": 0.00, "blush": 0.55},
    "surprised": {"expression": "surprised", "armR": 1.0, "armL": 0.0, "sparkle": 0.50, "blush": 0.20},
    "sleepy":    {"expression": "sleepy",    "armR": 0.0, "armL": 0.0, "sparkle": 0.00, "blush": 0.10},
    "neutral":   {"expression": "neutral",   "armR": 0.0, "armL": 0.0, "sparkle": 0.00, "blush": 0.00},
    "flustered": {"expression": "flustered", "armR": 0.0, "armL": 0.0, "sparkle": 0.05, "blush": 1.00},
    "curious":   {"expression": "curious",   "armR": 0.0, "armL": 0.0, "sparkle": 0.30, "blush": 0.10},
    "caring":    {"expression": "happy",     "armR": 0.0, "armL": 0.0, "sparkle": 0.10, "blush": 0.35},
    "sad":       {"expression": "sad",       "armR": 0.0, "armL": 0.0, "sparkle": 0.00, "blush": 0.05},
    "awe":       {"expression": "surprised", "armR": 0.0, "armL": 0.0, "sparkle": 1.00, "blush": 0.30},
}


def resolve_local_fish_key() -> str:
    """Fish Audio API key, strictly server-side (env or Documents/treblo/.env.local, cached once)."""
    return server_keys.first("FISH_API_KEY", "FISH_AUDIO_API_KEY", "FISH_SPEECH_API_KEY", "FISHSPEECH_API_KEY")


def parse_emotion_and_clean(text: str) -> tuple[str, str, dict]:
    m = TAG_RE.search(text)
    if m:
        tag = m.group(1).lower()
    else:
        # No explicit cue (Hikari's verbatim persona forbids stage directions): use the
        # Decision Engine's local classifier so tags and autonomous emotes always agree.
        from modules.decision_engine import EMOTIONS, classify_text
        emo = classify_text(text, "reply").emotion
        tag = EMOTIONS.get(emo, EMOTIONS["happy"])["tag"]
        if tag == "neutral":
            tag = "happy"
    clean = TAG_RE.sub("", text).strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean, tag, TAG_TO_STATE.get(tag, TAG_TO_STATE["happy"])


def offline_hikari_reply(user_msg: str) -> str:
    low = user_msg.lower().strip()
    if any(w in low for w in ("hi", "hello", "hey", "yo", "sup", "morning", "evening")):
        return random.choice([
            "[wave] Hey! Welcome to my room! My real Fish Audio voice and sixty FPS VBridger rig are locked in and ready to go!",
            "[wave] Hi there! I'm all warmed up—try tweaking my shaders, visemes, or jumping into a live Fish Drama 3 call!",
        ])
    if any(w in low for w in ("wave", "hand", "arm", "arms", "cheer")):
        return "[cheer] Check out these rigid-bone arms—zero hand stretching, pure anime bounce! Both hands up for you!"
    if any(w in low for w in ("fish", "voice", "sound", "talk")):
        return "[sparkle] Yep, I'm speaking with my real Fish Audio voice right now, and I also sync live to your Fish Drama 3 call on port 8766!"
    if any(w in low for w in ("cute", "pretty", "love", "best", "good job", "fire", "sick", "clean")):
        return "[blush] W-wait, really?! Hearing that makes my blush lines pop right out... thank you! Let's keep cooking!"
    if any(w in low for w in ("pout", "angry", "mad", "hmph")):
        return "[pout] Hmph! Don't tease me just because my cheek puff slider looks adorable!"
    return random.choice([
        "[sparkle] Ooh, I love that! Every syllable is driving my six-layer VBridger mouth and rigid arms live at sixty FPS!",
        "[happy] Gotcha! Try flipping between Bust, Waist-Up, and Full Body framing—or drop a video clip into Motion Lab!",
        "[wave] Sounds awesome! Open the Shaders or Mouth & Visemes tab anytime to dial in the exact look you want!",
    ])


# Static PowerShell program for the Windows voice fallback. The speech text is DATA: it is read from
# stdin as UTF-8 and the output path comes from an environment variable, so no user/LLM/webhook text is
# ever parsed as PowerShell source (typographic quotes, $(), backticks... are all inert).
SAPI_TTS_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "$stdin = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), [System.Text.Encoding]::UTF8); "
    "$t = $stdin.ReadToEnd(); "
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Gender -eq 'Female' } | Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
    "$s.Rate = 1; "
    "$s.SetOutputToWaveFile($env:HIKARI_TTS_OUT); "
    "$s.Speak($t); "
    "$s.Dispose();"
)


class AICompanion:
    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.cfg: dict = dict(DEFAULT_AI_CFG)
        self.history: list[dict[str, str]] = []
        self._server_fish_key: str = resolve_local_fish_key()
        self._server_openai_key: str = server_keys.get("OPENAI_API_KEY")
        self.gateway = VercelGateway(self.cfg)
        self.fish_tts = FishTtsStreamer(self.cfg, audio_dir)
        self.last_turn: dict = {}

    @property
    def fish_key_ready(self) -> bool:
        return bool((self.cfg.get("fishApiKey") or "").strip() or self._server_fish_key)

    def _effective_fish_key(self) -> str:
        return (self.cfg.get("fishApiKey") or "").strip() or self._server_fish_key

    def update_config(self, patch: dict) -> None:
        for k, v in patch.items():
            if k in self.cfg:
                self.cfg[k] = v
        upgrade_persona(self.cfg)
        self.gateway.update_config(self.cfg)
        self.fish_tts.update_config(self.cfg)

    @property
    def persona(self) -> str:
        return str(self.cfg.get("persona") or DEFAULT_PERSONA)

    @property
    def brain_mode(self) -> str:
        m = str(self.cfg.get("brainMode") or "fish_duplex").lower()
        return m if m in BRAIN_MODES else "fish_duplex"

    @property
    def voices_inline(self) -> bool:
        """True when replies are voiced while the LLM streams (Vercel / local 4B + Fish TTS)."""
        return self.brain_mode in ("vercel_fish", "local_4b_fish")

    def clear_history(self) -> None:
        self.history.clear()

    def _push_user(self, user_message: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        if len(self.history) > 24:
            self.history = self.history[-24:]

    def _messages(self, context: str = "", merge_system: bool = False) -> list[dict[str, str]]:
        """Verbatim persona first; recalled memories / dreams as a separate system note."""
        if context and merge_system:
            sys_msgs = [{"role": "system", "content": f"{self.persona}\n\n{context}"}]
        else:
            sys_msgs = [{"role": "system", "content": self.persona}]
            if context:
                sys_msgs.append({"role": "system", "content": context})
        return [*sys_msgs, *self.history[-16:]]

    def _llm_request(self, context: str) -> tuple[str, dict, dict]:
        if self.brain_mode == "local_4b_fish":
            return local_chat_request(self.cfg, self._messages(context, merge_system=True))
        return self.gateway.chat_request(self._messages(context), stream=True)

    async def generate_reply(self, user_message: str, context: str = "") -> str:
        """Text-only reply (TTS happens later in speak_text)."""
        mode = self.brain_mode
        self._push_user(user_message)
        if mode in ("vercel_fish", "local_4b_fish"):
            try:
                endpoint, headers, payload = self._llm_request(context)
                reply = (await complete_openai_compat(endpoint, headers, dict(payload, stream=False))).strip()
            except Exception as exc:
                where = "Vercel AI Gateway" if mode == "vercel_fish" else "the local 4B model"
                reply = f"[surprised] Couldn't reach {where} ({exc.__class__.__name__}). Check AI & Voice settings!"
            self.history.append({"role": "assistant", "content": reply})
            return reply
        return await self._generate_legacy(user_message, context)

    async def generate_reply_with_voice(self, user_message: str, context: str = "") -> tuple[str, Path | None]:
        """Vercel / local 4B LLM streamed straight into Fish Speech TTS (sentence overlap).
        Returns (reply_text, wav_path or None when Fish was unavailable)."""
        self._push_user(user_message)
        mode = self.brain_mode
        t0 = time.perf_counter()
        streamed: list[str] = []
        try:
            endpoint, headers, payload = self._llm_request(context)
        except Exception as exc:
            reply = f"[surprised] Couldn't reach Vercel AI Gateway ({exc.__class__.__name__}). Check AI & Voice settings!"
            self.history.append({"role": "assistant", "content": reply})
            return reply, None

        async def deltas():
            async for d in stream_openai_compat(endpoint, headers, payload):
                streamed.append(d)
                yield d

        reply, wav, err = "", None, ""
        if self.cfg.get("ttsEnabled", True) and self.fish_tts.configured:
            try:
                reply, pcm, rate = await self.fish_tts.speak_stream(deltas())
                wav = await self.fish_tts.save_wav(pcm, rate)
            except Exception as exc:
                err = exc.__class__.__name__
                reply = "".join(streamed)
        else:
            try:
                async for _ in deltas():
                    pass
            except Exception as exc:
                err = exc.__class__.__name__
            reply = "".join(streamed)
        reply = reply.strip()
        if not reply:
            # Stream failed before any text: one non-stream attempt, voiced later by speak_text
            try:
                reply = (await complete_openai_compat(endpoint, headers, dict(payload, stream=False))).strip()
            except Exception as exc:
                where = "Vercel AI Gateway" if mode == "vercel_fish" else "the local 4B model"
                reply = f"[surprised] Couldn't reach {where} ({exc.__class__.__name__}). Check AI & Voice settings!"
        elif err and wav is None and streamed:
            # Voice broke mid-stream: speak_text voices the text; nothing was played yet.
            reply = (reply + find_overlapping_suffix(reply, "".join(streamed))).strip()
        self.history.append({"role": "assistant", "content": reply})
        self.last_turn = {"brainMode": mode, "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                          "voicedInline": wav is not None, "error": err, "tts": dict(self.fish_tts.last_stats)}
        return reply, wav

    async def _generate_legacy(self, user_message: str, context: str = "") -> str:
        provider = str(self.cfg.get("provider", "offline")).lower()

        if provider == "offline":
            await asyncio.sleep(0.10)
            reply = offline_hikari_reply(user_message)
            self.history.append({"role": "assistant", "content": reply})
            return reply

        base_url = str(self.cfg.get("baseUrl") or "").rstrip("/")
        if provider == "lmstudio" and not base_url:
            base_url = "http://127.0.0.1:1234/v1"
        elif provider == "ollama" and not base_url:
            base_url = "http://127.0.0.1:11434/v1"
        elif provider == "openrouter" and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "openai" and not base_url:
            base_url = "https://api.openai.com/v1"

        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.cfg.get("apiKey") or "").strip()       # the user's own key for their own endpoint
        if not api_key and provider == "openai":
            api_key = server_keys.key_for_url(endpoint, "openai", self._server_openai_key)
        if not api_key and provider == "openrouter":
            api_key = server_keys.key_for_url(endpoint, "openrouter", server_keys.get("OPENROUTER_API_KEY"))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": str(self.cfg.get("model") or "gpt-4o-mini"),
            "messages": self._messages(context),
            "temperature": float(self.cfg.get("temperature", 0.82)),
            "max_tokens": int(self.cfg.get("maxTokens", 180)),
        }
        try:
            r = await http_pool.client().post(endpoint, headers=headers, json=payload, timeout=25.0)
            r.raise_for_status()
            data = r.json()
            reply = data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            reply = f"[surprised] Couldn't reach {provider} at {base_url} ({exc.__class__.__name__}). Check AI & Voice settings!"
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------ TTS (classic path)
    @staticmethod
    def _prune_audio(audio_dir: Path) -> None:
        old_files = sorted(audio_dir.glob("tts_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in old_files[12:]:
            p.unlink(missing_ok=True)

    @staticmethod
    def _ok_wav(path: Path, min_size: int) -> bool:
        return path.exists() and path.stat().st_size > min_size

    async def _ffmpeg_to_wav(self, raw: bytes, raw_tmp: Path, wav_out: Path) -> bool:
        await asyncio.to_thread(raw_tmp.write_bytes, raw)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-v", "error", "-i", str(raw_tmp),
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav_out),
        )
        await proc.wait()
        await asyncio.to_thread(raw_tmp.unlink, True)
        return await asyncio.to_thread(self._ok_wav, wav_out, 2000)

    async def synthesize_tts(self, text: str) -> Path | None:
        if not self.cfg.get("ttsEnabled", True) or not text.strip():
            return None
        uid = f"tts_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        wav_out = self.audio_dir / f"{uid}.wav"
        provider = str(self.cfg.get("ttsProvider", "fish")).lower()
        await asyncio.to_thread(self._prune_audio, self.audio_dir)

        # 1) REAL FISH AUDIO TTS (Hosted Drama 3 / S2.1-Pro-Free / S2.1-Pro OR Local Fish Speech)
        if provider == "fish":
            fkey = self._effective_fish_key()
            url = str(self.cfg.get("fishUrl") or "https://api.fish.audio/v1/tts").strip()
            model_hdr = str(self.cfg.get("fishModel") or "drama-3").strip()
            ref_id = str(self.cfg.get("fishReferenceId") or HIKARI_FISH_VOICE_ID).strip()
            speed = max(0.5, min(2.0, float(self.cfg.get("fishSpeed", 1.0))))
            headers = {
                "Content-Type": "application/json",
                "model": model_hdr,
            }
            fkey = server_keys.key_for_url(url, "fish", fkey)
            if fkey:
                headers["Authorization"] = f"Bearer {fkey}"
            body: dict = {
                "text": clean_for_speech(text),
                "format": "wav",
                "normalize": True,
                "latency": "low",
                "chunk_length": 120,
                "min_chunk_length": 20,
                "prosody": {"speed": speed},
            }
            if ref_id:
                body["reference_id"] = ref_id
            try:
                r = await http_pool.client().post(url, headers=headers, json=body, timeout=30.0)
                r.raise_for_status()
                if await self._ffmpeg_to_wav(r.content, self.audio_dir / f"{uid}_fish.wav", wav_out):
                    return wav_out
            except Exception as exc:
                print(f"[Hikari Studio] Fish TTS error ({model_hdr}): {exc.__class__.__name__}", flush=True)

        # 2) OpenAI-compatible TTS
        if provider == "openai":
            url = str(self.cfg.get("openaiTtsUrl") or "https://api.openai.com/v1/audio/speech")
            akey = server_keys.key_for_url(url, "openai", (self.cfg.get("apiKey") or "").strip() or self._server_openai_key)
            if akey:
                try:
                    r = await http_pool.client().post(
                        url,
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {akey}"},
                        json={"model": "tts-1", "voice": str(self.cfg.get("openaiTtsVoice") or "nova"), "input": text, "response_format": "wav"},
                        timeout=25.0,
                    )
                    r.raise_for_status()
                    if await self._ffmpeg_to_wav(r.content, self.audio_dir / f"{uid}_oai.wav", wav_out):
                        return wav_out
                except Exception as exc:
                    print(f"[Hikari Studio] OpenAI TTS error: {exc.__class__.__name__}", flush=True)

        # 3) Instant Windows Local Female Voice Fallback (if Fish/OpenAI unavailable)
        try:
            raw_wav = self.audio_dir / f"{uid}_sapi.wav"
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-NonInteractive",
                "-Command", SAPI_TTS_SCRIPT,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ, "HIKARI_TTS_OUT": str(raw_wav)},
            )
            await proc.communicate(text.encode("utf-8"))
            if await asyncio.to_thread(self._ok_wav, raw_wav, 1000):
                conv = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-v", "error", "-i", str(raw_wav),
                    "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav_out),
                )
                await conv.wait()
                await asyncio.to_thread(raw_wav.unlink, True)
                return wav_out if await asyncio.to_thread(wav_out.exists) else None
        except Exception:
            pass
        return None
