"""Vercel AI Gateway (OpenAI-compatible, https://ai-gateway.vercel.sh/v1) + local OpenAI-compat LLMs.

Mirrors WebWaifu4's gateway wiring (runtimeProviderRouting.ts, provider-defaults.ts,
AiSdkGatewayProvider.ts) and treblo/lib/ai-gateway.ts:
  * key: AI_GATEWAY_API_KEY (or VERCEL_OIDC_TOKEN), server-side only; model: config -> AI_GATEWAY_MODEL
    -> 'openai/gpt-5-nano'. Models must be routed ids ('provider/model'); premium o1/*-pro ids are
    refused and replaced by the default (WebWaifu4 isPremiumCostModelId / pickProviderModel).
  * providerOptions.gateway: order / only / models (fallback models) / byok, exactly as the gateway
    expects; providerOptions.openai: reasoningEffort 'minimal' + reasoningSummary 'auto' for gpt-5.
  * streaming via SSE `data:` chunks; embeddings via /v1/embeddings (openai/text-embedding-3-small).
All network I/O uses the shared pooled client (modules.http_pool).
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import httpx

from modules import http_pool, server_keys

VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_AI_GATEWAY_MODEL = "openai/gpt-5-nano"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Gateway app-attribution headers (safe, non-secret)
DEFAULT_GATEWAY_HEADERS = {"http-referer": "https://hikari.studio.local", "x-title": "Hikari Studio"}
_BLOCKED_HEADERS = {"authorization", "content-type", "content-length", "host", "cookie"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class GatewayNotConfigured(RuntimeError):
    pass


# --------------------------------------------------------------------------- model rules
def is_premium_cost_model_id(value: Any) -> bool:
    """Port of WebWaifu4 isPremiumCostModelId (o1 / *-pro OpenAI models are refused)."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    parts = normalized.split("/")
    provider = parts[0] if len(parts) > 1 else ""
    leaf = normalized.split("/")[-1].replace("_", ".")
    is_openai = not provider or provider == "openai"
    is_o1 = leaf == "o1" or leaf.startswith("o1-") or leaf.startswith("o1.") or leaf.startswith("o1pro")
    is_pro = re.search(r"(^|[.-])pro([.-]|$)", leaf) is not None
    return is_openai and (is_o1 or is_pro)


def is_openai_gpt5_model(model: str) -> bool:
    normalized = model.strip().lower()
    provider = normalized.split("/")[0] if "/" in normalized else "openai"
    leaf = re.sub(r"[_ .]+", "-", normalized.split("/")[-1])
    return provider == "openai" and leaf.startswith("gpt-5")


def default_model() -> str:
    env = server_keys.get("AI_GATEWAY_MODEL")
    return env if env and "/" in env and not is_premium_cost_model_id(env) else DEFAULT_AI_GATEWAY_MODEL


def pick_gateway_model(value: Any) -> str:
    """WebWaifu4 pickProviderModel for 'vercel-gateway': routed ids only, premium ids refused."""
    model = str(value or "").strip()
    if not model or is_premium_cost_model_id(model) or "/" not in model:
        return default_model()
    return model


def _id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value) if v]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for v in value:
        s = str(v).strip().lower()
        if _ID_RE.match(s) and s not in out:
            out.append(s)
    return out[:12]


def _model_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [v for v in re.split(r"[,\s]+", value) if v]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for v in value:
        s = str(v).strip()
        if "/" in s and not is_premium_cost_model_id(s) and s not in out:
            out.append(s)
    return out[:6]


# --------------------------------------------------------------------------- payloads
def provider_only(cfg: dict[str, Any]) -> list[str]:
    """`vercelProviderOnly`: True = restrict strictly to `vercelProviderOrder`; a list/string = explicit set."""
    raw = cfg.get("vercelProviderOnly")
    if raw is True:
        return _id_list(cfg.get("vercelProviderOrder"))
    if raw is False or raw is None:
        return []
    return _id_list(raw)


def build_provider_options(cfg: dict[str, Any], model: str, *, byok_allowed: bool = False) -> dict[str, Any] | None:
    """providerOptions for the gateway body (createProviderOptions in AiSdkGatewayProvider.ts,
    plus the gateway routing controls order / only / models). The BYOK OpenAI key is only placed in
    a body bound for the official gateway host (`byok_allowed`)."""
    provider_options: dict[str, Any] = {}
    if is_openai_gpt5_model(model):
        provider_options["openai"] = {"reasoningEffort": "minimal", "reasoningSummary": "auto"}
    gateway: dict[str, Any] = {}
    order = _id_list(cfg.get("vercelProviderOrder"))
    only = provider_only(cfg)
    fallbacks = [m for m in _model_list(cfg.get("vercelFallbackModels")) if m != model]
    if order:
        gateway["order"] = order
    if only:
        gateway["only"] = only
    if fallbacks:
        gateway["models"] = fallbacks
    if cfg.get("vercelByokOpenAi") and byok_allowed:
        byok_key = server_keys.get("OPENAI_API_KEY")
        if byok_key:
            gateway["byok"] = {"openai": [{"apiKey": byok_key}]}
    if gateway:
        provider_options["gateway"] = gateway
    return provider_options or None


def gateway_headers(cfg: dict[str, Any], key: str) -> dict[str, str]:
    headers = dict(DEFAULT_GATEWAY_HEADERS)
    extra = cfg.get("vercelHeaders")
    if isinstance(extra, dict):
        for k, v in extra.items():
            k2 = str(k).strip().lower()
            if k2 and k2 not in _BLOCKED_HEADERS and re.match(r"^[a-z0-9-]{1,64}$", k2):
                headers[k2] = str(v)[:256]
    headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def resolve_gateway_key(cfg: dict[str, Any]) -> str:
    return str(cfg.get("vercelApiKey") or "").strip() or server_keys.gateway_key()


def build_chat_payload(cfg: dict[str, Any], messages: list[dict[str, str]], *, stream: bool,
                       max_tokens: int | None = None, temperature: float | None = None,
                       json_mode: bool = False, byok_allowed: bool = False) -> dict[str, Any]:
    model = pick_gateway_model(cfg.get("vercelModel"))
    tokens = int(max_tokens if max_tokens is not None else cfg.get("maxTokens", 180))
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if is_openai_gpt5_model(model):
        # gpt-5 family: fixed temperature; leave room for (minimal) reasoning tokens
        payload["max_tokens"] = tokens + 256
    else:
        payload["max_tokens"] = tokens
        payload["temperature"] = float(temperature if temperature is not None else cfg.get("temperature", 0.82))
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    po = build_provider_options(cfg, model, byok_allowed=byok_allowed)
    if po:
        payload["providerOptions"] = po
    return payload


# --------------------------------------------------------------------------- transport
async def _iter_sse_deltas(resp: httpx.Response) -> AsyncIterator[str]:
    async for line in resp.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("error"):
            raise RuntimeError("LLM stream reported an error")
        for ch in obj.get("choices") or []:
            delta = (ch.get("delta") or {}).get("content")
            if isinstance(delta, str) and delta:
                yield delta


async def stream_openai_compat(endpoint: str, headers: dict[str, str], payload: dict[str, Any],
                               timeout: float = 45.0) -> AsyncIterator[str]:
    """Yield text deltas from any OpenAI-compatible /chat/completions SSE stream."""
    body = dict(payload, stream=True)
    async with http_pool.client().stream("POST", endpoint, headers=headers, json=body,
                                         timeout=httpx.Timeout(timeout, connect=10.0)) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            raise RuntimeError(f"LLM HTTP {resp.status_code}")
        async for d in _iter_sse_deltas(resp):
            yield d


async def complete_openai_compat(endpoint: str, headers: dict[str, str], payload: dict[str, Any],
                                 timeout: float = 45.0) -> str:
    r = await http_pool.client().post(endpoint, headers=headers, json=dict(payload, stream=False),
                                      timeout=httpx.Timeout(timeout, connect=10.0))
    if r.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {r.status_code}")
    content = ((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("LLM returned an empty completion")
    return content


class VercelGateway:
    """Chat + embeddings through Vercel AI Gateway with WebWaifu4 provider pinning."""

    def __init__(self, cfg: dict[str, Any] | None = None, base_url: str = VERCEL_AI_GATEWAY_BASE_URL) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.base_url = base_url.rstrip("/")

    def update_config(self, cfg: dict[str, Any]) -> None:
        self.cfg.update(cfg or {})

    @property
    def official(self) -> bool:
        return server_keys.is_official_url(self.base_url, "vercel")

    @property
    def configured(self) -> bool:
        return (self.official and bool(resolve_gateway_key(self.cfg))) or server_keys.is_local_url(self.base_url)

    def _key(self) -> str:
        """Gateway key for the official host only; localhost targets get none; anything else is refused."""
        if self.official:
            key = resolve_gateway_key(self.cfg)
            if not key:
                raise GatewayNotConfigured("AI Gateway is not configured (AI_GATEWAY_API_KEY).")
            return key
        if server_keys.is_local_url(self.base_url):
            return ""
        raise GatewayNotConfigured("Refusing to send gateway credentials to a non-official host.")

    def chat_request(self, messages: list[dict[str, str]], *, stream: bool = True, **kw: Any) -> tuple[str, dict[str, str], dict[str, Any]]:
        return (f"{self.base_url}/chat/completions", gateway_headers(self.cfg, self._key()),
                build_chat_payload(self.cfg, messages, stream=stream, byok_allowed=self.official, **kw))

    async def stream_chat(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        endpoint, headers, payload = self.chat_request(messages, stream=True, **kw)
        async for d in stream_openai_compat(endpoint, headers, payload):
            yield d

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False, **kw: Any) -> str:
        endpoint, headers, payload = self.chat_request(messages, stream=False, json_mode=json_mode, **kw)
        try:
            return await complete_openai_compat(endpoint, headers, payload)
        except RuntimeError as exc:
            # treblo ai-gateway.ts: retry once without response_format if a model rejects it
            if json_mode and "HTTP 400" in str(exc):
                payload.pop("response_format", None)
                return await complete_openai_compat(endpoint, headers, payload)
            raise

    def embedding_request(self, texts: list[str], model: str | None = None) -> tuple[str, dict[str, str], dict[str, Any]]:
        m = str(model or self.cfg.get("vercelEmbeddingModel") or DEFAULT_EMBEDDING_MODEL).strip()
        if "/" not in m:
            m = DEFAULT_EMBEDDING_MODEL
        return (f"{self.base_url}/embeddings", gateway_headers(self.cfg, self._key()),
                {"model": m, "input": [t[:8000] for t in texts]})

    async def embed(self, texts: list[str], model: str | None = None, timeout: float = 8.0) -> list[list[float]]:
        if not texts:
            return []
        endpoint, headers, payload = self.embedding_request(texts, model)
        r = await http_pool.client().post(endpoint, headers=headers, json=payload,
                                          timeout=httpx.Timeout(timeout, connect=5.0))
        if r.status_code >= 400:
            raise RuntimeError(f"Embeddings HTTP {r.status_code}")
        data = sorted(r.json().get("data") or [], key=lambda d: d.get("index", 0))
        out = [d.get("embedding") for d in data]
        if len(out) != len(texts) or not all(isinstance(e, list) and e for e in out):
            raise RuntimeError("Embeddings response malformed")
        return out


# --------------------------------------------------------------------------- local 4B
def local_chat_request(cfg: dict[str, Any], messages: list[dict[str, str]]) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Local 4B via Ollama (/v1) or llama-server; optional SLEEP LoRA on llama-server."""
    backend = str(cfg.get("localBackend") or "ollama").lower()
    base = str(cfg.get("localBaseUrl") or "").strip().rstrip("/")
    if not base:
        base = "http://127.0.0.1:8081/v1" if backend == "llama-server" else "http://127.0.0.1:11434/v1"
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    payload: dict[str, Any] = {
        "model": str(cfg.get("localModel") or "hikari-sleep-4b"),
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.82)),
        "max_tokens": int(cfg.get("maxTokens", 180)),
    }
    if backend == "llama-server" and str(cfg.get("localLoraPath") or "").strip():
        # llama-server loads adapters with --lora at start; per-request scale selects it.
        payload["lora"] = [{"id": 0, "scale": float(cfg.get("localLoraScale", 1.0))}]
    return endpoint, {"Content-Type": "application/json"}, payload


def local_backend_hint(cfg: dict[str, Any]) -> dict[str, str]:
    lora = str(cfg.get("localLoraPath") or "hikari-sleep-lora.gguf")
    model = str(cfg.get("localModel") or "hikari-sleep-4b")
    return {
        "ollamaModelfile": f"FROM gemma3:4b\nADAPTER {lora}\n",
        "ollamaCreate": f"ollama create {model} -f Modelfile",
        "llamaServer": f"llama-server -m gemma-3-4b-it-Q4_K_M.gguf --lora {lora} -c 4096 --port 8081",
    }


def find_overlapping_suffix(prefix: str, final_text: str) -> str:
    """Port of WebWaifu4 streaming-overlap.ts: the part of final_text not already streamed."""
    p, f = prefix.strip(), final_text.strip()
    if not p or not f:
        return f
    if f.startswith(p):
        return f[len(p):]
    for length in range(min(len(p), len(f)), 31, -1):
        if p[-length:] == f[:length]:
            return f[length:]
    return ""
