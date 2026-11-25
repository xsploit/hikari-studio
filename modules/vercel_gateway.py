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

import asyncio
import json
import weakref
import time
import re
from typing import Any, AsyncIterator
from urllib.parse import quote as urllib_quote

import httpx

from modules import http_pool, server_keys

VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
DEFAULT_AI_GATEWAY_MODEL = "openai/gpt-5-nano"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Screen Vision "Auto": the Vision Eye / direct vision model used when vision.model is blank.
DEFAULT_VISION_MODEL = "openai/gpt-5-mini"
# Gateway app-attribution headers (safe, non-secret)
DEFAULT_GATEWAY_HEADERS = {"http-referer": "https://hikari.studio.local", "x-title": "Hikari Studio"}
_BLOCKED_HEADERS = {"authorization", "content-type", "content-length", "host", "cookie"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")       # CR, LF, NUL, tabs, DEL: never allowed in a header


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


def is_vision_model_id(model_id: Any) -> bool:
    """A routed, non-premium id usable as a one-off vision model override ('provider/model')."""
    m = model_id.strip() if isinstance(model_id, str) else ""
    return (0 < len(m) <= 160 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*(?:/[A-Za-z0-9._:@+-]+)+", m)
            is not None and not is_premium_cost_model_id(m))


def pick_vision_model(value: Any) -> str:
    return value.strip() if is_vision_model_id(value) else DEFAULT_VISION_MODEL


def override_cfg(cfg: dict[str, Any], model_override: str | None = None,
                 provider_order_override: str | list[str] | None = None) -> dict[str, Any]:
    """A per-request copy of the AI config for a one-off model (Vision Eye / direct vision turn). The chat model's
    provider pins and fallback models belong to that model, so they are dropped when the model is overridden;
    `provider_order_override` (comma string or list) becomes the new order. `cfg` itself is never mutated."""
    if model_override is None and provider_order_override is None:
        return cfg
    out = dict(cfg)
    if model_override is not None:
        out["vercelModel"] = pick_vision_model(model_override)
        out["vercelFallbackModels"] = []
        out["vercelProviderOrder"] = []
        out["vercelProviderOnly"] = False
    if provider_order_override is not None:
        out["vercelProviderOrder"] = _id_list(provider_order_override)
        out["vercelProviderOnly"] = False
    return out


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
        for k, v in list(extra.items())[:16]:
            # CR / LF / NUL (and other control chars): a name containing one is dropped (never glued into a
            # different header name); a value has them stripped. A pasted newline can never inject a header or
            # make httpx reject every gateway call.
            k2 = str(k).strip().lower()
            if not k2 or _CTRL_RE.search(k2) or k2 in _BLOCKED_HEADERS or not re.fullmatch(r"[a-z0-9-]{1,64}", k2):
                continue
            if v is None or isinstance(v, (dict, list, tuple, bool)):
                continue
            v2 = _CTRL_RE.sub("", str(v)).strip()[:256]
            if v2:
                headers[k2] = v2
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

    def chat_request(self, messages: list[dict[str, str]], *, stream: bool = True,
                     model_override: str | None = None, provider_order_override: str | list[str] | None = None,
                     **kw: Any) -> tuple[str, dict[str, str], dict[str, Any]]:
        """(endpoint, headers, payload). `model_override` / `provider_order_override` target a one-off model (e.g.
        vision.model + vision.providerOrder) without mutating self.cfg; None = the configured chat model."""
        cfg = override_cfg(self.cfg, model_override, provider_order_override)
        return (f"{self.base_url}/chat/completions", gateway_headers(self.cfg, self._key()),
                build_chat_payload(cfg, messages, stream=stream, byok_allowed=self.official, **kw))

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


# --------------------------------------------------------------------------- evaluation-model catalog
# Port of Waifu Studio model_catalog.rs (CatalogKind::Decision): GET <gateway>/v1/models, keep rows whose
# declared `type` is "evaluation", ids <= 256 chars without control characters, sorted + de-duplicated,
# 5 MB response cap. Cached in memory for 10 minutes; offline / failure falls back to the curated list.
EVALUATION_CATALOG_URL = f"{VERCEL_AI_GATEWAY_BASE_URL}/models"
CATALOG_TTL_SEC = 600.0
CATALOG_RETRY_SEC = 60.0                # after a failure, don't hammer the catalog
CATALOG_MAX_BYTES = 5_000_000           # model_catalog.rs MAX_BYTES
CATALOG_MAX_ROWS = 10_000
CURATED_EVALUATION_MODELS: tuple[str, ...] = (
    "perplexity/pplx-decider-flash-v1.3", "perplexity/pplx-decider-v1.3",
    "perplexity/pplx-decider-flash-v1.2", "perplexity/pplx-decider-v1.2",
    "openai/gpt-6-luna-decisions",
    "typesafe-ai/jev", "typesafe-ai/jev-1.14.0", "typesafe-ai/jev-1.13.0",
    "google/gemini-decision-1",
    "microsoft/decision-1.1", "microsoft/microsoft-decision-1",
    "liquid/d1.2", "liquid/d1.2-ultra", "liquid/d1-ultra",
    "parallel/parallel-decisions-fast",
    "heygen/heygen-decision-1-turbo", "heygen/heygen-decision-1",
    "baseten/joey-1", "baseten/elephant-1", "baseten/mammoth-1",
    "convaiinnovations/laya", "scaledown/scaledown-classify-1", "modaic/mo",
)
_VENDOR_LABELS = {
    "perplexity": "Perplexity", "openai": "OpenAI", "typesafe-ai": "Typesafe", "google": "Google",
    "microsoft": "Microsoft", "liquid": "Liquid", "parallel": "Parallel", "heygen": "HeyGen", "baseten": "Baseten",
    "convaiinnovations": "Convai", "scaledown": "ScaleDown", "modaic": "Modaic", "nace-ai": "Nace AI",
}
_MODEL_LABELS = {
    "perplexity/pplx-decider-flash-v1.3": "Perplexity Decider Flash v1.3",
    "perplexity/pplx-decider-v1.3": "Perplexity Decider v1.3",
    "perplexity/pplx-decider-flash-v1.2": "Perplexity Decider Flash v1.2",
    "perplexity/pplx-decider-v1.2": "Perplexity Decider v1.2",
    "openai/gpt-6-luna-decisions": "OpenAI GPT-6 Luna Decisions",
    "typesafe-ai/jev": "Typesafe Jev", "typesafe-ai/jev-1.14.0": "Typesafe Jev 1.14.0",
    "typesafe-ai/jev-1.13.0": "Typesafe Jev 1.13.0",
    "google/gemini-decision-1": "Google Gemini Decision 1",
    "microsoft/decision-1.1": "Microsoft Decision 1.1", "microsoft/microsoft-decision-1": "Microsoft Decision 1",
    "liquid/d1.2": "Liquid D1.2", "liquid/d1.2-ultra": "Liquid D1.2 Ultra", "liquid/d1-ultra": "Liquid D1 Ultra",
    "parallel/parallel-decisions-fast": "Parallel Decisions Fast",
    "heygen/heygen-decision-1-turbo": "HeyGen Decision 1 Turbo", "heygen/heygen-decision-1": "HeyGen Decision 1",
    "baseten/joey-1": "Baseten Joey 1", "baseten/elephant-1": "Baseten Elephant 1", "baseten/mammoth-1": "Baseten Mammoth 1",
    "convaiinnovations/laya": "Convai Laya", "scaledown/scaledown-classify-1": "ScaleDown Classify 1", "modaic/mo": "Modaic Mo",
}
_catalog: dict[str, Any] = {"models": None, "at": 0.0, "failedAt": -1e9, "error": ""}
_catalog_inflight: "weakref.WeakKeyDictionary[Any, asyncio.Future]" = weakref.WeakKeyDictionary()


def model_label(model_id: str) -> str:
    if model_id in _MODEL_LABELS:
        return _MODEL_LABELS[model_id]
    vendor, _, name = model_id.partition("/")
    vlabel = _VENDOR_LABELS.get(vendor, re.sub(r"\bAi\b", "AI", vendor.replace("-", " ").title()))
    words = []
    for w in re.split(r"[-_]", name):
        if not w or w.lower() == vendor.lower():
            continue
        words.append(w if re.match(r"^v?\d", w) else (w.upper() if len(w) <= 3 and w.isalpha() and w.lower() in ("gpt", "ai") else w.capitalize()))
    return f"{vlabel} {' '.join(words)}".strip()


def evaluation_model_ids(value: Any) -> list[str]:
    """model_catalog.rs `model_ids` for CatalogKind::Decision (declared type == "evaluation")."""
    rows = value.get("data") if isinstance(value, dict) else None
    if rows is None and isinstance(value, dict):
        rows = value.get("models")
    if not isinstance(rows, list):
        raise RuntimeError("Expected a model list in catalog")
    if len(rows) > CATALOG_MAX_ROWS:
        raise RuntimeError("Model catalog exceeds 10,000 entries")
    ids = set()
    for model in rows:
        if not isinstance(model, dict):
            continue
        mid = model.get("id")
        if not isinstance(mid, str) or not mid or len(mid) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in mid):
            continue
        if model.get("type") == "evaluation":
            ids.add(mid)
    return sorted(ids)


def known_evaluation_models() -> set[str]:
    """Curated + last live catalog (used to validate a shared `decisionModel` as an evaluation model)."""
    return set(CURATED_EVALUATION_MODELS) | set(_catalog.get("models") or ())


def catalog_entries(ids: list[str]) -> list[dict[str, str]]:
    return [{"id": m, "provider": m.split("/", 1)[0], "label": model_label(m)} for m in ids]


async def _fetch_catalog(cfg: dict[str, Any], timeout: float) -> list[str]:
    url = EVALUATION_CATALOG_URL
    # The /v1/models listing is public (WebWaifu4 providerModelsCanBeListedWithoutKey): no credential sent.
    headers = {"Accept": "application/json"}
    buf = bytearray()
    async with http_pool.client().stream("GET", url, headers=headers,
                                         timeout=httpx.Timeout(timeout, connect=min(timeout, 4.0))) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"Model catalog returned HTTP {r.status_code}")
        async for chunk in r.aiter_bytes():
            buf += chunk
            if len(buf) > CATALOG_MAX_BYTES:
                raise RuntimeError("Model catalog exceeds size limit")
    try:
        value = json.loads(bytes(buf))
    except json.JSONDecodeError:
        raise RuntimeError("Model catalog returned invalid JSON") from None
    return evaluation_model_ids(value)


async def evaluation_catalog(cfg: dict[str, Any] | None = None, *, force: bool = False, timeout: float = 6.0) -> dict[str, Any]:
    """{"models": [ids], "source": "live" | "cache" | "stale" | "curated", "error": str}. Never raises."""
    now = time.monotonic()
    cached = _catalog.get("models")
    if not force and cached and now - _catalog["at"] < CATALOG_TTL_SEC:
        return {"models": list(cached), "source": "cache", "error": ""}
    if not force and now - _catalog["failedAt"] < CATALOG_RETRY_SEC:
        return {"models": list(cached) if cached else list(CURATED_EVALUATION_MODELS),
                "source": "stale" if cached else "curated", "error": _catalog["error"]}
    loop = asyncio.get_running_loop()
    fut = _catalog_inflight.get(loop)
    if fut is None or fut.done():                       # single flight: concurrent callers share one fetch
        fut = asyncio.ensure_future(_fetch_catalog(cfg or {}, timeout))
        _catalog_inflight[loop] = fut
    try:
        ids = await asyncio.shield(fut)
        if not ids:
            raise RuntimeError("Model catalog listed no evaluation models")
        _catalog.update({"models": ids, "at": time.monotonic(), "error": ""})
        return {"models": list(ids), "source": "live", "error": ""}
    except Exception as exc:
        msg = str(exc) if isinstance(exc, RuntimeError) else "Could not reach model catalog"
        _catalog.update({"failedAt": time.monotonic(), "error": msg[:120]})
        return {"models": list(cached) if cached else list(CURATED_EVALUATION_MODELS),
                "source": "stale" if cached else "curated", "error": msg[:120]}


# --------------------------------------------------------------------------- chat-model + upstream provider catalog
# Port of Waifu Studio provider.rs `chat_model_ids` (Vercel: keep `type == "language"` rows with routed
# 'creator/model' ids) + model_catalog.rs bounds, extended with the exact per-model upstream providers from
# GET /v1/models/{id}/endpoints (`endpoints[].provider_name`: the slugs providerOptions.gateway.order / only take).
# The listing is cached 5 min; each model's provider list 30 min (a refresh only re-asks for stale ones).
CHAT_CATALOG_TTL_SEC = 300.0
CHAT_CATALOG_RETRY_SEC = 30.0
ENDPOINTS_TTL_SEC = 1800.0
ENDPOINTS_CONCURRENCY = 12
ENDPOINTS_BUDGET_SEC = 25.0
FALLBACK_CHAT_MODELS: tuple[tuple[str, str], ...] = (
    ("openai/gpt-5-nano", "GPT-5 nano"), ("openai/gpt-5-mini", "GPT-5 mini"), ("openai/gpt-5", "GPT-5"),
    ("openai/gpt-4.1-mini", "GPT-4.1 mini"), ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5"),
    ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"), ("google/gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro"), ("deepseek/deepseek-v3.2", "DeepSeek V3.2"),
    ("xai/grok-4-fast-reasoning", "Grok 4 Fast Reasoning"), ("meta/llama-4-maverick", "Llama 4 Maverick"),
    ("mistral/mistral-medium", "Mistral Medium"), ("alibaba/qwen-3-235b", "Qwen3 235B"),
    ("moonshotai/kimi-k2", "Kimi K2"),
)
_chat_catalog: dict[str, Any] = {"models": None, "at": 0.0, "failedAt": -1e9, "error": ""}
_endpoint_cache: dict[str, tuple[float, list[str] | None]] = {}     # model id -> (at, providers | None = 404)
_chat_inflight: "weakref.WeakKeyDictionary[Any, asyncio.Future]" = weakref.WeakKeyDictionary()


def _safe_id(mid: Any) -> bool:
    return (isinstance(mid, str) and 0 < len(mid) <= 256 and "/" in mid
            and not any(ord(c) < 32 or ord(c) == 127 for c in mid))


def chat_model_rows(value: Any) -> list[dict[str, Any]]:
    """`/v1/models` JSON -> sorted, de-duplicated language-model rows (no providers yet)."""
    rows = value.get("data") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Expected a data array from /models")
    if len(rows) > CATALOG_MAX_ROWS:
        raise RuntimeError("Model catalog exceeds 10,000 entries")
    out: dict[str, dict[str, Any]] = {}
    for m in rows:
        if not isinstance(m, dict) or m.get("type") != "language" or not _safe_id(m.get("id")):
            continue
        mid = m["id"]
        pricing = m.get("pricing") if isinstance(m.get("pricing"), dict) else None
        out[mid] = {
            "id": mid,
            "name": str(m.get("name") or model_label(mid))[:120],
            "owner": str(m.get("owned_by") or mid.split("/", 1)[0])[:64],
            "type": "language",
            "contextWindow": m.get("context_window") if isinstance(m.get("context_window"), int) else None,
            "maxTokens": m.get("max_tokens") if isinstance(m.get("max_tokens"), int) else None,
            "pricing": {k: str(v) for k, v in pricing.items() if isinstance(v, (str, int, float))} if pricing else None,
            "tags": [str(t)[:32] for t in (m.get("tags") or []) if isinstance(t, str)][:12],
            "providers": [],
        }
    return [out[k] for k in sorted(out)]


def endpoint_providers(value: Any) -> list[str]:
    """`/v1/models/{id}/endpoints` JSON -> the exact upstream provider slugs (order kept, de-duplicated)."""
    data = value.get("data") if isinstance(value, dict) else None
    eps = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(eps, list):
        raise RuntimeError("Expected an endpoints array")
    out: list[str] = []
    for e in eps:
        p = str(e.get("provider_name") or "").strip().lower() if isinstance(e, dict) else ""
        if _ID_RE.match(p) and p not in out:
            out.append(p)
    return out


def _catalog_client() -> httpx.AsyncClient:
    """A short-lived client just for the catalog fan-out, so ~300 endpoint lookups never queue behind (or
    starve) live chat on the shared pool. Tests replace this factory with an httpx.MockTransport client."""
    return httpx.AsyncClient(limits=httpx.Limits(max_connections=ENDPOINTS_CONCURRENCY,
                                                 max_keepalive_connections=ENDPOINTS_CONCURRENCY),
                             follow_redirects=False, headers={"User-Agent": "HikariStudio/1.0"})


async def _get_json(client: httpx.AsyncClient, url: str, headers: dict[str, str], timeout: float,
                    limit: int = CATALOG_MAX_BYTES) -> tuple[int, Any]:
    buf = bytearray()
    async with client.stream("GET", url, headers=headers,
                                         timeout=httpx.Timeout(timeout, connect=min(timeout, 5.0))) as r:
        if r.status_code >= 400:
            await r.aread()
            return r.status_code, None
        async for chunk in r.aiter_bytes():
            buf += chunk
            if len(buf) > limit:
                raise RuntimeError("Model catalog exceeds size limit")
    try:
        return r.status_code, json.loads(bytes(buf))
    except json.JSONDecodeError:
        raise RuntimeError("Model catalog returned invalid JSON") from None


async def _fetch_chat_catalog(api_key: str, base_url: str) -> list[dict[str, Any]]:
    headers = {"Accept": "application/json"}
    if api_key and server_keys.is_official_url(base_url, "vercel"):
        headers["Authorization"] = f"Bearer {api_key}"
    async with _catalog_client() as client:
        return await _fetch_chat_catalog_with(client, headers, base_url)


async def _fetch_chat_catalog_with(client: httpx.AsyncClient, headers: dict[str, str],
                                   base_url: str) -> list[dict[str, Any]]:
    status, value = await _get_json(client, f"{base_url}/models", headers, 10.0)
    if value is None:
        raise RuntimeError(f"Model catalog returned HTTP {status}")
    rows = chat_model_rows(value)
    if not rows:
        raise RuntimeError("Model catalog listed no language models")
    sem = asyncio.Semaphore(ENDPOINTS_CONCURRENCY)
    deadline = time.monotonic() + ENDPOINTS_BUDGET_SEC

    async def providers_for(mid: str) -> None:
        hit = _endpoint_cache.get(mid)
        if hit and time.monotonic() - hit[0] < ENDPOINTS_TTL_SEC:
            return
        async with sem:
            left = deadline - time.monotonic()
            if left <= 0.5:
                return
            path = "/".join(urllib_quote(part, safe="") for part in mid.split("/"))
            try:
                st, val = await _get_json(client, f"{base_url}/models/{path}/endpoints", headers, min(10.0, left), 1_000_000)
            except Exception:
                return                                   # transient: keep any older answer, retry next refresh
            if st == 404:
                _endpoint_cache[mid] = (time.monotonic(), None)     # listed but not routable (e.g. *-batch)
            elif val is not None:
                try:
                    _endpoint_cache[mid] = (time.monotonic(), endpoint_providers(val))
                except RuntimeError:
                    pass

    await asyncio.gather(*(providers_for(r["id"]) for r in rows))
    out = []
    for r in rows:
        hit = _endpoint_cache.get(r["id"])
        if hit is not None and hit[1] is None:
            continue                                     # the gateway says this id cannot be routed
        r["providers"] = list(hit[1]) if hit else []
        r["providersKnown"] = hit is not None
        out.append(r)
    return out


_FALLBACK_VISION_IDS = frozenset({
    "openai/gpt-5-nano", "openai/gpt-5-mini", "openai/gpt-5", "openai/gpt-4.1-mini",
    "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash", "google/gemini-2.5-pro", "meta/llama-4-maverick",
})


def _fallback_models() -> list[dict[str, Any]]:
    return [{"id": mid, "name": name, "owner": mid.split("/", 1)[0], "type": "language", "contextWindow": None,
             "maxTokens": None, "pricing": None, "tags": ["vision"] if mid in _FALLBACK_VISION_IDS else [],
             "providers": [], "providersKnown": False}
            for mid, name in FALLBACK_CHAT_MODELS]


def _with_providers(models: list[dict[str, Any]], source: str, error: str) -> dict[str, Any]:
    providers = sorted({p for m in models for p in m.get("providers") or ()})
    return {"models": [dict(m, providers=list(m.get("providers") or [])) for m in models],
            "providers": providers, "source": source, "error": error}


async def list_chat_models_and_providers(api_key: str = "", force: bool = False, *,
                                         base_url: str = VERCEL_AI_GATEWAY_BASE_URL) -> dict[str, Any]:
    """{"models": [{id, name, owner, type, contextWindow, pricing, providers[]}], "providers": [sorted slugs],
    "source": "live" | "cache" | "stale" | "fallback", "error": str}. Never raises."""
    now = time.monotonic()
    cached = _chat_catalog.get("models")
    if not force and cached and now - _chat_catalog["at"] < CHAT_CATALOG_TTL_SEC:
        return _with_providers(cached, "cache", "")
    if not force and now - _chat_catalog["failedAt"] < CHAT_CATALOG_RETRY_SEC:
        return _with_providers(cached or _fallback_models(), "stale" if cached else "fallback", _chat_catalog["error"])
    loop = asyncio.get_running_loop()
    fut = _chat_inflight.get(loop)
    if fut is None or fut.done():                       # single flight: concurrent callers share one fan-out
        fut = asyncio.ensure_future(_fetch_chat_catalog((api_key or "").strip(), base_url.rstrip("/")))
        _chat_inflight[loop] = fut
    try:
        models = await asyncio.shield(fut)
        _chat_catalog.update({"models": models, "at": time.monotonic(), "error": ""})
        return _with_providers(models, "live", "")
    except Exception as exc:
        msg = str(exc) if isinstance(exc, RuntimeError) else "Could not reach model catalog"
        _chat_catalog.update({"failedAt": time.monotonic(), "error": msg[:120]})
        return _with_providers(cached or _fallback_models(), "stale" if cached else "fallback", msg[:120])


def clear_chat_catalog_cache() -> None:
    _chat_catalog.update({"models": None, "at": 0.0, "failedAt": -1e9, "error": ""})
    _endpoint_cache.clear()
