"""One shared, connection-pooled httpx.AsyncClient per event loop (never one client per turn).

httpx clients are bound to the loop they first run on, so the pool is keyed by loop. Per-request
timeouts are passed at call sites (`timeout=` on each request) while the pool keeps keep-alive
TLS connections to Vercel AI Gateway, Fish Audio and local LLM servers warm.
"""
from __future__ import annotations

import asyncio
import weakref

import httpx

_LIMITS = httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=45.0)
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = weakref.WeakKeyDictionary()


def client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    c = _clients.get(loop)
    if c is None or c.is_closed:
        c = httpx.AsyncClient(limits=_LIMITS, timeout=_DEFAULT_TIMEOUT, follow_redirects=False,
                              headers={"User-Agent": "HikariStudio/1.0"})
        _clients[loop] = c
    return c


async def aclose() -> None:
    """Close the current loop's client (call from app cleanup)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    c = _clients.pop(loop, None)
    if c is not None and not c.is_closed:
        await c.aclose()
