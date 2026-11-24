"""Backwards-compatible name for Hikari's memory module.

The SLEEP / dreaming / Colab LoRA pipeline is gone from the live app. Hikari's real memory now lives in
modules/memory_store.py (persistent conversation turns + real-time Waifu Studio-style facts). This module only
re-exports it so older imports (plugins, scripts) keep working.
"""
from modules.memory_store import *  # noqa: F401,F403
from modules.memory_store import (  # noqa: F401
    CATEGORIES, DEFAULT_MEMORY_CFG, MEMORY_KINDS, MemoryStore, PinLimitError, PromptContext, clean_card_text,
    is_injection, quote_data,
)

SleepMemory = MemoryStore
