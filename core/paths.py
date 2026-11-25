"""Frozen-safe root path resolution for dev (`python server.py`) and compiled (`hikari-server.exe`) runs."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def studio_root() -> Path:
    if os.environ.get("HIKARI_STUDIO_DIR"):
        return Path(os.environ["HIKARI_STUDIO_DIR"]).resolve()
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir.parent if (exe_dir.parent / "web").is_dir() else exe_dir
    return Path(__file__).resolve().parents[1]
