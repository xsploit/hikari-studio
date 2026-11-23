#!/usr/bin/env python3
"""Safety net for a Colab run: the GPU VM must never be left running.

Runs as its own detached process next to `run_colab_sleep.py --mode colab` and checks once a minute:
  * the launcher process died (crash, killed, chat closed) -> stop the Colab session if it still exists
  * the hard time limit passed -> kill the launcher and stop the Colab session
Then it confirms with `colab sessions` and writes a final line to its log.

  python sleep_lab/colab_watchdog.py --pid 1234 --session hikari-sleep --max-hours 2.5
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "outputs" / "colab_watchdog.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def colab_base() -> list[str]:
    if shutil.which("colab"):
        return [shutil.which("colab")]
    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / ("uvx.exe" if os.name == "nt" else "uvx"))
    return [uvx, "--from", "google-colab-cli", "colab"]


ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def colab(*args: str, timeout: int = 180) -> str:
    try:
        r = subprocess.run([*colab_base(), *args], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=ENV, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "<timeout>"


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def save_partial(session: str) -> None:
    """Grab the notebook's latest per-cycle archive before the VM goes away (best effort)."""
    dst = HERE / "dist" / f"colab_outputs_watchdog_{datetime.now():%Y%m%d_%H%M}.zip"
    dst.parent.mkdir(exist_ok=True)
    out = colab("download", "-s", session, "/content/sleep_lab/sleep_outputs.zip", str(dst), timeout=600)
    log(f"partial download -> {dst.name}: {'ok' if dst.exists() else 'none (' + out.strip()[-120:] + ')'}")


def ensure_stopped(session: str) -> bool:
    """Stop the session if the server still lists it; retry a few times. True = confirmed gone."""
    for attempt in range(5):
        listing = colab("sessions")
        if f"[{session}]" not in listing and "<timeout>" not in listing:
            log(f"confirmed: no '{session}' session on the server")
            return True
        log(f"'{session}' still running -> colab stop (attempt {attempt + 1})")
        colab("stop", "-s", session)
        time.sleep(20)
    log(f"WARNING: could not confirm '{session}' stopped - check https://colab.research.google.com")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pid", type=int, required=True, help="launcher process id to watch")
    ap.add_argument("--session", default="hikari-sleep")
    ap.add_argument("--max-hours", type=float, default=2.5, help="hard limit from now")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    deadline = datetime.now() + timedelta(hours=args.max_hours)
    log(f"watching launcher pid {args.pid}, session '{args.session}', hard stop at {deadline:%H:%M}")
    while True:
        if not pid_alive(args.pid):
            log("launcher exited")
            if f"[{args.session}]" in colab("sessions"):     # launcher died without cleaning up
                save_partial(args.session)
            ok = ensure_stopped(args.session)
            log(f"WATCHDOG DONE reason=launcher_exited vm_stopped={ok}")
            return 0
        if datetime.now() >= deadline:
            log(f"hard time limit ({args.max_hours} h) reached -> killing launcher and stopping the VM")
            kill_tree(args.pid)
            save_partial(args.session)
            ok = ensure_stopped(args.session)
            log(f"WATCHDOG DONE reason=time_limit vm_stopped={ok}")
            return 1
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
