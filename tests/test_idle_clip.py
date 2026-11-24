"""Verification: Motion Lab "Use as idle" loop (core/rig_controller.py IDLE_TRACKS layer + server restore).

  1. prepare_idle_clip keeps ONLY head / body / drift tracks, zero-mean, with a seamless loop point
  2. with an idle clip her head and body move more than her procedural idle alone; mix 0 = no clip motion
  3. the idle loop never raises her arms and never stops her own blinks
  4. it fades out for an emote / Motion Lab clip and comes back afterwards; speaking keeps it at ~0.75
  5. set_idle_clip(None) clears it; a malformed clip is rejected before touching live state
  6. StudioRuntime._restore_idle_clip re-installs the saved clip, and clears a missing one

    python tests/test_idle_clip.py        (or: python -m pytest tests/test_idle_clip.py)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
import sys
import tempfile
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from test_wave_decisioning import CLOCK, DT, new_rig as make_rig  # noqa: E402  (installs the virtual clock)
from core.rig_controller import IDLE_TRACKS, MOTION_TRACKS, prepare_idle_clip, validate_motion_clip  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


def synth_clip(seconds: float = 8.0) -> dict[str, Any]:
    """A fake 60 FPS clip: big slow head / body sway, plus wild eye / arm tracks the idle layer must ignore."""
    n = int(seconds * 60)
    t = [i / 60.0 for i in range(n)]
    tracks = {
        "AngleX": [12.0 * math.sin(2 * math.pi * x / 3.1) + 5.0 for x in t],     # +5 offset: must be removed
        "AngleY": [6.0 * math.sin(2 * math.pi * x / 2.3) for x in t],
        "AngleZ": [15.0 * math.sin(2 * math.pi * x / 4.0) for x in t],
        "BodyX": [4.0 * math.sin(2 * math.pi * x / 3.7) for x in t],
        "BodyZ": [5.0 * math.sin(2 * math.pi * x / 5.0) for x in t],
        "EyeOpen": [0.0] * n,                                                    # eyes shut the whole clip
        "ArmR": [1.0] * n,                                                       # arms up the whole clip
        "ArmL": [1.0] * n,
        "StageX": [0.3 * math.sin(2 * math.pi * x / 6.0) for x in t],
        "StageY": [0.2 * math.sin(2 * math.pi * x / 2.9) for x in t],
        "StageZoom": [0.15 * math.sin(2 * math.pi * x / 4.6) for x in t],
        "ArmRGesture": [1.0] * n,
        "ArmLGesture": [1.0] * n,
    }
    return {"id": "synth_idle", "name": "Synth idle", "fps": 60, "frames": n, "duration": seconds, "tracks": tracks}


def stats(vals: list[float]) -> float:
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def run(rig, cfg, seconds: float, lip=None, sink=None) -> list[dict[str, Any]]:
    out = []
    for _ in range(int(round(seconds / DT))):
        CLOCK.now += DT
        p, _, _ = rig.step(DT, cfg, None)
        out.append({"yaw": p["Angle"][0], "roll": p["Angle Z"][0], "bodyx": p["Body Angle X"][0],
                    "bodyy": p["Body Angle Y"][0], "bodyz": p["Body Angle Z"][0], "lean": p["Body Lean"][0],
                    "stageX": p["Stage X"][0], "stageY": p["Stage Y"][0], "stageZoom": p["Stage Zoom"][0],
                    "stageRoll": p["Stage Roll"][0], "shrug": p["Shoulder Shrug"][0],
                    "armR": p["Arm R Pose"][0], "armL": p["Arm L Pose"][0], "eyeL": p["Left Eye Open"][0],
                    "w": rig._idle_layer_w})
    return out


def test_prepare() -> None:
    c = prepare_idle_clip(validate_motion_clip(synth_clip()))
    check("prepare. only head / body / drift tracks are kept", set(c["tracks"]) == set(IDLE_TRACKS),
          str(sorted(c["tracks"])))
    check("prepare. eyes / arms / gestures are never part of the idle loop",
          not ({"EyeOpen", "ArmR", "ArmL", "ArmRGesture"} & set(c["tracks"])))
    ax = c["tracks"]["AngleX"]
    check("prepare. tracks are zero-mean offsets (clip's +5 average removed)", abs(sum(ax) / len(ax)) < 1e-6,
          f"{sum(ax) / len(ax):.4f}")
    seam = max(abs(tr[-1] - tr[0]) for tr in c["tracks"].values())
    step = max(abs(tr[i + 1] - tr[i]) for tr in c["tracks"].values() for i in range(len(tr) - 1))
    check("prepare. loop point is seamless (seam no bigger than a normal frame step)", seam <= step * 1.5 + 1e-6,
          f"seam {seam:.3f} step {step:.3f}")
    check("prepare. loop is shorter than the clip by the 1 s crossfade", c["frames"] == 8 * 60 - 60, str(c["frames"]))


def new_rig():
    # Compare the same procedural motion with and without the clip. Phase offsets
    # and idle schedules are sampled during construction, before the first step.
    CLOCK.now = 1000.0
    with patch("core.rig_controller.random", SimpleNamespace(Random=lambda: random.Random(42))):
        return make_rig()


def test_rig(tmp: Path) -> None:
    clip_path = tmp / "synth_idle.json"
    clip_path.write_text(json.dumps(synth_clip()), encoding="utf-8")

    rig0, _, cfg0 = new_rig()
    base = run(rig0, cfg0, 12.0)
    check("rig. procedural Neuro rock drives grounded Stage Roll + torso sway even without an idle clip",
          stats([f["stageRoll"] for f in base[120:]]) > 0.25 and stats([f["bodyz"] for f in base[120:]]) > 2.0,
          f"sr={stats([f['stageRoll'] for f in base[120:]]):.2f} bz={stats([f['bodyz'] for f in base[120:]]):.2f}")

    rig, lip, cfg = new_rig()
    st = rig.set_idle_clip(clip_path)
    check("rig. set_idle_clip reports the clip", st.get("id") == "synth_idle" and st.get("name") == "Synth idle", str(st))
    cfg["rig"]["idleClipMix"] = 1.0
    frames = run(rig, cfg, 12.0)
    roll_b, roll_i = stats([f["roll"] for f in base[120:]]), stats([f["roll"] for f in frames[120:]])
    check("rig. idle loop adds clearly more head roll than the procedural idle", roll_i > roll_b * 1.5,
          f"idle {roll_i:.2f} vs base {roll_b:.2f}")
    check("rig. idle loop moves the body too",
          stats([f["bodyx"] for f in frames[120:]]) > 3.0, f"{stats([f['bodyx'] for f in frames[120:]]):.2f}")
    check("rig. idle loop moves whole-body torso roll, dip & lean",
          stats([f["bodyz"] for f in frames[120:]]) > 3.0
          and stats([f["bodyy"] for f in frames[120:]]) > 1.5
          and stats([f["lean"] for f in frames[120:]]) > 1.2,
          f"bz={stats([f['bodyz'] for f in frames[120:]]):.2f} by={stats([f['bodyy'] for f in frames[120:]]):.2f} lean={stats([f['lean'] for f in frames[120:]]):.2f}")
    check("rig. idle loop moves whole-body Stage Roll/Y/Zoom & subtle shoulder bounce",
          stats([f["stageRoll"] for f in frames[120:]]) > 0.25
          and stats([f["stageY"] for f in frames[120:]]) > 0.12
          and stats([f["stageZoom"] for f in frames[120:]]) > 0.08
          and max(f["shrug"] for f in frames[120:]) > 0.06,
          f"sr={stats([f['stageRoll'] for f in frames[120:]]):.2f} sy={stats([f['stageY'] for f in frames[120:]]):.2f} sz={stats([f['stageZoom'] for f in frames[120:]]):.2f}")
    check("rig. idle loop never raises an arm (clip had ArmR / ArmL = 1)",
          max(max(f["armR"], f["armL"]) for f in frames) == 0.0)
    check("rig. her own blinks keep happening (clip had eyes shut)",
          min(f["eyeL"] for f in frames) < 0.3 and max(f["eyeL"] for f in frames) > 0.8)

    rig2, _, cfg2 = new_rig()
    rig2.set_idle_clip(clip_path)
    cfg2["rig"]["idleClipMix"] = 0.0
    sampled = []
    real_sample = rig2._idle_sample
    rig2._idle_sample = lambda *a: sampled.append(1) or real_sample(*a)    # type: ignore[method-assign]
    run(rig2, cfg2, 6.0)
    check("rig. mix 0 adds no clip motion (clip never sampled)", not sampled, f"{len(sampled)} samples")

    # fade around an emote
    w_before = frames[-1]["w"]
    rig.trigger_emote("curious_lean", duration=3.0)
    during = run(rig, cfg, 1.5)
    check("rig. idle loop was fully in before the emote", w_before > 0.95, f"{w_before:.2f}")
    check("rig. idle loop fades out while an emote plays", during[-1]["w"] < 0.05, f"{during[-1]['w']:.3f}")
    after = run(rig, cfg, 4.0)
    check("rig. idle loop fades back in after the emote", after[-1]["w"] > 0.9, f"{after[-1]['w']:.3f}")

    lip.open, lip.energy = 0.5, 0.5
    talk = run(rig, cfg, 3.0)
    check("rig. while speaking she keeps swaying at ~0.75", abs(talk[-1]["w"] - 0.75) < 0.05, f"{talk[-1]['w']:.3f}")
    lip.open, lip.energy = 0.0, 0.0

    rig.load_motion_clip(clip_path, loop=True, mix=0.85)
    fg = run(rig, cfg, 2.0)
    check("rig. a playing Motion Lab clip takes over (idle loop fades out)", fg[-1]["w"] < 0.05, f"{fg[-1]['w']:.3f}")
    rig.stop_motion()

    st = rig.set_idle_clip(None)
    check("rig. set_idle_clip(None) clears it", st == {"id": "", "name": ""}, str(st))
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"id": "bad", "frames": 3, "tracks": {}}), encoding="utf-8")
    rig.set_idle_clip(clip_path)
    try:
        rig.set_idle_clip(bad)
        rejected = False
    except (ValueError, KeyError):
        rejected = True
    check("rig. a malformed clip is rejected", rejected)
    check("rig. ...and the previous idle clip stays installed", rig.idle_clip_status()["id"] == "synth_idle")
    check("rig. all core MOTION_TRACKS still exist (format unchanged)", all(k in synth_clip()["tracks"] for k in MOTION_TRACKS))


def test_restore(tmp: Path) -> None:
    import server

    class _Rig:
        def __init__(self) -> None:
            self.loaded: Path | None = None

        def set_idle_clip(self, path: Path | None, phrase_sec: float = 4.5) -> dict[str, Any]:
            self.loaded = path
            return {}

    class _Cfg:
        def __init__(self, cid: str) -> None:
            self.data = {"rig": {"idleClip": cid}}

    old = server.MOTIONS_DIR
    server.MOTIONS_DIR = tmp
    try:
        (tmp / "synth_idle.json").write_text(json.dumps(synth_clip()), encoding="utf-8")
        fake = type("R", (), {})()
        fake.rig, fake.config = _Rig(), _Cfg("synth_idle")
        server.StudioRuntime._restore_idle_clip(fake)
        check("server. saved idle clip is re-installed on startup",
              fake.rig.loaded is not None and fake.rig.loaded.name == "synth_idle.json", str(fake.rig.loaded))
        fake.rig, fake.config = _Rig(), _Cfg("gone_clip")
        server.StudioRuntime._restore_idle_clip(fake)
        check("server. a missing saved clip is cleared, not crashed on",
              fake.rig.loaded is None and fake.config.data["rig"]["idleClip"] == "")
        fake.rig, fake.config = _Rig(), _Cfg("../evil")
        server.StudioRuntime._restore_idle_clip(fake)
        check("server. a path-like clip id is refused", fake.rig.loaded is None and fake.config.data["rig"]["idleClip"] == "")
    finally:
        server.MOTIONS_DIR = old


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_prepare()
        test_rig(tmp)
        test_restore(tmp)
    ok = sum(1 for _, c, _ in RESULTS if c)
    print(f"\n{ok}/{len(RESULTS)} checks passed")
    return 0 if ok == len(RESULTS) else 1


def test_all() -> None:          # pytest entry point
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
