"""Unit tests for Hikari v30 Kuroama Head & Soft-Volume upgrades:
- 3D eyeball perspective + nose tip projection on Angle (11x5) (strictly zero at 0 deg rest)
- Cheek squish on Mouth Shrug + Angle X Cheek Squish + Cheek L/R PY1 spring blend shapes
- Side-independent Arm R/L Fit clothing coupling
- Snappy Kuroama bouncy SimplePhysics tuning
- Live RigController drivers: corner Angle Z roll, Body Z2 counter-sway, conditional saccade/blink eye-bounce,
  cheek springs, Angle X Cheek Squish, and raised-arm clothing fit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

STUDIO = Path(__file__).resolve().parents[1]
if str(STUDIO) not in sys.path:
    sys.path.insert(0, str(STUDIO))

from core.config import default_config
from core.customizer import _upgrade_spring_physics
from core.events import HookRegistry
from core.rig_controller import RigController
from core.texture_upscaler import _unpack_inp
from core.vfx_rig import FACE, NOSE_NODE, apply_vfx_rig, build_vfx_textures
from modules.lipsync import LiveLipSyncEngine


def _load_puppet(inp: Path) -> dict:
    raw = inp.read_bytes()
    end = 12 + int.from_bytes(raw[8:12], "big")
    return json.loads(raw[12:end])


def _param_defaults(pup: dict) -> dict[str, tuple[float, float]]:
    return {
        pr["name"]: (float(pr["defaults"][0]), float(pr["defaults"][1]))
        for pr in pup.get("param", [])
    }


def test_kuroama_v30_rig_bindings_and_zero_rest() -> None:
    inp = STUDIO / "models" / "Hikari-v28-Custom.inp"
    if not inp.exists():
        inp = STUDIO / "models" / "Hikari-v28-BaseHD-v5.inp"
    header, texs, _ = _unpack_inp(inp)
    pup = _load_puppet(inp)
    by_name = {pr["name"]: pr for pr in pup.get("param", [])}

    # If testing against BaseHD (pre-VFX), apply VFX + Kuroama pass in memory
    if "Angle X Cheek Squish" not in by_name:
        tid_of, sizes = {}, {}
        out_texs = list(texs)
        for slot, png, w, h in build_vfx_textures(out_texs[51][1]):
            tid_of[slot] = len(out_texs)
            out_texs.append((0, png))
            sizes[slot] = (w, h)
        new_h = apply_vfx_rig(header, tid_of, sizes, {"R": out_texs[23][1], "L": out_texs[24][1]})
        pup = json.loads(new_h[12:])
        by_name = {pr["name"]: pr for pr in pup.get("param", [])}

    # 1. Required Kuroama v30 params exist with deform bindings
    for name in ("Angle X Cheek Squish", "Cheek L PY1", "Cheek R PY1", "Arm R Fit", "Arm L Fit"):
        assert name in by_name, f"missing Kuroama param {name}"
        assert any(b.get("param_name") == "deform" for b in by_name[name]["bindings"]), f"{name} missing deform"

    # 2. Mouth Shrug now binds to ArtMeshFace (1011) for cheek squish while keeping VB Mouth bindings
    ms = by_name["Mouth Shrug"]
    ms_nodes = {b["node"] for b in ms["bindings"]}
    assert FACE in ms_nodes, "Mouth Shrug missing ArtMeshFace cheek-squish deform"
    assert 40006 in ms_nodes, "Mouth Shrug lost VB Mouth upper lip binding"

    # 3. Angle (11x5) preserves exact 0.0 rest at center yaw (index 5) and adds 3D nose tip projection at +-45 deg
    ang = by_name["Angle"]
    nose_b = next(b for b in ang["bindings"] if b["node"] == NOSE_NODE and b["param_name"] == "deform")
    V_nose = np.asarray(nose_b["values"], np.float64)
    assert np.allclose(V_nose[5, 2], 0.0, atol=1e-4), "Nose rest deform at yaw=0 must remain zero"
    tip_span_neg = float(V_nose[0, 2, :, 0].max() - V_nose[0, 2, :, 0].min())
    tip_span_pos = float(V_nose[10, 2, :, 0].max() - V_nose[10, 2, :, 0].min())
    assert tip_span_neg >= 2.0 and tip_span_pos >= 2.0, f"Expected 3D nose tip projection, got {tip_span_neg}, {tip_span_pos}"

    # 4. Far iris foreshortens cleanly at +-45 deg while rest yaw=0 stays zero
    ir_r = next(b for b in ang["bindings"] if b["node"] == 1023 and b["param_name"] == "deform")
    V_ir = np.asarray(ir_r["values"], np.float64)
    assert np.allclose(V_ir[5, 2], 0.0, atol=1e-4), "Iris R rest deform at yaw=0 must remain zero"
    assert float(V_ir[0, 2, :, 0].max() - V_ir[0, 2, :, 0].min()) > 12.0, "Far iris R should foreshorten at -45 deg"

    # 5. Snappy Kuroama spring physics frequencies applied cleanly
    up_h = _upgrade_spring_physics(header)
    up_pup = json.loads(up_h[12:])
    freqs = {}
    def walk(n):
        if n.get("type") == "SimplePhysics":
            freqs[str(n.get("name", "")).lower()] = float(n.get("frequency", 0.0))
        for c in n.get("children", []):
            walk(c)
    walk(up_pup["nodes"])
    assert any("ahoge" in k and v >= 2.25 for k, v in freqs.items()), f"Ahoge frequency not bumped: {freqs}"
    assert any("eye jelly" in k and v >= 2.60 for k, v in freqs.items()), f"Eye jelly frequency not bumped: {freqs}"


def test_kuroama_v30_live_controller_drivers() -> None:
    inp = STUDIO / "models" / "Hikari-v28-Custom.inp"
    assert inp.exists(), "Build Hikari-v28-Custom.inp before running controller test"
    pup = _load_puppet(inp)
    defaults = _param_defaults(pup)
    rc = RigController(defaults, LiveLipSyncEngine(), HookRegistry())

    # 1. Corner Angle Z roll + Angle X Cheek Squish + Cheek L/R PY1 when happy & turned
    cfg = default_config()
    cfg["rig"]["expression"] = "happy"
    cfg["rig"]["autoBlink"] = False
    rc.mouse_x = 0.85
    rc.mouse_y = -0.65
    rc._mouse_blend = 1.0
    rc._mouse_last_move = time.perf_counter()
    p = defaults
    for _ in range(80):
        p, _, _ = rc.step(1.0 / 60.0, cfg, None)
    assert p["Angle X Cheek Squish"][0] > 0.04, f"Expected 3/4 cheek squish > 0.04, got {p['Angle X Cheek Squish']}"
    assert p["Mouth Shrug"][0] > 0.10, f"Expected happy smile Mouth Shrug > 0.10, got {p['Mouth Shrug']}"
    assert abs(p["Cheek L PY1"][0]) > 0.01 and abs(p["Cheek R PY1"][0]) > 0.01, "Cheek L/R PY1 springs inactive"
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in p.values()), "Non-finite parameter emitted"

    # 2. Conditional saccade eye-bounce triggers sp_eye_jelly impulse
    rc.sp_eye_jelly.v = 0.0
    rc._prev_eye_x = -0.55
    rc._prev_eye_y = 0.0
    rc.sp_eye_x.y = 0.55
    rc.step(1.0 / 60.0, cfg, None)
    assert abs(rc.sp_eye_jelly.v) > 0.5 or abs(rc.sp_eye_jelly.y) > 0.01, "Rapid saccade should trigger eye-bounce"

    # 3. Arm-raise greet_wave emote triggers Arm R Fit + subtle Shoulder Shrug
    rc.trigger_emote("greet_wave")
    rc._emote_start = time.perf_counter() - 0.85
    for _ in range(12):
        p, _, _ = rc.step(1.0 / 60.0, cfg, None)
    assert p["Arm R Pose"][0] > 0.7, f"Wave emote arm not raised: {p['Arm R Pose']}"
    assert p["Arm R Fit"][0] > 0.7, f"Arm R Fit should couple to raised wave arm, got {p['Arm R Fit']}"
    assert p["Arm L Fit"][0] < 0.1, f"Resting left arm should not trigger Arm L Fit, got {p['Arm L Fit']}"
    assert p["Shoulder Shrug"][0] > 0.05, f"Expected subtle shoulder lift with raised arm, got {p['Shoulder Shrug']}"


if __name__ == "__main__":
    test_kuroama_v30_rig_bindings_and_zero_rest()
    test_kuroama_v30_live_controller_drivers()
    print("PASS test_kuroama_v30.py")
