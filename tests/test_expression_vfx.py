"""Verification: Mao / Neuro-style expression VFX & blend-shape layer.

  * all 10 VFX params ease in / out smoothly (no jumps) and animate in their ranges (twinkle / bob / throb)
  * Cat Mouth ducks while the lip-sync mouth is open (> 0.15) and eases back to :3 in pauses
  * auto emotion mapping (rig.autoVfx), manual overrides beat auto, thinking / neutral calm everything
  * every emote (original + new) runs without NaN / errors; wrists stay within +-0.65; shoulders untouched
  * decision engine knows the new emotions / emotes
  * GET / POST /api/vfx, GET /api/emotes, POST /api/config rig.autoVfx on the real server.py handlers
    (random loopback port, never 8800; no engine / mic / network)

    python tests/test_expression_vfx.py        (or: python -m pytest tests/test_expression_vfx.py)
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from core.config import ConfigStore, default_config  # noqa: E402
from core.events import HookRegistry  # noqa: E402
import core.rig_controller as rig_mod  # noqa: E402
from core.rig_controller import EMOTE_CATALOG, VFX_MOTION, VFX_PARAMS, RigController  # noqa: E402


class VirtualClock:
    """The controller clocks itself with time.perf_counter(); the test advances it exactly DT per frame so
    oscillation frequencies, emote envelopes and holds are measured in simulated (not wall) time."""

    def __init__(self) -> None:
        import time as _t
        self.now = _t.perf_counter()

    def perf_counter(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        import time as _t
        return getattr(_t, name)


CLOCK = VirtualClock()
rig_mod.time = CLOCK          # type: ignore[assignment]

RESULTS: list[tuple[str, bool, str]] = []
DT = 1.0 / 60.0
NEW_EMOTES = ("starry_eyes", "heart_crush", "pout_puff", "cat_smug", "anime_sweat", "anger_vein", "vibe_music")
WRIST_EMOTES = ("greet_wave", "cheer_bounce", "excited_hop", "flustered_shy", "wink_wave")


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


class FakeLipSync:
    """Stands in for LiveLipSyncEngine: a settable mouth-open / energy frame."""

    def __init__(self) -> None:
        self.open = 0.0
        self.energy = 0.0

    def step(self, dt: float, t_now: float) -> dict[str, float]:
        return {"open": self.open, "form": 0.1, "funnel": 0.0, "shrug": 0.0, "x": 0.0, "tongue": 0.0, "press": 0.0,
                "energy": self.energy, "stress": 0.3 if self.energy else 0.0, "onset": 0.0, "pitch": 0.5,
                "voiced": 1.0 if self.energy else 0.0}


def rig_param_defaults() -> dict[str, tuple[float, float]]:
    """Every parameter name the controller writes (scraped from its source) + the 10 VFX params + shoulders."""
    src = (HERE / "core" / "rig_controller.py").read_text(encoding="utf-8")
    names = set(re.findall(r'p\["([^"]+)"\]', src))
    names |= {f"Arm {s} {j}" for s in "RL" for j in ("Elbow", "Wrist")}
    names |= {"Brow L Angle", "Brow R Angle", "Cheek", "Proportion (Chibi)", "Gloom", "Emissive Glow",
              "Chest X", "Chest Y", "Eye Jelly", "Mouth Spring", "Arm R Swing", "Arm L Swing", "Lips Press / Part",
              "Shoulder R", "Shoulder L"}
    names |= set(VFX_PARAMS)
    d = {n: (0.0, 0.0) for n in names}
    d["Emissive Glow"] = (1.0, 0.0)
    d["Left Eye Open"] = d["Right Eye Open"] = (1.0, 0.0)
    return d


def new_rig() -> tuple[RigController, FakeLipSync, dict[str, Any]]:
    lip = FakeLipSync()
    rig = RigController(rig_param_defaults(), lip, HookRegistry())   # type: ignore[arg-type]
    cfg = default_config()
    cfg["rig"]["autoGaze"] = True
    return rig, lip, cfg


def run(rig: RigController, cfg: dict[str, Any], seconds: float) -> list[dict[str, tuple[float, float]]]:
    frames = []
    for _ in range(int(round(seconds / DT))):
        CLOCK.now += DT
        p, _, _ = rig.step(DT, cfg, None)
        frames.append(dict(p))
    return frames


def series(frames: list[dict[str, tuple[float, float]]], name: str) -> list[float]:
    return [f[name][0] for f in frames]


# ------------------------------------------------------------------------------------------------ rig
def test_rig_layer() -> None:
    import time as _time
    rig, lip, cfg = new_rig()
    run(rig, cfg, 0.5)
    base = rig.get_vfx_state()
    check("rig. all 10 VFX params known, start at 0", base["available"] == list(VFX_PARAMS)
          and all(v == 0.0 for v in base["active"].values()), str(base))

    # Manual overrides: every param eases in smoothly, then animates within its range.
    for k in VFX_PARAMS:
        rig.set_vfx_override(k, 1.0)
    rise = run(rig, cfg, 1.0)
    hold = run(rig, cfg, 2.0)
    for k in VFX_PARAMS:
        s_rise = series(rise, k)
        s_hold = series(hold, k)
        jumps = max(abs(b - a) for a, b in zip([0.0] + s_rise, s_rise + s_hold))
        lo = VFX_MOTION.get(k, (1.0, 0.0))[0]
        if k == "Cat Mouth":
            lo = 1.0
        ok_range = min(s_hold) >= lo - 0.03 and max(s_hold) <= 1.0 + 1e-9
        check(f"rig. {k}: smooth ease-in (max step {jumps:.3f}) and holds in [{lo:.2f}, 1.0]",
              jumps < 0.2 and ok_range and s_rise[0] < 0.2, f"first={s_rise[0]:.3f} min={min(s_hold):.3f} max={max(s_hold):.3f}")
        if k in VFX_MOTION:
            check(f"rig. {k}: lively motion while active", max(s_hold) - min(s_hold) > 0.08,
                  f"span={max(s_hold) - min(s_hold):.3f}")
    # Anger throbs at ~3.5 Hz (count peaks over 2 s), Music bobs at ~2.2 Hz
    def peaks(s: list[float]) -> int:
        return sum(1 for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > (min(s) + max(s)) / 2)
    pa, pm = peaks(series(hold, "VFX Anger")), peaks(series(hold, "VFX Music"))
    check("rig. VFX Anger throbs ~3.5 Hz", 6 <= pa <= 8, str(pa))
    check("rig. VFX Music bobs ~2.2 Hz", 4 <= pm <= 5, str(pm))

    # Cat Mouth ducks during open talking vowels, returns to :3 in pauses
    lip.open, lip.energy = 0.6, 0.35
    talk = run(rig, cfg, 0.5)
    cat_talk = series(talk, "Cat Mouth")
    lip.open, lip.energy = 0.0, 0.0
    pause = run(rig, cfg, 1.5)
    cat_pause = series(pause, "Cat Mouth")
    check("rig. Cat Mouth ducks to ~0 while mouth_open > 0.15", cat_talk[-1] < 0.03 and max(cat_talk[10:]) < 0.12,
          f"end={cat_talk[-1]:.3f}")
    check("rig. Cat Mouth ducks smoothly (no pop)", max(abs(b - a) for a, b in zip(cat_talk, cat_talk[1:])) < 0.4)
    check("rig. Cat Mouth eases back to :3 in the pause", cat_pause[-1] > 0.95 and cat_pause[3] < 0.5,
          f"end={cat_pause[-1]:.3f}")
    lip.open, lip.energy = 0.10, 0.2                                  # small mouth movement under the threshold
    small = run(rig, cfg, 0.6)
    check("rig. Cat Mouth stays on for tiny mouth movement (<= 0.15)", min(series(small, "Cat Mouth")[-20:]) > 0.9)
    lip.open, lip.energy = 0.0, 0.0

    # Clear: everything fades back to 0 smoothly
    rig.clear_vfx_overrides()
    fall = run(rig, cfg, 2.0)
    check("rig. clear_vfx_overrides fades all VFX to 0 smoothly",
          all(series(fall, k)[-1] < 0.01 for k in VFX_PARAMS if k != "Cheek Puff")
          and series(fall, "Cheek Puff")[-1] <= 0.01 + cfg["rig"]["cheekPuff"]
          and all(max(abs(b - a) for a, b in zip(series(fall, k), series(fall, k)[1:])) < 0.2 for k in VFX_PARAMS))

    # Auto emotion mapping
    def settle(emo: str, secs: float = 1.2) -> dict[str, float]:
        rig.set_vfx_emotion(emo, hold=30.0)
        run(rig, cfg, secs)
        return rig.get_vfx_state()["active"]

    a = settle("excited")
    check("auto. excited -> sparkles (~0.85) + star eyes", 0.85 * 0.76 <= a["VFX Sparkles"] <= 0.86 and a["Eye Star"] > 0.95, str(a))
    a = settle("love")
    check("auto. love -> hearts (~0.9) + heart eyes", 0.9 * 0.8 <= a["VFX Hearts"] <= 0.91 and a["Eye Heart"] > 0.95 and a["Eye Star"] < 0.05, str(a))
    a = settle("flustered")
    check("auto. flustered -> hearts ~0.65, no heart eyes", 0.65 * 0.8 <= a["VFX Hearts"] <= 0.66 and a["Eye Heart"] < 0.05, str(a))
    a = settle("angry")
    check("auto. angry -> anger vein ~0.95", 0.95 * 0.7 <= a["VFX Anger"] <= 0.96, str(a))
    a = settle("pout")
    check("auto. pout / annoyed -> anger + cheek puff ~0.55", a["VFX Anger"] > 0.6 and 0.5 <= a["Cheek Puff"] <= 0.6, str(a))
    a = settle("nervous")
    check("auto. nervous -> sweat ~0.95", 0.93 <= a["VFX Sweat"] <= 0.96, str(a))
    a = settle("surprised")
    check("auto. surprised -> shock ~0.9", 0.88 <= a["VFX Shock"] <= 0.91, str(a))
    a = settle("singing")
    check("auto. singing -> music notes ~0.95", 0.95 * 0.78 <= a["VFX Music"] <= 0.96, str(a))
    rig.set_vfx_emotion("smug", hold=30.0)
    frames = run(rig, cfg, 1.5)
    a = rig.get_vfx_state()["active"]
    lb, rb = series(frames, "Left Brow Y")[-1], series(frames, "Right Brow Y")[-1]
    check("auto. smug -> :3 cat mouth + asymmetric brow / half-lid look",
          a["Cat Mouth"] > 0.95 and lb - rb > 0.3 and series(frames, "Eye L Squint")[-1] >= 0.4, f"{a} lb={lb:.2f} rb={rb:.2f}")
    rig.set_interaction_state("thinking")
    run(rig, cfg, 2.5)
    a = rig.get_vfx_state()["active"]
    check("auto. thinking calms all VFX to 0", max(a.values()) < 0.03, str(a))
    rig.set_interaction_state("idle")
    a = settle("neutral", 2.0)
    check("auto. neutral -> all VFX smoothly back to 0", max(a.values()) < 0.02, str(a))
    rig.set_mood_state(0.0, 0.0)
    a = settle("happy", 1.5)
    check("auto. ordinary happy (low mood) -> no sparkles", a["VFX Sparkles"] < 0.02, str(a))

    # Override beats auto; autoVfx off stops auto but keeps overrides
    rig.set_vfx_emotion("love", hold=30.0)
    rig.set_vfx_override("VFX Hearts", 0.3)
    rig.set_vfx_override("VFX Sparkles", 1.0)
    run(rig, cfg, 1.5)
    a = rig.get_vfx_state()["active"]
    check("override. manual value beats the auto-emotion value", a["VFX Hearts"] <= 0.31 and a["VFX Sparkles"] > 0.75, str(a))
    rig.set_auto_vfx(False)
    run(rig, cfg, 2.0)
    a = rig.get_vfx_state()["active"]
    check("override. autoVfx off: auto Eye Heart gone, manual overrides stay",
          a["Eye Heart"] < 0.02 and a["VFX Sparkles"] > 0.75 and a["VFX Hearts"] > 0.2, str(a))
    rig.set_vfx_override("VFX Sparkles", 0.0)
    check("override. value 0 toggles the override off", "VFX Sparkles" not in rig.get_vfx_state()["overrides"])
    for bad in ("VFX Lasers", "", "Sparkle Eyes"):
        try:
            rig.set_vfx_override(bad, 1.0)
            check(f"override. unknown param {bad!r} rejected", False)
        except KeyError:
            check(f"override. unknown param {bad!r} rejected", True)
    try:
        rig.set_vfx_override("VFX Hearts", float("nan"))
        check("override. NaN value rejected", False)
    except ValueError:
        check("override. NaN value rejected", True)
    rig.clear_vfx_overrides()
    rig.set_auto_vfx(True)
    rig.set_vfx_emotion("neutral")
    run(rig, cfg, 2.0)

    # Older rigs without the VFX params: nothing written, no warnings, no errors
    d = rig_param_defaults()
    for k in VFX_PARAMS:
        if k != "Cheek Puff":                  # Cheek Puff predates the VFX layer (expression puff)
            d.pop(k, None)
    old = RigController(d, FakeLipSync(), HookRegistry())   # type: ignore[arg-type]
    old.set_vfx_emotion("excited", hold=10.0)
    p = run(old, cfg, 0.5)[-1]
    check("rig. older rig without VFX params: skipped silently", not any(k in p for k in VFX_PARAMS if k != "Cheek Puff")
          and not old._warned_unknown, str(old._warned_unknown))
    _ = _time


def test_emotes() -> None:
    rig, lip, cfg = new_rig()
    check("emotes. catalog has all 15 originals + the new set", len(EMOTE_CATALOG) == 21
          and all(e in EMOTE_CATALOG for e in NEW_EMOTES), str(len(EMOTE_CATALOG)))
    defaults = rig.defaults
    for eid in EMOTE_CATALOG:
        res = rig.trigger_emote(eid)
        frames = []
        bad = ""
        for i in range(int((res.get("duration", 0) + 0.6) / DT)):
            lip.open, lip.energy = ((0.5, 0.3) if (i // 30) % 3 == 1 else (0.0, 0.0))   # talk now and then
            try:
                CLOCK.now += DT
                p, _, _ = rig.step(DT, cfg, None)
            except Exception as exc:                                  # noqa: BLE001
                bad = f"{exc.__class__.__name__}: {exc}"
                break
            frames.append(p)
            if any(not (math.isfinite(v[0]) and math.isfinite(v[1])) for v in p.values()):
                bad = "non-finite value"
                break
        lip.open, lip.energy = 0.0, 0.0
        ok = res.get("ok") and not bad and frames
        vfx_ok = all(0.0 <= f[k][0] <= 1.0 for f in frames for k in VFX_PARAMS)
        wr = max(abs(f[n][0]) for f in frames for n in ("Arm R Wrist", "Arm L Wrist")) if frames else 9.0
        shoulders = all(f[n] == defaults[n] for f in frames for n in ("Shoulder R", "Shoulder L"))
        check(f"emotes. {eid} runs clean (finite, VFX in [0,1], wrists <= 0.65, shoulders locked)",
              bool(ok) and vfx_ok and wr <= 0.65 + 1e-9 and shoulders, bad or f"wrist={wr:.3f}")
        if eid in WRIST_EMOTES:
            check(f"emotes. {eid} has lively wrist follow-through", wr > 0.25, f"wrist={wr:.3f}")
        for k in EMOTE_CATALOG[eid].get("vfx", ()):
            peak = max(f[k][0] for f in frames)
            check(f"emotes. {eid} fires {k}", peak > 0.6, f"peak={peak:.3f}")
        run(rig, cfg, 1.5)                                           # let VFX settle between emotes
    # autoVfx gates the extra sparkles / hearts on the upgraded raised-arm emotes
    for auto in (True, False):
        rig.set_auto_vfx(auto)
        rig.trigger_emote("cheer_bounce")
        peak = max(f["VFX Sparkles"][0] for f in run(rig, cfg, 2.8))
        check(f"emotes. cheer_bounce sparkles {'on' if auto else 'off'} with autoVfx={auto}",
              (peak > 0.4) if auto else (peak < 0.02), f"peak={peak:.3f}")
        run(rig, cfg, 1.5)
        rig.trigger_emote("flustered_shy")
        peak = max(f["VFX Hearts"][0] for f in run(rig, cfg, 3.0))
        check(f"emotes. flustered_shy hearts {'on' if auto else 'off'} with autoVfx={auto}",
              (peak > 0.3) if auto else (peak < 0.02), f"peak={peak:.3f}")
        run(rig, cfg, 1.5)
    rig.set_auto_vfx(True)


# ------------------------------------------------------------------------------------------------ decider
def test_decider() -> None:
    from modules.decision_engine import EMOTIONS, IDLE_EMOTES, classify_text, hikari_questions
    offered = {e for v in EMOTIONS.values() for e in v["emotes"]}
    check("decider. every offered emote exists in the rig catalog", offered <= set(EMOTE_CATALOG), str(offered - set(EMOTE_CATALOG)))
    check("decider. offers all 7 rich emotes", set(NEW_EMOTES) <= offered, str(set(NEW_EMOTES) - offered))
    check("decider. idle emotes exist in the rig catalog", set(IDLE_EMOTES) <= set(EMOTE_CATALOG))
    for emo in ("smug", "playful", "nervous", "love", "pout", "singing"):
        check(f"decider. emotion {emo} exists", emo in EMOTIONS)
    qs = {q.name: q for q in hikari_questions("reply")}
    check("decider. LLM emote question lists the rich emotes with VFX labels",
          set(NEW_EMOTES) <= qs["emote"].ids and "sparkles" in dict(qs["emote"].options)["starry_eyes"])
    cases = [
        ("Hehe, just kidding~ :3", "reply", "playful"),
        ("Aww, love you too <3", "reply", "love"),
        ("Grr! How dare you say that!", "user", "angry"),
        ("Uh oh... this is awkward, ahaha...", "reply", "nervous"),
        ("La la la, I'm singing my favorite song!", "reply", "singing"),
        ("hmph, baka!", "user", "pout"),
        ("hi hikari!", "user", "greeting"),
        ("I love you hikari", "user", "flustered"),
    ]
    for text, src, want in cases:
        got = classify_text(text, src).emotion
        check(f"decider. {text!r} ({src}) -> {want}", got == want, got)


# ------------------------------------------------------------------------------------------------ HTTP
async def test_http(tmp: Path) -> None:
    import server
    from modules.decision_engine import DecisionEngine

    class FakeRuntime:
        sync_rig_flags = server.StudioRuntime.sync_rig_flags

        def __init__(self) -> None:
            self.config = ConfigStore(tmp / "settings.json")
            self.rig, self.lip, _ = new_rig()
            self.decisions = DecisionEngine(self.config.data["ai"])
            self._shutdown_task = None
            self._fish_ended_on_shutdown = True
            self.broadcasts: list = []
            self.sync_rig_flags()

        def start_loop(self, loop) -> None:
            pass

        def stop_loop(self) -> None:
            pass

        def note_stage_event(self, text: str, kind: str) -> None:
            self.broadcasts.append({"stage": text, "kind": kind})

        async def broadcast_json(self, msg, exclude=None) -> None:
            self.broadcasts.append(msg)

    class _Mem:
        async def close(self) -> None:
            pass

    rt = FakeRuntime()
    rt.memory = _Mem()
    app = server.build_app(rt)
    cfg = rt.config.data
    async with TestClient(TestServer(app, host="127.0.0.1", port=0)) as cl:
        check("http. test server is NOT on port 8800", cl.server.port != 8800, str(cl.server.port))
        r = await cl.get("/api/vfx")
        j = await r.json()
        check("http. GET /api/vfx shape", r.status == 200 and j["ok"] is True and j["autoVfx"] is True
              and j["params"] == list(VFX_PARAMS) and j["overrides"] == {} and set(j["active"]) == set(VFX_PARAMS), str(j))

        r = await cl.post("/api/vfx", json={"param": "VFX Sparkles", "value": 1.0})
        j = await r.json()
        check("http. POST /api/vfx param+value", r.status == 200 and j["ok"] and j["overrides"] == {"VFX Sparkles": 1.0}
              and "active" in j and "autoVfx" in j, str(j))
        run(rt.rig, cfg, 1.0)
        j = await (await cl.get("/api/vfx")).json()
        check("http. override animates live", j["active"]["VFX Sparkles"] > 0.7, str(j["active"]))

        r = await cl.post("/api/vfx", json={"overrides": {"Cat Mouth": 1.0, "Eye Heart": 0.8}})
        j = await r.json()
        check("http. POST /api/vfx overrides replaces the set", r.status == 200 and j["overrides"] == {"Cat Mouth": 1.0, "Eye Heart": 0.8}, str(j))
        r = await cl.post("/api/vfx", json={"param": "Cat Mouth", "value": 0})
        check("http. value 0 toggles one off", (await r.json())["overrides"] == {"Eye Heart": 0.8})
        r = await cl.post("/api/vfx", json={"param": "VFX Lasers", "value": 1.0})
        check("http. unknown param -> 400", r.status == 400 and (await r.json())["ok"] is False)
        r = await cl.post("/api/vfx", json={"overrides": {"VFX Hearts": 1.0, "Nope": 1.0}})
        check("http. unknown param in overrides -> 400, nothing applied",
              r.status == 400 and rt.rig.get_vfx_state()["overrides"] == {"Eye Heart": 0.8})
        r = await cl.post("/api/vfx", json={"param": "VFX Hearts", "value": "lots"})
        check("http. non-numeric value -> 400", r.status == 400)
        r = await cl.post("/api/vfx", json={})
        check("http. empty body -> 400", r.status == 400)
        r = await cl.post("/api/vfx", json={"clear": True})
        j = await r.json()
        check("http. POST /api/vfx clear", r.status == 200 and j["overrides"] == {}, str(j))
        r = await cl.post("/api/vfx", json={"clear": True}, headers={"Origin": "https://evil.example"})
        check("http. foreign Origin blocked by the local-origin guard", r.status == 403)

        r = await cl.post("/api/config", json={"rig": {"autoVfx": False}, "save": False})
        check("http. POST /api/config rig.autoVfx=false reaches the rig", r.status == 200 and rt.rig.auto_vfx is False
              and cfg["rig"]["autoVfx"] is False)
        j = await (await cl.get("/api/vfx")).json()
        check("http. GET /api/vfx reports autoVfx=false", j["autoVfx"] is False)
        r = await cl.post("/api/config", json={"rig": {"autoVfx": True}, "save": False})
        check("http. POST /api/config rig.autoVfx=true", rt.rig.auto_vfx is True)
        r = await cl.post("/api/config", json={"rig": {"autoVfx": "nope"}, "save": False})
        check("http. non-bool autoVfx rejected by the sanitizer", rt.rig.auto_vfx is True and cfg["rig"]["autoVfx"] is True)

        r = await cl.get("/api/emotes")
        j = await r.json()
        ids = {e["id"]: e for e in j["emotes"]}
        check("http. GET /api/emotes exposes the 7 rich emotes with their VFX",
              set(NEW_EMOTES) <= set(ids) and ids["starry_eyes"]["vfx"] == ["VFX Sparkles", "Eye Star"]
              and ids["starry_eyes"]["new"] is True and len(ids) == 21, str(sorted(ids)))
        r = await cl.post("/api/emote", json={"emote": "anger_vein"})
        check("http. POST /api/emote plays a new emote", r.status == 200 and (await r.json()).get("ok") is True)

    # startup wiring: a saved autoVfx=false reaches a fresh runtime's rig
    rt.config.data["rig"]["autoVfx"] = False
    rt2 = FakeRuntime.__new__(FakeRuntime)
    rt2.config = rt.config
    rt2.rig, _, _ = new_rig()
    FakeRuntime.sync_rig_flags(rt2)
    check("startup. config rig.autoVfx is applied to the RigController", rt2.rig.auto_vfx is False)


def main() -> int:
    test_rig_layer()
    test_emotes()
    test_decider()
    with tempfile.TemporaryDirectory(prefix="hikari-vfx-test-") as td:
        asyncio.run(test_http(Path(td)))
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}", flush=True)
    return 1 if failed else 0


def test_all() -> None:                 # pytest entry point
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
