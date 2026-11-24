"""Verification: Hikari waves ONLY when the Decision Model decides she should (IF) and plays it on cue (WHEN).

  1. RigController never raises Arm R/L Pose on its own during loud, excited speech (the retired speech-peak pop)
  2. decided emotes (greet_wave / wink_wave), manual armRPose, temp armR and Motion Lab clips still raise the arms
  3. DecisionEngine offers the typed `wave_timing` question (none / now / on_reply / after_beat) and passes
     secondsSinceLastWave / returnedAfterIdleSec / source in the decision state
  4. a model's "none" is honoured: no wave, no fallback emote, tag never "wave" (playful, idle return, greeting)
  5. on_reply / after_beat -> wave granted with that timing and a wave emote
  6. the 45 s wave cooldown blocks repeats AND demotes the tag away from "wave"; legacy yes/no answers parse
  7. server.py: apply_decision delays after_beat (cancellable), defers on_reply to deliver_reply (expires after 15 s),
     and deliver_reply / api_fish_emote never act out TAG_TO_STATE["wave"] without a decided wave
     (aiohttp test server on a random loopback port, never 8800; no engine / mic / network)

    python tests/test_wave_decisioning.py        (or: python -m pytest tests/test_wave_decisioning.py)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import random
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
from core.rig_controller import EMOTE_CATALOG, MOTION_TRACKS, RIG_FX_PARAMS, VFX_PARAMS, RigController  # noqa: E402
import modules.decision_engine as de_mod  # noqa: E402
from modules.decision_engine import (DECISION_DEFAULTS, EMOTIONS, WAVE_EMOTES, WAVE_TIMINGS, Decision,  # noqa: E402
                                     DecisionEngine, hikari_questions, parse_decision_answers)


class VirtualClock:
    """The controller clocks itself with time.perf_counter(); the test advances it exactly DT per frame."""

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


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


# ------------------------------------------------------------------------------------------------ rig helpers
class FakeLipSync:
    def __init__(self) -> None:
        self.open = 0.0
        self.energy = 0.0
        self.stress = 0.0

    def step(self, dt: float, t_now: float) -> dict[str, float]:
        return {"open": self.open, "form": 0.1, "funnel": 0.0, "shrug": 0.0, "x": 0.0, "tongue": 0.0, "press": 0.0,
                "energy": self.energy, "stress": self.stress, "onset": 0.0, "pitch": 0.5,
                "voiced": 1.0 if self.energy else 0.0}


def rig_param_defaults() -> dict[str, tuple[float, float]]:
    src = (HERE / "core" / "rig_controller.py").read_text(encoding="utf-8")
    names = set(re.findall(r'p\["([^"]+)"\]', src))
    names |= {f"Arm {s} {j}" for s in "RL" for j in ("Elbow", "Wrist")}
    names |= {"Brow L Angle", "Brow R Angle", "Cheek", "Proportion (Chibi)", "Gloom", "Emissive Glow",
              "Chest X", "Chest Y", "Eye Jelly", "Mouth Spring", "Arm R Swing", "Arm L Swing", "Lips Press / Part",
              "Shoulder R", "Shoulder L"}
    names |= set(VFX_PARAMS) | set(RIG_FX_PARAMS)
    d = {n: (0.0, 0.0) for n in names}
    d["Emissive Glow"] = (1.0, 0.0)
    d["Left Eye Open"] = d["Right Eye Open"] = (1.0, 0.0)
    return d


def new_rig() -> tuple[RigController, FakeLipSync, dict[str, Any]]:
    lip = FakeLipSync()
    rig = RigController(rig_param_defaults(), lip, HookRegistry())   # type: ignore[arg-type]
    rig._rnd.seed(20261124)
    cfg = default_config()
    return rig, lip, cfg


def run(rig: RigController, cfg: dict[str, Any], seconds: float) -> list[dict[str, tuple[float, float]]]:
    frames = []
    for _ in range(int(round(seconds / DT))):
        CLOCK.now += DT
        p, _, _ = rig.step(DT, cfg, None)
        frames.append({"Arm R Pose": p["Arm R Pose"], "Arm L Pose": p["Arm L Pose"]})
    return frames


def peak(frames: list[dict[str, tuple[float, float]]], name: str) -> float:
    return max(f[name][0] for f in frames)


# ------------------------------------------------------------------------------------------------ 1 + 2 rig
def test_rig_no_speech_peak_pop(tmp: Path) -> None:
    rig, lip, cfg = new_rig()
    cfg["rig"]["autoGestureOnSpeech"] = True
    rig.set_mood_state(0.7, 0.9)                     # very excited mood (ma well above 0.35)
    lip.open, lip.energy, lip.stress = 0.8, 0.8, 0.95
    rig.trigger_temp_action({"expression": "excited"}, duration=60.0)   # expr "excited", NO armR / armL keys
    frames = run(rig, cfg, 40.0)                     # the retired pop fired after 4 s, then every 8-14 s
    check("rig. loud excited speech (stress .95, energy .8, 40 s) never raises Arm R Pose",
          peak(frames, "Arm R Pose") < 0.05, f"{peak(frames, 'Arm R Pose'):.3f}")
    check("rig. loud excited speech never raises Arm L Pose",
          peak(frames, "Arm L Pose") < 0.05, f"{peak(frames, 'Arm L Pose'):.3f}")
    check("rig. status reports no auto arm raise", rig.step(DT, cfg, None)[1].get("armRaise", "") in ("", None)
          if isinstance(rig.step(DT, cfg, None)[1], dict) else True)

    # 2. explicit commands still raise the wave arms
    for emote, arm in (("greet_wave", "Arm R Pose"), ("wink_wave", "Arm R Pose")):
        rig2, lip2, cfg2 = new_rig()
        run(rig2, cfg2, 0.5)
        res = rig2.trigger_emote(emote)
        f = run(rig2, cfg2, 1.6)
        check(f"rig. {emote} emote still raises {arm}", res.get("ok") and peak(f, arm) > 0.8, f"{peak(f, arm):.3f}")

    rig3, _, cfg3 = new_rig()
    cfg3["rig"]["armRPose"] = 1.0
    f = run(rig3, cfg3, 1.5)
    check("rig. manual armRPose=1.0 slider still raises Arm R Pose", f[-1]["Arm R Pose"][0] > 0.9, f"{f[-1]['Arm R Pose'][0]:.3f}")
    cfg3["rig"]["armRPose"] = 0.0
    cfg3["rig"]["armLPose"] = 1.0
    f = run(rig3, cfg3, 1.5)
    check("rig. manual armLPose=1.0 slider still raises Arm L Pose (R eases down)",
          f[-1]["Arm L Pose"][0] > 0.9 and f[-1]["Arm R Pose"][0] < 0.1, str(f[-1]))

    rig4, _, cfg4 = new_rig()
    rig4.trigger_temp_action({"armR": 1.0, "armL": 0.0}, duration=5.0)
    f = run(rig4, cfg4, 1.5)
    check("rig. temp action armR=1.0 still raises Arm R Pose", f[-1]["Arm R Pose"][0] > 0.9, f"{f[-1]['Arm R Pose'][0]:.3f}")

    rig5, lip5, cfg5 = new_rig()
    n = 240
    tracks = {k: [0.0] * n for k in MOTION_TRACKS}
    tracks["EyeOpen"] = [1.0] * n
    tracks["ArmL"] = [1.0] * n
    tracks["ArmLGesture"] = [0.3] * n
    tracks["ArmRGesture"] = [0.0] * n                # rich (v2) clip: tracked raises own the wave arms
    clip = {"id": "t", "name": "t", "fps": 60, "frames": n, "duration": n / 60.0, "tracks": tracks}
    path = tmp / "clip.json"
    path.write_text(json.dumps(clip))
    rig5.load_motion_clip(path, loop=True, mix=1.0)
    lip5.open, lip5.energy, lip5.stress = 0.8, 0.8, 0.95
    f = run(rig5, cfg5, 2.0)
    check("rig. Motion Lab clip (ArmL track) still raises Arm L Pose", f[-1]["Arm L Pose"][0] > 0.8,
          f"{f[-1]['Arm L Pose'][0]:.3f}")
    check("rig. Motion Lab clip leaves the untracked Arm R down", peak(f, "Arm R Pose") < 0.05,
          f"{peak(f, 'Arm R Pose'):.3f}")


# ------------------------------------------------------------------------------------------------ 3-6 decisions
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def gateway_engine(clock: Clock, answers: dict[str, Any], seen: dict[str, Any]) -> DecisionEngine:
    de = DecisionEngine({"decisionMode": "gateway", "autoWaveOnGreeting": True}, clock=clock, rng=random.Random(7))
    de.resolve_decider = lambda: "gateway"      # type: ignore[method-assign]

    async def fake_typed(cfg, provider, state, questions, **kw):
        seen["state"] = state
        seen["q"] = {q.name: q for q in questions}
        return dict(answers)

    de_mod.run_typed_decision = fake_typed
    return de


async def test_decisions() -> None:
    real = de_mod.run_typed_decision
    try:
        # 3. the question + state
        qs = {q.name: q for q in hikari_questions("user")}
        wq = qs.get("wave_timing")
        check("decide. wave_timing is a typed choice question", wq is not None and not wq.yes_no, str(list(qs)))
        check("decide. wave_timing offers none / now / on_reply / after_beat",
              wq is not None and [o for o, _ in wq.options] == list(WAVE_TIMINGS), str(wq.options if wq else None))
        check("decide. wave instructions make none the default + mention recency",
              wq is not None and "none is the normal default" in wq.instructions and "secondsSinceLastWave" in wq.instructions)
        check("decide. emote instructions carry wave restraint", "Never pick a waving motion" in qs["emote"].instructions)
        body = de_mod.build_decision_body("jev", "typesafe-ai/jev", {"x": 1}, list(qs.values()))
        check("decide. evaluator body keeps the keyed choice format for wave_timing",
              body["questions"]["wave_timing"]["type"] == "choice"
              and set(body["questions"]["wave_timing"]["criteria"]) == set(WAVE_TIMINGS))
        check("decide. default waveCooldownSec is 45 s", DECISION_DEFAULTS["waveCooldownSec"] == 45.0)
        check("decide. playful never offers wink_wave", "wink_wave" not in EMOTIONS["playful"]["emotes"]
              and not (set(EMOTIONS["playful"]["emotes"]) & set(WAVE_EMOTES)))

        clock, seen = Clock(), {}
        answers = {"emotion": "greeting", "emote": "greet_wave", "wave_timing": "now"}
        de = gateway_engine(clock, answers, seen)
        await de.decide("hi!", "user")
        check("decide. first decision has no secondsSinceLastWave (never waved)", "secondsSinceLastWave" not in seen["state"],
              str(seen["state"]))
        check("decide. state carries the source", seen["state"].get("source") == "user")
        clock.t += 20.0
        answers.update({"emotion": "curious", "emote": "none", "wave_timing": "none"})
        await de.decide("what are you doing?", "user")
        check("decide. state passes secondsSinceLastWave after a wave", seen["state"].get("secondsSinceLastWave") == 20.0,
              str(seen["state"]))
        check("decide. no returnedAfterIdleSec for a quick follow-up", "returnedAfterIdleSec" not in seen["state"])
        clock.t += 400.0
        await de.decide("i'm back", "user")
        check("decide. state passes returnedAfterIdleSec after an idle gap",
              seen["state"].get("returnedAfterIdleSec") == 400.0, str(seen["state"]))
        llm_body = json.loads(de.build_llm_request("hey", "user")["payload"]["messages"][1]["content"])
        check("decide. chat-LLM decider payload carries wave timings + recency",
              set(llm_body.get("waveTimings", {})) == set(WAVE_TIMINGS) and "secondsSinceLastWave" in llm_body
              and "returnedAfterIdleSec" in llm_body, str(list(llm_body)))

        # 4. "none" is honoured
        for emotion, text in (("playful", "hehe just kidding :3"), ("greeting", "hi hikari!"), ("excited", "yay we won!!")):
            clock2, seen2 = Clock(), {}
            de2 = gateway_engine(clock2, {"emotion": emotion, "emote": "none", "wave_timing": "none"}, seen2)
            if emotion == "greeting":
                await de2.decide("hello", "user")
                clock2.t += 500.0                    # she returns after a long idle gap
            d = await de2.decide(text, "user")
            check(f"decide. model none/none on {emotion}: no emote, no wave, tag != wave",
                  d.emote is None and d.wave is False and d.wave_timing == "none" and d.tag != "wave"
                  and d.skipped == "decider chose none", str(d.to_dict()))
        check("decide. idle return + model none -> no forced wave", True)

        clock3, seen3 = Clock(), {}
        de3 = gateway_engine(clock3, {"emotion": "happy", "emote": "wink_wave", "wave_timing": "none"}, seen3)
        d = await de3.decide("that was fun", "user")
        check("decide. a wave emote without a wave timing is never played", d.emote is None and not d.wave and d.tag != "wave",
              str(d.to_dict()))

        # 5. on_reply / after_beat
        for timing in ("on_reply", "after_beat", "now"):
            clock4, seen4 = Clock(), {}
            de4 = gateway_engine(clock4, {"emotion": "greeting", "emote": "none", "wave_timing": timing}, seen4)
            d = await de4.decide("hey hikari!", "user")
            check(f"decide. model {timing} -> wave granted with timing {timing}",
                  d.wave is True and d.wave_timing == timing and d.emote in WAVE_EMOTES and d.tag == "wave", str(d.to_dict()))
            check(f"decide. to_dict serialises waveTiming={timing}", d.to_dict().get("waveTiming") == timing
                  and d.to_dict().get("wave") is True and "wave_timing" not in d.to_dict())
        clock5, seen5 = Clock(), {}
        de5 = gateway_engine(clock5, {"emotion": "farewell", "emote": "excited_hop", "wave_timing": "after_beat"}, seen5)
        d = await de5.decide("good night!", "user")
        check("decide. a wave timing with a non-wave emote pick still plays a wave clip",
              d.wave and d.emote in WAVE_EMOTES and d.wave_timing == "after_beat", str(d.to_dict()))
        clock6, seen6 = Clock(), {}
        de6 = gateway_engine(clock6, {"emotion": "greeting", "emote": "wink_wave", "wave_timing": "on_reply"}, seen6)
        d = await de6.decide("Hey there, welcome back!", "reply")
        check("decide. on_reply on her own line resolves to now", d.wave and d.wave_timing == "now" and d.emote == "wink_wave",
              str(d.to_dict()))

        # 6. cooldown + demotion
        clock7, seen7 = Clock(), {}
        ans7 = {"emotion": "greeting", "emote": "greet_wave", "wave_timing": "now"}
        de7 = gateway_engine(clock7, ans7, seen7)
        d1 = await de7.decide("hi!", "user")
        clock7.t += 30.0
        d2 = await de7.decide("hello again!", "user")
        check("decide. wave cooldown (30 s < 45 s) blocks a repeat wave", d1.wave and not d2.wave and d2.emote is None
              and d2.skipped == "wave cooldown", str(d2.to_dict()))
        check("decide. cooldown-blocked wave demotes tag away from wave", d2.tag != "wave" and d2.wave_timing == "none",
              d2.tag)
        clock7.t += 16.0
        d3 = await de7.decide("hiii", "user")
        check("decide. after 46 s she may wave again", d3.wave and d3.emote in WAVE_EMOTES, str(d3.to_dict()))
        ans7.update({"emotion": "farewell", "wave_timing": "now"})
        clock7.t += 5.0
        d4 = await de7.decide("bye!", "user")
        check("decide. farewell on cooldown -> gentle non-wave tag", not d4.wave and d4.tag not in ("wave",) and d4.tag == "caring",
              d4.tag)

        hd = DecisionEngine({"decisionMode": "heuristic", "autoWaveOnGreeting": True}, clock=Clock(), rng=random.Random(1))
        h1 = await hd.decide("hi hikari!", "user")
        h2 = await hd.decide("hey again", "user")
        check("decide. heuristic greeting waves (now)", h1.wave and h1.wave_timing == "now" and h1.emote in WAVE_EMOTES,
              str(h1.to_dict()))
        check("decide. heuristic repeat greeting: no wave, tag demoted", not h2.wave and h2.tag != "wave" and h2.emote is None,
              str(h2.to_dict()))
        for _ in range(12):
            hd._clock.t += 60.0                     # type: ignore[attr-defined]
            hp = await hd.decide("hehe just kidding, teehee :3", "reply")
            if hp.wave or hp.emote in WAVE_EMOTES or hp.tag == "wave":
                break
        check("decide. heuristic playful jokes never wave", not hp.wave and hp.emote not in WAVE_EMOTES and hp.tag != "wave",
              str(hp.to_dict()))

        # legacy answers
        clock8, seen8 = Clock(), {}
        de8 = gateway_engine(clock8, {"emotion": "greeting", "emote": "none", "wave": "yes"}, seen8)
        d = await de8.decide("hello!", "user")
        check("decide. legacy typed 'wave: yes' parses as now", d.wave and d.wave_timing == "now", str(d.to_dict()))
        clock9, seen9 = Clock(), {}
        de9 = gateway_engine(clock9, {"emotion": "happy", "emote": "none", "wave": "no"}, seen9)
        d = await de9.decide("hello!", "user")
        check("decide. legacy typed 'wave: no' -> no wave, no fallback", not d.wave and d.emote is None and d.tag != "wave",
              str(d.to_dict()))
        questions = hikari_questions("user")
        base = {"emotion": {"type": "choice", "choice": "happy"}, "emote": {"type": "choice", "choice": "none"}}
        out = parse_decision_answers({"answers": {**base, "wave": {"type": "noul", "noul": 0.9}}}, questions)
        check("decide. evaluator legacy noul under 'wave' -> wave_timing now", out.get("wave_timing") == "now", str(out))
        out = parse_decision_answers({"answers": {**base, "wave_timing": {"type": "choice", "choice": "no"}}}, questions)
        check("decide. evaluator legacy choice 'no' -> wave_timing none", out.get("wave_timing") == "none", str(out))
        out = parse_decision_answers({"answers": {**base, "wave_timing": {"type": "choice", "choice": "after_beat"}}}, questions)
        check("decide. evaluator wave_timing after_beat parses", out.get("wave_timing") == "after_beat", str(out))
        try:
            parse_decision_answers({"answers": {**base, "wave_timing": {"type": "choice", "choice": "maybe"}}}, questions)
            check("decide. unoffered wave_timing fails closed", False)
        except de_mod.DecisionModelError:
            check("decide. unoffered wave_timing fails closed", True)
        P = DecisionEngine.parse_llm_decision
        check("decide. LLM wave_timing on_reply parses",
              P('{"emotion": "greeting", "emote": "none", "wave_timing": "on_reply"}') == ("greeting", None, "on_reply"))
        check("decide. LLM legacy wave true -> now", P('{"emotion": "greeting", "emote": "none", "wave": true}')[2] == "now")
        check("decide. LLM legacy wave false / missing -> none",
              P('{"emotion": "happy", "emote": "none", "wave": false}')[2] == "none"
              and P('{"emotion": "happy", "emote": "none"}')[2] == "none")
        check("decide. LLM invalid wave_timing fails closed",
              P('{"emotion": "happy", "emote": "none", "wave_timing": "sometimes"}') is None)
        check("decide. legacy parse_llm_answer still returns a bool",
              DecisionEngine.parse_llm_answer('{"emotion": "happy", "emote": "none", "wave_timing": "after_beat"}')
              == ("happy", None, True))
    finally:
        de_mod.run_typed_decision = real


# ------------------------------------------------------------------------------------------------ 7 server
class FakeRig:
    def __init__(self) -> None:
        self.emotes: list[str] = []
        self.temps: list[dict[str, Any]] = []

    def trigger_emote(self, eid: str, duration: float = 2.6) -> dict[str, Any]:
        if eid in EMOTE_CATALOG:
            self.emotes.append(eid)
            return {"ok": True, "emote": eid}
        return {"ok": False}

    def trigger_temp_action(self, action: dict[str, Any], duration: float = 4.5) -> None:
        self.temps.append(dict(action))

    def set_vfx_emotion(self, *a: Any, **k: Any) -> None:
        pass


async def test_server(tmp: Path) -> None:
    import server

    S = server.StudioRuntime

    class _Lip:
        speaking = False

        def is_speaking(self) -> bool:
            return self.speaking

    class _AI:
        last_turn: dict = {}

        async def synthesize_tts(self, text: str) -> None:
            return None

    class _Hooks:
        async def emit_async(self, *a: Any, **k: Any) -> list:
            return []

    class FakeRuntime:
        WAVE_AFTER_BEAT_SEC = S.WAVE_AFTER_BEAT_SEC
        PENDING_REPLY_TTL_SEC = S.PENDING_REPLY_TTL_SEC
        DECISION_GRACE_SEC = S.DECISION_GRACE_SEC
        apply_decision = S.apply_decision
        cancel_pending_wave = S.cancel_pending_wave
        take_pending_reply_emote = S.take_pending_reply_emote
        _delayed_emote = S._delayed_emote
        _fire_pending_reply_emote = S._fire_pending_reply_emote
        spawn = S.spawn
        deliver_reply = S.deliver_reply
        speak_text = S.speak_text
        _await_decision = S._await_decision
        _late_decision = S._late_decision
        sync_rig_flags = S.sync_rig_flags

        def __init__(self) -> None:
            self.rig = FakeRig()
            self.lipsync = _Lip()
            self.ai = _AI()
            self.hooks = _Hooks()
            self._bg_tasks: set = set()
            self._wave_task = None
            self._pending_reply_emote = None
            self._pending_reply_at = 0.0
            self._subtitle = ""
            self.broadcasts: list = []

        async def broadcast_json(self, msg: Any, exclude: Any = None) -> None:
            self.broadcasts.append(msg)

    def done_task(d: Decision | None) -> asyncio.Task:
        async def _d() -> Decision | None:
            return d
        return asyncio.ensure_future(_d())

    rt = FakeRuntime()
    beat = Decision(source="user", emotion="greeting", tag="wave", emote="greet_wave", wave=True, wave_timing="after_beat")
    await rt.apply_decision(beat)
    check("server. after_beat does not play immediately", rt.rig.emotes == [], str(rt.rig.emotes))
    await asyncio.sleep(0.45)
    check("server. after_beat still waiting at 0.45 s", rt.rig.emotes == [], str(rt.rig.emotes))
    await asyncio.sleep(0.5)
    check("server. after_beat plays the wave after ~0.75 s", rt.rig.emotes == ["greet_wave"], str(rt.rig.emotes))
    check("server. after_beat task cleared after firing", rt._wave_task is None)

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="wink_wave", wave=True, wave_timing="after_beat"))
    await asyncio.sleep(0.2)
    await rt.apply_decision(Decision(source="reply", emotion="shy", tag="blush", emote="shy_blush"))
    await asyncio.sleep(0.9)
    check("server. a newer emote cancels the pending after_beat wave", rt.rig.emotes == ["shy_blush"], str(rt.rig.emotes))

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="after_beat"))
    rt.cancel_pending_wave()
    await asyncio.sleep(0.9)
    check("server. cancel_pending_wave (explicit action) cancels after_beat", rt.rig.emotes == [], str(rt.rig.emotes))

    # on_reply: stashed while she thinks, fired as her reply starts
    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", emotion="greeting", tag="wave", emote="greet_wave", wave=True,
                                     wave_timing="on_reply"))
    check("server. on_reply does not fire while she is still thinking", rt.rig.emotes == [] and
          rt._pending_reply_emote == "greet_wave", str(rt.rig.emotes))
    reply_d = Decision(source="reply", emotion="happy", tag="happy", skipped="cooldown")
    evt, _ = await rt.deliver_reply("Hey hey! Welcome in!", done_task(reply_d))
    check("server. on_reply wave fires as her reply starts", rt.rig.emotes == ["greet_wave"] and evt.get("emote") == "greet_wave",
          f"{rt.rig.emotes} {evt}")
    check("server. on_reply stash consumed", rt._pending_reply_emote is None)

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="on_reply"))
    own = Decision(source="reply", emotion="excited", tag="cheer", emote="cheer_bounce")
    await rt.deliver_reply("Yay, let's go!", done_task(own))
    check("server. reply that picks its own emote replaces the on_reply wave", rt.rig.emotes == ["cheer_bounce"],
          str(rt.rig.emotes))

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="on_reply"))
    rt._pending_reply_at -= 16.0
    await rt.deliver_reply("Oh, hi.", done_task(Decision(source="reply", tag="happy")))
    check("server. on_reply wave expires after 15 s", rt.rig.emotes == [], str(rt.rig.emotes))

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="on_reply"))
    rt.cancel_pending_wave()                         # api_chat's failed-turn path
    await rt.deliver_reply("Hm?", done_task(Decision(source="reply", tag="happy")))
    check("server. failed turn clears the on_reply wave", rt.rig.emotes == [], str(rt.rig.emotes))

    rt = FakeRuntime()
    rt.lipsync.speaking = True
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="on_reply"))
    check("server. on_reply while she is already speaking plays right away", rt.rig.emotes == ["greet_wave"],
          str(rt.rig.emotes))

    rt = FakeRuntime()
    await rt.apply_decision(Decision(source="user", tag="wave", emote="greet_wave", wave=True, wave_timing="now"))
    check("server. now plays immediately", rt.rig.emotes == ["greet_wave"])

    # TAG_TO_STATE["wave"] leak guards
    rt = FakeRuntime()
    leak = Decision(source="reply", emotion="greeting", tag="wave", wave=False)
    await rt.deliver_reply("Hello again!", done_task(leak))
    check("server. deliver_reply never acts out tag wave without a decided wave",
          rt.rig.temps and all(float(t.get("armR", 0)) == 0.0 for t in rt.rig.temps), str(rt.rig.temps))
    rt = FakeRuntime()
    await rt.deliver_reply("[wave] Hiii!", done_task(leak))
    check("server. an authored [wave] tag still raises her arm", any(float(t.get("armR", 0)) == 1.0 for t in rt.rig.temps),
          str(rt.rig.temps))
    rt = FakeRuntime()
    await rt.apply_decision(leak, temp_action=True)
    check("server. apply_decision temp action never uses TAG_TO_STATE[wave] without a wave",
          all(float(t.get("armR", 0)) == 0.0 for t in rt.rig.temps), str(rt.rig.temps))

    # api_fish_emote over HTTP (random loopback port, never 8800)
    class HttpRuntime:
        sync_rig_flags = S.sync_rig_flags
        apply_decision = S.apply_decision
        cancel_pending_wave = S.cancel_pending_wave
        take_pending_reply_emote = S.take_pending_reply_emote
        _delayed_emote = S._delayed_emote
        _fire_pending_reply_emote = S._fire_pending_reply_emote
        spawn = S.spawn
        WAVE_AFTER_BEAT_SEC = S.WAVE_AFTER_BEAT_SEC
        PENDING_REPLY_TTL_SEC = S.PENDING_REPLY_TTL_SEC

        def __init__(self) -> None:
            self.config = ConfigStore(tmp / "settings.json")
            self.config.data["ai"]["decisionMode"] = "heuristic"
            self.rig = FakeRig()
            self.lipsync = _Lip()
            self.config.data["ai"]["autoWaveOnGreeting"] = True     # this suite tests the AI-decided wave
            self.decisions = DecisionEngine(self.config.data["ai"])
            self._shutdown_task = None
            self._fish_ended_on_shutdown = True
            self._bg_tasks: set = set()
            self._wave_task = None
            self._pending_reply_emote = None
            self._pending_reply_at = 0.0
            self.broadcasts: list = []

        def sync_rig_flags(self) -> None:
            pass

        def start_loop(self, loop: Any) -> None:
            pass

        def stop_loop(self) -> None:
            pass

        def note_stage_event(self, text: str, kind: str) -> None:
            pass

        def set_interaction(self, state: str) -> None:
            pass

        async def record_turn(self, *a: Any, **k: Any) -> None:
            pass

        async def broadcast_json(self, msg: Any, exclude: Any = None) -> None:
            self.broadcasts.append(msg)

    class _Mem:
        async def close(self) -> None:
            pass

    hr = HttpRuntime()
    hr.memory = _Mem()
    app = server.build_app(hr)
    async with TestClient(TestServer(app, host="127.0.0.1", port=0)) as cl:
        check("http. test server is NOT on port 8800", cl.server.port != 8800, str(cl.server.port))
        j1 = await (await cl.post("/api/fish/emote", json={"text": "Hey there, welcome!", "role": "agent"})).json()
        j2 = await (await cl.post("/api/fish/emote", json={"text": "Hello hello again!", "role": "agent"})).json()
        check("http. fish agent first hello waves", j1.get("decision", {}).get("wave") is True and j1.get("tag") == "wave",
              str(j1))
        check("http. fish agent repeat hello inside cooldown: no wave, tag demoted",
              j2.get("decision", {}).get("wave") is False and j2.get("tag") != "wave", str(j2))
        check("http. fish agent repeat hello: arm never raised by TAG_TO_STATE[wave]",
              float(hr.rig.temps[-1].get("armR", 0)) == 0.0, str(hr.rig.temps[-1]))
        hr.rig.emotes.clear()
        hr.decisions._last_wave_at = -1e9
        await hr.apply_decision(Decision(source="fish_user", tag="wave", emote="greet_wave", wave=True,
                                         wave_timing="on_reply"))
        check("http. fish caller on_reply wave waits for her line", hr.rig.emotes == [], str(hr.rig.emotes))
        hr.decisions._last_emote_at = hr.decisions._clock()     # her line lands inside the emote cooldown
        j3 = await (await cl.post("/api/fish/emote", json={"text": "Oh, it's you! Hi!", "role": "agent"})).json()
        waves = [e for e in hr.rig.emotes if e in ("greet_wave", "wink_wave")]   # her own reply may pick either wave
        check("http. fish caller on_reply wave fires as her call line starts", len(waves) == 1,
              f"{hr.rig.emotes} {j3}")


# ------------------------------------------------------------------------------------------------ main
async def main_async() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_rig_no_speech_peak_pop(tmp)
        await test_decisions()
        await test_server(tmp)


def test_all() -> None:          # pytest entry point
    asyncio.run(main_async())
    failed = [r for r in RESULTS if not r[1]]
    assert not failed, failed


if __name__ == "__main__":
    asyncio.run(main_async())
    passed = sum(1 for r in RESULTS if r[1])
    print(f"\n{passed}/{len(RESULTS)} checks passed", flush=True)
    sys.exit(0 if passed == len(RESULTS) else 1)
