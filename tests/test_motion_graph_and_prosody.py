"""Verification: idle Motion Graph + causal prosody-to-body + anticipatory visual backchannel.

  1. modules/motion_graph.py: phrases (3-6 s) + classified styles + pose-matched edges from synthetic clips and the real
     motions/Neuro_Idle.json (if present); the walker never repeats a phrase back-to-back, never plays in fixed linear
     order, has no seam pops, adapts to arousal, costs < 0.1 ms / step, and degrades cleanly on short clips
  2. modules/lipsync.py: CausalProsody detects a rising 200 -> 340 Hz chirp (pitchSlope > 0), falling pitch (< 0),
     stressed syllable bursts (stressOnset > 0.3), sustained emphasis; silence = exact 0.0; strictly causal (no
     lookahead on playback); < 1 ms / frame; the 5 fields ride in LiveLipSyncEngine.step()/sample(); user-voice pulses
  3. core/rig_controller.py: prosody micro-nods / brow lift / lean-in while speaking, visual backchannel nods on user
     speech -> pause while listening (cooled down), and the idle clip is walked through the MotionGraph when
     rig.idleMotionGraph = True (classic linear loop when False, crossfaded toggle)

    python tests/test_motion_graph_and_prosody.py      (or: python -m pytest tests/test_motion_graph_and_prosody.py)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import time as _real_time
from typing import Any
import wave

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import core.rig_controller as rig_mod  # noqa: E402
from core.config import default_config, sanitize_section  # noqa: E402
from core.events import HookRegistry  # noqa: E402
from core.rig_controller import RIG_FX_PARAMS, VFX_PARAMS, RigController, prepare_idle_clip, validate_motion_clip  # noqa: E402
from modules.lipsync import (PROSODY_FIELDS, PROSODY_WINDOW_SEC, CausalProsody, LiveLipSyncEngine,  # noqa: E402
                             UserVoiceActivity)
from modules.motion_graph import (GRAPH_TRACKS, PHRASE_MAX_SEC, PHRASE_MIN_SEC, STYLES, MotionGraph,  # noqa: E402
                                  MotionGraphWalker)

RESULTS: list[tuple[str, bool, str]] = []
DT = 1.0 / 60.0
SR = 48_000


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""), flush=True)


class VirtualClock:
    def __init__(self) -> None:
        self.now = _real_time.perf_counter()

    def perf_counter(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


CLOCK = VirtualClock()


# ------------------------------------------------------------------------------------------------ synthetic clips
def synth_clip(seconds: float = 40.0, cid: str = "synth_graph", seed: int = 3) -> dict[str, Any]:
    """A 60 FPS clip made of 5 s 'moods' (calm sway / big rock / bouncy / duck / lean / tilt), smoothly blended."""
    rng = np.random.default_rng(seed)
    n = int(seconds * 60)
    t = np.arange(n) / 60.0
    moods = ["calm", "rock", "bounce", "duck", "lean", "tilt"]
    seg = 5.0
    tr = {k: np.zeros(n) for k in GRAPH_TRACKS}
    for i in range(int(math.ceil(seconds / seg))):
        mood = moods[i % len(moods)] if i < len(moods) else moods[int(rng.integers(len(moods)))]
        c = (i + 0.5) * seg
        env = np.exp(-0.5 * ((t - c) / (seg * 0.38)) ** 2)            # smooth bump: continuous clip
        ph = rng.uniform(0, 6.28)
        if mood == "calm":
            tr["AngleZ"] += env * 3.0 * np.sin(2 * np.pi * t / 4.2 + ph)
            tr["BodyZ"] += env * 1.0 * np.sin(2 * np.pi * t / 4.2 + ph)
        elif mood == "rock":
            tr["AngleZ"] += env * 14.0 * np.sin(2 * np.pi * t / 2.4 + ph)
            tr["BodyZ"] += env * 6.0 * np.sin(2 * np.pi * t / 2.4 + ph - 0.3)
            tr["StageX"] += env * 0.3 * np.sin(2 * np.pi * t / 2.4 + ph)
        elif mood == "bounce":
            tr["StageY"] += env * 0.25 * np.sin(2 * np.pi * t * 2.2 + ph)
            tr["AngleY"] += env * 4.0 * np.sin(2 * np.pi * t * 2.2 + ph)
        elif mood == "duck":
            tr["AngleY"] += env * -12.0
            tr["StageY"] += env * -0.3
        elif mood == "lean":
            tr["StageZoom"] += env * 0.5
            tr["AngleX"] += env * 2.0 * np.sin(2 * np.pi * t / 3.0 + ph)
        else:
            tr["AngleZ"] += env * 12.0
            tr["AngleX"] += env * 3.0 * np.sin(2 * np.pi * t / 3.3 + ph)
    tracks = {k: v.tolist() for k, v in tr.items()}
    tracks.update({"EyeOpen": [1.0] * n, "ArmR": [0.0] * n, "ArmL": [0.0] * n})
    return {"id": cid, "name": cid, "fps": 60, "frames": n, "duration": seconds, "tracks": tracks}


def prepared(seconds: float, cid: str = "synth_graph", seed: int = 3) -> dict[str, Any]:
    return prepare_idle_clip(validate_motion_clip(synth_clip(seconds, cid, seed)))


def z_steps(graph: MotionGraph, frames: list[dict[str, float]]) -> list[float]:
    sc = {k: float(graph._scale[i]) for i, k in enumerate(graph.tracks)}
    return [max(abs(b[k] - a[k]) / sc[k] for k in graph.tracks) for a, b in zip(frames, frames[1:])]


def walk(w: MotionGraphWalker, seconds: float, arousal: float = 0.5, speaking: bool = False
         ) -> tuple[list[dict[str, float]], list[int], float]:
    frames, seq = [], [w.cur]
    t0 = _real_time.perf_counter()
    for _ in range(int(seconds * 60)):
        frames.append(w.step(DT, arousal=arousal, speaking=speaking))
        if w.cur != seq[-1]:
            seq.append(w.cur)
    us = (_real_time.perf_counter() - t0) / max(1, len(frames)) * 1e6
    return frames, seq, us


# ------------------------------------------------------------------------------------------------ 1. motion graph
def test_motion_graph() -> None:
    clip = prepared(40.0)
    g = MotionGraph(clip)
    st = g.status()
    check("graph. builds >= 8 phrases from a 40 s clip", st["phraseCount"] >= 8, str(st))
    durs = [p.duration for p in g.phrases]
    check("graph. every phrase is 3.0-6.0 s", all(PHRASE_MIN_SEC - 0.02 <= d <= PHRASE_MAX_SEC + 0.02 for d in durs),
          f"{min(durs):.2f}..{max(durs):.2f}")
    check("graph. every phrase has outgoing edges, never to itself",
          all(p.edges and all(j != i for j, _ in p.edges) for i, p in enumerate(g.phrases)))
    check("graph. edges are sorted lowest transition cost first",
          all(all(a[1] <= b[1] for a, b in zip(p.edges, p.edges[1:])) for p in g.phrases))
    check("graph. styles are valid tags", all(p.style in STYLES for p in g.phrases))
    styles = set(p.style for p in g.phrases)
    check("graph. mixed moods classify into >= 3 distinct styles", len(styles) >= 3, str(g.style_counts()))
    check("graph. energy normalized 0..1 (min 0, max 1)",
          min(p.energy for p in g.phrases) == 0.0 and abs(max(p.energy for p in g.phrases) - 1.0) < 1e-9)
    check("graph. phrases carry start / end pose + velocity vectors",
          all(len(p.start_pose) == len(g.tracks) == len(p.end_vel) for p in g.phrases))
    check("graph. status exposes phraseCount / clipCount / transitions",
          {"phraseCount", "clipCount", "transitions"} <= set(st) and st["clipCount"] == 1 and st["transitions"] > 0)

    w = g.walker(random.Random(7))
    frames, seq, us = walk(w, 300.0)
    check("walker. never repeats the same phrase back-to-back", all(a != b for a, b in zip(seq, seq[1:])),
          str(seq[:30]))
    ph = g.phrases
    linear = sum(1 for a, b in zip(seq, seq[1:]) if ph[b].clip == ph[a].clip and ph[b].start == ph[a].start + ph[a].frames)
    check("walker. does not just play the clip in linear order", linear < 0.6 * (len(seq) - 1),
          f"{linear}/{len(seq) - 1} linear hops")
    check("walker. explores the graph (>= 50 % of phrases visited in 5 min)", len(set(seq)) >= 0.5 * len(ph),
          f"{len(set(seq))}/{len(ph)}")
    in_phrase = max(float(np.abs(np.diff(np.asarray(p.rows) / g._scale, axis=0)).max()) for p in ph)
    steps = z_steps(g, frames)
    check("walker. zero seam pops (max frame step across transitions <= 1.5x the clip's own max step)",
          max(steps) <= in_phrase * 1.5 + 1e-6, f"walk {max(steps):.4f} vs clip {in_phrase:.4f}")
    check("walker. < 0.1 ms per step", us < 100.0, f"{us:.1f} us")
    w_status = w.status()
    check("walker. status has activePhraseStyle / activePhraseEnergy",
          w_status["activePhraseStyle"] in STYLES and 0.0 <= w_status["activePhraseEnergy"] <= 1.0, str(w_status))
    check("walker. outputs every idle track", set(frames[0]) >= set(GRAPH_TRACKS))

    def mean_energy(arousal: float) -> float:
        wk = g.walker(random.Random(11))
        _, sq, _ = walk(wk, 400.0, arousal=arousal)
        return sum(ph[i].energy for i in sq[1:]) / max(1, len(sq) - 1)

    calm, hype = mean_energy(0.05), mean_energy(0.95)
    check("walker. adapts to arousal (upbeat picks bigger phrases than chill)", hype > calm + 0.08,
          f"chill {calm:.3f} upbeat {hype:.3f}")

    g2 = MotionGraph([prepared(20.0, "a", 5), prepared(16.0, "b", 9)])
    check("graph. multiple clips -> one graph", g2.status()["clipCount"] == 2 and
          len(set(p.clip for p in g2.phrases)) == 2, str(g2.status()))
    cross = sum(1 for p in g2.phrases for j, _ in p.edges if g2.phrases[j].clip != p.clip)
    check("graph. edges connect phrases across clips", cross > 0, str(cross))

    short = MotionGraph(prepared(8.0, "short8"))
    check("graph. 8 s clip (7 s loop) degrades to >= 2 phrases", short.status()["phraseCount"] >= 2,
          str(short.status()))
    sw = short.walker(random.Random(1))
    fr, sq, _ = walk(sw, 30.0)
    check("graph. short clip still walks without back-to-back repeats", all(a != b for a, b in zip(sq, sq[1:])))
    tiny = MotionGraph(prepared(4.0, "tiny4"))
    check("graph. very short clip -> one phrase smooth self-loop", tiny.status()["phraseCount"] == 1)
    tw = tiny.walker(random.Random(1))
    fr, _, _ = walk(tw, 12.0)
    ip = float(np.abs(np.diff(np.asarray(tiny.phrases[0].rows) / tiny._scale, axis=0)).max())
    check("graph. tiny self-loop has no seam pop", max(z_steps(tiny, fr)) <= ip * 2.0 + 1e-6,
          f"{max(z_steps(tiny, fr)):.4f} vs {ip:.4f}")

    real = HERE / "motions" / "Neuro_Idle.json"
    if real.exists():
        gr = MotionGraph.from_files([real])
        rs = gr.status()
        check("real. Neuro_Idle.json -> >= 20 phrases, >= 3 styles", rs["phraseCount"] >= 20 and len(rs["styles"]) >= 3,
              str(rs))
        wr = gr.walker(random.Random(5))
        fr, sq, us = walk(wr, 240.0, arousal=0.6)
        ipr = max(float(np.abs(np.diff(np.asarray(p.rows) / gr._scale, axis=0)).max()) for p in gr.phrases)
        check("real. walk has no back-to-back repeats and no seam pops",
              all(a != b for a, b in zip(sq, sq[1:])) and max(z_steps(gr, fr)) <= ipr * 1.5 + 1e-6,
              f"{max(z_steps(gr, fr)):.4f} vs {ipr:.4f}")
        check("real. < 0.1 ms per step", us < 100.0, f"{us:.1f} us")
    else:
        print("[SKIP] real Neuro_Idle.json not present")


# ------------------------------------------------------------------------------------------------ 2. prosody
def tone(f0_fn, dur: float, amp_fn=lambda t: 0.3) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f0_fn(t)) / SR
    sig = sum((0.6 ** k) * np.sin((k + 1) * ph) for k in range(4))
    return (sig * amp_fn(t)).astype(np.float32)


def run_prosody(sig: np.ndarray) -> tuple[list[dict[str, float]], list[float]]:
    pr = CausalProsody()
    hop, win = SR // 60, int(PROSODY_WINDOW_SEC * SR)
    out, ms = [], []
    for i in range(hop, len(sig) + 1, hop):
        t0 = _real_time.perf_counter()
        out.append(pr.update(sig[max(0, i - win):i], SR, DT))
        ms.append((_real_time.perf_counter() - t0) * 1000.0)
    return out, ms


def test_prosody() -> None:
    flat = tone(lambda t: np.full_like(t, 220.0), 1.0)
    chirp = tone(lambda t: 200.0 + 140.0 * t / 1.2, 1.2)
    o, ms = run_prosody(np.concatenate([flat, chirp]))
    check("prosody. 200 -> 340 Hz chirp yields positive pitchSlope", max(x["pitchSlope"] for x in o[60:]) > 0.3,
          f"{max(x['pitchSlope'] for x in o[60:]):.3f}")
    check("prosody. ...and pitchNorm ends above the speaker median", o[-1]["pitchNorm"] > 0.3, f"{o[-1]['pitchNorm']:.3f}")
    ms_sorted = sorted(ms[5:])
    check("prosody. < 1 ms / frame (mean < 0.5 ms)", ms_sorted[len(ms_sorted) // 2] < 1.0 and
          sum(ms_sorted) / len(ms_sorted) < 0.5, f"median {ms_sorted[len(ms_sorted) // 2]:.3f} ms")
    fall = tone(lambda t: 340.0 - 140.0 * t / 1.2, 1.2)
    o, _ = run_prosody(np.concatenate([tone(lambda t: np.full_like(t, 300.0), 1.0), fall]))
    check("prosody. falling statement pitch yields negative pitchSlope", min(x["pitchSlope"] for x in o[60:]) < -0.3)
    o, _ = run_prosody(tone(lambda t: np.full_like(t, 240.0), 2.0))
    check("prosody. steady pitch -> |pitchSlope| small", max(abs(x["pitchSlope"]) for x in o[20:]) < 0.15,
          f"{max(abs(x['pitchSlope']) for x in o[20:]):.3f}")
    bursts = tone(lambda t: np.full_like(t, 230.0), 2.5, lambda t: 0.35 * ((t % 0.25) < 0.12))
    o, _ = run_prosody(bursts)
    check("prosody. syllable stress bursts -> stressOnset > 0.3", max(x["stressOnset"] for x in o) > 0.3)
    check("prosody. 4 syllables / s -> speechRate > 0.35", o[-1]["speechRate"] > 0.35, f"{o[-1]['speechRate']:.3f}")
    slow = tone(lambda t: np.full_like(t, 230.0), 2.5, lambda t: 0.35 * ((t % 0.8) < 0.4))
    o2, _ = run_prosody(slow)
    check("prosody. slow speech has lower speechRate than fast chatter", o2[-1]["speechRate"] < o[-1]["speechRate"])
    loud = tone(lambda t: 250 + 40 * np.sin(2 * np.pi * 1.5 * t), 2.0, lambda t: 0.6)
    o, _ = run_prosody(loud)
    check("prosody. sustained loud expressive phrase -> emphasis > 0.5", o[-1]["emphasis"] > 0.5, f"{o[-1]['emphasis']:.3f}")
    o, _ = run_prosody(np.zeros(SR, np.float32))
    check("prosody. silent audio returns exact 0.0 for all 5 fields", all(v == 0.0 for x in o for v in x.values()))
    o, _ = run_prosody(np.concatenate([loud, np.zeros(SR * 2, np.float32)]))
    check("prosody. returns to exact 0.0 after speech ends", all(v == 0.0 for v in o[-1].values()), str(o[-1]))

    # --- engine integration: Fish duplex PCM + WAV playback (causal) + backwards compatible dict
    eng = LiveLipSyncEngine()
    d = eng.step(DT, 0.0)
    old_keys = {"open", "form", "funnel", "x", "shrug", "tongue", "press", "energy", "stress", "onset", "pitch",
                "voiced", "brightness"}
    check("engine. step() keeps every old key + the 5 prosody fields", old_keys | set(PROSODY_FIELDS) <= set(d))
    check("engine. silent -> prosody fields 0.0", all(d[k] == 0.0 for k in PROSODY_FIELDS))
    sig = np.concatenate([flat, chirp])
    hop = SR // 60
    best = 0.0
    for i in range(0, len(sig) - hop, hop):
        eng.ingest_fish_pcm((np.clip(sig[i:i + hop], -1, 1) * 32767).astype("<i2").tobytes())
        d = eng.step(DT, 0.0)
        best = max(best, d["pitchSlope"])
    check("engine. Fish duplex PCM frames feed prosody (chirp -> pitchSlope > 0.2)", best > 0.2, f"{best:.3f}")
    check("engine. sample() mirrors the last frame", eng.sample()["pitchSlope"] == d["pitchSlope"])
    check("engine. status() reports prosody", "prosody" in eng.status() and "cpuMs" in eng.status()["prosody"])

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "onset.wav"
        pcm = np.concatenate([np.zeros(SR, np.float32), tone(lambda t: np.full_like(t, 230.0), 0.5, lambda t: 0.5)])
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes((pcm * 32767).astype("<i2").tobytes())
        eng2 = LiveLipSyncEngine()
        eng2.load_wav_and_play(wav)
        pre = []
        for k in range(60):                               # frames up to 8 ms BEFORE the onset at 1.000 s
            eng2.sync_playback_time(0.992 * (k + 1) / 60)
            pre.append(eng2.step(DT, 0.0))
        check("engine. playback prosody is strictly causal (nothing before the onset, even 8 ms early)",
              all(f[k] == 0.0 for f in pre for k in PROSODY_FIELDS))
        eng2.sync_playback_time(1.03)
        post = eng2.step(DT, 0.0)
        check("engine. ...and reacts right after it (stressOnset > 0.3)", post["stressOnset"] > 0.3,
              f"{post['stressOnset']:.3f}")

    uv = UserVoiceActivity()
    t = 0.0
    fired = []
    for _ in range(60):                                   # 1 s of user speech
        t += DT
        fired.append(uv.feed(0.08, t))
    for _ in range(36):                                   # 0.6 s pause
        t += DT
        fired.append(uv.feed(0.0005, t))
    idx = [i for i, f in enumerate(fired) if f]
    check("user VAD. one micro-pause pulse ~0.25-0.4 s into the pause",
          len(idx) == 1 and 0.25 <= (idx[0] - 60) * DT + DT <= 0.45 if idx else False, str(idx))
    uv2 = UserVoiceActivity()
    t = 0.0
    for _ in range(10):                                   # 0.17 s blip: too short to be a turn
        t += DT
        uv2.feed(0.08, t)
    for _ in range(60):
        t += DT
        uv2.feed(0.0, t)
    check("user VAD. a short blip does not pulse", uv2.pulses == 0)
    eng.note_user_speech(True)
    check("engine. feed_user_energy / note_user_speech / user_voice_state hooks", isinstance(eng.feed_user_energy(0.05), bool)
          and eng.user_voice_state()["active"] is True)


# ------------------------------------------------------------------------------------------------ 3. rig
class ProsodyLip:
    """Stands in for LiveLipSyncEngine with settable speech energy + prosody fields."""

    def __init__(self) -> None:
        self.energy = 0.0
        self.fields = {k: 0.0 for k in PROSODY_FIELDS}

    def step(self, dt: float, t_now: float) -> dict[str, float]:
        e = self.energy
        return {"open": 0.4 * (e > 0), "form": 0.1, "funnel": 0.0, "shrug": 0.0, "x": 0.0, "tongue": 0.0, "press": 0.0,
                "energy": e, "stress": 0.2 if e else 0.0, "onset": 0.0, "pitch": 0.5, "voiced": 1.0 if e else 0.0,
                **self.fields}


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


class _SeededRandomModule:
    """rig_mod.random stand-in while constructing: the controller's own Random() becomes seeded, so A/B rigs match."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def Random(self, *a: Any) -> random.Random:          # noqa: N802
        return random.Random(a[0] if a else self.seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(random, name)


def new_rig(seed: int = 20261124) -> tuple[RigController, ProsodyLip, dict[str, Any]]:
    lip = ProsodyLip()
    CLOCK.now = 1000.0                                    # same absolute clock for every A/B rig
    old_random = rig_mod.random
    rig_mod.random = _SeededRandomModule(seed)            # type: ignore[assignment]
    try:
        rig = RigController(rig_param_defaults(), lip, HookRegistry())   # type: ignore[arg-type]
    finally:
        rig_mod.random = old_random
    cfg = default_config()
    cfg["rig"]["autoGaze"] = False                        # fewer random layers: cleaner A/B comparisons
    cfg["rig"]["autoBlink"] = False
    return rig, lip, cfg


def run(rig: RigController, cfg: dict[str, Any], seconds: float, per_frame=None) -> list[dict[str, float]]:
    out = []
    for i in range(int(round(seconds / DT))):
        CLOCK.now += DT
        if per_frame:
            per_frame(i)
        p, _, _ = rig.step(DT, cfg, None)
        out.append({"pitch": p["Angle"][1], "roll": p["Angle Z"][0], "bodyy": p["Body Angle Y"][0],
                    "lean": p["Body Lean"][0], "browL": p["Left Brow Y"][0], "browR": p["Right Brow Y"][0],
                    "armR": p["Arm R Pose"][0], "armL": p["Arm L Pose"][0], "zoom": p["Stage Zoom"][0],
                    "smile": p["Eye L Smile"][0]})
    return out


def mean(v: list[float]) -> float:
    return sum(v) / max(1, len(v))


def test_rig_prosody() -> None:
    def speech(rig_on: bool, fields_fn) -> tuple[list[dict[str, float]], RigController]:
        rig, lip, cfg = new_rig()
        cfg["rig"]["causalProsody"] = rig_on
        lip.energy = 0.5

        def pf(i: int) -> None:
            lip.fields.update(fields_fn(i))
        return run(rig, cfg, 4.0, pf), rig

    stress = lambda i: {"stressOnset": 0.9 if i % 20 < 3 else 0.1}          # stressed syllable every 1/3 s
    on, rig_on = speech(True, stress)
    off, rig_off = speech(False, stress)
    diff = max(abs(a["pitch"] - b["pitch"]) for a, b in zip(on, off))
    check("rig. stressed syllables -> spring micro-nods (head pitch differs from prosody-off)", diff > 1.0, f"{diff:.2f}")
    check("rig. ...counted as prosody nods (off: none)", rig_on._pros_nods >= 8 and rig_off._pros_nods == 0,
          f"{rig_on._pros_nods} vs {rig_off._pros_nods}")
    check("rig. ...with a gentle Body Angle Y dip", max(abs(a["bodyy"] - b["bodyy"]) for a, b in zip(on, off)) > 0.3)

    q = lambda i: {"pitchSlope": 0.65, "pitchNorm": 0.5}
    on, _ = speech(True, q)
    off, _ = speech(False, q)
    lift = mean([f["browL"] + f["browR"] for f in on[60:]]) / 2 - mean([f["browL"] + f["browR"] for f in off[60:]]) / 2
    check("rig. rising question pitch lifts both brows", lift > 0.12, f"{lift:.3f}")
    tilt = mean([abs(a["roll"] - b["roll"]) for a, b in zip(on[60:], off[60:])])
    check("rig. ...plus a slight curious head tilt", 1.0 < tilt < 6.0, f"{tilt:.2f}")

    e = lambda i: {"emphasis": 0.9}
    on, _ = speech(True, e)
    off, _ = speech(False, e)
    lean = mean([f["lean"] for f in on[60:]]) - mean([f["lean"] for f in off[60:]])
    check("rig. sustained emphasis -> forward torso lean", lean > 1.2, f"{lean:.2f}")
    zoom = mean([f["zoom"] for f in on[90:]]) - mean([f["zoom"] for f in off[90:]])
    check("rig. ...and a subtle Stage Zoom lean-in", zoom > 0.02, f"{zoom:.3f}")
    check("rig. prosody never raises a wave arm", max(max(f["armR"], f["armL"]) for f in on) == 0.0)

    rig, lip, cfg = new_rig()
    lip.energy = 0.0
    lip.fields.update({"stressOnset": 0.9, "pitchSlope": 0.8, "emphasis": 0.9})
    run(rig, cfg, 2.0)
    check("rig. no prosody body motion while she is silent", rig._pros_nods == 0 and rig._pros_q < 1e-3)

    rig, lip, cfg = new_rig()
    lip.energy = 0.5
    rig.trigger_emote("curious_lean", duration=3.0)
    run(rig, cfg, 0.5)
    n0 = rig._pros_nods
    run(rig, cfg, 1.5, lambda i: lip.fields.update(stress(i)))
    check("rig. prosody does not fight a playing emote", rig._pros_nods == n0, f"{rig._pros_nods - n0} nods")


def test_rig_backchannel() -> None:
    def session(enabled: bool, state: str = "listening", cycles: int = 1, talk: float = 1.2, pause: float = 1.4,
                manual: bool = False) -> tuple[list[dict[str, float]], RigController, dict]:
        rig, _, cfg = new_rig()
        cfg["rig"]["visualBackchannel"] = enabled
        rig.set_interaction_state(state)
        run(rig, cfg, 1.0)
        frames: list[dict[str, float]] = []
        n_talk, n_pause = int(talk * 60), int(pause * 60)
        for _ in range(cycles):
            if manual:
                rig.note_user_pause()
            frames += run(rig, cfg, talk + pause,
                          lambda i: rig.note_user_audio(0.08 if i < n_talk else 0.0005) if not manual else None)
        return frames, rig, cfg

    on, rig_on, _ = session(True)
    off, rig_off, _ = session(False)
    check("backchannel. user speech -> micro-pause in 'listening' triggers one backchannel",
          rig_on._bc_count == 1 and rig_off._bc_count == 0, f"{rig_on._bc_count} vs {rig_off._bc_count}")
    w0, w1 = 72, 72 + 70                                    # 1.2 s talk, pulse 0.25 s into the pause
    nod = max(abs(a["pitch"] - b["pitch"]) for a, b in zip(on[w0:w1], off[w0:w1]))
    check("backchannel. soft double-nod on the head", nod > 1.5, f"{nod:.2f}")
    lean = max(a["lean"] - b["lean"] for a, b in zip(on[w0:w1], off[w0:w1]))
    brow = max(a["browL"] - b["browL"] for a, b in zip(on[w0:w1], off[w0:w1]))
    smile = max(a["smile"] - b["smile"] for a, b in zip(on[w0:w1], off[w0:w1]))
    check("backchannel. attentive lean-in + brow lift + warm micro-smile", lean > 0.8 and brow > 0.08 and smile > 0.04,
          f"lean {lean:.2f} brow {brow:.3f} smile {smile:.3f}")
    check("backchannel. pulse fires only after the pause starts (not during user speech)",
          all(abs(a["lean"] - b["lean"]) < 1e-9 for a, b in zip(on[:72], off[:72])))

    spam, rig_spam, _ = session(True, cycles=8, talk=0.5, pause=0.45)   # a pause every 0.95 s for ~7.6 s
    allowed = int((8 * 0.95) / 2.2) + 1
    check("backchannel. cooldown 2.2-4.5 s: rapid pauses never spam", 1 <= rig_spam._bc_count <= allowed,
          f"{rig_spam._bc_count} (max {allowed})")
    _, rig_idle, _ = session(True, state="idle")
    check("backchannel. no automatic backchannel outside 'listening'", rig_idle._bc_count == 0)
    _, rig_man, _ = session(True, state="idle", manual=True)
    check("backchannel. note_user_pause() pulse still works outside 'listening'", rig_man._bc_count == 1)
    st = rig_on.backchannel_status()
    check("backchannel. backchannel_status() reports count / cooldown / user VAD",
          st["count"] == 1 and "cooldown" in st and "pulses" in st["user"], str(st))

    rig, _, cfg = new_rig()
    cfg["rig"]["idleSway"] = 0.0
    a = run(rig, cfg, 3.0)
    rig2, _, cfg2 = new_rig()
    cfg2["rig"]["idleSway"] = 0.0
    cfg2["rig"]["visualBackchannel"] = cfg2["rig"]["causalProsody"] = False
    b = run(rig2, cfg2, 3.0)
    check("rig. with no speech / user audio the new layers are exactly inert (idleSway 0)",
          all(x == y for x, y in zip(a, b)))


def test_rig_graph(tmp: Path) -> None:
    clip_path = tmp / "graph_idle.json"
    clip_path.write_text(json.dumps(synth_clip(30.0, "graph_idle")), encoding="utf-8")
    rig_l, _, cfg_l = new_rig()
    rig_l.set_idle_clip(clip_path)
    cfg_l["rig"]["idleMotionGraph"] = False
    lin = [f["roll"] for f in run(rig_l, cfg_l, 40.0)[120:]]
    lin_jump = max(abs(b - a) for a, b in zip(lin, lin[1:]))
    rig, _, cfg = new_rig()
    rig.set_idle_clip(clip_path)
    st = rig.idle_graph_status()
    check("rig graph. set_idle_clip builds the idle motion graph", st["active"] and st["phraseCount"] >= 5, str(st))
    sampled: list[int] = []
    real = rig._idle_sample
    rig._idle_sample = lambda *a: sampled.append(1) or real(*a)     # type: ignore[method-assign]
    frames = run(rig, cfg, 40.0)
    st = rig.idle_graph_status()
    check("rig graph. idleMotionGraph=True walks the graph (no linear sampling)",
          st["transitionsTaken"] >= 5 and not sampled, f"{st['transitionsTaken']} hops, {len(sampled)} linear")
    rolls = [f["roll"] for f in frames[120:]]
    sd = float(np.std(rolls))
    jump = max(abs(b - a) for a, b in zip(rolls, rolls[1:]))
    check("rig graph. the walked idle still moves her clearly", sd > 1.5, f"{sd:.2f}")
    check("rig graph. no seam pops in the rig output (max roll step <= 1.25x the linear loop's)",
          jump <= lin_jump * 1.25 + 0.02, f"graph {jump:.3f} vs linear {lin_jump:.3f}")
    check("rig graph. status exposes activePhraseStyle / energy",
          st["activePhraseStyle"] in STYLES and 0.0 <= st["activePhraseEnergy"] <= 1.0)

    cfg["rig"]["idleMotionGraph"] = False
    hops = st["transitionsTaken"]
    sampled.clear()
    tog = run(rig, cfg, 6.0)
    jump_t = max(abs(b["roll"] - a["roll"]) for a, b in zip(frames[-1:] + tog, tog))
    check("rig graph. idleMotionGraph=False -> classic linear _idle_sample", len(sampled) > 300, str(len(sampled)))
    check("rig graph. ...walker stops after the toggle crossfade",
          rig.idle_graph_status()["transitionsTaken"] - hops <= 1)
    check("rig graph. toggling crossfades (no pop)", jump_t <= lin_jump * 1.25 + 0.02, f"{jump_t:.3f} vs {lin_jump:.3f}")

    extra = tmp / "graph_idle2.json"
    extra.write_text(json.dumps(synth_clip(18.0, "graph_idle2", seed=17)), encoding="utf-8")
    st2 = rig.add_idle_graph_clips([extra])
    check("rig graph. add_idle_graph_clips adds a phrase source", st2["clipCount"] == 2 and
          st2["extraClips"] == ["graph_idle2"], str(st2))
    eight = tmp / "eight_idle.json"
    eight.write_text(json.dumps(synth_clip(8.0, "eight_idle")), encoding="utf-8")
    rig.set_idle_clip(eight)
    check("rig graph. an 8 s clip (too few phrases to vary) keeps the seamless linear loop",
          not rig.idle_graph_status()["active"])
    short = tmp / "short_idle.json"
    short.write_text(json.dumps(synth_clip(5.0, "short_idle")), encoding="utf-8")
    rig.set_idle_clip(short)
    check("rig graph. a clip < 6 s keeps the classic loop (graph inactive)", not rig.idle_graph_status()["active"])
    rig.set_idle_clip(None)
    check("rig graph. set_idle_clip(None) clears the graph", not rig.idle_graph_status()["active"])


def test_config() -> None:
    c = default_config()
    check("config. rig.idleMotionGraph / causalProsody / visualBackchannel default True",
          c["rig"]["idleMotionGraph"] is True and c["rig"]["causalProsody"] is True and c["rig"]["visualBackchannel"] is True)
    check("config. ai.fishVocalTags and vision.enabled default True",
          c["ai"].get("fishVocalTags") is True and c["vision"]["enabled"] is True)
    s = sanitize_section("rig", {"idlePhraseSec": 99, "prosodyGain": -3, "backchannelGain": float("nan"),
                                 "idleMotionGraph": 0})
    check("config. numeric bounds clamp (idlePhraseSec 3..6, gains 0..2.5)",
          s.get("idlePhraseSec") == 6.0 and s.get("prosodyGain") == 0.0 and "backchannelGain" not in s
          and s.get("idleMotionGraph") is False, str(s))


def main() -> int:
    test_motion_graph()
    test_prosody()
    old_time = rig_mod.time
    rig_mod.time = CLOCK                                  # type: ignore[assignment]
    try:
        test_rig_prosody()
        test_rig_backchannel()
        with tempfile.TemporaryDirectory() as td:
            test_rig_graph(Path(td))
    finally:
        rig_mod.time = old_time
    test_config()
    ok = sum(1 for _, c, _ in RESULTS if c)
    print(f"\n{ok}/{len(RESULTS)} checks passed")
    return 0 if ok == len(RESULTS) else 1


def test_all() -> None:          # pytest entry point
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
