"""Idle Motion Graph: pose-matched phrase graph over prepared Motion Lab idle clips.

After *Motion-Based Graph Retrieval* (Song et al. 2025) and *MECo* (2025): instead of looping one idle clip in a fixed
linear order, the clip(s) are cut into 3-6 s motion phrases at low-motion "energy valleys", every phrase is classified
(energy + style tag + start / end pose & velocity), and a directed graph links each phrase to its K best pose- and
velocity-matched successors. ``MotionGraphWalker`` walks that graph forever: it never plays the same phrase twice in a
row, leans toward calmer phrases when she is chill and bigger rocks / bounces when she is upbeat, and joins phrases with
a 0.55 s cosine crossfade (outgoing tail -> incoming head), so there is never a seam pop.

Input clips are the dicts returned by ``core.rig_controller.prepare_idle_clip`` ({"fps", "frames", "tracks"}, zero-mean
offsets). Universal: only the IDLE head / body / stage tracks are used, so it works for Hikari, Hiyori, Mao or any rig.
The build is numpy; the per-frame ``step`` is pure Python on tiny lists (a few microseconds).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Same tracks as core.rig_controller.IDLE_TRACKS (duplicated to avoid a circular import).
GRAPH_TRACKS = ("AngleX", "AngleY", "AngleZ", "BodyX", "BodyZ", "StageX", "StageY", "StageZoom")
PHRASE_MIN_SEC = 3.0
PHRASE_MAX_SEC = 6.0
DEFAULT_PHRASE_SEC = 4.5
GRAPH_XFADE_SEC = 0.55
GRAPH_TOP_K = 6
MIN_GRAPH_CLIP_SEC = 6.0           # shorter clips play as one smooth self-crossfaded loop
MAX_GRAPH_PHRASES = 512            # bounds all-pairs linking; larger libraries retain linear playback
MIN_WALK_PHRASES = 4               # fewer phrases add no variety: callers keep the seamless linear loop
STYLES = ("calm_sway", "big_rock", "curious_tilt", "deep_duck", "bouncy", "lean_in")
_VEL_W = 0.35                      # velocity term weight in the transition cost
_STYLE_ENERGY_BIAS = {"calm_sway": -0.10, "curious_tilt": -0.05, "deep_duck": 0.0, "lean_in": 0.0,
                      "big_rock": 0.12, "bouncy": 0.15}


@dataclass
class MotionPhrase:
    clip: int
    start: int                     # first frame in the clip
    frames: int
    fps: float
    rows: list[list[float]]        # frames x len(tracks) samples (python floats: fast per-frame lerp)
    energy: float = 0.0
    style: str = "calm_sway"
    start_pose: list[float] = field(default_factory=list)
    start_vel: list[float] = field(default_factory=list)
    end_pose: list[float] = field(default_factory=list)
    end_vel: list[float] = field(default_factory=list)
    raw_energy: float = 0.0
    edges: list[tuple[int, float]] = field(default_factory=list)   # (phrase idx, transition cost) ascending

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def sample(self, t: float) -> list[float]:
        f = t * self.fps
        n = self.frames
        if f <= 0.0:
            return self.rows[0]
        if f >= n - 1:
            return self.rows[n - 1]
        i0 = int(f)
        a = f - i0
        r0 = self.rows[i0]
        r1 = self.rows[i0 + 1]
        return [x + (y - x) * a for x, y in zip(r0, r1)]


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x
    k = np.ones(win) / win
    return np.convolve(np.pad(x, (win // 2, win - 1 - win // 2), mode="edge"), k, mode="valid")


class MotionGraph:
    """Phrase graph over one or more prepared idle clips."""

    def __init__(self, clips: dict[str, Any] | Iterable[dict[str, Any]] | None = None, *,
                 phrase_sec: float = DEFAULT_PHRASE_SEC, xfade_sec: float = GRAPH_XFADE_SEC,
                 top_k: int = GRAPH_TOP_K) -> None:
        self.phrase_sec = max(PHRASE_MIN_SEC, min(PHRASE_MAX_SEC, float(phrase_sec)))
        self.xfade_sec = max(0.1, float(xfade_sec))
        self.top_k = max(1, int(top_k))
        self.tracks: tuple[str, ...] = ()
        self.clips: list[dict[str, Any]] = []
        self.phrases: list[MotionPhrase] = []
        self._scale: np.ndarray | None = None
        if clips is not None:
            items = [clips] if isinstance(clips, dict) else list(clips)
            for c in items:
                self._add(c)
            self.build()

    # ------------------------------------------------------------------ construction
    def _add(self, clip: dict[str, Any]) -> None:
        tracks = clip.get("tracks")
        if not isinstance(tracks, dict) or not any(k in tracks for k in GRAPH_TRACKS):
            raise ValueError("motion graph clip has no idle head / body tracks")
        self.clips.append(clip)

    def add_clip(self, clip: dict[str, Any]) -> "MotionGraph":
        self._add(clip)
        self.build()
        return self

    @classmethod
    def from_files(cls, paths: Iterable[str | Path], **kw: Any) -> "MotionGraph":
        from core.rig_controller import prepare_idle_clip, validate_motion_clip   # lazy: avoids circular import
        clips = [prepare_idle_clip(validate_motion_clip(json.loads(Path(p).read_text(encoding="utf-8"))))
                 for p in paths]
        return cls(clips, **kw)

    def _matrix(self, clip: dict[str, Any]) -> np.ndarray:
        tr = clip["tracks"]
        n = int(clip.get("frames") or len(next(iter(tr.values()))))
        cols = [np.asarray(tr[k][:n], dtype=np.float64) if k in tr else np.zeros(n) for k in self.tracks]
        return np.stack(cols, axis=1)

    def build(self) -> None:
        present = [k for k in GRAPH_TRACKS if any(k in c["tracks"] for c in self.clips)]
        self.tracks = tuple(present)
        self.phrases = []
        if not self.clips or not self.tracks:
            return
        # Two segmentation passes each produce at most one cut per minimum phrase.
        possible = sum(2 * max(1, int(c["frames"]) // max(1, int(PHRASE_MIN_SEC * c["fps"]))) for c in self.clips)
        if possible > MAX_GRAPH_PHRASES:
            raise ValueError("Clip exceeds the motion graph phrase budget")
        mats = [self._matrix(c) for c in self.clips]
        allm = np.concatenate(mats, axis=0)
        self._scale = np.maximum(allm.std(axis=0), 1e-6)
        for ci, (clip, m) in enumerate(zip(self.clips, mats)):
            fps = float(clip.get("fps", 60.0)) or 60.0
            for s, e in self._segment(m, fps):
                self.phrases.append(self._make_phrase(ci, m, s, e, fps))
        self._classify()
        self._link()

    def _segment(self, m: np.ndarray, fps: float) -> list[tuple[int, int]]:
        n = len(m)
        dur = n / fps
        if dur < MIN_GRAPH_CLIP_SEC:
            return [(0, n)]                                         # too short: one phrase, smooth self-loop
        length = min(self.phrase_sec, dur / 2.0)
        lmin = int(PHRASE_MIN_SEC * fps)
        lmax = int(PHRASE_MAX_SEC * fps)
        ltgt = int(length * fps)
        z = m / self._scale
        vel = np.diff(z, axis=0, prepend=z[:1]) * fps
        speed = _smooth(np.sqrt((vel * vel).sum(axis=1)), max(1, int(0.3 * fps)))
        spread = float(speed.max() - speed.min()) + 1e-9
        cuts: list[tuple[int, int]] = []
        for offset in (0, ltgt // 2):                               # two passes -> overlapping phrase sets
            s = offset
            while s + lmin <= n:
                lo, hi = s + lmin, min(n, s + lmax)
                if hi <= lo:
                    break
                if n - s <= lmax and (n - s) >= lmin and n - s - ltgt < lmin:
                    e = n                                           # the remainder fits: take it whole
                else:
                    idx = np.arange(lo, hi)
                    # energy-valley cut near the target length
                    score = (speed[np.minimum(idx, n - 1)] - speed.min()) / spread + 0.6 * np.abs(idx - (s + ltgt)) / max(1, ltgt)
                    e = int(idx[int(np.argmin(score))])
                cuts.append((s, e))
                s = e
        cuts.sort()
        out: list[tuple[int, int]] = []
        for s, e in cuts:                                           # drop near-duplicate phrases
            if any(abs(s - s2) < 0.5 * fps and abs(e - e2) < 0.5 * fps for s2, e2 in out):
                continue
            out.append((s, e))
        return out or [(0, n)]

    def _make_phrase(self, ci: int, m: np.ndarray, s: int, e: int, fps: float) -> MotionPhrase:
        seg = m[s:e]
        z = seg / self._scale
        n = len(seg)
        x2 = min(n - 1, max(1, min(n // 4, int(self.xfade_sec * fps * 0.5))))   # pose sampled mid-crossfade window
        vw = max(1, int(0.1 * fps))

        def vel_at(i: int) -> np.ndarray:
            a, b = max(0, i - vw), min(n - 1, i + vw)
            return (z[b] - z[a]) * fps / max(1, b - a)

        vel = np.diff(z, axis=0) * fps if n > 1 else np.zeros((1, z.shape[1]))
        raw = float(np.sqrt((vel * vel).sum(axis=1).mean())) + 0.5 * float(z.std(axis=0).mean())
        return MotionPhrase(
            clip=ci, start=s, frames=n, fps=fps, rows=seg.tolist(), raw_energy=raw,
            start_pose=z[x2].tolist(), start_vel=vel_at(x2).tolist(),
            end_pose=z[n - 1 - x2].tolist(), end_vel=vel_at(n - 1 - x2).tolist())

    def _classify(self) -> None:
        if not self.phrases:
            return
        raws = [p.raw_energy for p in self.phrases]
        lo, hi = min(raws), max(raws)
        ti = {k: i for i, k in enumerate(self.tracks)}
        for p in self.phrases:
            p.energy = 0.5 if hi - lo < 1e-9 else (p.raw_energy - lo) / (hi - lo)
            z = np.asarray(p.rows) / self._scale
            mean = z.mean(axis=0)
            std = z.std(axis=0)

            def mz(k: str) -> float:
                return float(mean[ti[k]]) if k in ti else 0.0

            def sz(k: str) -> float:
                return float(std[ti[k]]) if k in ti else 0.0

            bounce_hz = 0.0
            for k in ("StageY", "AngleY"):
                if k in ti and len(z) > 4:
                    v = np.diff(z[:, ti[k]])
                    flips = int(np.sum(np.signbit(v[1:]) != np.signbit(v[:-1])))
                    bounce_hz = max(bounce_hz, 0.5 * flips / p.duration)
            rock = max(sz("BodyZ"), sz("AngleZ"), sz("StageX"))
            if mz("StageZoom") > 0.6:
                style = "lean_in"
            elif min(mz("AngleY"), mz("StageY")) < -0.6:
                style = "deep_duck"
            elif bounce_hz > 1.1 and p.energy > 0.35:
                style = "bouncy"
            elif p.energy > 0.62 or rock > 1.25:
                style = "big_rock"
            elif abs(mz("AngleZ")) > 0.55:
                style = "curious_tilt"
            else:
                style = "calm_sway"
            p.style = style

    def _link(self) -> None:
        ph = self.phrases
        n = len(ph)
        if n < 2:
            if n == 1:
                ph[0].edges = [(0, 0.0)]                            # single phrase: smooth self-crossfade loop
            return
        ep = np.asarray([p.end_pose for p in ph])
        ev = np.asarray([p.end_vel for p in ph])
        sp = np.asarray([p.start_pose for p in ph])
        sv = np.asarray([p.start_vel for p in ph])
        dp = ((ep[:, None, :] - sp[None, :, :]) ** 2).sum(axis=2)
        dv = ((ev[:, None, :] - sv[None, :, :]) ** 2).sum(axis=2)
        cost = dp + _VEL_W * dv
        for i, p in enumerate(ph):
            for j, q in enumerate(ph):
                if i == j:
                    cost[i, j] = np.inf
                elif p.clip == q.clip:
                    ov = min(p.start + p.frames, q.start + q.frames) - max(p.start, q.start)
                    if ov > 0:                                      # overlapping copy of the same motion
                        cost[i, j] += 2.0 * ov / min(p.frames, q.frames)
                    if q.start == p.start + p.frames:               # the plain linear successor: allowed, mildly
                        cost[i, j] += 0.15
            order = np.argsort(cost[i])
            k = min(self.top_k, n - 1)
            p.edges = [(int(j), float(cost[i, j])) for j in order[:k] if np.isfinite(cost[i, j])]

    # ------------------------------------------------------------------ queries
    @property
    def transitions(self) -> int:
        return sum(len(p.edges) for p in self.phrases)

    @property
    def duration(self) -> float:
        return sum(float(c.get("frames", 0)) / (float(c.get("fps", 60.0)) or 60.0) for c in self.clips)

    def style_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.phrases:
            out[p.style] = out.get(p.style, 0) + 1
        return out

    def status(self) -> dict[str, Any]:
        return {"phraseCount": len(self.phrases), "clipCount": len(self.clips), "transitions": self.transitions,
                "styles": self.style_counts(), "phraseSec": self.phrase_sec, "xfadeSec": self.xfade_sec,
                "tracks": list(self.tracks)}

    def walker(self, rng: random.Random | None = None) -> "MotionGraphWalker":
        return MotionGraphWalker(self, rng=rng)


class MotionGraphWalker:
    """Continuously walks a MotionGraph; ``step`` returns {track: offset} like the linear idle sampler."""

    def __init__(self, graph: MotionGraph, rng: random.Random | None = None) -> None:
        if not graph.phrases:
            raise ValueError("motion graph has no phrases")
        self.graph = graph
        self.rng = rng or random.Random()
        self.cur = self._initial(0.5)
        self.nxt: int | None = None
        self.t = 0.0
        self.history: list[int] = [self.cur]
        self.transitions_taken = 0
        self.visits: dict[int, int] = {self.cur: 1}
        self._zero = {k: 0.0 for k in GRAPH_TRACKS}

    def _target_energy(self, arousal: float, speaking: bool) -> float:
        a = max(0.0, min(1.0, float(arousal)))
        tgt = 0.12 + 0.78 * a
        return tgt * 0.85 if speaking else tgt

    def _initial(self, arousal: float) -> int:
        tgt = self._target_energy(arousal, False)
        ph = self.graph.phrases
        best = sorted(range(len(ph)), key=lambda i: abs(ph[i].energy - tgt))[:max(1, min(4, len(ph)))]
        return self.rng.choice(best)

    def _choose(self, arousal: float, speaking: bool, rng: random.Random) -> int:
        ph = self.graph.phrases
        edges = ph[self.cur].edges
        if not edges:
            return self.cur
        if len(ph) == 1:
            return 0
        tgt = self._target_energy(arousal, speaking)
        c0 = edges[0][1]
        spread = max(1e-6, edges[-1][1] - c0) if len(edges) > 1 else 1.0
        # Arousal steering needs a real phrase library: a tiny graph (short clip) covers its phrases evenly instead,
        # so it never loses the clip's motion amplitude by camping on its calmest phrase.
        e_w = 2.2 * min(1.0, max(0.0, (len(ph) - 2) / 6.0))
        recent = self.history[-8:]
        best_j, best_s = edges[0][0], math.inf
        for j, c in edges:
            if j == self.cur:
                continue
            q = ph[j]
            e_eff = q.energy + _STYLE_ENERGY_BIAS.get(q.style, 0.0)
            s = 0.9 * (c - c0) / spread + e_w * abs(e_eff - tgt)
            for age, h in enumerate(reversed(recent)):              # recency: avoid short fixed cycles
                if h == j:
                    s += 1.4 * (0.8 ** age)
            s += 0.4 * self.visits.get(j, 0) / (1.0 + self.transitions_taken / max(1, len(ph)))
            if speaking and q.style == "deep_duck":
                s += 0.6
            s += rng.random() * 0.55                                 # Gumbel-ish jitter: never a fixed order
            if s < best_s:
                best_j, best_s = j, s
        return best_j

    def step(self, dt: float, *, arousal: float = 0.0, speaking: bool = False,
             rng: random.Random | None = None) -> dict[str, float]:
        g = self.graph
        ph = g.phrases
        rng = rng or self.rng
        dt = max(0.0, min(0.25, float(dt)))
        self.t += dt
        xf = g.xfade_sec
        for _ in range(4):                                          # catch up after a long frame
            cur = ph[self.cur]
            xf_i = min(xf, cur.duration * 0.4)
            if self.t < cur.duration - xf_i:
                break
            if self.nxt is None:
                self.nxt = self._choose(arousal, speaking, rng)
            if self.t < cur.duration:
                break
            # crossfade finished: incoming phrase continues from where its head blend stopped
            self.t = self.t - (cur.duration - xf_i)
            self.cur = self.nxt
            self.nxt = None
            self.history.append(self.cur)
            self.visits[self.cur] = self.visits.get(self.cur, 0) + 1
            if len(self.history) > 16:
                del self.history[0]
            self.transitions_taken += 1
        cur = ph[self.cur]
        xf_i = min(xf, cur.duration * 0.4)
        row = cur.sample(self.t)
        t0 = cur.duration - xf_i
        if self.nxt is not None and self.t > t0:
            u = (self.t - t0) / xf_i
            w = 0.5 - 0.5 * math.cos(math.pi * min(1.0, u))
            inc = ph[self.nxt].sample(self.t - t0)
            row = [a + (b - a) * w for a, b in zip(row, inc)]
        out = dict(self._zero) if len(g.tracks) < len(GRAPH_TRACKS) else {}
        for k, v in zip(g.tracks, row):
            out[k] = v
        return out

    @property
    def active(self) -> MotionPhrase:
        return self.graph.phrases[self.cur]

    def status(self) -> dict[str, Any]:
        p = self.active
        return {**self.graph.status(), "activePhrase": self.cur, "activePhraseStyle": p.style,
                "activePhraseEnergy": round(p.energy, 3), "activePhraseSec": round(p.duration, 2),
                "phraseTime": round(self.t, 2), "transitionsTaken": self.transitions_taken,
                "nextPhrase": self.nxt}
