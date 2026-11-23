"""60 FPS VTuber Rig State Compositor.
Combines cleanly in strict layered priority:
  1. Living idle: uneven mood-coupled breathing (occasional sigh), held weight-shift postures eased by critically
     damped springs + subtle micro-drift, Neuro-style idle acts, normal / double / half / slow blinks
  2. Interaction state layer (idle | listening | thinking | speaking) + mood momentum (valence, arousal)
  3. Autonomous gaze + mouse look-at (eyes lead, head follows)
  4. Spring-smoothed Expression Preset + asymmetric brow micro-life + cheek blush
  5. Arms: the original seamless resting arms sway with body / breath physics (Arm L/R Swing); the raised wave arms
     (Right Wave / Left Gesture / Both Cheer) come up for emotes, AI actions and real excited peaks in speech
  6. Live 6-axis VBridger lip-sync + syllable-onset head beats + pitch-coupled brows / chin
  7. Motion Lab video-clip retargeting blend (0-100% mix)
  8. Raw user parameter overrides (finite-checked)
  9. Plugin hooks (on_lipsync_frame, on_frame_params, on_shader_uniforms)
Physics-owned parameters (Chest X/Y, Eye Jelly, Mouth Spring, Arm Swing, hair) are left to Inochi2D's SimplePhysics
while engine physics is on: the controller feeds them real body / head / breath / arm-root motion instead.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

from core.config import EXPRESSION_PRESETS, ROOM_AMBIENT_MAP
from core.events import HookRegistry
from modules.lipsync import DEFAULT_PARAM_MAP, LiveLipSyncEngine, Spring2nd

log = logging.getLogger("hikari.rig")


# Tweened emote catalog (smooth cubic envelope + spring impulse coupling). Exposed via GET /api/emotes.
# kick = instant spring impulse on start: (chest Y, chest X, eye jelly).
EMOTE_CATALOG: dict[str, dict[str, Any]] = {
    "excited_hop":   {"label": "Excited Hop",   "duration": 2.4, "category": "joy",      "kick": (11.5, 0.0, 9.0),  "desc": "Bouncy double hop with both arms up and sparkle eyes"},
    "shy_blush":     {"label": "Shy Blush",     "duration": 3.2, "category": "shy",      "kick": (0.0, 0.0, 0.0),   "desc": "Looks away and down, blush lines flare"},
    "wink_wave":     {"label": "Wink Wave",     "duration": 2.8, "category": "greeting", "kick": (0.0, 0.0, 0.0),   "desc": "Playful wink with a right-arm wave"},
    "gloom_shake":   {"label": "Gloom Shake",   "duration": 3.0, "category": "sad",      "kick": (0.0, 8.0, 0.0),   "desc": "Dramatic gloom shiver, glow dims"},
    "curious_lean":  {"label": "Curious Lean",  "duration": 2.8, "category": "curious",  "kick": (0.0, 0.0, 0.0),   "desc": "Head tilt and lean-in with raised brow"},
    "chibi_pop":     {"label": "Chibi Pop",     "duration": 2.6, "category": "joy",      "kick": (11.5, 0.0, 9.0),  "desc": "Pops into chibi proportions and back"},
    "pout_puff":     {"label": "Pout Puff",     "duration": 3.0, "category": "annoyed",  "kick": (0.0, 0.0, 0.0),   "desc": "Turns away with puffed cheeks"},
    "heart_sparkle": {"label": "Heart Sparkle", "duration": 3.0, "category": "love",     "kick": (11.5, 0.0, 9.0),  "desc": "Sparkly-eyed affectionate sway"},
    # --- New expressive emotes (inspired by Waifu Studio's affect set) ---
    "greet_wave":    {"label": "Greet Wave",    "duration": 2.9, "category": "greeting", "kick": (6.0, 2.0, 4.0),   "desc": "Big friendly right-arm wave with a head tilt and bounce"},
    "cheer_bounce":  {"label": "Cheer Bounce",  "duration": 2.8, "category": "joy",      "kick": (13.0, 0.0, 8.0),  "desc": "Both arms pumping while bouncing"},
    "smug_tease":    {"label": "Smug Tease",    "duration": 2.9, "category": "smug",     "kick": (0.0, 4.0, 2.0),   "desc": "One brow up, squinty smirk, little heh-heh bob"},
    "starry_awe":    {"label": "Starry Awe",    "duration": 3.2, "category": "awe",      "kick": (5.0, 0.0, 7.0),   "desc": "Looks up starry-eyed, mouth a small o"},
    "flustered_shy": {"label": "Flustered Shy", "duration": 3.0, "category": "shy",      "kick": (7.0, 3.0, 10.0),  "desc": "Max blush, eyes darting, flustered little head shakes"},
    "sleepy_yawn":   {"label": "Sleepy Yawn",   "duration": 3.6, "category": "sleepy",   "kick": (6.0, 0.0, 0.0),   "desc": "Big stretch-yawn, then droopy eyes"},
    "gentle_nod":    {"label": "Gentle Nod",    "duration": 3.0, "category": "caring",   "kick": (0.0, 0.0, 0.0),   "desc": "Soft sympathetic tilt with two slow nods"},
}


def _smooth(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# paramMap channel -> (Inochi2D parameter, axis index, hard clamp range)
MOUTH_CHANNELS: dict[str, tuple[str, int, float, float]] = {
    "mouthOpen":   ("VB Mouth", 1, 0.0, 1.0),
    "mouthForm":   ("VB Mouth", 0, -1.0, 1.0),
    "jawOpen":     ("Jaw Open", 0, 0.0, 1.0),
    "mouthFunnel": ("Mouth Funnel", 0, 0.0, 1.0),
    "lipsPress":   ("Lips Press / Part", 0, -1.0, 1.0),
    "mouthShrug":  ("Mouth Shrug", 0, 0.0, 1.0),
    "mouthX":      ("Mouth X", 0, -1.0, 1.0),
    "tongueOut":   ("Tongue Out", 0, 0.0, 1.0),
    "mouthSpring": ("Mouth Spring", 0, -1.0, 1.0),   # optional map entry; identity when absent
}


class MouthParamMapper:
    """VTube Studio-style per-channel remapper:
    clamp((v - inMin) / (inMax - inMin)) -> signed/unsigned power curve -> lerp(outMin, outMax)
    -> exponential smoothing (`smooth` = fraction of the previous value kept per 60 FPS frame)."""

    def __init__(self) -> None:
        self._y: dict[str, float] = {}

    @staticmethod
    def _num(m: dict[str, Any], key: str, default: float) -> float:
        try:
            v = float(m.get(key, default))
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def map(self, channel: str, value: float, dt: float, m: dict[str, Any] | None) -> float | None:
        """Mapped value, or None when the channel is disabled (lip-sync stops driving it)."""
        lo_hard, hi_hard = MOUTH_CHANNELS[channel][2], MOUTH_CHANNELS[channel][3]
        if m is None:
            out = max(lo_hard, min(hi_hard, value))
            self._y[channel] = out
            return out
        if not m.get("enabled", True):
            self._y.pop(channel, None)
            return None
        in_min = self._num(m, "inMin", lo_hard)
        in_max = self._num(m, "inMax", hi_hard)
        out_min = self._num(m, "outMin", lo_hard)
        out_max = self._num(m, "outMax", hi_hard)
        curve = max(0.2, min(5.0, self._num(m, "curve", 1.0)))
        smooth = max(0.0, min(0.98, self._num(m, "smooth", 0.0)))
        span = in_max - in_min
        t = 0.0 if abs(span) < 1e-6 else max(0.0, min(1.0, (value - in_min) / span))
        if curve != 1.0:
            if in_min < 0.0 < in_max:                     # signed channel: curve around the centre
                s = 2.0 * t - 1.0
                t = 0.5 * (1.0 + math.copysign(abs(s) ** curve, s))
            else:
                t = t ** curve
        out = out_min + (out_max - out_min) * t
        out = max(lo_hard, min(hi_hard, out))
        prev = self._y.get(channel)
        if prev is not None and smooth > 0.0:
            k = 1.0 - smooth ** max(0.05, min(3.0, dt * 60.0))
            out = prev + (out - prev) * k
        self._y[channel] = out
        return out


MOTION_TRACKS = ("AngleX", "AngleY", "AngleZ", "BodyX", "BodyZ", "EyeOpen", "ArmR", "ArmL")


def validate_motion_clip(data: Any) -> dict[str, Any]:
    """Reject a malformed Motion Lab clip before it can reach the 60 FPS step (raises ValueError)."""
    if not isinstance(data, dict):
        raise ValueError("Motion clip must be a JSON object")
    frames = data.get("frames")
    if isinstance(frames, bool) or not isinstance(frames, int) or not (1 <= frames <= 60 * 60 * 60):
        raise ValueError("Motion clip has an invalid frame count")
    fps = data.get("fps", 60.0)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not (1.0 <= float(fps) <= 240.0):
        raise ValueError("Motion clip has an invalid fps")
    tracks = data.get("tracks")
    if not isinstance(tracks, dict):
        raise ValueError("Motion clip has no tracks")
    for name in MOTION_TRACKS:
        tr = tracks.get(name)
        if not isinstance(tr, list) or len(tr) < frames:
            raise ValueError(f"Motion clip track {name} is missing or too short")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in tr[:frames]):
            raise ValueError(f"Motion clip track {name} has non-numeric samples")
    if not isinstance(data.get("id"), str) or not isinstance(data.get("name"), str):
        raise ValueError("Motion clip needs an id and a name")
    dur = data.get("duration", frames / float(fps))
    if isinstance(dur, bool) or not isinstance(dur, (int, float)) or not math.isfinite(dur):
        raise ValueError("Motion clip has an invalid duration")
    return data


INTERACTION_STATES = ("idle", "listening", "thinking", "speaking")

# Physics-owned parameters (SimplePhysics outputs): Inochi2D 0.8.7 ignores script values for these while drivers run.
PHYSICS_OWNED = ("Chest Y", "Chest X", "Eye Jelly", "Mouth Spring", "Arm R Swing", "Arm L Swing")

# Legacy parameter names some plugins still write -> real rig names.
LEGACY_PARAM_ALIASES = {"Brow L Y": "Left Brow Y", "Brow R Y": "Right Brow Y"}

def _ff(v: Any, default: float) -> float:
    """Finite float or default (a bad setting must never break the 60 FPS step)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class _Hold:
    """Held posture target: keeps a comfortable pose for a while, then picks a new one; a critically damped
    spring eases between holds so weight shifts read as decisions, not a screensaver loop."""
    __slots__ = ("spring", "target")

    def __init__(self) -> None:
        self.spring = Spring2nd(0.0)
        self.target = 0.0


class RigController:
    def __init__(self, param_defaults: dict[str, tuple[float, float]], lipsync: LiveLipSyncEngine, hooks: HookRegistry) -> None:
        self.defaults = dict(param_defaults)
        self._known = frozenset(self.defaults)
        self._warned_unknown: set[str] = set()
        self.lipsync = lipsync
        self.hooks = hooks
        self.lock = threading.Lock()
        now = time.perf_counter()
        rnd = random.Random()
        self._rnd = rnd

        # Optional rig features (feature-detected from the loaded .inp; older models just skip them)
        self._has_raised_joints = all(f"Arm {s} {j}" in self._known for s in "RL" for j in ("Elbow", "Wrist"))
        self._has_brow_angle = "Brow L Angle" in self._known and "Brow R Angle" in self._known
        self._has_cheek = "Cheek" in self._known

        # Pointer look-at normalized [-1..1] + smooth low-pass filter & blend
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self._mx_filt = 0.0
        self._my_filt = 0.0
        self._mouse_blend = 0.0
        self._mouse_last_move = 0.0

        # Autonomous gaze & eye-leads-head springs
        self._gaze_tx = 0.0
        self._gaze_ty = 0.0
        self._gaze_head_tilt = 0.0
        self._next_gaze_shift = now + 1.8
        self._next_micro_saccade = now + 0.5
        self._micro_x = 0.0
        self._micro_y = 0.0
        self.sp_eye_x = Spring2nd(0.0)
        self.sp_eye_y = Spring2nd(0.0)
        self.sp_head_gx = Spring2nd(0.0)
        self.sp_head_gy = Spring2nd(0.0)
        self.sp_head_gt = Spring2nd(0.0)

        # Blinks: normal, double, half (~12 %) and slow relaxed (~10 %)
        self._next_blink = now + 2.4
        self._blink_phase = 0.0
        self._blink_active = False
        self._blink_kind = "normal"
        self._blink_queue_double = False

        # Expression springs (+ cheek tint)
        self.expr_springs = {k: Spring2nd(0.0) for k in ("eyeSmile", "squint", "browY", "browForm", "mouthForm", "cheekPuff", "sparkle", "blush")}
        self.sp_cheek = Spring2nd(0.0)
        self.sp_sparkle = Spring2nd(0.0)         # AI / temp sparkle & blush fade in instead of popping
        self.sp_blush = Spring2nd(0.0)
        self._mouth_busy = 0.0                    # eased "is talking" for emote mouth shapes
        self.sp_brow_angle_l = Spring2nd(0.0)
        self.sp_brow_angle_r = Spring2nd(0.0)

        # Raised wave-arm springs
        self.sp_arm_r_pose = Spring2nd(0.0)
        self.sp_arm_l_pose = Spring2nd(0.0)
        self.sp_arm_r_gest = Spring2nd(0.0)
        self.sp_arm_l_gest = Spring2nd(0.0)
        self.sp_arm_r_lift = Spring2nd(0.0)
        self.sp_arm_l_lift = Spring2nd(0.0)
        self.sp_arm_r_wlag = Spring2nd(0.0)       # raised-wrist follow-through (lags the gesture)
        self.sp_arm_l_wlag = Spring2nd(0.0)

        # Excited-moment raise of the clean wave arm (rare, cooled down; normal chatter keeps the arms resting)
        self._raise_side = "R" if rnd.random() < 0.5 else "L"
        self._raise_until = 0.0
        self._raise_next = now + 4.0
        self._speech_on_since = 0.0
        self._speech_off_since = now
        self._was_speaking = False

        # "Body bounce" springs. Historic names kept (plugins kick them): they now drive REAL body motion, which
        # moves the chest / arm / hair physics anchors so Inochi2D's pendulums respond naturally.
        self.sp_chest_y = Spring2nd(0.0)      # -> Body Angle Y bounce
        self.sp_chest_x = Spring2nd(0.0)      # -> Body Angle X sway
        self.sp_eye_jelly = Spring2nd(0.0)    # -> quick eye squash (squint pulse)
        # Script-side fallback springs, only used while engine physics is OFF
        self.sp_fb_chest_y = Spring2nd(0.0)
        self.sp_fb_chest_x = Spring2nd(0.0)
        self.sp_fb_arm_swing = Spring2nd(0.0)
        self._prev_body_x = 0.0
        self._prev_body_y = 0.0
        self._prev_breath = 0.0

        # Temporary AI / action emote hold
        self._temp_action: dict[str, Any] | None = None
        self._temp_until: float = 0.0

        # Active tweened emote clip
        self._emote_id: str = ""
        self._emote_start: float = 0.0
        self._emote_dur: float = 0.0

        # Neuro-style idle acts (now sparser: the held postures carry most of the life)
        self._idle_act: str = ""
        self._idle_start: float = 0.0
        self._idle_dur: float = 0.0
        self._idle_side: float = 1.0
        self._next_idle_act: float = now + rnd.uniform(5.0, 8.0)
        self._idle_gate = 1.0                     # fades a running idle act out (emote / clip) instead of cutting it

        # Held posture targets (body x / z / lean, head x / y / z) + micro drift
        self._hold = {k: _Hold() for k in ("bx", "bz", "lean", "hx", "hy", "hz")}
        self._next_hold = now + rnd.uniform(1.2, 2.5)
        self._drift = {k: _Hold() for k in ("bx", "bz", "lean", "hx", "hy", "hz")}
        self._next_drift = now

        # Uneven breathing
        self._breath_phase = 0.0
        self._breath_period = 3.4
        self._breath_depth = 0.72
        self._breath_count = 0
        self._next_sigh = rnd.randint(6, 10)
        self._breath_is_sigh = False

        # Mood momentum (set by the backend) + interaction state
        self._mood_tv = 0.0
        self._mood_ta = 0.0
        self._mood_v = 0.0
        self._mood_a = 0.0
        self._istate = "idle"
        self._istate_since = now
        self._w_listen = 0.0
        self._w_think = 0.0
        self._w_speak = 0.0
        self._think_side = 1.0
        self._think_side_s = 1.0                 # eased copies: a side flip never pops the pose
        self._listen_side = 1.0
        self._listen_side_s = 1.0
        self._next_listen_nod = now + 1.5
        self.sp_nod = Spring2nd(0.0)

        # Prosody: onset head beats + pitch
        self.sp_beat_y = Spring2nd(0.0)
        self.sp_beat_z = Spring2nd(0.0)
        self._last_beat = 0.0
        self._beat_sign = 1.0
        self._pitch_base = 0.5
        self._pitch_rel = 0.0
        self._energy_face = 0.0                  # lightly smoothed energy / stress for brows & squint
        self._stress_face = 0.0

        # Asymmetric brow micro-life
        self._brow_ev_side = 1.0
        self._brow_ev_start = 0.0
        self._brow_ev_dur = 0.0
        self._brow_ev_amp = 0.0
        self._next_brow_ev = now + rnd.uniform(2.0, 4.0)

        # VTS-style mouth parameter mapper + Mouth Spring (telemetry / physics-off fallback)
        self.mouth_mapper = MouthParamMapper()
        self.sp_mouth_spring = Spring2nd(0.0)
        self._prev_lip_open = 0.0
        self.last_mapped: dict[str, float | None] = {}
        self.sp_chibi = Spring2nd(0.0)
        self.sp_gloom = Spring2nd(0.0)
        self.sp_emissive = Spring2nd(1.0)

        # Active Motion Lab video clip playback
        self._motion_clip: dict[str, Any] | None = None
        self._motion_start: float = 0.0
        self._motion_loop: bool = True
        self._motion_mix: float = 0.85
        self._motion_playing: bool = False

        # Live telemetry snapshot
        self.last_params: dict[str, tuple[float, float]] = dict(self.defaults)
        self.last_shader: dict[str, Any] = {}
        self.last_camera: dict[str, Any] = {}
        self.last_state: dict[str, Any] = {}
        self.frame_count = 0

    # ------------------------------------------------------------------ public API
    def set_pointer(self, nx: float, ny: float) -> None:
        with self.lock:
            nx_c = _clamp(_ff(nx, 0.0), -1.0, 1.0)
            ny_c = _clamp(_ff(ny, 0.0), -1.0, 1.0)
            if abs(nx_c - self.mouse_x) > 0.008 or abs(ny_c - self.mouse_y) > 0.008:
                self._mouse_last_move = time.perf_counter()
            self.mouse_x = nx_c
            self.mouse_y = ny_c

    def set_mood_state(self, valence: float, arousal: float) -> None:
        """Mood momentum from the backend, clamped to [-1, 1]. Shifts resting smile warmth, brow lift, cheek blush,
        breath rate / depth and sway energy (smoothed over a few seconds, so updates never pop)."""
        with self.lock:
            self._mood_tv = _clamp(_ff(valence, 0.0), -1.0, 1.0)
            self._mood_ta = _clamp(_ff(arousal, 0.0), -1.0, 1.0)

    def set_interaction_state(self, state: str) -> None:
        """'idle' | 'listening' (leans in, attentive nods) | 'thinking' (glance up-side, tilt, asymmetric brow)
        | 'speaking'. Unknown values fall back to 'idle'."""
        s = str(state or "idle").strip().lower()
        if s not in INTERACTION_STATES:
            s = "idle"
        with self.lock:
            if s != self._istate:
                self._istate = s
                self._istate_since = time.perf_counter()
                if s == "thinking":
                    self._think_side = -1.0 if self._rnd.random() < 0.5 else 1.0
                elif s == "listening":
                    self._listen_side = -1.0 if self._rnd.random() < 0.5 else 1.0

    def trigger_temp_action(self, action: dict[str, Any], duration: float = 4.5) -> None:
        with self.lock:
            self._temp_action = dict(action)
            self._temp_until = time.perf_counter() + _clamp(_ff(duration, 4.5), 1.0, 60.0)

    def trigger_emote(self, emote_id: str, duration: float = 2.6) -> dict[str, Any]:
        """Trigger a smooth cubic/spring tweened emote clip with a coupled body bounce."""
        eid = str(emote_id or "").strip().lower()
        meta = EMOTE_CATALOG.get(eid)
        if meta is None:
            return {"ok": False, "error": f"Unknown emote: {eid}", "available": sorted(EMOTE_CATALOG.keys())}
        dur = _ff(duration, 2.6)
        with self.lock:
            self._emote_id = eid
            self._emote_start = time.perf_counter()
            self._emote_dur = _clamp(dur, 0.3, 30.0) if dur and dur != 2.6 else float(meta["duration"])
            # Instant kick on start: real body bounce / sway the physics pendulums react to.
            ky, kx, ke = meta.get("kick", (0.0, 0.0, 0.0))
            self.sp_chest_y.v += ky
            self.sp_chest_x.v += kx
            self.sp_eye_jelly.v += ke * 0.25
        return {"ok": True, "emote": eid, "duration": self._emote_dur}

    def emote_status(self) -> dict[str, Any]:
        with self.lock:
            el = time.perf_counter() - self._emote_start
            active = bool(self._emote_id) and 0.0 <= el <= self._emote_dur
            return {"active": self._emote_id if active else "", "elapsed": round(el, 2) if active else 0.0,
                    "duration": round(self._emote_dur, 2) if active else 0.0}

    def load_motion_clip(self, clip_path: Path, loop: bool = True, mix: float = 0.85) -> dict[str, Any]:
        data = validate_motion_clip(json.loads(clip_path.read_text(encoding="utf-8")))   # before installing!
        with self.lock:
            self._motion_clip = data
            self._motion_start = time.perf_counter()
            self._motion_loop = bool(loop)
            self._motion_mix = _clamp(_ff(mix, 0.85), 0.0, 1.0)
            self._motion_playing = True
        return {"id": data["id"], "name": data["name"], "duration": data["duration"], "hasAudio": data.get("hasAudio", False)}

    def update_motion_playback(self, playing: bool | None = None, loop: bool | None = None, mix: float | None = None) -> None:
        with self.lock:
            if playing is not None:
                self._motion_playing = bool(playing)
                if playing:
                    self._motion_start = time.perf_counter()
            if loop is not None:
                self._motion_loop = bool(loop)
            if mix is not None:
                self._motion_mix = _clamp(_ff(mix, self._motion_mix), 0.0, 1.0)

    def stop_motion(self) -> None:
        with self.lock:
            self._motion_playing = False
            self._motion_clip = None

    def motion_status(self) -> dict[str, Any]:
        with self.lock:
            if not self._motion_clip:
                return {"playing": False, "id": "", "name": "", "mix": self._motion_mix, "loop": self._motion_loop}
            return {
                "playing": self._motion_playing,
                "id": self._motion_clip["id"],
                "name": self._motion_clip["name"],
                "mix": self._motion_mix,
                "loop": self._motion_loop,
            }

    # ------------------------------------------------------------------ living layers
    def _breath_step(self, dt: float, rate_mul: float, depth_mul: float, speaking: bool, mood_v: float) -> float:
        """Uneven breathing: each breath gets its own period / depth, an occasional deeper sigh every 6-10 breaths.
        Inhale (~40 %) is quicker than the exhale. Returns Breath in [0, 1]."""
        rnd = self._rnd
        self._breath_phase += dt / max(0.8, self._breath_period)
        if self._breath_phase >= 1.0:
            self._breath_phase -= 1.0
            self._breath_count += 1
            sigh = self._breath_count >= self._next_sigh and not speaking
            self._breath_is_sigh = sigh
            if sigh:
                self._breath_count = 0
                lo, hi = (4, 7) if mood_v < -0.3 else (6, 10)       # low mood sighs more often
                self._next_sigh = rnd.randint(lo, hi)
            base = 3.4 / max(0.55, rate_mul)
            if speaking:                                            # speech breathing: shallow, irregular
                self._breath_period = base * rnd.uniform(0.75, 1.35)
                self._breath_depth = rnd.uniform(0.38, 0.62) * depth_mul
            elif sigh:
                self._breath_period = base * rnd.uniform(1.45, 1.75)
                self._breath_depth = 1.0
            else:
                self._breath_period = base * rnd.uniform(0.86, 1.16)
                self._breath_depth = _clamp(rnd.uniform(0.62, 0.80) * depth_mul, 0.3, 0.92)
        ph = self._breath_phase
        if ph < 0.4:                                                # inhale
            u = ph / 0.4
            s = u * u * (3.0 - 2.0 * u)
        else:                                                       # slower exhale with a soft tail
            u = (ph - 0.4) / 0.6
            s = 1.0 - u * u * (3.0 - 2.0 * u)
        return _clamp(s * self._breath_depth, 0.0, 1.0)

    def _posture_step(self, dt: float, t_now: float, energy_mul: float, still: float) -> dict[str, float]:
        """Held weight shifts (3.5-9 s holds, eased) + tiny micro-drift. `still` (0..1) damps the amplitude
        (e.g. while speaking or listening she holds steadier)."""
        rnd = self._rnd
        amp = energy_mul * (1.0 - 0.45 * still)
        if t_now >= self._next_hold:
            side = rnd.choice((-1.0, -1.0, 0.0, 1.0, 1.0))
            h = self._hold
            h["bx"].target = side * rnd.uniform(1.4, 3.8) + rnd.uniform(-0.6, 0.6)
            h["bz"].target = side * rnd.uniform(1.0, 3.2)
            h["lean"].target = rnd.uniform(-1.2, 2.6)
            h["hz"].target = (-side if rnd.random() < 0.65 else side) * rnd.uniform(0.5, 4.5)   # usually counter-tilt
            h["hx"].target = rnd.uniform(-4.5, 4.5)
            h["hy"].target = rnd.uniform(-2.5, 2.8)
            self._next_hold = t_now + rnd.uniform(3.5, 9.0) / max(0.6, min(1.6, energy_mul))
        if t_now >= self._next_drift:
            for d in self._drift.values():
                d.target = rnd.uniform(-1.0, 1.0)
            self._next_drift = t_now + rnd.uniform(0.9, 2.2)
        out: dict[str, float] = {}
        drift_amp = {"bx": 0.45, "bz": 0.4, "lean": 0.35, "hx": 0.9, "hy": 0.7, "hz": 0.75}
        for k, hd in self._hold.items():
            v = hd.spring.step(hd.target * amp, dt, 1.75, 1.0)
            dv = self._drift[k].spring.step(self._drift[k].target, dt, 1.1, 1.0)
            out[k] = v + dv * drift_amp[k] * energy_mul
        return out

    def _blink_shape(self) -> float:
        """Eyelid closure 0..1 for the active blink."""
        ph = self._blink_phase
        kind = self._blink_kind
        if kind == "slow":                       # close 35 %, hold 20 %, slow open 45 %
            if ph < 0.35:
                u = ph / 0.35
                return u * u * (3.0 - 2.0 * u)
            if ph < 0.55:
                return 1.0
            u = (ph - 0.55) / 0.45
            return 1.0 - u * u * (3.0 - 2.0 * u)
        b = math.sin(ph * math.pi) ** 1.6
        return b * (0.5 if kind == "half" else 1.0)

    def _start_blink(self, t_now: float) -> None:
        r = self._rnd.random()
        if r < 0.12:
            self._blink_kind = "half"
        elif r < 0.22:
            self._blink_kind = "slow"
        else:
            self._blink_kind = "normal"
            self._blink_queue_double = r > 0.84      # ~16 % double blink
        self._blink_active = True
        self._blink_phase = 0.0

    # ------------------------------------------------------------------ frame
    def step(self, dt: float, cfg: dict[str, Any], ctx: Any) -> tuple[dict[str, tuple[float, float]], dict[str, Any], dict[str, Any]]:
        t_now = time.perf_counter()
        self.frame_count += 1
        dt = _clamp(_ff(dt, 1.0 / 60.0), 1e-4, 0.1)
        rnd = self._rnd

        rig_cfg = cfg["rig"]
        cam_cfg = dict(cfg["camera"])
        shader_cfg = dict(cfg["shader"])
        stage_cfg = cfg["stage"]
        lip_cfg = cfg["lipsync"]

        if stage_cfg.get("autoMatchLighting", True):
            rm = ROOM_AMBIENT_MAP.get(stage_cfg.get("room", ""), None)
            if rm:
                shader_cfg["ambientColor"] = rm["ambientColor"]

        # 1. Live lip-sync & hook
        lip = self.lipsync.step(dt, t_now)
        self.hooks.emit_sync("on_lipsync_frame", ctx, lip)
        energy = _clamp(_ff(lip.get("energy", 0.0), 0.0), 0.0, 1.0)
        stress = _clamp(_ff(lip.get("stress", 0.0), 0.0), 0.0, 1.0)
        onset = _clamp(_ff(lip.get("onset", 0.0), 0.0), 0.0, 1.0)
        pitch = _clamp(_ff(lip.get("pitch", 0.5), 0.5), 0.0, 1.0)
        voiced = _clamp(_ff(lip.get("voiced", 1.0 if energy > 0.06 else 0.0), 0.0), 0.0, 1.0)
        speaking = energy > 0.06 or bool(lip_cfg.get("holdViseme"))
        if speaking and not self._was_speaking:
            self._speech_on_since = t_now
        if not speaking and self._was_speaking:
            self._speech_off_since = t_now
        self._was_speaking = speaking

        p: dict[str, tuple[float, float]] = dict(self.defaults)

        # 2. Snapshot shared state
        with self.lock:
            temp = self._temp_action if t_now < self._temp_until else None
            if temp is None:
                self._temp_action = None
            em_id = self._emote_id
            em_el = t_now - self._emote_start
            em_dur = self._emote_dur
            if em_id and (em_el < 0.0 or em_el > em_dur):
                self._emote_id = ""
                em_id = ""
            mx_t, my_t = self.mouse_x, self.mouse_y
            m_clip = self._motion_clip if self._motion_playing else None
            m_start = self._motion_start
            m_loop = self._motion_loop
            m_mix = self._motion_mix
            mood_tv, mood_ta = self._mood_tv, self._mood_ta
            istate = self._istate
            think_side_t = self._think_side
            listen_side_t = self._listen_side

        # 2.5 Mood momentum + interaction-state blend weights (eased, never pop)
        k_mood = 1.0 - math.exp(-dt / 2.4)
        self._mood_v += (mood_tv - self._mood_v) * k_mood
        self._mood_a += (mood_ta - self._mood_a) * k_mood
        mv, ma = self._mood_v, self._mood_a
        k_st = 1.0 - math.exp(-dt * 2.6)
        self._w_listen += ((1.0 if istate == "listening" else 0.0) - self._w_listen) * k_st
        self._w_think += ((1.0 if istate == "thinking" else 0.0) - self._w_think) * k_st
        self._w_speak += ((1.0 if (istate == "speaking" or speaking) else 0.0) - self._w_speak) * k_st
        w_l, w_t, w_s = self._w_listen, self._w_think, self._w_speak
        k_side = 1.0 - math.exp(-dt * 3.0)
        self._think_side_s += (think_side_t - self._think_side_s) * k_side
        self._listen_side_s += (listen_side_t - self._listen_side_s) * k_side
        think_side = self._think_side_s
        listen_side = self._listen_side_s
        arousal_n = 0.5 * (ma + 1.0)                                  # 0..1

        # 3. Uneven mood-coupled breathing
        breath_val = 0.0
        if rig_cfg.get("autoBreath", True):
            b_amp = _clamp(_ff(rig_cfg.get("breathAmp", 1.0), 1.0), 0.0, 3.0)
            rate_mul = 1.0 + 0.24 * ma + 0.10 * w_s
            depth_mul = 1.0 + 0.12 * ma - 0.08 * mv * (1.0 if mv < 0.0 else 0.0)
            breath_val = self._breath_step(dt, rate_mul, depth_mul, speaking, mv) * b_amp
        p["Breath"] = (_clamp(breath_val, 0.0, 1.0), 0.0)

        # 3.2 Held weight-shift postures + micro drift (replaces the fixed sine "screensaver" sway)
        sway = _clamp(_ff(rig_cfg.get("idleSway", 1.0), 1.0), 0.0, 3.0)
        energy_mul = sway * (0.8 + 0.5 * arousal_n)
        pose = self._posture_step(dt, t_now, energy_mul, max(w_s * 0.6, w_l * 0.5, w_t * 0.7))
        head_x = pose["hx"]
        head_y = pose["hy"]
        head_z = pose["hz"]
        body_x = pose["bx"]
        body_z = pose["bz"]
        body_lean = pose["lean"]
        body_y = sway * 1.5 * (breath_val - 0.4)                    # chest rises with the breath
        # Low mood droops the head a little; high arousal lifts it.
        head_y += -2.2 * max(0.0, -mv) + 1.2 * max(0.0, ma)

        # 3.5 Neuro-style idle acts (sparser now; blocked while speaking / thinking)
        idle_brow_boost = 0.0
        idle_smile_boost = 0.0
        idle_block = bool(em_id) or m_clip is not None
        self._idle_gate += ((0.0 if idle_block else 1.0) - self._idle_gate) * (1.0 - math.exp(-dt / 0.15))
        if self._idle_act and idle_block and self._idle_gate < 0.02:
            self._idle_act = ""
            self._next_idle_act = t_now + rnd.uniform(4.0, 8.0)
        if sway > 0.05:
            if (not self._idle_act and not idle_block and not speaking and w_t < 0.3 and w_l < 0.3
                    and t_now >= self._next_idle_act):
                choices = ("curious_tilt", "vibe_sway", "happy_bounce", "chat_read", "daydream_sigh", "cheeky_lean")
                self._idle_act = rnd.choice(choices)
                self._idle_start = t_now
                self._idle_side = -1.0 if rnd.random() < 0.5 else 1.0
                self._idle_dur = {
                    "curious_tilt": rnd.uniform(2.4, 3.4),
                    "vibe_sway":    rnd.uniform(3.2, 4.4),
                    "happy_bounce": rnd.uniform(1.8, 2.5),
                    "chat_read":    rnd.uniform(2.6, 3.6),
                    "daydream_sigh": rnd.uniform(3.0, 4.0),
                    "cheeky_lean":  rnd.uniform(2.2, 3.1),
                }[self._idle_act]
            if self._idle_act:
                iu = (t_now - self._idle_start) / max(0.1, self._idle_dur)
                if iu >= 1.0:
                    self._idle_act = ""
                    self._next_idle_act = t_now + rnd.uniform(6.0, 12.0) / max(0.35, min(2.0, sway))
                else:
                    env = (math.sin(iu * math.pi) ** 1.25) * self._idle_gate
                    s_dir = self._idle_side
                    ia = self._idle_act
                    amt = min(1.5, sway)
                    if ia == "curious_tilt":
                        head_z += s_dir * 13.5 * env * amt
                        head_x += -s_dir * 6.5 * env * amt
                        body_z += -s_dir * 3.8 * env * amt
                        body_lean += 2.4 * env * amt
                        idle_brow_boost += 0.24 * env
                    elif ia == "vibe_sway":
                        groove = math.sin(iu * math.pi * 2.0) * env
                        body_x += s_dir * 5.2 * groove * amt
                        body_z += s_dir * 4.5 * groove * amt
                        head_z += -s_dir * 8.5 * groove * amt
                        head_y += 2.8 * abs(groove) * amt
                        body_y += 2.4 * abs(groove) * amt
                        idle_smile_boost += 0.14 * env
                    elif ia == "happy_bounce":
                        hop = math.sin(iu * math.pi * 2.0)
                        body_y += 5.4 * max(0.0, hop) * env * amt
                        head_y += 4.2 * hop * env * amt
                        head_z += s_dir * 5.5 * math.sin(iu * math.pi * 2.0) * env * amt
                        idle_smile_boost += 0.22 * env
                        idle_brow_boost += 0.16 * env
                    elif ia == "chat_read":
                        head_x += s_dir * 10.0 * env * amt
                        head_y += (-7.0 + 3.2 * math.sin(iu * math.pi * 3.0)) * env * amt
                        body_x += s_dir * 3.2 * env * amt
                    elif ia == "daydream_sigh":
                        breath_val = min(1.0, breath_val + 0.42 * env)
                        p["Breath"] = (breath_val, 0.0)
                        head_y += 6.5 * env * amt
                        head_x += s_dir * 7.5 * env * amt
                        head_z += s_dir * 6.0 * env * amt
                        body_y += 2.2 * env * amt
                    elif ia == "cheeky_lean":
                        body_lean += 5.0 * env * amt
                        head_x += s_dir * 8.5 * env * amt
                        head_z += -s_dir * 7.5 * env * amt
                        head_y += 2.5 * env * amt
                        idle_brow_boost += 0.18 * env
                        idle_smile_boost += 0.16 * env

        # 3.7 Interaction-state layers
        # Listening: lean in, small attentive nods on a loose rhythm.
        if w_l > 0.02:
            body_lean += 2.8 * w_l
            head_y -= 1.2 * w_l
            head_z += 2.5 * w_l * listen_side
            if istate == "listening" and t_now >= self._next_listen_nod:
                self.sp_nod.v -= rnd.uniform(26.0, 42.0)
                self._next_listen_nod = t_now + rnd.uniform(1.8, 4.2)
        head_y += self.sp_nod.step(0.0, dt, 7.5, 0.55)
        # Thinking: glance up to one side, thoughtful tilt, chin a touch up.
        if w_t > 0.02:
            head_z += think_side * 5.5 * w_t
            head_x += think_side * 4.0 * w_t
            head_y += 3.0 * w_t
            body_z += think_side * 1.2 * w_t

        # 4. Autonomous gaze + mouse follow (eye leads head)
        auto_gaze = bool(rig_cfg.get("autoGaze", True))
        liveliness = _clamp(_ff(rig_cfg.get("autoGazeLiveliness", 1.0), 1.0), 0.0, 3.0)
        mouse_on = bool(rig_cfg.get("mouseLookAt", True))
        mouse_recent = mouse_on and ((t_now - self._mouse_last_move) < 2.2)
        contact = max(w_l, w_s)                                      # attentive: more eye contact

        if auto_gaze:
            if t_now >= self._next_micro_saccade:
                self._next_micro_saccade = t_now + rnd.uniform(0.35, 0.95)
                self._micro_x = rnd.uniform(-0.07, 0.07) * liveliness
                self._micro_y = rnd.uniform(-0.05, 0.05) * liveliness
            if t_now >= self._next_gaze_shift:
                r_pick = rnd.random()
                old_tx = self._gaze_tx
                contact_p = 0.62 + 0.24 * contact
                if r_pick < contact_p:
                    self._gaze_tx = rnd.uniform(-0.14, 0.14) * liveliness
                    self._gaze_ty = rnd.uniform(-0.10, 0.08) * liveliness
                    self._gaze_head_tilt = rnd.uniform(-0.22, 0.22) * liveliness
                    self._next_gaze_shift = t_now + rnd.uniform(2.0, 4.4)
                elif r_pick < contact_p + (1.0 - contact_p) * 0.58:
                    side = -1.0 if rnd.random() < 0.5 else 1.0
                    self._gaze_tx = side * rnd.uniform(0.38, 0.68) * liveliness
                    self._gaze_ty = rnd.uniform(-0.48, -0.20) * liveliness
                    self._gaze_head_tilt = side * rnd.uniform(0.25, 0.55) * liveliness
                    self._next_gaze_shift = t_now + rnd.uniform(1.1, 2.3)
                else:
                    side = -1.0 if rnd.random() < 0.5 else 1.0
                    self._gaze_tx = side * rnd.uniform(0.42, 0.78) * liveliness
                    self._gaze_ty = rnd.uniform(-0.08, 0.22) * liveliness
                    self._gaze_head_tilt = -side * rnd.uniform(0.15, 0.40) * liveliness
                    self._next_gaze_shift = t_now + rnd.uniform(1.2, 2.4)
                if rig_cfg.get("autoBlink", True) and not self._blink_active and abs(self._gaze_tx - old_tx) > 0.42 and rnd.random() < 0.65:
                    self._start_blink(t_now)

        lp = 1.0 - math.exp(-8.5 * max(0.001, dt))
        self._mx_filt += (mx_t - self._mx_filt) * lp
        self._my_filt += (my_t - self._my_filt) * lp
        if mouse_recent:
            self._mouse_blend = min(1.0, self._mouse_blend + dt * 2.6)
        else:
            self._mouse_blend = max(0.0, self._mouse_blend - dt * 1.05)

        ml = _clamp(_ff(rig_cfg.get("mouseLookStrength", 0.55), 0.55), 0.0, 2.0)
        m_gx = self._mx_filt * ml
        m_gy = self._my_filt * ml
        m_tilt = self._mx_filt * 0.22 * ml

        if auto_gaze:
            a_gx = _clamp(self._gaze_tx + self._micro_x, -0.95, 0.95) * (1.0 - 0.35 * contact)
            a_gy = _clamp(self._gaze_ty + self._micro_y, -0.85, 0.85) * (1.0 - 0.35 * contact)
            a_tilt = self._gaze_head_tilt
        else:
            a_gx, a_gy, a_tilt = 0.0, 0.0, 0.0
        if w_t > 0.01:                                               # thinking glance (up + to one side)
            a_gx = a_gx * (1.0 - w_t) + think_side * 0.52 * w_t + self._micro_x * w_t
            a_gy = a_gy * (1.0 - w_t) + (-0.46) * w_t + self._micro_y * w_t
            a_tilt = a_tilt * (1.0 - w_t) + think_side * 0.25 * w_t

        mb_raw = self._mouse_blend * self._mouse_blend * (3.0 - 2.0 * self._mouse_blend)
        mb = mb_raw * min(1.0, max(0.0, ml) * 2.2)
        tgt_gx = a_gx * (1.0 - mb) + m_gx * mb
        tgt_gy = a_gy * (1.0 - mb) + m_gy * mb
        tgt_tilt = a_tilt * (1.0 - mb) + m_tilt * mb

        eye_om = 17.0 * (1.0 - mb) + 8.5 * mb
        eye_zeta = 0.90 * (1.0 - mb) + 0.98 * mb
        head_om = 5.6 * (1.0 - mb) + 3.8 * mb
        head_zeta = 0.90 * (1.0 - mb) + 0.98 * mb                  # (was `head_z`: it erased the head roll!)
        ex_s = self.sp_eye_x.step(tgt_gx, dt, eye_om, eye_zeta)
        ey_s = self.sp_eye_y.step(tgt_gy, dt, eye_om, eye_zeta)
        hx_s = self.sp_head_gx.step(tgt_gx, dt, head_om, head_zeta)
        hy_s = self.sp_head_gy.step(tgt_gy, dt, head_om, head_zeta)
        ht_s = self.sp_head_gt.step(tgt_tilt, dt, 3.6, 0.96)

        head_x += hx_s * 15.5
        head_y += -hy_s * 11.5
        head_z += ht_s * 9.5 + hx_s * 3.2
        body_x += hx_s * 4.8
        body_z += ht_s * 2.8
        eye_x = _clamp(ex_s * 0.88, -1.0, 1.0)
        eye_y = _clamp(-ey_s * 0.74, -1.0, 1.0)

        # 4.5 Prosody: syllable-onset head beats (spring impulses, not a metronome) + pitch-coupled chin / brows
        nod_gain = _clamp(_ff(lip_cfg.get("speechToHeadNod", 0.80), 0.80), 0.0, 3.0)
        jiggle = _clamp(_ff(rig_cfg.get("jiggleScale", 1.25), 1.25), 0.0, 2.5)
        beat_strength = 0.0
        if speaking and onset > 0.22 and (t_now - self._last_beat) > 0.17 and nod_gain > 0.0:
            beat_strength = _clamp(0.35 + 0.9 * stress + 0.5 * onset, 0.0, 1.5)
            self._last_beat = t_now
            if rnd.random() < 0.35:
                self._beat_sign = -self._beat_sign
            self.sp_beat_y.v -= 30.0 * beat_strength * nod_gain
            self.sp_beat_z.v += self._beat_sign * 10.0 * beat_strength * nod_gain
            self.sp_chest_y.v += 1.8 * beat_strength * min(1.5, jiggle)      # tiny body bounce -> chest physics
            self.sp_chest_x.v += 1.1 * self._beat_sign * beat_strength * min(1.5, jiggle)   # -> Arm L/R Swing
            if beat_strength > 0.95 and rnd.random() < 0.35 and t_now - self._brow_ev_start > self._brow_ev_dur:
                self._brow_ev_side = -1.0 if rnd.random() < 0.5 else 1.0     # asymmetric emphasis brow
                self._brow_ev_start, self._brow_ev_dur, self._brow_ev_amp = t_now, rnd.uniform(0.45, 0.8), rnd.uniform(0.18, 0.3)
        head_y += self.sp_beat_y.step(0.0, dt, 11.0, 0.42)
        head_z += self.sp_beat_z.step(0.0, dt, 7.0, 0.5)
        if energy > 0.04 and nod_gain > 0.0:
            head_y += nod_gain * 1.6 * stress
        body_z += 0.3 * self.sp_beat_z.y                            # the torso follows the beat tilt a little

        if speaking and voiced > 0.3:
            self._pitch_base += (pitch - self._pitch_base) * (1.0 - math.exp(-dt / 2.5))
            rel_t = _clamp((pitch - self._pitch_base) * 2.6, -1.0, 1.0)
        else:
            rel_t = 0.0
        self._pitch_rel += (rel_t - self._pitch_rel) * (1.0 - math.exp(-dt / 0.12))
        pr = self._pitch_rel
        head_y += 3.2 * pr * w_s                                     # chin lifts on rising / excited inflection
        brow_pitch = (0.30 * max(0.0, pr) - 0.10 * max(0.0, -pr)) * w_s
        eye_widen = 0.07 * max(0.0, pr) * w_s

        # 4.7 Body bounce springs (emote kicks, onset beats, plugins): real motion -> physics anchors respond
        bounce_y = self.sp_chest_y.step(0.0, dt, 9.0, 0.30)
        bounce_x = self.sp_chest_x.step(0.0, dt, 8.0, 0.32)
        squash = self.sp_eye_jelly.step(0.0, dt, 18.0, 0.5)
        body_y += 2.4 * bounce_y
        head_y += 1.1 * bounce_y
        body_x += 1.8 * bounce_x

        # 5. Blinks: normal / double / half / slow relaxed
        eye_open_l = 1.0
        eye_open_r = 1.0
        if rig_cfg.get("autoBlink", True):
            rate = _clamp(_ff(rig_cfg.get("blinkRate", 1.0), 1.0), 0.2, 6.0)
            if not self._blink_active and t_now >= self._next_blink:
                self._start_blink(t_now)
            if self._blink_active:
                dur = {"normal": 0.165, "half": 0.15, "slow": 0.42}[self._blink_kind]
                self._blink_phase += dt / dur
                if self._blink_phase >= 1.0:
                    self._blink_active = False
                    if self._blink_queue_double:
                        self._blink_queue_double = False
                        self._next_blink = t_now + 0.22
                    else:
                        gap = rnd.uniform(2.1, 4.8) / rate
                        gap *= 1.25 if istate == "thinking" else 1.0     # concentration: fewer blinks
                        self._next_blink = t_now + gap
                else:
                    b = self._blink_shape()
                    eye_open_l = eye_open_r = max(0.0, 1.0 - b)
        if eye_open_l > 0.95:
            eye_open_l = eye_open_r = eye_open_l + eye_widen

        # 6. Expression + mood + state
        expr_name = (temp.get("expression") if temp else None) or str(rig_cfg.get("expression", "happy"))
        expr_tgt = EXPRESSION_PRESETS.get(expr_name, EXPRESSION_PRESETS["neutral"])
        mood_tgt = {
            "eyeSmile": 0.16 * max(0.0, mv) + 0.05 * w_l,
            "squint": 0.10 * w_t,
            "browY": 0.12 * mv + 0.08 * ma + 0.08 * w_l,
            "browForm": 0.14 * mv,
            "mouthForm": 0.22 * mv - 0.10 * w_t,
            "cheekPuff": 0.0, "sparkle": 0.12 * max(0.0, ma) * max(0.0, mv), "blush": 0.0,
        }
        ev = {k: self.expr_springs[k].step(expr_tgt[k] + mood_tgt[k], dt, 11.0, 0.86) for k in self.expr_springs}

        sparkle_t = max(ev["sparkle"], _ff(rig_cfg.get("sparkleEyes", 0.0), 0.0), _ff(temp["sparkle"], 0.0) if temp and "sparkle" in temp else 0.0)
        blush_t = max(ev["blush"], _ff(rig_cfg.get("blushLines", 0.0), 0.0), _ff(temp["blush"], 0.0) if temp and "blush" in temp else 0.0)
        sparkle = _clamp(self.sp_sparkle.step(_clamp(sparkle_t, 0.0, 1.0), dt, 12.0, 1.0), 0.0, 1.0)
        blush = _clamp(self.sp_blush.step(_clamp(blush_t, 0.0, 1.0), dt, 9.0, 1.0), 0.0, 1.0)
        self._mouth_busy += ((1.0 if speaking else 0.0) - self._mouth_busy) * (1.0 - math.exp(-dt / 0.12))
        quiet = 1.0 - self._mouth_busy
        cheek_puff = _clamp(max(ev["cheekPuff"], _ff(rig_cfg.get("cheekPuff", 0.0), 0.0)), 0.0, 1.0)
        k_face = 1.0 - math.exp(-dt / 0.06)
        self._energy_face += (energy - self._energy_face) * k_face
        self._stress_face += (stress - self._stress_face) * k_face
        e_f, s_f = self._energy_face, self._stress_face
        squint = _clamp(max(ev["squint"], _ff(rig_cfg.get("eyeSquint", 0.0), 0.0)) + 0.14 * e_f + 0.25 * max(0.0, squash), 0.0, 1.0)
        brow_speech = _clamp(_ff(lip_cfg.get("speechToBrows", 0.72), 0.72), 0.0, 3.0) * (0.25 * e_f + 0.35 * s_f) + brow_pitch
        cheek_tgt = 0.55 * blush + 0.18 * max(0.0, mv) + 0.10 * max(0.0, ma) + 0.10 * energy * max(0.0, mv)
        cheek = _clamp(self.sp_cheek.step(cheek_tgt, dt, 4.0, 1.0), 0.0, 1.0)

        # 6.5 Asymmetric brow micro-life (curious / smug / thinking / emphasis)
        if t_now >= self._next_brow_ev and t_now - self._brow_ev_start > self._brow_ev_dur:
            self._brow_ev_side = (think_side_t if w_t > 0.5 else (-1.0 if rnd.random() < 0.5 else 1.0))
            self._brow_ev_start = t_now
            self._brow_ev_dur = rnd.uniform(0.5, 1.4)
            self._brow_ev_amp = rnd.uniform(0.10, 0.26) * (1.3 if expr_name in ("smug", "surprised") else 1.0)
            busy = 0.55 if (w_t > 0.5 or expr_name == "smug" or speaking) else 1.0
            self._next_brow_ev = t_now + rnd.uniform(2.5, 7.0) * busy
        bu = (t_now - self._brow_ev_start) / max(0.05, self._brow_ev_dur)
        brow_ev = self._brow_ev_amp * (math.sin(bu * math.pi) ** 1.3 if 0.0 <= bu <= 1.0 else 0.0)
        # Thinking: the glance-side brow lifts, the other one dips slightly.
        tl = 0.5 * (1.0 + _clamp(think_side, -1.0, 1.0))           # 1 = glancing to the L-brow side (continuous)
        tr = 1.0 - tl
        brow_l_extra = brow_ev * (1.0 if self._brow_ev_side > 0 else -0.25) + w_t * (0.38 * tl - 0.08)
        brow_r_extra = brow_ev * (1.0 if self._brow_ev_side < 0 else -0.25) + w_t * (0.38 * tr - 0.08)
        # Brow angle: + = worried / soft (inner ends up), - = cross. Sad / shy / thinking soften, pout sharpens.
        ang_base = 0.30 * max(0.0, -mv) + (0.25 if expr_name == "shy" else 0.0) - (0.45 if expr_name == "pout" else 0.0)
        ang_l = self.sp_brow_angle_l.step(ang_base + 0.25 * w_t * (0.4 + 0.6 * tl), dt, 6.0, 0.9)
        ang_r = self.sp_brow_angle_r.step(ang_base + 0.25 * w_t * (0.4 + 0.6 * tr), dt, 6.0, 0.9)

        # 7. Raised wave arms (manual / emote / temp action) + speech gestures
        auto_gest = bool(rig_cfg.get("autoGestureOnSpeech", True))
        ag_gain = _clamp(_ff(lip_cfg.get("speechToArmGesture", 0.75), 0.75), 0.0, 2.0)
        t_arm_r = _ff(temp["armR"], 0.0) if temp and "armR" in temp else _ff(rig_cfg.get("armRPose", 0.0), 0.0)
        t_arm_l = _ff(temp["armL"], 0.0) if temp and "armL" in temp else _ff(rig_cfg.get("armLPose", 0.0), 0.0)

        # 7.1 Excited / greeting peaks: raise the clean wave arm for one phrase, rarely (8-14 s cooldown). Normal chatter
        #     keeps the original resting arms down; they sway through Arm L/R Swing physics with posture & beats.
        rest_free = (t_arm_r < 0.5 and t_arm_l < 0.5) and not em_id and m_clip is None
        excited_peak = speaking and stress > 0.72 and (ma > 0.35 or expr_name == "excited")
        if auto_gest and ag_gain > 0.0 and rest_free and excited_peak and t_now >= self._raise_next and t_now >= self._raise_until:
            self._raise_side = "L" if self._raise_side == "R" else "R"
            self._raise_until = t_now + rnd.uniform(1.4, 2.6)
            self._raise_next = self._raise_until + rnd.uniform(8.0, 14.0)
        if t_now < self._raise_until:
            if not auto_gest or not rest_free or (not speaking and t_now - self._speech_off_since > 0.4):
                self._raise_until = t_now                            # phrase over / something else owns the arms
            elif self._raise_side == "R":
                t_arm_r = max(t_arm_r, 1.0)
            else:
                t_arm_l = max(t_arm_l, 1.0)

        arm_r_pose = _clamp(self.sp_arm_r_pose.step(t_arm_r, dt, 22.0, 0.92), 0.0, 1.0)
        arm_l_pose = _clamp(self.sp_arm_l_pose.step(t_arm_l, dt, 22.0, 0.92), 0.0, 1.0)
        g_r_tgt = _ff(rig_cfg.get("armRGesture", 0.0), 0.0)
        g_l_tgt = _ff(rig_cfg.get("armLGesture", 0.0), 0.0)
        l_r_tgt = _ff(rig_cfg.get("armRLift", 0.0), 0.0)
        l_l_tgt = _ff(rig_cfg.get("armLLift", 0.0), 0.0)
        if arm_r_pose > 0.5:
            if temp and _ff(temp.get("armR", 0), 0.0) > 0.5:
                g_r_tgt += 0.25 + 0.55 * math.sin(t_now * 7.2)
            elif auto_gest and speaking:
                g_r_tgt += ag_gain * (0.18 + 0.36 * math.sin(t_now * 3.1) + 0.42 * stress)
                l_r_tgt += ag_gain * (0.30 * energy + 0.25 * max(0.0, pr))
            else:
                g_r_tgt += 0.14 * math.sin(t_now * 1.9)
        if arm_l_pose > 0.5:
            if temp and _ff(temp.get("armL", 0), 0.0) > 0.5:
                g_l_tgt += 0.22 + 0.52 * math.sin(t_now * 6.8 + 0.6)
            elif auto_gest and speaking:
                g_l_tgt += ag_gain * (0.18 + 0.36 * math.sin(t_now * 2.8 + 0.9) + 0.42 * stress)
                l_l_tgt += ag_gain * (0.30 * energy + 0.25 * max(0.0, pr))
            else:
                g_l_tgt += 0.14 * math.sin(t_now * 1.7 + 0.8)

        arm_r_gest = _clamp(self.sp_arm_r_gest.step(g_r_tgt, dt, 14.0, 0.85), -0.48, 0.92)
        arm_l_gest = _clamp(self.sp_arm_l_gest.step(g_l_tgt, dt, 14.0, 0.85), -0.48, 0.92)
        arm_r_lift = _clamp(self.sp_arm_r_lift.step(l_r_tgt, dt, 11.0, 0.88), -0.75, 0.85)
        arm_l_lift = _clamp(self.sp_arm_l_lift.step(l_l_tgt, dt, 11.0, 0.88), -0.75, 0.85)
        # Raised wrist follow-through: a slow copy of the gesture; the difference flicks the hand.
        lag_r = self.sp_arm_r_wlag.step(arm_r_gest, dt, 6.0, 0.7)
        lag_l = self.sp_arm_l_wlag.step(arm_l_gest, dt, 6.0, 0.7)

        # 8. Motion Lab clip blend (head, body, blinks & arms only — never the live mouth)
        if m_clip is not None:
            tracks = m_clip["tracks"]
            n_f = int(m_clip["frames"])
            fps = float(m_clip.get("fps", 60.0))
            el = (t_now - m_start) * fps
            if el >= n_f:
                if m_loop:
                    el = el % n_f
                else:
                    with self.lock:
                        self._motion_playing = False
                    el = n_f - 1
            fi = max(0, min(n_f - 1, int(el)))
            w = m_mix
            head_x = head_x * (1.0 - w) + tracks["AngleX"][fi] * w
            head_y = head_y * (1.0 - w) + tracks["AngleY"][fi] * w
            head_z = head_z * (1.0 - w) + tracks["AngleZ"][fi] * w
            body_x = body_x * (1.0 - w) + tracks["BodyX"][fi] * w
            body_z = body_z * (1.0 - w) + tracks["BodyZ"][fi] * w
            eye_open_l = eye_open_l * (1.0 - w) + tracks["EyeOpen"][fi] * w
            eye_open_r = eye_open_r * (1.0 - w) + tracks["EyeOpen"][fi] * w
            if tracks["ArmR"][fi] > 0.35 and arm_r_pose > 0.5:
                arm_r_gest = _clamp(arm_r_gest + (tracks["ArmR"][fi] - 0.35) * 0.9 * w, -0.48, 0.92)
            if tracks["ArmL"][fi] > 0.35 and arm_l_pose > 0.5:
                arm_l_gest = _clamp(arm_l_gest + (tracks["ArmL"][fi] - 0.35) * 0.9 * w, -0.48, 0.92)

        # 9. Mouth (VTube Studio / VBridger mapping of every lip-sync channel)
        pmap_cfg = lip_cfg.get("paramMap") if isinstance(lip_cfg.get("paramMap"), dict) else {}
        default_map = DEFAULT_PARAM_MAP

        def _chan(name: str, raw: float) -> float | None:
            m = pmap_cfg.get(name)
            if not isinstance(m, dict):
                m = default_map.get(name)
            elif name in default_map:
                m = {**default_map[name], **m}
            return self.mouth_mapper.map(name, _ff(raw, 0.0), dt, m)

        form_src = lip["form"] if speaking else ev["mouthForm"]
        m_open = _chan("mouthOpen", lip["open"])
        m_form = _chan("mouthForm", form_src)
        m_jaw = _chan("jawOpen", lip["open"])
        m_funnel = _chan("mouthFunnel", lip["funnel"])
        m_press = _chan("lipsPress", lip.get("press", 0.0))
        m_shrug = _chan("mouthShrug", lip["shrug"])
        m_x = _chan("mouthX", lip["x"])
        m_tongue = _chan("tongueOut", lip["tongue"])

        # Mouth Spring: computed for telemetry; only WRITTEN when engine physics is off (otherwise the rig's
        # SimplePhysics owns it - its anchor now follows VB Mouth / Jaw Open).
        physics_on = bool(rig_cfg.get("physicsEnabled", True))
        v_open = (_ff(lip["open"], 0.0) - self._prev_lip_open) / max(0.008, dt)
        self._prev_lip_open = _ff(lip["open"], 0.0)
        bounce = _clamp(_ff(lip_cfg.get("jawBounce", 1.0), 1.0), 0.0, 3.0)
        self.sp_mouth_spring.v += bounce * (onset * 260.0 + _clamp(v_open, -6.0, 6.0) * 1.6) * dt
        m_spring = _chan("mouthSpring", self.sp_mouth_spring.step(0.0, dt, 19.0, 0.28))

        self.last_mapped = {
            "mouthOpen": m_open, "mouthForm": m_form, "jawOpen": m_jaw, "mouthFunnel": m_funnel,
            "lipsPress": m_press, "mouthShrug": m_shrug, "mouthX": m_x, "tongueOut": m_tongue,
            "mouthSpring": m_spring,
        }
        vb_open = m_open if m_open is not None else 0.0
        vb_form = m_form if m_form is not None else _clamp(ev["mouthForm"], -1.0, 1.0)
        vb_tongue = max(m_tongue if m_tongue is not None else 0.0, _ff(rig_cfg.get("tongueOut", 0.0), 0.0))
        mouth_x_v = m_x if m_x is not None else self.defaults.get("Mouth X", (0.0, 0.0))[0]
        funnel_v = m_funnel if m_funnel is not None else self.defaults.get("Mouth Funnel", (0.0, 0.0))[0]
        if not speaking and w_t > 0.02:                              # thinking: pursed, pulled to one side
            mouth_x_v = _clamp(mouth_x_v + 0.28 * think_side * w_t, -1.0, 1.0)
            funnel_v = _clamp(funnel_v + 0.12 * w_t, 0.0, 1.0)

        # 10. Assign Inochi2D parameters
        p["Angle"] = (_clamp(head_x, -30.0, 30.0), _clamp(head_y, -24.0, 24.0))
        p["Angle Z"] = (_clamp(head_z, -25.0, 25.0), 0.0)
        p["Body Angle X"] = (_clamp(body_x, -10.0, 10.0), 0.0)
        p["Body Angle Y"] = (_clamp(body_y, -10.0, 10.0), 0.0)
        p["Body Angle Z"] = (_clamp(body_z, -10.0, 10.0), 0.0)
        p["Body Lean"] = (_clamp(body_lean, -10.0, 10.0), 0.0)

        smile = _clamp(ev["eyeSmile"] + idle_smile_boost, 0.0, 1.0)
        brow_base = ev["browY"] + brow_speech + idle_brow_boost
        p["EyeBall"] = (eye_x, eye_y)
        p["Left Eye Open"] = (_clamp(eye_open_l, 0.0, 1.2), 0.0)
        p["Right Eye Open"] = (_clamp(eye_open_r, 0.0, 1.2), 0.0)
        p["Eye L Smile"] = (smile, 0.0)
        p["Eye R Smile"] = (smile, 0.0)
        p["Eye L Squint"] = (squint, 0.0)
        p["Eye R Squint"] = (squint, 0.0)
        p["Left Brow Y"] = (_clamp(brow_base + brow_l_extra, -1.0, 1.0), 0.0)
        p["Right Brow Y"] = (_clamp(brow_base + brow_r_extra, -1.0, 1.0), 0.0)
        p["Brow L Form"] = (_clamp(ev["browForm"], -1.0, 1.0), 0.0)
        p["Brow R Form"] = (_clamp(ev["browForm"], -1.0, 1.0), 0.0)
        if self._has_brow_angle:
            p["Brow L Angle"] = (_clamp(ang_l, -1.0, 1.0), 0.0)
            p["Brow R Angle"] = (_clamp(ang_r, -1.0, 1.0), 0.0)
        if self._has_cheek:
            p["Cheek"] = (cheek, 0.0)

        p["VB Mouth"] = (_clamp(vb_form, -1.0, 1.0), _clamp(vb_open, 0.0, 1.0))
        p["Mouth Funnel"] = (funnel_v, 0.0)
        p["Mouth X"] = (mouth_x_v, 0.0)
        p["Mouth Shrug"] = (m_shrug if m_shrug is not None else self.defaults.get("Mouth Shrug", (0.0, 0.0))[0], 0.0)
        if "Lips Press / Part" in p:
            p["Lips Press / Part"] = (m_press if m_press is not None else self.defaults["Lips Press / Part"][0], 0.0)
        p["Tongue Out"] = (_clamp(vb_tongue, 0.0, 1.0), 0.0)
        p["Jaw Open"] = (m_jaw if m_jaw is not None else self.defaults.get("Jaw Open", (0.0, 0.0))[0], 0.0)
        p["Cheek Puff"] = (cheek_puff, 0.0)

        p["Arm R Pose"] = (arm_r_pose, 0.0)
        p["Arm R Gesture"] = (arm_r_gest, 0.0)
        p["Arm R Lift"] = (arm_r_lift, 0.0)
        p["Arm L Pose"] = (arm_l_pose, 0.0)
        p["Arm L Gesture"] = (arm_l_gest, 0.0)
        p["Arm L Lift"] = (arm_l_lift, 0.0)
        if self._has_raised_joints:
            p["Arm R Elbow"] = (_clamp(0.10 * arm_r_pose * math.sin(t_now * 1.3), -1.0, 1.0), 0.0)
            p["Arm L Elbow"] = (_clamp(0.10 * arm_l_pose * math.sin(t_now * 1.1 + 0.7), -1.0, 1.0), 0.0)
            p["Arm R Wrist"] = (_clamp(1.4 * (lag_r - arm_r_gest) * arm_r_pose, -1.0, 1.0), 0.0)
            p["Arm L Wrist"] = (_clamp(1.4 * (lag_l - arm_l_gest) * arm_l_pose, -1.0, 1.0), 0.0)
        p["Sparkle Eyes"] = (sparkle, 0.0)
        p["Anime Blush Lines"] = (blush, 0.0)

        # 10.2 Tweened emote clips (Chibi / Gloom / Emissive glow)
        chibi_tgt = _clamp(_ff(rig_cfg.get("chibiMode", 0.0), 0.0), 0.0, 1.0)
        gloom_tgt = _clamp(_ff(rig_cfg.get("gloom", 0.0), 0.0), 0.0, 1.0)
        em_glow_boost = 0.0

        if em_id and em_dur > 0.1:
            u_t = _clamp(em_el / em_dur, 0.0, 1.0)
            env_in = min(1.0, u_t / 0.16)
            env_in = env_in * env_in * (3.0 - 2.0 * env_in)
            env_out = min(1.0, (1.0 - u_t) / 0.28)
            env_out = env_out * env_out * (3.0 - 2.0 * env_out)
            env = env_in * env_out

            def _bump(name: str, amount: float, lo: float = -1.0, hi: float = 1.0) -> None:
                if name in p:
                    p[name] = (_clamp(p[name][0] + amount, lo, hi), p[name][1])

            def _atleast(name: str, value: float) -> None:
                if name in p:
                    p[name] = (_clamp(max(p[name][0], value), 0.0, 1.0), p[name][1])

            if em_id == "excited_hop":
                hop = math.sin(u_t * math.pi * 4.0)
                body_y += 8.2 * env * hop
                head_y += 9.0 * env * hop
                head_z += 12.0 * env * math.sin(u_t * math.pi * 2.0)
                _atleast("Eye L Smile", 0.92 * env)
                _atleast("Eye R Smile", 0.92 * env)
                _atleast("Sparkle Eyes", 1.0 * env)
                _atleast("Arm R Pose", env)
                _atleast("Arm L Pose", env)
                p["Arm R Gesture"] = (0.45 * math.sin(u_t * math.pi * 6.0) * env, 0.0)
                p["Arm L Gesture"] = (0.45 * math.sin(u_t * math.pi * 6.0 + 0.5) * env, 0.0)
                _bump("Left Brow Y", 0.30 * env)
                _bump("Right Brow Y", 0.30 * env)
                em_glow_boost = 0.65 * env
            elif em_id == "shy_blush":
                head_x -= 11.0 * env
                head_y -= 7.5 * env
                head_z += 11.0 * env
                body_lean += 4.5 * env
                p["EyeBall"] = (max(-1.0, eye_x - 0.35 * env), max(-1.0, eye_y - 0.28 * env))
                _atleast("Anime Blush Lines", 1.0 * env)
                _atleast("Eye L Smile", 0.55 * env)
                _atleast("Eye R Smile", 0.55 * env)
                _bump("Left Brow Y", -0.25 * env)
                _bump("Right Brow Y", -0.25 * env)
                if self._has_brow_angle:
                    _bump("Brow L Angle", 0.45 * env)
                    _bump("Brow R Angle", 0.45 * env)
                if self._has_cheek:
                    _atleast("Cheek", 0.85 * env)
            elif em_id == "wink_wave":
                head_z += 10.5 * env
                head_x += 7.0 * env
                p["Right Eye Open"] = (max(0.0, p["Right Eye Open"][0] * (1.0 - env)), 0.0)
                _atleast("Eye R Smile", 0.95 * env)
                _atleast("Eye L Smile", 0.55 * env)
                _atleast("Arm R Pose", env)
                p["Arm R Gesture"] = ((0.30 + 0.55 * math.sin(u_t * math.pi * 6.0)) * env, 0.0)
                _atleast("Sparkle Eyes", 0.75 * env)
                _bump("Left Brow Y", 0.22 * env)
                em_glow_boost = 0.40 * env
            elif em_id == "gloom_shake":
                shiver = math.sin(u_t * math.pi * 24.0) * env
                head_x += 5.5 * shiver
                head_y -= 10.0 * env
                body_y -= 4.0 * env
                gloom_tgt = max(gloom_tgt, env)
                p["Brow L Form"] = (-0.85 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.85 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                _bump("Left Brow Y", -0.20 * env)
                _bump("Right Brow Y", -0.20 * env)
                if self._has_brow_angle:
                    _bump("Brow L Angle", 0.70 * env)
                    _bump("Brow R Angle", 0.70 * env)
                em_glow_boost = -0.55 * env
            elif em_id == "curious_lean":
                head_z -= 16.0 * env
                head_y += 4.0 * env
                body_lean += 7.5 * env
                body_z -= 4.5 * env
                _bump("Left Brow Y", 0.55 * env)
                _bump("Right Brow Y", 0.20 * env)
                if self._has_brow_angle:
                    _bump("Brow R Angle", -0.25 * env)
                em_glow_boost = 0.25 * env
            elif em_id == "chibi_pop":
                chibi_tgt = max(chibi_tgt, env)
                body_y += 5.5 * math.sin(u_t * math.pi * 3.0) * env
                _atleast("Sparkle Eyes", 0.90 * env)
                em_glow_boost = 0.50 * env
            elif em_id == "pout_puff":
                head_x += 12.0 * env
                head_z -= 8.0 * env
                _atleast("Cheek Puff", 1.0 * env)
                _atleast("Anime Blush Lines", 0.65 * env)
                p["Brow L Form"] = (-0.75 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.75 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                _bump("Left Brow Y", -0.30 * env)
                _bump("Right Brow Y", -0.30 * env)
                if self._has_brow_angle:
                    _bump("Brow L Angle", -0.65 * env)
                    _bump("Brow R Angle", -0.65 * env)
                if self._has_cheek:
                    _atleast("Cheek", 0.55 * env)
                p["VB Mouth"] = (-0.65 * env + p["VB Mouth"][0] * (1.0 - env), p["VB Mouth"][1])
            elif em_id == "heart_sparkle":
                head_y += 5.0 * env
                head_z += 7.5 * math.sin(u_t * math.pi * 2.0) * env
                _atleast("Sparkle Eyes", 1.0 * env)
                _atleast("Anime Blush Lines", 0.85 * env)
                _atleast("Eye L Smile", 0.72 * env)
                _atleast("Eye R Smile", 0.72 * env)
                if self._has_cheek:
                    _atleast("Cheek", 0.7 * env)
                em_glow_boost = 0.95 * env
            elif em_id == "greet_wave":
                wave = math.sin(u_t * math.pi * 8.0)
                head_z += 8.0 * env + 2.5 * wave * env
                head_x += 4.0 * env
                body_y += 3.0 * abs(math.sin(u_t * math.pi * 4.0)) * env
                body_z += 2.0 * env
                _atleast("Arm R Pose", env)
                _atleast("Arm R Lift", 0.45 * env)
                p["Arm R Gesture"] = ((0.20 + 0.62 * wave) * env, 0.0)
                _atleast("Eye L Smile", 0.80 * env)
                _atleast("Eye R Smile", 0.80 * env)
                _atleast("Sparkle Eyes", 0.50 * env)
                _bump("Left Brow Y", 0.25 * env)
                _bump("Right Brow Y", 0.25 * env)
                em_glow_boost = 0.45 * env
            elif em_id == "cheer_bounce":
                bnc = abs(math.sin(u_t * math.pi * 5.0))
                pump = math.sin(u_t * math.pi * 10.0)
                body_y += 9.0 * bnc * env
                head_y += 7.0 * bnc * env
                head_z += 6.0 * math.sin(u_t * math.pi * 2.5) * env
                _atleast("Arm R Pose", env)
                _atleast("Arm L Pose", env)
                _atleast("Arm R Lift", (0.55 + 0.25 * pump) * env)
                _atleast("Arm L Lift", (0.55 - 0.25 * pump) * env)
                p["Arm R Gesture"] = (0.35 * pump * env, 0.0)
                p["Arm L Gesture"] = (-0.35 * pump * env, 0.0)
                _atleast("Eye L Smile", 1.0 * env)
                _atleast("Eye R Smile", 1.0 * env)
                _atleast("Sparkle Eyes", 1.0 * env)
                _bump("Left Brow Y", 0.35 * env)
                _bump("Right Brow Y", 0.35 * env)
                em_glow_boost = 0.80 * env
            elif em_id == "smug_tease":
                heh = math.sin(u_t * math.pi * 6.0) * _smooth(u_t / 0.3) * (1.0 - _smooth((u_t - 0.5) / 0.3))
                head_z -= 9.0 * env
                head_x += 6.0 * env
                head_y += 2.5 * heh * env
                body_lean -= 3.0 * env
                _bump("Left Brow Y", 0.50 * env)
                _bump("Right Brow Y", -0.15 * env)
                if self._has_brow_angle:
                    _bump("Brow R Angle", -0.35 * env)
                _atleast("Eye L Squint", 0.55 * env)
                _atleast("Eye R Squint", 0.55 * env)
                _atleast("Eye L Smile", 0.45 * env)
                _atleast("Eye R Smile", 0.45 * env)
                eq = env * quiet
                p["VB Mouth"] = (p["VB Mouth"][0] + (max(p["VB Mouth"][0], 0.60) - p["VB Mouth"][0]) * eq, p["VB Mouth"][1])
                p["Mouth X"] = (_clamp(p["Mouth X"][0] + 0.35 * eq, -1.0, 1.0), 0.0)
                em_glow_boost = 0.20 * env
            elif em_id == "starry_awe":
                head_y += 8.0 * env
                head_z += 6.0 * math.sin(u_t * math.pi * 2.0) * env
                body_lean -= 2.0 * env
                p["EyeBall"] = (p["EyeBall"][0] * (1.0 - 0.6 * env), min(1.0, p["EyeBall"][1] + 0.45 * env))
                _bump("Left Brow Y", 0.70 * env)
                _bump("Right Brow Y", 0.70 * env)
                _atleast("Sparkle Eyes", 1.0 * env)
                _atleast("Anime Blush Lines", 0.35 * env)
                eq = env * quiet
                p["VB Mouth"] = (-0.20 * eq + p["VB Mouth"][0] * (1.0 - eq), max(p["VB Mouth"][1], 0.32 * eq))
                p["Mouth Funnel"] = (max(p["Mouth Funnel"][0], 0.45 * eq), 0.0)
                p["Jaw Open"] = (max(p["Jaw Open"][0], 0.25 * eq), 0.0)
                em_glow_boost = 0.85 * env
            elif em_id == "flustered_shy":
                shake = math.sin(u_t * math.pi * 18.0) * env
                dart = math.sin(u_t * math.pi * 8.0)
                head_x += 4.5 * shake - 6.0 * env
                head_y -= 9.0 * env
                head_z += 7.0 * env
                body_lean += 3.5 * env
                p["EyeBall"] = (_clamp(0.40 * dart * env + p["EyeBall"][0] * (1.0 - env), -1.0, 1.0), max(-1.0, p["EyeBall"][1] - 0.25 * env))
                _atleast("Anime Blush Lines", 1.0 * env)
                _atleast("Eye L Squint", 0.50 * env)
                _atleast("Eye R Squint", 0.50 * env)
                _atleast("Cheek Puff", 0.35 * env)
                p["Brow L Form"] = (-0.45 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.45 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                _bump("Left Brow Y", 0.30 * env)
                _bump("Right Brow Y", 0.30 * env)
                if self._has_brow_angle:
                    _bump("Brow L Angle", 0.55 * env)
                    _bump("Brow R Angle", 0.55 * env)
                if self._has_cheek:
                    _atleast("Cheek", 1.0 * env)
                _atleast("Arm R Pose", env)
                _atleast("Arm L Pose", env)
                p["Arm R Gesture"] = (0.30 * math.sin(u_t * math.pi * 14.0) * env, 0.0)
                p["Arm L Gesture"] = (0.30 * math.sin(u_t * math.pi * 14.0 + 1.2) * env, 0.0)
                em_glow_boost = 0.30 * env
            elif em_id == "sleepy_yawn":
                yawn = math.sin(math.pi * _clamp(u_t / 0.55, 0.0, 1.0))
                droop = _smooth((u_t - 0.45) / 0.25) * env
                head_y += 7.0 * yawn * env - 6.0 * droop
                head_z += 5.0 * droop
                body_y += 3.0 * yawn * env
                lid = max(0.0, 1.0 - 0.85 * yawn * env - 0.50 * droop)
                p["Left Eye Open"] = (p["Left Eye Open"][0] * lid, 0.0)
                p["Right Eye Open"] = (p["Right Eye Open"][0] * lid, 0.0)
                _atleast("Eye L Squint", 0.60 * droop)
                _atleast("Eye R Squint", 0.60 * droop)
                _bump("Left Brow Y", -0.35 * env)
                _bump("Right Brow Y", -0.35 * env)
                _atleast("Arm R Pose", yawn * env)
                _atleast("Arm L Pose", yawn * env)
                _atleast("Arm R Lift", 0.50 * yawn * env)
                _atleast("Arm L Lift", 0.50 * yawn * env)
                yq = yawn * env * quiet
                p["VB Mouth"] = (-0.10 * yq + p["VB Mouth"][0] * (1.0 - yq), max(p["VB Mouth"][1], 0.85 * yq))
                p["Jaw Open"] = (max(p["Jaw Open"][0], 0.75 * yq), 0.0)
                p["Mouth Funnel"] = (max(p["Mouth Funnel"][0], 0.35 * yq), 0.0)
                em_glow_boost = -0.25 * env
            elif em_id == "gentle_nod":
                nod = math.sin(u_t * math.pi * 4.0) ** 2
                head_y -= 6.0 * nod * env
                head_z += 7.0 * env
                body_lean += 2.0 * env
                _atleast("Eye L Smile", 0.50 * env)
                _atleast("Eye R Smile", 0.50 * env)
                p["Brow L Form"] = (-0.30 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.30 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                _bump("Left Brow Y", 0.15 * env)
                _bump("Right Brow Y", 0.15 * env)
                if self._has_brow_angle:
                    _bump("Brow L Angle", 0.30 * env)
                    _bump("Brow R Angle", 0.30 * env)
                _atleast("Anime Blush Lines", 0.30 * env)
                em_glow_boost = 0.15 * env

            p["Angle"] = (_clamp(head_x, -30.0, 30.0), _clamp(head_y, -24.0, 24.0))
            p["Angle Z"] = (_clamp(head_z, -25.0, 25.0), 0.0)
            p["Body Angle Y"] = (_clamp(body_y, -10.0, 10.0), 0.0)
            p["Body Angle Z"] = (_clamp(body_z, -10.0, 10.0), 0.0)
            p["Body Lean"] = (_clamp(body_lean, -10.0, 10.0), 0.0)

        chibi_val = _clamp(self.sp_chibi.step(chibi_tgt, dt, 12.0, 0.55), 0.0, 1.0)
        gloom_val = _clamp(self.sp_gloom.step(gloom_tgt, dt, 10.0, 0.80), 0.0, 1.0)
        if "Proportion (Chibi)" in p:
            p["Proportion (Chibi)"] = (chibi_val, 0.0)
        if "Gloom" in p:
            p["Gloom"] = (gloom_val, 0.0)

        em_base = _clamp(_ff(rig_cfg.get("emissiveGlow", 1.0), 1.0), 0.0, 2.0)
        em_tgt = _clamp(em_base * (0.90 + 0.10 * (breath_val - 0.4) + 0.30 * energy + 0.32 * p["Sparkle Eyes"][0] + em_glow_boost), 0.0, 2.0)
        em_val = _clamp(self.sp_emissive.step(em_tgt, dt, 10.0, 0.85), 0.0, 2.0)
        if "Emissive Glow" in p:
            p["Emissive Glow"] = (em_val, 0.0)

        # 10.5 Physics-owned params: Inochi2D owns them while drivers run (the body / head / breath / arm-root
        # motion above moves their anchors). Script fallback springs only while engine physics is OFF.
        if not physics_on:
            inv_dt = 1.0 / max(0.008, min(0.05, dt))
            v_bx = (body_x - self._prev_body_x) * inv_dt
            v_by = (body_y - self._prev_body_y) * inv_dt
            v_br = (breath_val - self._prev_breath) * inv_dt
            self.sp_fb_chest_y.v += jiggle * (v_by * 0.42 + v_br * 2.4 + stress * 2.8) * dt * 16.0
            self.sp_fb_chest_x.v += jiggle * (-v_bx * 0.36) * dt * 14.0
            cy_tgt = jiggle * (0.28 * (breath_val - 0.5) + 0.22 * energy * math.sin(t_now * 6.5))
            cx_tgt = jiggle * (-0.032 * body_x - 0.028 * body_z)
            if "Chest Y" in p:
                p["Chest Y"] = (_clamp(self.sp_fb_chest_y.step(cy_tgt, dt, 15.5, 0.26), -1.0, 1.0), 0.0)
            if "Chest X" in p:
                p["Chest X"] = (_clamp(self.sp_fb_chest_x.step(cx_tgt, dt, 13.5, 0.28), -1.0, 1.0), 0.0)
            if "Eye Jelly" in p:
                p["Eye Jelly"] = (_clamp(0.6 * squash, -0.22, 0.22), 0.0)
            if "Mouth Spring" in p and m_spring is not None:
                p["Mouth Spring"] = (m_spring, 0.0)
            self.sp_fb_arm_swing.v += jiggle * (-v_bx * 0.05) * dt * 12.0
            swing = _clamp(self.sp_fb_arm_swing.step(0.0, dt, 4.6, 0.35), -1.0, 1.0)
            if "Arm R Swing" in p:
                p["Arm R Swing"] = (swing, 0.0)
            if "Arm L Swing" in p:
                p["Arm L Swing"] = (swing, 0.0)
        self._prev_body_x = body_x
        self._prev_body_y = body_y
        self._prev_breath = breath_val

        # 11. Raw user overrides (finite-checked and clamped: a NaN must never reach the engine)
        overrides = rig_cfg.get("overrides") or {}
        if isinstance(overrides, dict):
            for pname, ov in overrides.items():
                if pname in p and isinstance(ov, (list, tuple)) and len(ov) == 2:
                    ox, oy = _ff(ov[0], float("nan")), _ff(ov[1], float("nan"))
                    if ox == ox and oy == oy:
                        p[pname] = (_clamp(ox, -1000.0, 1000.0), _clamp(oy, -1000.0, 1000.0))

        # 12. Plugin hooks
        frame_bundle = {"params": p, "camera": cam_cfg, "energy": energy, "stress": stress, "dt": dt, "time": t_now}
        self.hooks.emit_sync("on_frame_params", ctx, frame_bundle)
        self.hooks.emit_sync("on_shader_uniforms", ctx, shader_cfg)

        # 13. Final guard: fold legacy names, warn once about unknown names, drop non-finite values
        self._finalize_params(p)

        self.last_params = p
        self.last_shader = shader_cfg
        self.last_camera = cam_cfg
        self.last_state = {
            "interaction": istate, "mood": (round(mv, 3), round(ma, 3)), "armRaise": (self._raise_side if t_now < self._raise_until else ""), "breathSigh": self._breath_is_sigh, "blink": self._blink_kind if self._blink_active else "",
            "pitchRel": round(pr, 3),
        }
        return p, shader_cfg, cam_cfg

    def _finalize_params(self, p: dict[str, tuple[float, float]]) -> None:
        if len(p) != len(self.defaults):
            for old, new in LEGACY_PARAM_ALIASES.items():
                if old in p and new in self._known:
                    ov = p.pop(old)
                    try:
                        dv = float(ov[0]) - self.defaults[new][0]     # plugins wrote "default + boost"
                    except (TypeError, ValueError, IndexError):
                        dv = 0.0
                    if math.isfinite(dv):
                        cur = p.get(new, self.defaults[new])
                        p[new] = (_clamp(cur[0] + dv, -1.0, 1.0), cur[1])
            unknown = [k for k in p if k not in self._known]
            for k in unknown:
                if k not in self._warned_unknown:
                    self._warned_unknown.add(k)
                    log.warning("RigController wrote parameter %r which the loaded rig does not have (dropped)", k)
                p.pop(k, None)
        isf = math.isfinite
        for k, v in p.items():
            try:
                x, y = v
                if isf(x) and isf(y):
                    continue
                d = self.defaults[k]
                p[k] = (x if isf(x) else d[0], y if isf(y) else d[1])
            except (TypeError, ValueError):
                p[k] = self.defaults[k]
