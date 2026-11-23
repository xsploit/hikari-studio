"""60 FPS VTuber Rig State Compositor.
Combines cleanly in strict layered priority:
  1. Base idle animation (auto-breathing, randomized natural blinks, gentle figure-8 head/body sway)
  2. Mouse cursor look-at (eyes + head + subtle torso)
  3. Spring-smoothed Expression Preset (Neutral, Happy, Excited, Shy, Smug, Pout, Surprised, Sleepy)
  4. 3-Bone Rigid Toggle Arms (Rest / Right Wave / Left Gesture / Both Cheer + speech-driven gestures)
  5. Live 6-Axis VBridger Lip-Sync + speech head-nod & brow lift
  6. Motion Lab Video-Clip retargeting blend (0-100% mix)
  7. Raw user parameter overrides
  8. Plugin hooks (on_lipsync_frame, on_frame_params, on_shader_uniforms)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

from core.config import EXPRESSION_PRESETS, ROOM_AMBIENT_MAP
from core.events import HookRegistry
from modules.lipsync import DEFAULT_PARAM_MAP, LiveLipSyncEngine, Spring2nd


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


class RigController:
    def __init__(self, param_defaults: dict[str, tuple[float, float]], lipsync: LiveLipSyncEngine, hooks: HookRegistry) -> None:
        self.defaults = dict(param_defaults)
        self.lipsync = lipsync
        self.hooks = hooks
        self.lock = threading.Lock()

        # Pointer look-at normalized [-1..1] + smooth low-pass filter & blend (zero jerky snapping!)
        self.mouse_x = 0.0
        self.mouse_y = 0.0
        self._mx_filt = 0.0
        self._my_filt = 0.0
        self._mouse_blend = 0.0
        self._mouse_last_move = 0.0

        # Autonomous Natural VTuber Gaze & Eye-Leads-Head Springs
        self._gaze_tx = 0.0
        self._gaze_ty = 0.0
        self._gaze_head_tilt = 0.0
        self._next_gaze_shift = time.perf_counter() + 1.8
        self._next_micro_saccade = time.perf_counter() + 0.5
        self._micro_x = 0.0
        self._micro_y = 0.0

        self.sp_eye_x = Spring2nd(0.0)   # Fast saccade spring (eyes move first!)
        self.sp_eye_y = Spring2nd(0.0)
        self.sp_head_gx = Spring2nd(0.0) # Organic slower head follow behind eyes!
        self.sp_head_gy = Spring2nd(0.0)
        self.sp_head_gt = Spring2nd(0.0)

        # Blink scheduler
        self._next_blink = time.perf_counter() + 2.4
        self._blink_phase = 0.0
        self._blink_active = False

        # Expression springs
        self.expr_springs = {k: Spring2nd(0.0) for k in ("eyeSmile", "squint", "browY", "browForm", "mouthForm", "cheekPuff", "sparkle", "blush")}
        # Arm springs
        self.sp_arm_r_pose = Spring2nd(0.0)
        self.sp_arm_l_pose = Spring2nd(0.0)
        self.sp_arm_r_gest = Spring2nd(0.0)
        self.sp_arm_l_gest = Spring2nd(0.0)
        self.sp_arm_r_lift = Spring2nd(0.0)
        self.sp_arm_l_lift = Spring2nd(0.0)

        # Secondary Live2D Elastic Bounce Springs (Bust XY, Eye Jelly & Velocity Trackers)
        self.sp_chest_y = Spring2nd(0.0)
        self.sp_chest_x = Spring2nd(0.0)
        self.sp_eye_jelly = Spring2nd(0.0)
        self._prev_body_x = 0.0
        self._prev_body_y = 0.0
        self._prev_breath = 0.0
        self._prev_eye_open = 1.0
        self._prev_eye_x = 0.0
        self._prev_eye_y = 0.0
        self._prev_arm_sum = 0.0

        # Temporary AI / action emote hold (auto-relaxes smoothly after speech!)
        self._temp_action: dict[str, Any] | None = None
        self._temp_until: float = 0.0

        # Active Tweened Emote Clip (smooth cubic/spring animated actions!)
        self._emote_id: str = ""
        self._emote_start: float = 0.0
        self._emote_dur: float = 0.0

        # Neuro-style Procedural Idle Choreographer (scheduled living micro-behaviors every 4.5..8.5s)
        self._idle_act: str = ""
        self._idle_start: float = 0.0
        self._idle_dur: float = 0.0
        self._idle_side: float = 1.0
        self._next_idle_act: float = time.perf_counter() + random.uniform(2.5, 4.5)
        # VTS-style mouth parameter mapper + Mouth Spring (plosive / syllable-onset anime bounce)
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
        self.frame_count = 0

    def set_pointer(self, nx: float, ny: float) -> None:
        with self.lock:
            nx_c = max(-1.0, min(1.0, float(nx)))
            ny_c = max(-1.0, min(1.0, float(ny)))
            if abs(nx_c - self.mouse_x) > 0.008 or abs(ny_c - self.mouse_y) > 0.008:
                self._mouse_last_move = time.perf_counter()
            self.mouse_x = nx_c
            self.mouse_y = ny_c

    def trigger_temp_action(self, action: dict[str, Any], duration: float = 4.5) -> None:
        with self.lock:
            self._temp_action = dict(action)
            self._temp_until = time.perf_counter() + max(1.0, float(duration))

    def trigger_emote(self, emote_id: str, duration: float = 2.6) -> dict[str, Any]:
        """Trigger a smooth cubic/spring tweened Live2D emote clip with coupled bounce."""
        eid = str(emote_id or "").strip().lower()
        meta = EMOTE_CATALOG.get(eid)
        if meta is None:
            return {"ok": False, "error": f"Unknown emote: {eid}", "available": sorted(EMOTE_CATALOG.keys())}
        with self.lock:
            self._emote_id = eid
            self._emote_start = time.perf_counter()
            self._emote_dur = float(duration) if duration and duration != 2.6 else float(meta["duration"])
            # Instant spring kick on start for juicy Live2D punch (chest Y, chest X, eye jelly)!
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
            self._motion_mix = max(0.0, min(1.0, float(mix)))
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
                self._motion_mix = max(0.0, min(1.0, float(mix)))

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

    def step(self, dt: float, cfg: dict[str, Any], ctx: Any) -> tuple[dict[str, tuple[float, float]], dict[str, Any], dict[str, Any]]:
        t_now = time.perf_counter()
        self.frame_count += 1

        rig_cfg = cfg["rig"]
        cam_cfg = dict(cfg["camera"])
        shader_cfg = dict(cfg["shader"])
        stage_cfg = cfg["stage"]
        lip_cfg = cfg["lipsync"]

        # Auto-match room lighting if enabled
        if stage_cfg.get("autoMatchLighting", True):
            rm = ROOM_AMBIENT_MAP.get(stage_cfg.get("room", ""), None)
            if rm:
                shader_cfg["ambientColor"] = rm["ambientColor"]

        # 1. Step Live 6-Axis Lip-Sync & fire hook
        lip = self.lipsync.step(dt, t_now)
        self.hooks.emit_sync("on_lipsync_frame", ctx, lip)
        energy = lip["energy"]
        stress = lip["stress"]
        speaking = energy > 0.06 or bool(lip_cfg.get("holdViseme"))

        # Start from clean neutral defaults
        p: dict[str, tuple[float, float]] = dict(self.defaults)

        # 2. Check active temporary AI/emote action & tweened emote clip
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

        # 3. Auto-Breathing & Natural Idle Sway
        breath_val = 0.0
        if rig_cfg.get("autoBreath", True):
            b_amp = float(rig_cfg.get("breathAmp", 1.0))
            breath_val = (0.5 + 0.5 * math.sin(t_now * 1.85)) * b_amp
        p["Breath"] = (max(0.0, min(1.0, breath_val)), 0.0)

        sway = float(rig_cfg.get("idleSway", 1.0))
        head_x = sway * (4.6 * math.sin(t_now * 0.58) + 2.1 * math.sin(t_now * 1.23))
        head_y = sway * (3.2 * math.sin(t_now * 0.74 + 0.6))
        head_z = sway * (3.8 * math.sin(t_now * 0.51 + 1.1))
        body_x = sway * (3.1 * math.sin(t_now * 0.45 + 0.3))
        body_y = sway * (2.3 * math.sin(t_now * 1.85 - 0.35))
        body_z = sway * (2.8 * math.sin(t_now * 0.42 + 0.9))
        body_lean = sway * (2.0 * math.sin(t_now * 0.61))

        # 3.5 Neuro-style Scheduled Live Idle Choreographer (periodic organic VTuber micro-behaviors!)
        idle_brow_boost = 0.0
        idle_smile_boost = 0.0
        if sway > 0.05 and not em_id and m_clip is None:
            if not self._idle_act and not speaking and t_now >= self._next_idle_act:
                choices = ("curious_tilt", "vibe_sway", "happy_bounce", "chat_read", "daydream_sigh", "cheeky_lean")
                self._idle_act = random.choice(choices)
                self._idle_start = t_now
                self._idle_side = -1.0 if random.random() < 0.5 else 1.0
                self._idle_dur = {
                    "curious_tilt": random.uniform(2.4, 3.4),
                    "vibe_sway":    random.uniform(3.2, 4.4),
                    "happy_bounce": random.uniform(1.8, 2.5),
                    "chat_read":    random.uniform(2.6, 3.6),
                    "daydream_sigh":random.uniform(3.0, 4.0),
                    "cheeky_lean":  random.uniform(2.2, 3.1),
                }[self._idle_act]
            if self._idle_act:
                iu = (t_now - self._idle_start) / max(0.1, self._idle_dur)
                if iu >= 1.0:
                    self._idle_act = ""
                    self._next_idle_act = t_now + random.uniform(3.8, 7.5) / max(0.35, min(2.0, sway))
                else:
                    # Silky smoothstep fade-in / hold / fade-out envelope
                    env = math.sin(iu * math.pi) ** 1.25
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

        # 4. Autonomous Natural VTuber Gaze + Smart Hybrid Mouse Follow (Eye-Leads-Head!)
        auto_gaze = bool(rig_cfg.get("autoGaze", True))
        liveliness = max(0.0, float(rig_cfg.get("autoGazeLiveliness", 1.0)))
        mouse_on = bool(rig_cfg.get("mouseLookAt", True))
        mouse_recent = mouse_on and ((t_now - self._mouse_last_move) < 2.2)

        if auto_gaze:
            # Tiny realistic eye micro-saccades every 0.35..0.95s
            if t_now >= self._next_micro_saccade:
                self._next_micro_saccade = t_now + random.uniform(0.35, 0.95)
                self._micro_x = random.uniform(-0.07, 0.07) * liveliness
                self._micro_y = random.uniform(-0.05, 0.05) * liveliness

            # Organic focal shifts: 62% warm eye contact, 22% thinking glance up-side, 16% curious room glance
            if t_now >= self._next_gaze_shift:
                r_pick = random.random()
                old_tx = self._gaze_tx
                if r_pick < 0.62:
                    # Warm eye contact with viewer (slight natural offset, holds 2.0..4.5s)
                    self._gaze_tx = random.uniform(-0.14, 0.14) * liveliness
                    self._gaze_ty = random.uniform(-0.10, 0.08) * liveliness
                    self._gaze_head_tilt = random.uniform(-0.22, 0.22) * liveliness
                    self._next_gaze_shift = t_now + random.uniform(2.0, 4.4)
                elif r_pick < 0.84:
                    # Thinking / daydreaming glance slightly up-left or up-right (holds 1.1..2.3s)
                    side = -1.0 if random.random() < 0.5 else 1.0
                    self._gaze_tx = side * random.uniform(0.38, 0.68) * liveliness
                    self._gaze_ty = random.uniform(-0.48, -0.20) * liveliness
                    self._gaze_head_tilt = side * random.uniform(0.25, 0.55) * liveliness
                    self._next_gaze_shift = t_now + random.uniform(1.1, 2.3)
                else:
                    # Curious room glance left/right (holds 1.2..2.5s)
                    side = -1.0 if random.random() < 0.5 else 1.0
                    self._gaze_tx = side * random.uniform(0.42, 0.78) * liveliness
                    self._gaze_ty = random.uniform(-0.08, 0.22) * liveliness
                    self._gaze_head_tilt = -side * random.uniform(0.15, 0.40) * liveliness
                    self._next_gaze_shift = t_now + random.uniform(1.2, 2.4)

                # Natural blink synced to larger gaze shifts (real human vestibulo-ocular behavior!)
                if rig_cfg.get("autoBlink", True) and not self._blink_active and abs(self._gaze_tx - old_tx) > 0.42 and random.random() < 0.65:
                    self._blink_active = True
                    self._blink_phase = 0.0

        # Silky-smooth low-pass mouse filter + gradual blend (eliminates ALL jerky snap/twitch!)
        lp = 1.0 - math.exp(-8.5 * max(0.001, dt))
        self._mx_filt += (mx_t - self._mx_filt) * lp
        self._my_filt += (my_t - self._my_filt) * lp

        if mouse_recent:
            self._mouse_blend = min(1.0, self._mouse_blend + dt * 2.6)
        else:
            self._mouse_blend = max(0.0, self._mouse_blend - dt * 1.05)

        ml = float(rig_cfg.get("mouseLookStrength", 0.55))
        m_gx = self._mx_filt * ml
        m_gy = self._my_filt * ml
        m_tilt = self._mx_filt * 0.22 * ml

        if auto_gaze:
            a_gx = max(-0.95, min(0.95, self._gaze_tx + self._micro_x))
            a_gy = max(-0.85, min(0.85, self._gaze_ty + self._micro_y))
            a_tilt = self._gaze_head_tilt
        else:
            a_gx, a_gy, a_tilt = 0.0, 0.0, 0.0

        # Scale mouse dominance by mouseLookStrength so weak mouse follow never freezes autonomous gaze!
        mb_raw = self._mouse_blend * self._mouse_blend * (3.0 - 2.0 * self._mouse_blend)
        mb = mb_raw * min(1.0, max(0.0, ml) * 2.2)
        tgt_gx = a_gx * (1.0 - mb) + m_gx * mb
        tgt_gy = a_gy * (1.0 - mb) + m_gy * mb
        tgt_tilt = a_tilt * (1.0 - mb) + m_tilt * mb

        # Critically-damped silky springs when tracking mouse (fast saccades only for autonomous gaze!)
        eye_om = 17.0 * (1.0 - mb) + 8.5 * mb
        eye_z = 0.90 * (1.0 - mb) + 0.98 * mb
        head_om = 5.6 * (1.0 - mb) + 3.8 * mb
        head_z = 0.90 * (1.0 - mb) + 0.98 * mb
        ex_s = self.sp_eye_x.step(tgt_gx, dt, eye_om, eye_z)
        ey_s = self.sp_eye_y.step(tgt_gy, dt, eye_om, eye_z)
        hx_s = self.sp_head_gx.step(tgt_gx, dt, head_om, head_z)
        hy_s = self.sp_head_gy.step(tgt_gy, dt, head_om, head_z)
        ht_s = self.sp_head_gt.step(tgt_tilt, dt, 3.6, 0.96)

        head_x += hx_s * 15.5
        head_y += -hy_s * 11.5
        head_z += ht_s * 9.5 + hx_s * 3.2
        body_x += hx_s * 4.8
        body_z += ht_s * 2.8
        eye_x = max(-1.0, min(1.0, ex_s * 0.88))
        eye_y = max(-1.0, min(1.0, -ey_s * 0.74))

        # Speech cadence head-nod, subtle reactive eye-smile & brow emphasis (Reactive V2)
        nod_gain = float(lip_cfg.get("speechToHeadNod", 0.80))
        if energy > 0.04 and nod_gain > 0.0:
            head_y += nod_gain * (3.8 * energy * math.sin(t_now * 5.2) + 5.2 * stress)
            head_z += nod_gain * (3.4 * energy * math.sin(t_now * 2.5 + 0.5) + 2.6 * stress * math.cos(t_now * 1.9))
            body_y += nod_gain * (2.0 * stress)
            body_z += nod_gain * (1.4 * energy * math.sin(t_now * 2.2))

        # 5. Auto-Blink scheduler
        eye_open_l = 1.0
        eye_open_r = 1.0
        if rig_cfg.get("autoBlink", True):
            rate = max(0.2, float(rig_cfg.get("blinkRate", 1.0)))
            if not self._blink_active and t_now >= self._next_blink:
                self._blink_active = True
                self._blink_phase = 0.0
            if self._blink_active:
                self._blink_phase += dt / 0.165
                if self._blink_phase >= 1.0:
                    self._blink_active = False
                    # 18% chance of cute natural double-blink
                    gap = 0.24 if random.random() < 0.18 else random.uniform(2.1, 4.8) / rate
                    self._next_blink = t_now + gap
                else:
                    # Smooth close-open curve
                    b = math.sin(self._blink_phase * math.pi) ** 1.6
                    eye_open_l = eye_open_r = max(0.0, 1.0 - b)

        # 6. Expression Preset Springs + manual sliders
        expr_name = (temp.get("expression") if temp else None) or str(rig_cfg.get("expression", "happy"))
        expr_tgt = EXPRESSION_PRESETS.get(expr_name, EXPRESSION_PRESETS["neutral"])
        ev = {k: self.expr_springs[k].step(expr_tgt[k], dt, 11.0, 0.86) for k in self.expr_springs}

        # Combine expression + manual FX sliders + temp AI boost
        sparkle = max(0.0, min(1.0, max(ev["sparkle"], float(rig_cfg.get("sparkleEyes", 0.0)), float(temp["sparkle"]) if temp and "sparkle" in temp else 0.0)))
        blush = max(0.0, min(1.0, max(ev["blush"], float(rig_cfg.get("blushLines", 0.0)), float(temp["blush"]) if temp and "blush" in temp else 0.0)))
        cheek_puff = max(0.0, min(1.0, max(ev["cheekPuff"], float(rig_cfg.get("cheekPuff", 0.0)))))
        squint = max(0.0, min(1.0, max(ev["squint"], float(rig_cfg.get("eyeSquint", 0.0))) + 0.14 * energy))
        brow_speech = float(lip_cfg.get("speechToBrows", 0.72)) * (0.35 * energy + 0.50 * stress)

        # 7. 3-Bone Rigid Toggle Arms + Speech Auto-Gesture
        t_arm_r = float(temp["armR"]) if temp and "armR" in temp else float(rig_cfg.get("armRPose", 0.0))
        t_arm_l = float(temp["armL"]) if temp and "armL" in temp else float(rig_cfg.get("armLPose", 0.0))
        arm_r_pose = max(0.0, min(1.0, self.sp_arm_r_pose.step(t_arm_r, dt, 22.0, 0.92)))
        arm_l_pose = max(0.0, min(1.0, self.sp_arm_l_pose.step(t_arm_l, dt, 22.0, 0.92)))

        g_r_tgt = float(rig_cfg.get("armRGesture", 0.0))
        g_l_tgt = float(rig_cfg.get("armLGesture", 0.0))
        l_r_tgt = float(rig_cfg.get("armRLift", 0.0))
        l_l_tgt = float(rig_cfg.get("armLLift", 0.0))

        # Playful live wave / conversational gesture whenever an arm is raised!
        if arm_r_pose > 0.5:
            if temp and temp.get("armR", 0) > 0.5:
                g_r_tgt += 0.25 + 0.55 * math.sin(t_now * 7.2)
            elif rig_cfg.get("autoGestureOnSpeech", True) and speaking:
                ag = float(lip_cfg.get("speechToArmGesture", 0.75))
                g_r_tgt += ag * (0.18 + 0.36 * math.sin(t_now * 3.1) + 0.42 * stress)
                l_r_tgt += ag * (0.30 * energy)
            else:
                g_r_tgt += 0.14 * math.sin(t_now * 1.9)
        if arm_l_pose > 0.5:
            if temp and temp.get("armL", 0) > 0.5:
                g_l_tgt += 0.22 + 0.52 * math.sin(t_now * 6.8 + 0.6)
            elif rig_cfg.get("autoGestureOnSpeech", True) and speaking:
                ag = float(lip_cfg.get("speechToArmGesture", 0.75))
                g_l_tgt += ag * (0.18 + 0.36 * math.sin(t_now * 2.8 + 0.9) + 0.42 * stress)
                l_l_tgt += ag * (0.30 * energy)
            else:
                g_l_tgt += 0.14 * math.sin(t_now * 1.7 + 0.8)

        arm_r_gest = max(-0.48, min(0.92, self.sp_arm_r_gest.step(g_r_tgt, dt, 14.0, 0.85)))
        arm_l_gest = max(-0.48, min(0.92, self.sp_arm_l_gest.step(g_l_tgt, dt, 14.0, 0.85)))
        arm_r_lift = max(-0.75, min(0.85, self.sp_arm_r_lift.step(l_r_tgt, dt, 11.0, 0.88)))
        arm_l_lift = max(-0.75, min(0.85, self.sp_arm_l_lift.step(l_l_tgt, dt, 11.0, 0.88)))

        # 8. Motion Lab Video-Clip Retargeting Blend (Head, Body, Blinks & Arms ONLY — never overrides live VBridger mouth!)
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
                arm_r_gest = max(-0.48, min(0.92, arm_r_gest + (tracks["ArmR"][fi] - 0.35) * 0.9 * w))
            if tracks["ArmL"][fi] > 0.35 and arm_l_pose > 0.5:
                arm_l_gest = max(-0.48, min(0.92, arm_l_gest + (tracks["ArmL"][fi] - 0.35) * 0.9 * w))

        # 9. Assign final native Inochi2D v28 parameters!
        # 9.0 VTube Studio / VBridger mouth parameter mapping (every lip-sync channel is remapped)
        pmap_cfg = lip_cfg.get("paramMap") if isinstance(lip_cfg.get("paramMap"), dict) else {}
        default_map = DEFAULT_PARAM_MAP

        def _chan(name: str, raw: float) -> float | None:
            m = pmap_cfg.get(name)
            if not isinstance(m, dict):
                m = default_map.get(name)
            elif name in default_map:
                m = {**default_map[name], **m}
            return self.mouth_mapper.map(name, raw, dt, m)

        form_src = lip["form"] if speaking else ev["mouthForm"]
        m_open = _chan("mouthOpen", lip["open"])
        m_form = _chan("mouthForm", form_src)
        m_jaw = _chan("jawOpen", lip["open"])
        m_funnel = _chan("mouthFunnel", lip["funnel"])
        m_press = _chan("lipsPress", float(lip.get("press", 0.0)))
        m_shrug = _chan("mouthShrug", lip["shrug"])
        m_x = _chan("mouthX", lip["x"])
        m_tongue = _chan("tongueOut", lip["tongue"])

        # Mouth Spring: underdamped bounce kicked by plosive releases / syllable onsets + jaw snaps
        onset = float(lip.get("onset", 0.0))
        v_open = (lip["open"] - self._prev_lip_open) / max(0.008, dt)
        self._prev_lip_open = lip["open"]
        bounce = float(lip_cfg.get("jawBounce", 1.0))
        self.sp_mouth_spring.v += bounce * (onset * 260.0 + max(-6.0, min(6.0, v_open)) * 1.6) * dt
        m_spring = _chan("mouthSpring", self.sp_mouth_spring.step(0.0, dt, 19.0, 0.28))

        self.last_mapped = {
            "mouthOpen": m_open, "mouthForm": m_form, "jawOpen": m_jaw, "mouthFunnel": m_funnel,
            "lipsPress": m_press, "mouthShrug": m_shrug, "mouthX": m_x, "tongueOut": m_tongue,
            "mouthSpring": m_spring,
        }
        # Disabled channel -> lip-sync stops driving it: rest pose (form falls back to the expression)
        vb_open = m_open if m_open is not None else 0.0
        vb_form = m_form if m_form is not None else max(-1.0, min(1.0, ev["mouthForm"]))
        vb_tongue = max(m_tongue if m_tongue is not None else 0.0, float(rig_cfg.get("tongueOut", 0.0)))

        p["Angle"] = (max(-30.0, min(30.0, head_x)), max(-24.0, min(24.0, head_y)))
        p["Angle Z"] = (max(-25.0, min(25.0, head_z)), 0.0)
        p["Body Angle X"] = (max(-10.0, min(10.0, body_x)), 0.0)
        p["Body Angle Y"] = (max(-10.0, min(10.0, body_y)), 0.0)
        p["Body Angle Z"] = (max(-10.0, min(10.0, body_z)), 0.0)
        p["Body Lean"] = (max(-10.0, min(10.0, body_lean)), 0.0)

        p["EyeBall"] = (eye_x, eye_y)
        p["Left Eye Open"] = (max(0.0, min(1.0, eye_open_l)), 0.0)
        p["Right Eye Open"] = (max(0.0, min(1.0, eye_open_r)), 0.0)
        p["Eye L Smile"] = (max(0.0, min(1.0, ev["eyeSmile"] + idle_smile_boost)), 0.0)
        p["Eye R Smile"] = (max(0.0, min(1.0, ev["eyeSmile"] + idle_smile_boost)), 0.0)
        p["Eye L Squint"] = (squint, 0.0)
        p["Eye R Squint"] = (squint, 0.0)
        p["Brow L Y"] = (max(-1.0, min(1.0, ev["browY"] + brow_speech + idle_brow_boost)), 0.0)
        p["Brow R Y"] = (max(-1.0, min(1.0, ev["browY"] + brow_speech + idle_brow_boost)), 0.0)
        p["Brow L Form"] = (max(-1.0, min(1.0, ev["browForm"])), 0.0)
        p["Brow R Form"] = (max(-1.0, min(1.0, ev["browForm"])), 0.0)

        p["VB Mouth"] = (max(-1.0, min(1.0, vb_form)), max(0.0, min(1.0, vb_open)))
        p["Mouth Funnel"] = (m_funnel if m_funnel is not None else self.defaults.get("Mouth Funnel", (0.0, 0.0))[0], 0.0)
        p["Mouth X"] = (m_x if m_x is not None else self.defaults.get("Mouth X", (0.0, 0.0))[0], 0.0)
        p["Mouth Shrug"] = (m_shrug if m_shrug is not None else self.defaults.get("Mouth Shrug", (0.0, 0.0))[0], 0.0)
        if "Lips Press / Part" in p:
            p["Lips Press / Part"] = (m_press if m_press is not None else self.defaults["Lips Press / Part"][0], 0.0)
        p["Tongue Out"] = (max(0.0, min(1.0, vb_tongue)), 0.0)
        p["Jaw Open"] = (m_jaw if m_jaw is not None else self.defaults.get("Jaw Open", (0.0, 0.0))[0], 0.0)
        if "Mouth Spring" in p:
            p["Mouth Spring"] = (m_spring if m_spring is not None else self.defaults["Mouth Spring"][0], 0.0)
        p["Cheek Puff"] = (cheek_puff, 0.0)

        p["Arm R Pose"] = (arm_r_pose, 0.0)
        p["Arm R Gesture"] = (arm_r_gest, 0.0)
        p["Arm R Lift"] = (arm_r_lift, 0.0)
        p["Arm L Pose"] = (arm_l_pose, 0.0)
        p["Arm L Gesture"] = (arm_l_gest, 0.0)
        p["Arm L Lift"] = (arm_l_lift, 0.0)

        p["Sparkle Eyes"] = (sparkle, 0.0)
        p["Anime Blush Lines"] = (blush, 0.0)

        # 9.2 Apply Smooth Cubic/Spring Tweened Emote Clips & Transforms (Chibi / Gloom / Emissive Glow)
        chibi_tgt = max(0.0, min(1.0, float(rig_cfg.get("chibiMode", 0.0))))
        gloom_tgt = max(0.0, min(1.0, float(rig_cfg.get("gloom", 0.0))))
        em_glow_boost = 0.0

        if em_id and em_dur > 0.1:
            u_t = max(0.0, min(1.0, em_el / em_dur))
            # Smooth cubic envelope: quick ease-in (0..0.16), hold, silky ease-out (0.72..1.0)
            env_in = min(1.0, u_t / 0.16)
            env_in = env_in * env_in * (3.0 - 2.0 * env_in)
            env_out = min(1.0, (1.0 - u_t) / 0.28)
            env_out = env_out * env_out * (3.0 - 2.0 * env_out)
            env = env_in * env_out

            if em_id == "excited_hop":
                hop = math.sin(u_t * math.pi * 4.0)
                body_y += 8.2 * env * hop
                head_y += 9.0 * env * hop
                head_z += 12.0 * env * math.sin(u_t * math.pi * 2.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.92 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.92 * env)), 0.0)
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 1.0 * env)), 0.0)
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], env), 0.0)
                p["Arm L Pose"] = (max(p["Arm L Pose"][0], env), 0.0)
                p["Arm R Gesture"] = (0.45 * math.sin(u_t * math.pi * 6.0) * env, 0.0)
                p["Arm L Gesture"] = (0.45 * math.sin(u_t * math.pi * 6.0 + 0.5) * env, 0.0)
                em_glow_boost = 0.65 * env
            elif em_id == "shy_blush":
                head_x -= 11.0 * env
                head_y -= 7.5 * env
                head_z += 11.0 * env
                body_lean += 4.5 * env
                p["EyeBall"] = (max(-1.0, eye_x - 0.35 * env), max(-1.0, eye_y - 0.28 * env))
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 1.0 * env)), 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.55 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.55 * env)), 0.0)
                p["Brow L Y"] = (min(1.0, max(-1.0, p["Brow L Y"][0] - 0.25 * env)), 0.0)
                p["Brow R Y"] = (min(1.0, max(-1.0, p["Brow R Y"][0] - 0.25 * env)), 0.0)
            elif em_id == "wink_wave":
                head_z += 10.5 * env
                head_x += 7.0 * env
                p["Right Eye Open"] = (max(0.0, p["Right Eye Open"][0] * (1.0 - env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.95 * env)), 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.55 * env)), 0.0)
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], env), 0.0)
                p["Arm R Gesture"] = (0.30 + 0.55 * math.sin(u_t * math.pi * 6.0) * env, 0.0)
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 0.75 * env)), 0.0)
                em_glow_boost = 0.40 * env
            elif em_id == "gloom_shake":
                shiver = math.sin(u_t * math.pi * 24.0) * env
                head_x += 5.5 * shiver
                head_y -= 10.0 * env
                body_y -= 4.0 * env
                gloom_tgt = max(gloom_tgt, env)
                p["Brow L Form"] = (-0.85 * env, 0.0)
                p["Brow R Form"] = (-0.85 * env, 0.0)
                em_glow_boost = -0.55 * env
            elif em_id == "curious_lean":
                head_z -= 16.0 * env
                head_y += 4.0 * env
                body_lean += 7.5 * env
                body_z -= 4.5 * env
                p["Brow L Y"] = (min(1.0, p["Brow L Y"][0] + 0.55 * env), 0.0)
                p["Brow R Y"] = (min(1.0, p["Brow R Y"][0] + 0.20 * env), 0.0)
                em_glow_boost = 0.25 * env
            elif em_id == "chibi_pop":
                chibi_tgt = max(chibi_tgt, env)
                body_y += 5.5 * math.sin(u_t * math.pi * 3.0) * env
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 0.90 * env)), 0.0)
                em_glow_boost = 0.50 * env
            elif em_id == "pout_puff":
                head_x += 12.0 * env
                head_z -= 8.0 * env
                p["Cheek Puff"] = (min(1.0, max(p["Cheek Puff"][0], 1.0 * env)), 0.0)
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 0.65 * env)), 0.0)
                p["Brow L Form"] = (-0.75 * env, 0.0)
                p["Brow R Form"] = (-0.75 * env, 0.0)
                p["VB Mouth"] = (-0.65 * env, p["VB Mouth"][1])
            elif em_id == "heart_sparkle":
                head_y += 5.0 * env
                head_z += 7.5 * math.sin(u_t * math.pi * 2.0) * env
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 1.0 * env)), 0.0)
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 0.85 * env)), 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.72 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.72 * env)), 0.0)
                em_glow_boost = 0.95 * env

            elif em_id == "greet_wave":
                # Four big waves, head tilting into the wave, tiny happy bounce
                wave = math.sin(u_t * math.pi * 8.0)
                head_z += 8.0 * env + 2.5 * wave * env
                head_x += 4.0 * env
                body_y += 3.0 * abs(math.sin(u_t * math.pi * 4.0)) * env
                body_z += 2.0 * env
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], env), 0.0)
                p["Arm R Lift"] = (max(p["Arm R Lift"][0], 0.45 * env), 0.0)
                p["Arm R Gesture"] = ((0.20 + 0.62 * wave) * env, 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.80 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.80 * env)), 0.0)
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 0.50 * env)), 0.0)
                em_glow_boost = 0.45 * env
            elif em_id == "cheer_bounce":
                bounce = abs(math.sin(u_t * math.pi * 5.0))
                pump = math.sin(u_t * math.pi * 10.0)
                body_y += 9.0 * bounce * env
                head_y += 7.0 * bounce * env
                head_z += 6.0 * math.sin(u_t * math.pi * 2.5) * env
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], env), 0.0)
                p["Arm L Pose"] = (max(p["Arm L Pose"][0], env), 0.0)
                p["Arm R Lift"] = (max(p["Arm R Lift"][0], (0.55 + 0.25 * pump) * env), 0.0)
                p["Arm L Lift"] = (max(p["Arm L Lift"][0], (0.55 - 0.25 * pump) * env), 0.0)
                p["Arm R Gesture"] = (0.35 * pump * env, 0.0)
                p["Arm L Gesture"] = (-0.35 * pump * env, 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 1.0 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 1.0 * env)), 0.0)
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 1.0 * env)), 0.0)
                em_glow_boost = 0.80 * env
            elif em_id == "smug_tease":
                # Asymmetric brow, squinty smirk, little "heh heh" bob, leaning back
                heh = math.sin(u_t * math.pi * 6.0) * _smooth(u_t / 0.3) * (1.0 - _smooth((u_t - 0.5) / 0.3))
                head_z -= 9.0 * env
                head_x += 6.0 * env
                head_y += 2.5 * heh * env
                body_lean -= 3.0 * env
                p["Brow L Y"] = (min(1.0, p["Brow L Y"][0] + 0.50 * env), 0.0)
                p["Brow R Y"] = (max(-1.0, p["Brow R Y"][0] - 0.15 * env), 0.0)
                p["Eye L Squint"] = (min(1.0, max(p["Eye L Squint"][0], 0.55 * env)), 0.0)
                p["Eye R Squint"] = (min(1.0, max(p["Eye R Squint"][0], 0.55 * env)), 0.0)
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.45 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.45 * env)), 0.0)
                if not speaking:
                    p["VB Mouth"] = (max(p["VB Mouth"][0], 0.60 * env), p["VB Mouth"][1])
                    p["Mouth X"] = (max(-1.0, min(1.0, p["Mouth X"][0] + 0.35 * env)), 0.0)
                em_glow_boost = 0.20 * env
            elif em_id == "starry_awe":
                head_y += 8.0 * env
                head_z += 6.0 * math.sin(u_t * math.pi * 2.0) * env
                body_lean -= 2.0 * env
                p["EyeBall"] = (p["EyeBall"][0] * (1.0 - 0.6 * env), min(1.0, p["EyeBall"][1] + 0.45 * env))
                p["Brow L Y"] = (min(1.0, p["Brow L Y"][0] + 0.70 * env), 0.0)
                p["Brow R Y"] = (min(1.0, p["Brow R Y"][0] + 0.70 * env), 0.0)
                p["Sparkle Eyes"] = (min(1.0, max(p["Sparkle Eyes"][0], 1.0 * env)), 0.0)
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 0.35 * env)), 0.0)
                if not speaking:
                    p["VB Mouth"] = (-0.20 * env + p["VB Mouth"][0] * (1.0 - env), max(p["VB Mouth"][1], 0.32 * env))
                    p["Mouth Funnel"] = (max(p["Mouth Funnel"][0], 0.45 * env), 0.0)
                    p["Jaw Open"] = (max(p["Jaw Open"][0], 0.25 * env), 0.0)
                em_glow_boost = 0.85 * env
            elif em_id == "flustered_shy":
                shake = math.sin(u_t * math.pi * 18.0) * env
                dart = math.sin(u_t * math.pi * 8.0)
                head_x += 4.5 * shake - 6.0 * env
                head_y -= 9.0 * env
                head_z += 7.0 * env
                body_lean += 3.5 * env
                p["EyeBall"] = (max(-1.0, min(1.0, 0.40 * dart * env + p["EyeBall"][0] * (1.0 - env))), max(-1.0, p["EyeBall"][1] - 0.25 * env))
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 1.0 * env)), 0.0)
                p["Eye L Squint"] = (min(1.0, max(p["Eye L Squint"][0], 0.50 * env)), 0.0)
                p["Eye R Squint"] = (min(1.0, max(p["Eye R Squint"][0], 0.50 * env)), 0.0)
                p["Cheek Puff"] = (min(1.0, max(p["Cheek Puff"][0], 0.35 * env)), 0.0)
                p["Brow L Form"] = (-0.45 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.45 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                p["Brow L Y"] = (min(1.0, p["Brow L Y"][0] + 0.30 * env), 0.0)
                p["Brow R Y"] = (min(1.0, p["Brow R Y"][0] + 0.30 * env), 0.0)
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], env), 0.0)
                p["Arm L Pose"] = (max(p["Arm L Pose"][0], env), 0.0)
                p["Arm R Gesture"] = (0.30 * math.sin(u_t * math.pi * 14.0) * env, 0.0)
                p["Arm L Gesture"] = (0.30 * math.sin(u_t * math.pi * 14.0 + 1.2) * env, 0.0)
                em_glow_boost = 0.30 * env
            elif em_id == "sleepy_yawn":
                # Phase 1 (0..0.55): stretch-yawn. Phase 2: droopy, heavy-lidded settle.
                yawn = math.sin(math.pi * max(0.0, min(1.0, u_t / 0.55)))
                droop = _smooth((u_t - 0.45) / 0.25) * env
                head_y += 7.0 * yawn * env - 6.0 * droop
                head_z += 5.0 * droop
                body_y += 3.0 * yawn * env
                lid = max(0.0, 1.0 - 0.85 * yawn * env - 0.50 * droop)
                p["Left Eye Open"] = (p["Left Eye Open"][0] * lid, 0.0)
                p["Right Eye Open"] = (p["Right Eye Open"][0] * lid, 0.0)
                p["Eye L Squint"] = (min(1.0, max(p["Eye L Squint"][0], 0.60 * droop)), 0.0)
                p["Eye R Squint"] = (min(1.0, max(p["Eye R Squint"][0], 0.60 * droop)), 0.0)
                p["Brow L Y"] = (max(-1.0, p["Brow L Y"][0] - 0.35 * env), 0.0)
                p["Brow R Y"] = (max(-1.0, p["Brow R Y"][0] - 0.35 * env), 0.0)
                p["Arm R Pose"] = (max(p["Arm R Pose"][0], yawn * env), 0.0)
                p["Arm L Pose"] = (max(p["Arm L Pose"][0], yawn * env), 0.0)
                p["Arm R Lift"] = (max(p["Arm R Lift"][0], 0.50 * yawn * env), 0.0)
                p["Arm L Lift"] = (max(p["Arm L Lift"][0], 0.50 * yawn * env), 0.0)
                if not speaking:
                    p["VB Mouth"] = (-0.10 * yawn * env + p["VB Mouth"][0] * (1.0 - yawn * env), max(p["VB Mouth"][1], 0.85 * yawn * env))
                    p["Jaw Open"] = (max(p["Jaw Open"][0], 0.75 * yawn * env), 0.0)
                    p["Mouth Funnel"] = (max(p["Mouth Funnel"][0], 0.35 * yawn * env), 0.0)
                em_glow_boost = -0.25 * env
            elif em_id == "gentle_nod":
                nod = math.sin(u_t * math.pi * 4.0) ** 2
                head_y -= 6.0 * nod * env
                head_z += 7.0 * env
                body_lean += 2.0 * env
                p["Eye L Smile"] = (min(1.0, max(p["Eye L Smile"][0], 0.50 * env)), 0.0)
                p["Eye R Smile"] = (min(1.0, max(p["Eye R Smile"][0], 0.50 * env)), 0.0)
                p["Brow L Form"] = (-0.30 * env + p["Brow L Form"][0] * (1.0 - env), 0.0)
                p["Brow R Form"] = (-0.30 * env + p["Brow R Form"][0] * (1.0 - env), 0.0)
                p["Brow L Y"] = (min(1.0, p["Brow L Y"][0] + 0.15 * env), 0.0)
                p["Brow R Y"] = (min(1.0, p["Brow R Y"][0] + 0.15 * env), 0.0)
                p["Anime Blush Lines"] = (min(1.0, max(p["Anime Blush Lines"][0], 0.30 * env)), 0.0)
                em_glow_boost = 0.15 * env

            # Re-apply updated head/body angles from emote clip
            p["Angle"] = (max(-30.0, min(30.0, head_x)), max(-24.0, min(24.0, head_y)))
            p["Angle Z"] = (max(-25.0, min(25.0, head_z)), 0.0)
            p["Body Angle Y"] = (max(-10.0, min(10.0, body_y)), 0.0)
            p["Body Angle Z"] = (max(-10.0, min(10.0, body_z)), 0.0)
            p["Body Lean"] = (max(-10.0, min(10.0, body_lean)), 0.0)

        chibi_val = max(0.0, min(1.0, self.sp_chibi.step(chibi_tgt, dt, 12.0, 0.55)))
        gloom_val = max(0.0, min(1.0, self.sp_gloom.step(gloom_tgt, dt, 10.0, 0.80)))
        if "Proportion (Chibi)" in p:
            p["Proportion (Chibi)"] = (chibi_val, 0.0)
        if "Gloom" in p:
            p["Gloom"] = (gloom_val, 0.0)

        # Live GPU Emissive Glow (pulses subtly with breath, speech energy, sparkle eyes & emotes!)
        em_base = max(0.0, min(2.0, float(rig_cfg.get("emissiveGlow", 1.0))))
        em_tgt = max(0.0, min(2.0, em_base * (0.90 + 0.10 * math.sin(t_now * 2.4) + 0.30 * energy + 0.32 * p["Sparkle Eyes"][0] + em_glow_boost)))
        em_val = max(0.0, min(2.0, self.sp_emissive.step(em_tgt, dt, 10.0, 0.85)))
        if "Emissive Glow" in p:
            p["Emissive Glow"] = (em_val, 0.0)

        # 9.5 Coupled 60 FPS Underdamped Spring-Bounce (Bust XY + Eye Jelly)
        if rig_cfg.get("physicsEnabled", True):
            jiggle = max(0.0, min(2.5, float(rig_cfg.get("jiggleScale", 1.25))))
            inv_dt = 1.0 / max(0.008, min(0.05, dt))
            v_bx = (body_x - self._prev_body_x) * inv_dt
            v_by = (body_y - self._prev_body_y) * inv_dt
            v_br = (breath_val - self._prev_breath) * inv_dt
            arm_sum = arm_r_pose + arm_l_pose + 0.5 * (arm_r_gest + arm_l_gest)
            v_arm = (arm_sum - self._prev_arm_sum) * inv_dt
            v_eo = (eye_open_r - self._prev_eye_open) * inv_dt
            v_eye = math.hypot(eye_x - self._prev_eye_x, eye_y - self._prev_eye_y) * inv_dt

            # Inject natural inertia impulses into underdamped springs (zeta=0.26 -> juicy Live2D bounce!)
            self.sp_chest_y.v += jiggle * (v_by * 0.42 + v_br * 2.4 + abs(v_arm) * 0.48 + stress * 2.8) * dt * 16.0
            self.sp_chest_x.v += jiggle * (-v_bx * 0.36 + (arm_r_gest - arm_l_gest) * 0.12 - v_arm * 0.22) * dt * 14.0
            # Subtle eye jelly micro-settle on blink reopen & fast eye saccades (heavily damped by default)
            eye_kick = (max(0.0, v_eo) * 0.10 + v_eye * 0.04) * min(1.0, jiggle)
            self.sp_eye_jelly.v += eye_kick * dt * 10.0

            cy_tgt = jiggle * (0.28 * (breath_val - 0.5) + 0.22 * energy * math.sin(t_now * 6.5))
            cx_tgt = jiggle * (-0.032 * body_x - 0.028 * body_z)
            chest_y = max(-1.0, min(1.0, self.sp_chest_y.step(cy_tgt, dt, 15.5, 0.26)))
            chest_x = max(-1.0, min(1.0, self.sp_chest_x.step(cx_tgt, dt, 13.5, 0.28)))
            eye_j = max(-0.22, min(0.22, self.sp_eye_jelly.step(0.0, dt, 22.0, 0.78)))

            p["Chest Y"] = (chest_y, 0.0)
            p["Chest X"] = (chest_x, 0.0)
            p["Eye Jelly"] = (eye_j, 0.0)

            self._prev_body_x = body_x
            self._prev_body_y = body_y
            self._prev_breath = breath_val
            self._prev_arm_sum = arm_sum
            self._prev_eye_open = eye_open_r
            self._prev_eye_x = eye_x
            self._prev_eye_y = eye_y

        # 10. Apply raw user parameter overrides (from the All-91-Param Inspector)
        overrides = rig_cfg.get("overrides") or {}
        for pname, ov in overrides.items():
            if pname in p and isinstance(ov, (list, tuple)) and len(ov) == 2:
                p[pname] = (float(ov[0]), float(ov[1]))

        # 11. Fire Plugin Hooks (hot 60 FPS extensible layer!)
        frame_bundle = {"params": p, "camera": cam_cfg, "energy": energy, "stress": stress, "dt": dt, "time": t_now}
        self.hooks.emit_sync("on_frame_params", ctx, frame_bundle)
        self.hooks.emit_sync("on_shader_uniforms", ctx, shader_cfg)

        self.last_params = p
        self.last_shader = shader_cfg
        self.last_camera = cam_cfg
        return p, shader_cfg, cam_cfg
