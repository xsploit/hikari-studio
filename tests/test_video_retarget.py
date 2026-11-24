"""Verification: Motion Lab general-motion video tracker (modules/video_retarget.py) + clip playback (core/rig_controller.py).

  * synthetic anime VTuber video with KNOWN events: left wink, full blink, hop, head turn, head roll, right-arm raise,
    lean-in zoom, mouth flaps (must NOT leak anywhere) and flickering background sparkles (must NOT raise arms)
  * clip schema: 20 tracks (8 legacy + 12 rich), no mouth tracks, no audio, trackerMode / trackCount metadata
  * zero drift: the clip returns to rest (stage / roll ~ 0) after all the motion
  * RigController: validation (legacy 9-track clips, malformed optional tracks), rich playback on Hikari AND Hiyori
    (independent winks, gaze, stage, shrug, arm raise), the live lip-sync keeps 100% of the mouth during playback
  * real anime VTuber clip (if present on disk): face lock rate, the 2.5 s blink and the 2.5 s wave

    python tests/test_video_retarget.py        (or: python -m pytest tests/test_video_retarget.py)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

import core.puppet_adapter as pa  # noqa: E402
from core.events import HookRegistry  # noqa: E402
from core.rig_controller import MOTION_OPTIONAL_TRACKS, MOTION_TRACKS, RigController, validate_motion_clip  # noqa: E402
import modules.video_retarget as vr  # noqa: E402
from test_universal_puppets import HIYORI, hikari_defaults, quiet_cfg  # noqa: E402

REAL_CLIP = Path(r"C:\Users\SUBSECT\.gemini\antigravity\brain\858c1f53-84e5-4c6a-b57b-4290df562ea3"
                 r"\.user_uploaded\media_1795460053813_df4c74ba.mp4")
RESULTS: list[tuple[str, bool, str]] = []
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Japanese puppet param names
except Exception:
    pass
FPS = 30
DUR = 7.5


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail if not ok else ""))
    print(("  PASS " if ok else "  FAIL ") + name + ("" if ok else f"   -> {detail[:300]}"), flush=True)


# ------------------------------------------------------------------------------------------------ synthetic video
SKIN = (196, 214, 250)          # BGR peach
HAIR = (150, 70, 235)           # BGR pink
JACKET = (110, 110, 112)


def _env(t: float, a: float, b: float) -> float:
    """0 -> 1 -> 0 smooth bump over [a, b]."""
    if t <= a or t >= b:
        return 0.0
    return math.sin(math.pi * (t - a) / (b - a)) ** 2


def _head_layer(t: float, size: int = 260) -> tuple[np.ndarray, np.ndarray]:
    """Head (hair + face + eyes + mouth) on its own canvas, centred; returns (bgr, alpha)."""
    img = np.zeros((size, size, 3), np.uint8)
    al = np.zeros((size, size), np.uint8)
    c = size // 2
    turn = 9.0 * _env(t, 3.5, 4.4)                          # inner features shift right = yaw to screen-right
    for tgt, col in ((img, HAIR), (al, 255)):
        cv2.ellipse(tgt, (c, c - 10), (95, 105), 0, 0, 360, col, -1)                 # back hair
    cv2.ellipse(img, (c, c + 8), (62, 74), 0, 0, 360, SKIN, -1)                      # face
    cv2.ellipse(img, (c, c + 8), (62, 74), 0, 0, 360, (60, 60, 120), 2, cv2.LINE_AA)
    # bangs (zig-zag polygon = KLT corners)
    pts = [(c - 80, c - 40)]
    for k in range(9):
        x = c - 70 + k * 17
        pts += [(x, c - 30 + (14 if k % 2 else -6)), (x + 9, c - 52)]
    pts += [(c + 80, c - 40), (c + 85, c - 100), (c - 85, c - 100)]
    cv2.fillPoly(img, [np.int32(pts)], HAIR)
    for k in range(6):                                        # hair strand lines
        x = c - 60 + k * 24
        cv2.line(img, (x, c - 95), (x + 6, c - 45), (110, 40, 190), 2, cv2.LINE_AA)
    wink = _env(t, 0.95, 1.45) > 0.25
    blink = _env(t, 1.95, 2.25) > 0.25
    for side, ex in (("R", c - 28), ("L", c + 28)):           # screen-left eye = character's RIGHT eye
        ex = int(ex + turn)
        ey = c - 2
        closed = blink or (wink and side == "L")
        if closed:
            cv2.line(img, (ex - 14, ey + 4), (ex + 14, ey + 4), (30, 20, 40), 3, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (ex, ey), (13, 15), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, (ex, ey + 2), (10, 12), 0, 0, 360, (110, 30, 150), -1)  # iris
            cv2.circle(img, (ex, ey + 3), 5, (25, 10, 40), -1)                        # pupil
            cv2.circle(img, (ex - 4, ey - 4), 3, (255, 255, 255), -1)                 # highlight
            cv2.line(img, (ex - 15, ey - 14), (ex + 15, ey - 14), (30, 20, 40), 3, cv2.LINE_AA)  # lashes
        cv2.line(img, (ex - 12, ey - 26), (ex + 12, ey - 28), (90, 50, 120), 2, cv2.LINE_AA)    # brow
    nx = int(c + turn * 1.1)
    cv2.circle(img, (nx, c + 22), 2, (120, 140, 200), -1)                                      # nose
    flap = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 4.0) if 0.3 < t < 1.8 or 5.9 < t < 6.6 else 0.0
    cv2.ellipse(img, (int(c + turn), c + 44), (14, int(2 + 10 * flap)), 0, 0, 360, (60, 40, 150), -1)
    for bx in (-36, 36):                                                                       # blush
        cv2.ellipse(img, (int(c + bx + turn * 0.6), c + 26), (9, 4), 0, 0, 360, (170, 180, 245), -1)
    return img, al


def _frame(t: float, rng: np.random.Generator) -> np.ndarray:
    W, H = 640, 360
    f = np.zeros((H, W, 3), np.uint8)
    f[:] = np.linspace(40, 70, W, dtype=np.uint8)[None, :, None]
    f[..., 0] = np.clip(f[..., 0].astype(int) + 25, 0, 255)
    # flickering sparkles everywhere (background noise must never raise arms)
    for _ in range(14):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(f, (x, y), int(rng.integers(1, 4)), (230, 240, 255), -1)
    hop = 34.0 * _env(t, 2.6, 3.2)
    zoom = 1.0 + 0.22 * _env(t, 5.9, 7.0)
    roll = 13.0 * _env(t, 4.5, 5.3)                           # head top toward screen-left (CCW on screen)
    cx, cy = 320, int(150 - hop)
    # torso + shoulders + neck
    cv2.rectangle(f, (cx - 14, cy + 60), (cx + 14, cy + 105), SKIN, -1)
    torso = np.int32([(cx - 120, H), (cx - 112, cy + 108), (cx - 30, cy + 92), (cx + 30, cy + 92),
                      (cx + 112, cy + 108), (cx + 120, H)])
    cv2.fillPoly(f, [torso], JACKET)
    cv2.polylines(f, [torso], False, (60, 60, 60), 2, cv2.LINE_AA)
    for k in range(5):                                        # jacket seams / buttons (torso KLT texture)
        cv2.circle(f, (cx - 40 + k * 20, cy + 150 + (k % 2) * 14), 4, (40, 40, 40), -1)
        cv2.line(f, (cx - 90 + k * 45, cy + 125), (cx - 80 + k * 45, cy + 200), (85, 85, 88), 2)
    # right arm (screen-left): raise 5.0 .. 5.9 s, otherwise hanging hand at the hip
    up = _env(t, 4.95, 5.95)
    up = min(1.0, up * 1.6)
    sh = (cx - 100, cy + 115)
    hand = (int(cx - 150 - 10 * up), int((cy + 250) * (1 - up) + (cy - 20) * up))
    cv2.line(f, sh, hand, JACKET, 26, cv2.LINE_AA)
    cv2.line(f, (cx + 100, cy + 115), (cx + 150, cy + 250), JACKET, 26, cv2.LINE_AA)
    for hp in (hand, (cx + 150, cy + 250)):
        cv2.ellipse(f, hp, (13, 17), 0, 0, 360, SKIN, -1)
        cv2.ellipse(f, hp, (13, 17), 0, 0, 360, (40, 40, 90), 2, cv2.LINE_AA)
    # head layer: rotate about the neck, composite
    head, al = _head_layer(t)
    hs = head.shape[0]
    M = cv2.getRotationMatrix2D((hs / 2, hs / 2 + 70), roll, 1.0)
    head = cv2.warpAffine(head, M, (hs, hs))
    al = cv2.warpAffine(al, M, (hs, hs))
    x0, y0 = cx - hs // 2, cy - hs // 2
    roi = f[max(0, y0):y0 + hs, x0:x0 + hs]
    a = (al[max(0, -y0):max(0, -y0) + roi.shape[0], :roi.shape[1]] / 255.0)[..., None]
    roi[:] = (head[max(0, -y0):max(0, -y0) + roi.shape[0], :roi.shape[1]] * a + roi * (1 - a)).astype(np.uint8)
    if zoom > 1.0:                                            # lean-in: whole image scales about the face
        Mz = cv2.getRotationMatrix2D((cx, cy), 0.0, zoom)
        f = cv2.warpAffine(f, Mz, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return f


def make_synthetic(path: Path) -> None:
    rng = np.random.default_rng(7)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    for i in range(int(DUR * FPS)):
        wr.write(_frame(i / FPS, rng))
    wr.release()


def at(tr: list[float], t: float) -> float:
    return float(tr[min(len(tr) - 1, int(round(t * 60)))])


def span(tr: list[float], a: float, b: float) -> np.ndarray:
    return np.asarray(tr[int(a * 60):int(b * 60) + 1], dtype=np.float64)


# ------------------------------------------------------------------------------------------------ tracker
def test_synthetic(tmp: Path) -> dict:
    vid = tmp / "synthetic_anime.mp4"
    make_synthetic(vid)
    meta = vr.extract_motion_from_video(vid, tmp, "synthetic anime")
    clip = json.loads((tmp / f"{meta['id']}.json").read_text())
    tr = clip["tracks"]
    check("schema. 20 tracks = 8 legacy + 12 rich, exactly", set(tr) == set(vr.TRACK_KEYS) and len(tr) == 20
          and clip["trackCount"] == 20, str(sorted(tr)))
    check("schema. legacy core tracks all present", all(k in tr for k in MOTION_TRACKS))
    check("schema. rich tracks = RigController optional tracks", set(vr.RICH_TRACKS) == set(MOTION_OPTIONAL_TRACKS))
    check("schema. NO mouth / audio (live lip-sync owns the mouth)",
          not any(k.startswith("Mouth") for k in tr) and clip["hasAudio"] is False and clip["audioFile"] == ""
          and not list(tmp.glob("*.wav")))
    check("schema. 60 FPS, frames match duration", clip["fps"] == 60 and abs(clip["frames"] - DUR * 60) <= 2,
          f"{clip['frames']}")
    check("lock. anime dual-eye lock, 100% lock rate, no losses", clip["trackerMode"] == "anime_klt"
          and clip["trackerMetrics"]["lockRate"] == 1.0 and clip["trackerMetrics"]["lostFrames"] == 0,
          json.dumps(clip["trackerMetrics"]))
    check("validate. extracted clip passes RigController validation", validate_motion_clip(clip) is clip)

    # eyes
    check("eyes. open at rest (both > 0.85 @0.5s, @3.0s)",
          min(at(tr["EyeLOpen"], 0.5), at(tr["EyeROpen"], 0.5), at(tr["EyeLOpen"], 3.0), at(tr["EyeROpen"], 3.0)) > 0.85,
          f"{at(tr['EyeLOpen'], 0.5):.2f} {at(tr['EyeROpen'], 0.5):.2f} {at(tr['EyeLOpen'], 3.0):.2f}")
    wl, wr_ = span(tr["EyeLOpen"], 1.1, 1.3).mean(), span(tr["EyeROpen"], 1.1, 1.3).mean()
    check("eyes. LEFT wink: EyeLOpen closes, EyeROpen stays open", wl < 0.3 and wr_ > 0.75, f"L {wl:.2f} R {wr_:.2f}")
    bl, br = span(tr["EyeLOpen"], 2.05, 2.15).mean(), span(tr["EyeROpen"], 2.05, 2.15).mean()
    check("eyes. full blink: both close", bl < 0.35 and br < 0.35, f"L {bl:.2f} R {br:.2f}")
    check("eyes. legacy EyeOpen = mean(L, R)", abs(at(tr["EyeOpen"], 1.2) - 0.5 * (at(tr["EyeLOpen"], 1.2)
                                                                                  + at(tr["EyeROpen"], 1.2))) < 0.02)
    check("eyes. hop / roll / zoom never fake a blink (eyes > 0.75 through 2.6-7.0 s except arm-up)",
          min(span(tr["EyeLOpen"], 2.6, 4.9).min(), span(tr["EyeROpen"], 2.6, 4.9).min(),
              span(tr["EyeLOpen"], 6.0, 7.0).min()) > 0.75,
          f"{span(tr['EyeLOpen'], 2.6, 4.9).min():.2f} {span(tr['EyeROpen'], 2.6, 4.9).min():.2f}")

    # head / stage
    hop = at(tr["StageY"], 2.9) - at(tr["StageY"], 2.3)
    check("stage. hop -> StageY rises (> +0.2)", hop > 0.2, f"{hop:.2f}")
    check("stage. hop is not a head pitch / roll", abs(at(tr["AngleZ"], 2.9)) < 3.0, f"roll {at(tr['AngleZ'], 2.9):.1f}")
    yaw = at(tr["AngleX"], 3.95) - at(tr["AngleX"], 3.2)
    check("head. turn toward screen-right -> AngleX positive (> +4 deg)", yaw > 4.0, f"{yaw:.1f}")
    roll = at(tr["AngleZ"], 4.9)
    check("head. roll (head top to screen-left) -> AngleZ +10..+16 deg (true degrees)", 10.0 < roll < 16.0, f"{roll:.1f}")
    zm = at(tr["StageZoom"], 6.45) - at(tr["StageZoom"], 5.6)
    check("stage. lean-in zoom -> StageZoom rises (> +0.25)", zm > 0.25, f"{zm:.2f}")

    # arms
    ar = span(tr["ArmR"], 5.35, 5.6).mean()
    check("arms. right-arm raise (screen-left) -> ArmR > 0.6", ar > 0.6, f"{ar:.2f}")
    check("arms. raise is one-sided (ArmL stays < 0.15)", span(tr["ArmL"], 0, DUR).max() < 0.15,
          f"{span(tr['ArmL'], 0, DUR).max():.2f}")
    quiet = span(tr["ArmR"], 0.0, 4.6).max()
    check("arms. sparkles / hop / head motion never raise an arm (ArmR < 0.15 before the raise)", quiet < 0.15, f"{quiet:.2f}")
    check("arms. gesture only while raised", np.abs(span(tr["ArmRGesture"], 0.0, 4.6)).max() < 0.05)

    # drift
    end = {k: abs(span(tr[k], 7.25, 7.45).mean()) for k in ("StageX", "StageY", "StageZoom", "AngleZ")}
    check("drift. clip returns to rest after hop / turn / roll / zoom (stage < 0.06, roll < 1.5 deg)",
          end["StageX"] < 0.06 and end["StageY"] < 0.06 and end["StageZoom"] < 0.06 and end["AngleZ"] < 1.5, str(end))
    for k, v in tr.items():
        if not all(math.isfinite(x) for x in v):
            check(f"finite. {k}", False)
    check("finite. every sample finite and in range", all(abs(x) <= 30.0 for v in tr.values() for x in v))
    return clip


# ------------------------------------------------------------------------------------------------ rig playback
class LoudLipSync:
    """Live lip-sync source mid-word (the mouth must stay ITS output while a clip plays)."""

    def step(self, dt: float, t_now: float) -> dict:
        return {"open": 0.62, "form": 0.3, "funnel": 0.25, "x": 0.0, "shrug": 0.0, "tongue": 0.0, "energy": 0.5,
                "stress": 0.3, "onset": 0.0, "pitch": 0.5, "voiced": 1.0, "press": 0.0}


def _clip(n: int, rich: dict[str, float] | None = None, legacy_mouth: bool = False) -> dict:
    tracks = {k: [0.0] * n for k in MOTION_TRACKS}
    tracks["EyeOpen"] = [1.0] * n
    if legacy_mouth:
        tracks["MouthOpen"] = [1.0] * n
    for k, v in (rich or {}).items():
        tracks[k] = [float(v)] * n
    return {"id": "t", "name": "t", "fps": 60, "frames": n, "duration": n / 60.0, "tracks": tracks}


def _play(rig: RigController, clip: dict, tmp: Path, frames: int = 90) -> dict:
    p = tmp / "play.json"
    p.write_text(json.dumps(clip))
    rig.load_motion_clip(p, loop=True, mix=1.0)
    cfg = quiet_cfg()
    out = {}
    for _ in range(frames):
        out, _, _ = rig.step(1.0 / 60.0, cfg, None)
    return out


def test_validation(tmp: Path) -> None:
    legacy = _clip(30, legacy_mouth=True)
    check("validate. legacy 9-track clip (with MouthOpen) still loads", validate_motion_clip(legacy) is legacy)
    check("validate. legacy 8-track clip (no MouthOpen) loads", validate_motion_clip(_clip(30)) is not None)
    bad = _clip(30, {"EyeLOpen": 0.5})
    bad["tracks"]["EyeLOpen"][3] = "x"
    try:
        validate_motion_clip(bad)
        check("validate. malformed optional track rejected", False)
    except ValueError:
        check("validate. malformed optional track rejected", True)
    short = _clip(30, {"StageX": 0.1})
    short["tracks"]["StageX"] = short["tracks"]["StageX"][:10]
    try:
        validate_motion_clip(short)
        check("validate. too-short optional track rejected", False)
    except ValueError:
        check("validate. too-short optional track rejected", True)
    nan = _clip(30)
    nan["tracks"]["AngleX"][0] = float("nan")
    try:
        validate_motion_clip(nan)
        check("validate. NaN core sample rejected", False)
    except ValueError:
        check("validate. NaN core sample rejected", True)


RICH_POSE = {"EyeLOpen": 0.0, "EyeROpen": 1.0, "EyeLookX": 0.8, "EyeLookY": 0.0, "BrowLY": 0.6, "BrowRY": 0.0,
             "StageX": 0.0, "StageY": 0.7, "StageZoom": 0.4, "ShoulderShrug": 0.8, "ArmRGesture": 0.0,
             "ArmLGesture": 0.0, "AngleX": 18.0, "ArmR": 1.0}


def test_rig_hikari(tmp: Path) -> None:
    rig = RigController(hikari_defaults(), LoudLipSync(), HookRegistry())
    base = _play(rig, _clip(120), tmp)                       # neutral clip, same lip-sync
    rig = RigController(hikari_defaults(), LoudLipSync(), HookRegistry())
    p = _play(rig, _clip(120, RICH_POSE), tmp)
    check("hikari. independent wink: Left Eye Open closed, Right open",
          p["Left Eye Open"][0] < 0.15 and p["Right Eye Open"][0] > 0.85, f"{p['Left Eye Open']} {p['Right Eye Open']}")
    check("hikari. gaze -> EyeBall x", p["EyeBall"][0] > 0.6, str(p["EyeBall"]))
    check("hikari. head yaw -> Angle x 18", abs(p["Angle"][0] - 18.0) < 1.0, str(p["Angle"]))
    check("hikari. brow -> Left Brow Y raised vs right", p["Left Brow Y"][0] > p["Right Brow Y"][0] + 0.4,
          f"{p['Left Brow Y']} {p['Right Brow Y']}")
    check("hikari. StageY / StageZoom / Shoulder Shrug through the rig-fx springs",
          p["Stage Y"][0] > 0.55 and p["Stage Zoom"][0] > 0.3 and p["Shoulder Shrug"][0] > 0.6,
          f"{p['Stage Y']} {p['Stage Zoom']} {p['Shoulder Shrug']}")
    check("hikari. tracked raise -> Arm R Pose up", p["Arm R Pose"][0] > 0.9 and p["Arm L Pose"][0] < 0.1,
          f"{p['Arm R Pose']} {p['Arm L Pose']}")
    mouth = ("VB Mouth", "Mouth Funnel", "Jaw Open")
    same = all(abs(p[k][0] - base[k][0]) < 1e-6 and abs(p[k][1] - base[k][1]) < 1e-6 for k in mouth)
    check("hikari. live lip-sync owns the mouth (VB Mouth / Funnel / Jaw identical with a rich clip)", same,
          str({k: (p[k], base[k]) for k in mouth}))
    rig = RigController(hikari_defaults(), LoudLipSync(), HookRegistry())
    pl = _play(rig, _clip(120, legacy_mouth=True), tmp)
    check("hikari. legacy clip MouthOpen=1 is ignored (mouth = live lip-sync)",
          all(abs(pl[k][0] - base[k][0]) < 1e-6 and abs(pl[k][1] - base[k][1]) < 1e-6 for k in mouth),
          str({k: (pl[k], base[k]) for k in mouth}))
    check("hikari. live mouth really is open mid-word (sanity)", base["VB Mouth"][1] > 0.3, str(base["VB Mouth"]))
    # non-loop clip ends cleanly
    rig.load_motion_clip(tmp / "play.json", loop=False, mix=1.0)
    rig._motion_start -= 10.0
    rig.step(1 / 60, quiet_cfg(), None)
    check("hikari. one-shot clip stops at its end", rig.motion_status()["playing"] is False)


def test_rig_hiyori(tmp: Path) -> None:
    rig = RigController(hikari_defaults(), LoudLipSync(), HookRegistry())
    info = pa.inspect_puppet(HIYORI, "Hiyori")
    rig.reload_puppet_meta(info.param_meta, "Hiyori")
    base = _play(rig, _clip(120), tmp)
    rig = RigController(hikari_defaults(), LoudLipSync(), HookRegistry())
    rig.reload_puppet_meta(info.param_meta, "Hiyori")
    p = _play(rig, _clip(120, RICH_POSE), tmp)
    le, re_ = "左目\u3000開閉", "右目\u3000開閉"
    check("hiyori. output is Hiyori's own params", le in p and re_ in p and "角度" in p, str(list(p)[:8]))
    check("hiyori. independent wink flows through UniversalParamMapper", p[le][0] < 0.25 and p[re_][0] > 0.75,
          f"{p[le]} {p[re_]}")
    check("hiyori. head yaw flows (角度 x moves toward +)", p["角度"][0] > base["角度"][0] + 8.0,
          f"{p['角度']} vs {base['角度']}")
    mk = [k for k in p if "口" in k]
    check("hiyori. mouth params identical to live lip-sync baseline",
          bool(mk) and all(abs(p[k][0] - base[k][0]) < 1e-6 and abs(p[k][1] - base[k][1]) < 1e-6 for k in mk),
          str({k: (p[k], base[k]) for k in mk}))
    check("hiyori. no non-finite values", all(math.isfinite(v[0]) and math.isfinite(v[1]) for v in p.values()))


# ------------------------------------------------------------------------------------------------ real clip
def test_real_clip(tmp: Path) -> None:
    if not REAL_CLIP.is_file():
        check("real. VTuber clip present (skipped)", True)
        return
    meta = vr.extract_motion_from_video(REAL_CLIP, tmp, "real_vtuber")
    tr = json.loads((tmp / "real_vtuber.json").read_text())["tracks"]
    m = meta["trackerMetrics"]
    check("real. anime lock 100%, no lost frames, sub-pixel reference residual",
          meta["trackerMode"] == "anime_klt" and m["lockRate"] == 1.0 and m["lostFrames"] == 0
          and m["medianRefResidualPx"] < 1.5, json.dumps(m))
    check("real. 2.5 s smile-blink: both eyes closed", at(tr["EyeLOpen"], 2.5) < 0.3 and at(tr["EyeROpen"], 2.5) < 0.3,
          f"{at(tr['EyeLOpen'], 2.5):.2f} {at(tr['EyeROpen'], 2.5):.2f}")
    check("real. eyes open at 0.5 / 3.5 / 5.5 s", min(at(tr["EyeOpen"], t) for t in (0.5, 3.5, 5.5)) > 0.85)
    check("real. 2.5 s wave (character's left hand, screen-right) -> ArmL > 0.5, ArmR quiet",
          at(tr["ArmL"], 2.5) > 0.5 and at(tr["ArmR"], 2.5) < 0.15, f"{at(tr['ArmL'], 2.5):.2f} {at(tr['ArmR'], 2.5):.2f}")
    check("real. 3.5 s head tilt -> AngleZ > +8 deg", at(tr["AngleZ"], 3.5) > 8.0, f"{at(tr['AngleZ'], 3.5):.1f}")
    check("real. camera push-in -> StageZoom rises 0.5 -> 5.5 s", at(tr["StageZoom"], 5.5) - at(tr["StageZoom"], 0.5) > 0.5)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("== synthetic anime video tracker")
        test_synthetic(tmp)
        print("== clip validation")
        test_validation(tmp)
        print("== RigController playback: Hikari")
        test_rig_hikari(tmp)
        print("== RigController playback: Hiyori")
        test_rig_hiyori(tmp)
        print("== real anime VTuber clip")
        test_real_clip(tmp)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


def test_all() -> None:                 # pytest entry
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
