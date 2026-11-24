"""Universal Inochi2D puppet support: inspect ANY .inp / .inx, auto-frame it, map Hikari's rig onto it, import Live2D.

  * `inspect_puppet(path)`   -> PuppetInfo: meta (name / artist), full param_meta, rest-pose visible bounding box
                                (node transforms + Part mesh verts + default-pose deforms / opacity), face landmarks
                                and a `recommended_camera` for a 960x1080 upper-body VTuber shot.
  * `UniversalParamMapper`   -> per-frame translation of Hikari Studio's canonical rig output (Hikari param names)
                                onto any puppet's params: English Live2D (ParamAngleX ...), Japanese Live2D / MINT
                                (角度, 左目　開閉, 口　変形 / 口　開閉, あ/い/う/え/お ...) and official Inochi2D
                                (Head:: Yaw-Pitch, Eye:: Left:: Open ...). Every frame starts from the puppet's own
                                defaults, so untouched params keep their author-intended values.
  * `import_puppet_file(...)`-> .inp / .inx copied (validated) or a Live2D .zip (.moc3 + textures) converted with
                                MINT (`mint-cli export <moc3> <stem> --inp`) into models/library/.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable

INP_MAGIC = b"TRNSRTS\x00"
PUPPET_EXTS = (".inp", ".inx")
IMPORT_EXTS = (".inp", ".inx", ".zip")
MAX_IMPORT_BYTES = 150 * 1024 * 1024                 # uploaded file (puppet or zip)
MAX_ZIP_UNCOMPRESSED = 400 * 1024 * 1024             # zip bomb guard (sum of member sizes)
MAX_ZIP_MEMBERS = 4000
MINT_TIMEOUT_SEC = 600.0
DEFAULT_MINT_CLI = Path("D:/Hikari-Rig-Tools/MINT-1.2.2/mint-cli.exe")
HIKARI_ID = "hikari"
FRAME_W, FRAME_H = 960, 1080

# ----------------------------------------------------------------------------- reading


class PuppetFormatError(ValueError):
    """Not a readable Inochi2D puppet."""


def read_puppet_json(path: Path) -> dict[str, Any]:
    """Parse the JSON payload of an .inp / .inx (validates the `TRNSRTS\\0` magic and payload length)."""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:8] != INP_MAGIC:
            raise PuppetFormatError(f"{path.name} is not an Inochi2D puppet (bad magic header)")
        jlen = int.from_bytes(head[8:12], "big")
        size = path.stat().st_size
        if jlen <= 2 or 12 + jlen > size:
            raise PuppetFormatError(f"{path.name} has a corrupt payload length")
        raw = f.read(jlen)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise PuppetFormatError(f"{path.name} has an unreadable JSON payload") from None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), dict) or not isinstance(data.get("param"), list):
        raise PuppetFormatError(f"{path.name} is missing nodes / params")
    return data


def _f(v: Any, d: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return x if math.isfinite(x) else d


def _pair(v: Any, d: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _f(v[0], d[0]), _f(v[1], d[1])
    if isinstance(v, (int, float)):
        return _f(v, d[0]), d[1]
    return d


def extract_param_meta(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Every param (bound or not) with its real range / defaults. `bound` = drives at least one binding."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in model.get("param") or []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        mn, mx, df = _pair(p.get("min"), (0.0, 0.0)), _pair(p.get("max"), (1.0, 1.0)), _pair(p.get("defaults"))
        binds = p.get("bindings") or []
        out.append({
            "name": name,
            "isVec2": bool(p.get("is_vec2", False)),
            "min": [mn[0], mn[1]],
            "max": [mx[0], mx[1]],
            "defaults": [df[0], df[1]],
            "bound": bool(binds),
            "bindingCount": len(binds),
        })
    return out


def param_defaults_of(meta: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    return {m["name"]: (float(m["defaults"][0]), float(m["defaults"][1])) for m in meta}


# ----------------------------------------------------------------------------- rest-pose geometry

def _mat(trans: tuple[float, float], rot_z: float, scale: tuple[float, float]) -> tuple[float, ...]:
    """2D affine (a, b, c, d, tx, ty): T * Rz * S (Inochi2D's node transform, projected to the screen plane)."""
    c, s = math.cos(rot_z), math.sin(rot_z)
    return (c * scale[0], s * scale[0], -s * scale[1], c * scale[1], trans[0], trans[1])


def _mul(m: tuple[float, ...], n: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, tx, ty = m
    a2, b2, c2, d2, tx2, ty2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2,
            a * tx2 + c * ty2 + tx, b * tx2 + d * ty2 + ty)


_IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _axis_weights(points: list[float], t: float) -> list[tuple[int, float]]:
    """Linear interpolation weights of normalized value `t` over binding axis points."""
    if not points:
        return [(0, 1.0)]
    if len(points) == 1 or t <= points[0]:
        return [(0, 1.0)]
    if t >= points[-1]:
        return [(len(points) - 1, 1.0)]
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a <= t <= b:
            w = 0.0 if b - a <= 1e-9 else (t - a) / (b - a)
            return [(i, 1.0 - w), (i + 1, w)]
    return [(len(points) - 1, 1.0)]


def _binding_at_default(param: dict[str, Any], binding: dict[str, Any]) -> list[tuple[tuple[int, int], float]]:
    """Grid cells + weights of a binding evaluated at the param's default value."""
    axes = param.get("axis_points") or [[0.0, 1.0], [0.0]]
    mn, mx, df = _pair(param.get("min")), _pair(param.get("max"), (1.0, 1.0)), _pair(param.get("defaults"))
    tx = 0.0 if mx[0] == mn[0] else (df[0] - mn[0]) / (mx[0] - mn[0])
    ty = 0.0 if mx[1] == mn[1] else (df[1] - mn[1]) / (mx[1] - mn[1])
    ax = [float(v) for v in (axes[0] if len(axes) > 0 else [0.0])]
    ay = [float(v) for v in (axes[1] if len(axes) > 1 else [0.0])] if param.get("is_vec2") else [0.0]
    wy = _axis_weights(ay, ty) if param.get("is_vec2") else [(0, 1.0)]
    cells = []
    for ix, wxv in _axis_weights(ax, tx):
        for iy, wyv in wy:
            if wxv * wyv > 1e-6:
                cells.append(((ix, iy), wxv * wyv))
    return cells


def _rest_state(model: dict[str, Any]) -> tuple[dict[int, list[tuple[float, float]]], dict[int, float]]:
    """Per-node deform offsets and opacity multipliers at every param's default (the pose the puppet loads in)."""
    deform: dict[int, list[tuple[float, float]]] = {}
    opacity: dict[int, float] = {}
    for p in model.get("param") or []:
        if not isinstance(p, dict):
            continue
        for b in p.get("bindings") or []:
            kind = b.get("param_name")
            if kind not in ("deform", "opacity"):
                continue
            node = b.get("node")
            vals = b.get("values")
            if not isinstance(node, int) or not isinstance(vals, list):
                continue
            cells = _binding_at_default(p, b)
            if kind == "opacity":
                acc = 0.0
                for (ix, iy), w in cells:
                    try:
                        acc += w * _f(vals[ix][iy], 1.0)
                    except (IndexError, TypeError):
                        acc += w
                opacity[node] = opacity.get(node, 1.0) * acc
            else:
                acc_v: list[list[float]] | None = None
                for (ix, iy), w in cells:
                    try:
                        offs = vals[ix][iy]
                    except (IndexError, TypeError):
                        continue
                    if not isinstance(offs, list):
                        continue
                    if acc_v is None:
                        acc_v = [[0.0, 0.0] for _ in offs]
                    for k, o in enumerate(offs[: len(acc_v)]):
                        if isinstance(o, (list, tuple)) and len(o) >= 2:
                            acc_v[k][0] += w * _f(o[0])
                            acc_v[k][1] += w * _f(o[1])
                if acc_v:
                    cur = deform.get(node)
                    if cur is None:
                        deform[node] = [(x, y) for x, y in acc_v]
                    else:
                        n = min(len(cur), len(acc_v))
                        deform[node] = [(cur[k][0] + acc_v[k][0], cur[k][1] + acc_v[k][1]) for k in range(n)] + cur[n:]
    return deform, opacity


_SKIP_PART_RE = re.compile(r"hit\s*area|^hit_|hitarea|^collider", re.I)


@dataclass
class _PartGeo:
    uuid: int
    name: str
    group: str
    xs: list[float]
    ys: list[float]

    @property
    def box(self) -> tuple[float, float, float, float]:
        return min(self.xs), max(self.xs), min(self.ys), max(self.ys)


def _visible_parts(model: dict[str, Any]) -> list[_PartGeo]:
    deform, opacity = _rest_state(model)
    out: list[_PartGeo] = []

    def walk(node: dict[str, Any], parent_m: tuple[float, ...], group: str, depth: int) -> None:
        if not isinstance(node, dict) or depth > 64 or node.get("enabled", True) is False:
            return                                     # disabled nodes hide their whole subtree
        tr = node.get("transform") or {}
        t = tr.get("trans") or [0.0, 0.0, 0.0]
        r = tr.get("rot") or [0.0, 0.0, 0.0]
        s = tr.get("scale") or [1.0, 1.0]
        local = _mat((_f(t[0]) if len(t) > 0 else 0.0, _f(t[1]) if len(t) > 1 else 0.0),
                     _f(r[2]) if len(r) > 2 else 0.0,
                     (_f(s[0], 1.0) if len(s) > 0 else 1.0, _f(s[1], 1.0) if len(s) > 1 else 1.0))
        world = local if node.get("lockToRoot") else _mul(parent_m, local)
        name = str(node.get("name") or "")
        if depth == 1:
            group = name
        if node.get("type") in ("Part", "AnimatedPart") and not _SKIP_PART_RE.search(name):
            mesh = node.get("mesh") or {}
            verts = mesh.get("verts") or []
            uid = node.get("uuid")
            op = _f(node.get("opacity", 1.0), 1.0) * opacity.get(uid, 1.0)
            if len(verts) >= 6 and op > 0.05:
                offs = deform.get(uid) or []
                a, b, c, d, tx, ty = world
                xs, ys = [], []
                for k in range(len(verts) // 2):
                    vx, vy = _f(verts[2 * k]), _f(verts[2 * k + 1])
                    if k < len(offs):
                        vx += offs[k][0]
                        vy += offs[k][1]
                    xs.append(a * vx + c * vy + tx)
                    ys.append(b * vx + d * vy + ty)
                out.append(_PartGeo(int(uid) if isinstance(uid, int) else -1, name, group, xs, ys))
        for ch in node.get("children") or []:
            walk(ch, world, group, depth + 1)

    walk(model["nodes"], _IDENT, "", 0)
    return out


# ----------------------------------------------------------------------------- name normalization / channel rules

def norm_name(name: str) -> str:
    """NFKC, lowercase, strip spaces / separators: '左目　開閉' -> '左目開閉', 'Head:: Yaw-Pitch' -> 'headyawpitch'."""
    s = unicodedata.normalize("NFKC", str(name)).lower()
    s = re.sub(r"[\s_\-:./()\[\]|]+", "", s)
    return s


def _strip_param(n: str) -> str:
    return n[5:] if n.startswith("param") and len(n) > 5 else n


# Target rule: (axis-x spec, axis-y spec). A spec = (canonical channel, mode). mode: bi (-1..1 about the default),
# uni (0..1 from the default to the far end), eye (open multiplier of the default), inv (1 - value, uni).
Spec = tuple[str, str] | None
_RULES: list[tuple[tuple[str, ...], Spec, Spec, str]] = [
    # --- head
    (("角度", "angle", "headyawpitch", "headangle", "faceangle", "head"), ("head_x", "bi"), ("head_y", "bi"), "head"),
    (("anglex", "角度x", "headyaw", "headx", "faceanglex"), ("head_x", "bi"), None, "head"),
    (("angley", "角度y", "headpitch", "heady", "faceangley"), ("head_y", "bi"), None, "head"),
    (("anglez", "角度z", "headroll", "headz", "headtilt", "faceanglez"), ("head_z", "bi"), None, "head"),
    # --- body
    (("bodyyawpitch", "bodyangle", "体の回転"), ("body_x", "bi"), ("body_y", "bi"), "body"),
    (("bodyanglex", "体の回転x", "bodyyaw", "bodyx"), ("body_x", "bi"), None, "body"),
    (("bodyangley", "体の回転y", "bodypitch", "bodyy"), ("body_y", "bi"), None, "body"),
    (("bodyanglez", "体の回転z", "bodyroll", "bodyz"), ("body_z", "bi"), None, "body"),
    # --- eyes
    (("eyelopen", "lefteyeopen", "左目開閉", "eyeleftopen", "lefteye"), ("eye_l_open", "eye"), None, "eyes"),
    (("eyeropen", "righteyeopen", "右目開閉", "eyerightopen", "righteye"), ("eye_r_open", "eye"), None, "eyes"),
    (("eyeleftblink", "eyelblink", "leftblink", "blinkleft"), ("eye_l_open", "inv"), None, "eyes"),
    (("eyerightblink", "eyerblink", "rightblink", "blinkright"), ("eye_r_open", "inv"), None, "eyes"),
    (("eyeopen", "目開閉", "eyesopen"), ("eye_open", "eye"), None, "eyes"),
    (("blink", "eyeblink", "eyesblink"), ("eye_open", "inv"), None, "eyes"),
    (("eyelsmile", "lefteyesmile", "左目笑顔", "eyeleftsmile"), ("eye_smile", "uni"), None, "eyes"),
    (("eyersmile", "righteyesmile", "右目笑顔", "eyerightsmile"), ("eye_smile", "uni"), None, "eyes"),
    (("eyesmile", "目笑顔"), ("eye_smile", "uni"), None, "eyes"),
    # --- gaze
    (("目玉", "eyeball", "eyeballxy", "eyexy", "eyelook", "eyeleftxy", "eyerightxy", "eyesxy", "gaze", "pupil"),
     ("eye_x", "bi"), ("eye_y", "bi"), "gaze"),
    (("eyeballx", "目玉x", "eyex", "gazex", "eyelookx", "eyeleftx", "eyerightx"), ("eye_x", "bi"), None, "gaze"),
    (("eyebally", "目玉y", "eyey", "gazey", "eyelooky", "eyelefty", "eyerighty"), ("eye_y", "bi"), None, "gaze"),
    # --- brows
    (("browly", "leftbrowy", "左眉上下", "browlefty", "eyebrowlefty", "browleftheight", "eyebrowly"), ("brow_l_y", "bi"), None, "brows"),
    (("browry", "rightbrowy", "右眉上下", "browrighty", "eyebrowrighty", "browrightheight", "eyebrowry"), ("brow_r_y", "bi"), None, "brows"),
    (("browy", "眉上下", "eyebrowy", "browsy"), ("brow_y", "bi"), None, "brows"),
    (("browlangle", "左眉角度", "browleftangle", "eyebrowlangle"), ("brow_l_angle", "bi"), None, "brows"),
    (("browrangle", "右眉角度", "browrightangle", "eyebrowrangle"), ("brow_r_angle", "bi"), None, "brows"),
    (("browlform", "左眉変形", "browleftform", "eyebrowlform"), ("brow_form", "bi"), None, "brows"),
    (("browrform", "右眉変形", "browrightform", "eyebrowrform"), ("brow_form", "bi"), None, "brows"),
    # --- mouth
    (("口変形口開閉", "vbmouth", "mouth", "mouthformopen", "mouthshapeopen"), ("mouth_form", "bi"), ("mouth_open", "uni"), "mouth"),
    (("mouthopeny", "mouthopen", "口開閉", "jawopen", "mouthy", "mouthopenness"), ("mouth_open", "uni"), None, "mouth"),
    (("mouthform", "口変形", "mouthshape", "mouthsmile"), ("mouth_form", "bi"), None, "mouth"),
    (("あ", "a", "moutha", "vowela", "mouthvowela"), ("vowel_a", "uni"), None, "mouth"),
    (("い", "i", "mouthi", "voweli", "mouthvoweli"), ("vowel_i", "uni"), None, "mouth"),
    (("う", "u", "mouthu", "vowelu", "mouthvowelu"), ("vowel_u", "uni"), None, "mouth"),
    (("え", "e", "mouthe", "vowele", "mouthvowele"), ("vowel_e", "uni"), None, "mouth"),
    (("お", "o", "moutho", "vowelo", "mouthvowelo"), ("vowel_o", "uni"), None, "mouth"),
    (("上がり口", "mouthup", "mouthcornerup"), ("mouth_smile", "uni"), None, "mouth"),
    (("下がり口", "mouthdown", "mouthcornerdown", "mouthfrown"), ("mouth_frown", "uni"), None, "mouth"),
    (("むくれ口", "cheekpuff", "頬膨らみ", "puff", "mouthpuff"), ("cheek_puff", "uni"), None, "mouth"),
    # --- breath / blush / vfx
    (("呼吸", "breath", "breathing", "bodybreath"), ("breath", "uni"), None, "breath"),
    (("照れ", "cheek", "blush", "頬", "頬染め", "cheekblush", "faceblush"), ("blush", "uni"), None, "blush"),
    (("目エフェクト", "eyestar", "sparkleeyes", "eyeeffect", "eyesparkle", "eyehighlight"), ("vfx_sparkle", "uni"), None, "vfx"),
    (("顔インク表示", "faceink", "facepaint"), ("vfx_ink", "uni"), None, "vfx"),
    (("gloom", "faceshadow", "shadowface", "顔影", "青ざめ"), ("gloom", "uni"), None, "vfx"),
    (("eyeheart", "hearteyes"), ("vfx_heart", "uni"), None, "vfx"),
]

_RULE_INDEX: dict[str, tuple[Spec, Spec, str]] = {}
for _names, _sx, _sy, _ch in _RULES:
    for _n in _names:
        _RULE_INDEX.setdefault(norm_name(_n), (_sx, _sy, _ch))

CHANNEL_ORDER = ("head", "body", "eyes", "gaze", "brows", "mouth", "arms", "breath", "blush", "vfx")
# Hikari's own rig (passthrough when a puppet carries these natively)
HIKARI_NATIVE_PARAMS = ("Angle", "Angle Z", "Left Eye Open", "Right Eye Open", "EyeBall", "VB Mouth", "Breath")


def match_rule(name: str) -> tuple[Spec, Spec, str] | None:
    n = norm_name(name)
    r = _RULE_INDEX.get(n)
    if r is None:
        r = _RULE_INDEX.get(_strip_param(n))
    return r


# ----------------------------------------------------------------------------- arms / shoulders / hands
# Japanese Live2D / MINT (左腕 A, 右腕A　肩の回転, 左手B 回転, 左肩の上下, 肩　すくみ), English Live2D (ParamArmLA,
# ParamLeftShoulderUp) and official Inochi2D (Arm:: Left:: Move). Positive = arm out / raised (Live2D convention,
# visually verified on Hiyori 左腕 A and Mao 腕A 肩の回転 / 肘の回転).
_TOGGLE_WORDS_RE = re.compile(r"表示|出現|消滅|show|toggle|enable|visible|(?:^|[^a-z])(?:on|off)(?:$|[^a-z])", re.I)
_SIDE_L_RE = re.compile(r"左|left|(?:^|[^a-z])l(?:$|[^a-z])|arml|shoulderl|handl|elbowl|wristl", re.I)
_SIDE_R_RE = re.compile(r"右|right|(?:^|[^a-z])r(?:$|[^a-z])|armr|shoulderr|handr|elbowr|wristr", re.I)


def arm_rule(name: str) -> str | None:
    """Canonical arm channel for an arm / shoulder / hand param, or None. Hand-shape swaps (左手A, 右手B,
    ParamHandL) are left alone - they are poses, not rotations."""
    raw = unicodedata.normalize("NFKC", str(name))
    low = raw.lower()
    n = norm_name(raw)
    if _TOGGLE_WORDS_RE.search(raw) or "杖" in raw or "wand" in low:
        return None
    sides = [s for s, rx in (("l", _SIDE_L_RE), ("r", _SIDE_R_RE)) if rx.search(raw) or rx.search(n)]
    side = sides[0] if len(sides) == 1 else None
    is_arm = "腕" in raw or re.search(r"arm", low) is not None
    is_hand = "手" in raw or re.search(r"hand|wrist", low) is not None
    is_shoulder = "肩" in raw or "shoulder" in low
    if "手首" in raw or "wrist" in low or (is_hand and ("回転" in raw or re.search(r"rot|angle|turn", low))):
        return f"arm_{side}_wrist" if side else None
    if "肘" in raw or "elbow" in low:
        return f"arm_{side}_elbow" if side else None
    if is_arm and not ("腕のy" in n or "army" in n):
        return f"arm_{side}_raise" if side else None
    if is_shoulder:
        return f"shoulder_{side}" if side else "shoulder"
    return None


# ----------------------------------------------------------------------------- physics detection
_PHYS_JA = ("揺れ", "ふわ", "めくれ", "サイドアップ", "前髪", "後髪", "後ろ髪", "横髪", "アホ毛")
_PHYS_EN_RE = re.compile(r"hair|ribbon|skirt|bust|chest|cape|robe|hat|feather|sleeve|tail|(?:^|[^a-z])ears?(?:$|[^a-z])"
                         r"|phys|sway|bounce|jiggle|swing", re.I)
_PHYS_CORE_RE = re.compile(r"角度|体の回転|目|眉|口|腕|手|肩|呼吸|照れ|angle|eye|brow|mouth|arm|hand|shoulder|breath|"
                           r"cheek|blink|smile", re.I)
_CHAIN_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+)$")


def is_physics_param(name: str) -> bool:
    raw = unicodedata.normalize("NFKC", str(name))
    if _PHYS_CORE_RE.search(raw) and not any(k in raw for k in ("髪", "揺れ", "ふわ")):
        return False
    if match_rule(raw) is not None or _TOGGLE_WORDS_RE.search(raw):
        return False
    return any(k in raw for k in _PHYS_JA) or _PHYS_EN_RE.search(raw) is not None


def _phys_profile(name: str) -> tuple[dict[str, float], float, float, float]:
    """(input weights over canonical channels, omega rad/s, zeta, gain) by what the param hangs from."""
    raw = unicodedata.normalize("NFKC", str(name)).lower()
    vertical = any(k in raw for k in ("ふわ", "めくれ", "胸 揺れ", "胸揺れ", "bust", "bounce", "jiggle")) or raw.endswith("y")
    body_hung = any(k in raw for k in ("スカート", "ローブ", "胸", "首飾り", "パーカー", "ひも", "skirt", "robe", "cape",
                                        "chest", "bust", "sleeve", "tail", "necklace", "string"))
    if vertical and body_hung:
        return {"body_y": 1.0, "breath": 0.7, "head_y": 0.25, "body_x": 0.2}, 9.0, 0.30, 2.4
    if vertical:                                           # fluffy hair / hat lift: head nods
        return {"head_y": 0.9, "body_y": 0.5, "breath": 0.35, "head_x": 0.25}, 7.5, 0.32, 2.6
    if body_hung:
        return {"body_x": 0.9, "body_z": 0.8, "head_x": 0.25, "body_y": 0.2}, 6.5, 0.36, 3.0
    # hair, ahoge, hat brim, feather, ribbons on the head
    return {"head_x": 0.9, "head_z": 0.8, "body_x": 0.35, "body_z": 0.3, "head_y": 0.15}, 7.0, 0.30, 2.0


class _PhysParam:
    __slots__ = ("name", "weights", "omega", "zeta", "gain", "chain", "index", "d", "mn", "mx",
                 "p", "v", "out")

    def __init__(self, m: dict[str, Any], sp: dict[str, Any] | None) -> None:
        self.name = m["name"]
        self.weights, omega, zeta, gain = _phys_profile(self.name)
        if sp:                                             # MINT kept the Cubism physics settings: use their feel
            f = _f(sp.get("frequency"), 0.0)
            if f > 0.05:
                omega = 0.5 * omega + 0.5 * max(3.5, min(14.0, 2.0 * math.pi * f * 1.1))
            ad = _f(sp.get("angle_damping"), 0.0)
            if ad > 0.0:
                zeta = max(0.18, min(0.75, 0.5 * zeta + 0.5 * ad * 0.75))
        cm = _CHAIN_RE.match(unicodedata.normalize("NFKC", self.name))
        self.chain = cm.group(2).strip() if cm else ""
        self.index = int(cm.group(1)) if cm else 0
        if cm:                                             # outer segments: softer, slightly lazier
            omega *= max(0.6, 1.0 - 0.05 * self.index)
            zeta = max(zeta, 0.62 + 0.02 * self.index)     # followers must not resonate down the chain
            gain *= (0.16 if "前髪" in self.chain else 0.2) * (1.0 - 0.05 * self.index)   # 7 segments add up
        self.omega, self.zeta, self.gain = omega, zeta, gain
        self.d, self.mn, self.mx = float(m["defaults"][0]), float(m["min"][0]), float(m["max"][0])
        self.p = 0.0
        self.v = 0.0
        self.out = 0.0

    def value(self, s: float) -> float:
        s = max(-1.0, min(1.0, s))
        up, dn = self.mx - self.d, self.d - self.mn
        x = self.d + s * up if s >= 0.0 else self.d + s * dn
        lo, hi = min(self.mn, self.mx), max(self.mn, self.mx)
        return lo if x < lo else hi if x > hi else x


class SpringPhysics:
    """Universal 2nd-order damped spring physics for puppets whose physics arrived as plain params (MINT / Live2D).
    Each param is a pendulum tip p chasing its anchor A (a weighted mix of head / body / breath channels):
        p'' = w^2 (A - p) - 2 zeta w (p' - A')        output = gain * (p - A)   (lag behind the motion)
    Numbered chains ([0]..[6]サイドアップ左) cascade: segment k chases segment k-1, so a wave runs down the tail.
    Semi-implicit Euler, sub-stepped at <= 1/240 s (unconditionally stable for w <= 14). Disabled -> anchors
    freeze and an envelope eases the output back to the author default."""

    SUBSTEP = 1.0 / 240.0

    def __init__(self, params: list[_PhysParam]) -> None:
        self.params = params
        chains: dict[str, list[_PhysParam]] = {}
        self.solo: list[_PhysParam] = []
        for pp in params:
            (chains.setdefault(pp.chain, []) if pp.chain else self.solo).append(pp)
        self.chains = [sorted(c, key=lambda q: q.index) for c in chains.values()]
        self._prev_in: dict[str, float] | None = None
        self.env = 1.0

    def __len__(self) -> int:
        return len(self.params)

    def reset(self) -> None:
        for pp in self.params:
            pp.p = pp.v = pp.out = 0.0
        self._prev_in = None

    @staticmethod
    def _anchor(pp: _PhysParam, ch: dict[str, float]) -> float:
        return sum(w * ch.get(k, 0.0) for k, w in pp.weights.items())

    def step(self, dt: float, ch: dict[str, float], enabled: bool) -> dict[str, tuple[float, float]]:
        dt = max(1e-4, min(0.1, _f(dt, 1.0 / 60.0)))
        cur = {k: max(-1.5, min(1.5, _f(ch.get(k), 0.0)))
               for k in ("head_x", "head_y", "head_z", "body_x", "body_y", "body_z", "breath")}
        if self._prev_in is None:
            self._prev_in = dict(cur)
            for pp in self.params:                       # start at rest on the current pose (no first-frame kick)
                pp.p, pp.v = self._anchor(pp, cur), 0.0
        if not enabled:
            cur = dict(self._prev_in)                    # frozen anchors: springs ring down
        self.env += ((1.0 if enabled else 0.0) - self.env) * (1.0 - math.exp(-dt / 0.25))
        prev = self._prev_in
        n = max(1, int(math.ceil(dt / self.SUBSTEP)))
        h = dt / n
        for i in range(n):
            a = (i + 1) / n
            inp = {k: prev[k] + (cur[k] - prev[k]) * a for k in cur}
            vel = {k: (cur[k] - prev[k]) / dt for k in cur}
            for pp in self.solo:
                A = self._anchor(pp, inp)
                Av = self._anchor(pp, vel)
                acc = pp.omega * pp.omega * (A - pp.p) - 2.0 * pp.zeta * pp.omega * (pp.v - Av)
                pp.v += acc * h
                pp.p += pp.v * h
                pp.out = A
            for chain in self.chains:
                A = self._anchor(chain[0], inp)
                Av = self._anchor(chain[0], vel)
                for pp in chain:
                    acc = pp.omega * pp.omega * (A - pp.p) - 2.0 * pp.zeta * pp.omega * (pp.v - Av)
                    pp.v += acc * h
                    pp.p += pp.v * h
                    pp.out = A
                    A, Av = pp.p, pp.v                   # next segment hangs from this one
        self._prev_in = cur
        out: dict[str, tuple[float, float]] = {}
        env = self.env
        for pp in self.params:
            if not (math.isfinite(pp.p) and math.isfinite(pp.v)):
                pp.p, pp.v = pp.out, 0.0
            s = env * math.tanh(pp.gain * (pp.p - pp.out))     # soft limit: hair never slams into its range end
            out[pp.name] = (pp.value(s), 0.0)
        return out


def _canon_span(v: float, span: float) -> float:
    return max(-1.0, min(1.0, v / span))


def canonical_channels(p: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Hikari rig output (Hikari param names / units) -> normalized canonical channels.
    Angles: degrees / 30 (head) or / 10 (body); eye open raw (1 = normally open); everything else -1..1 / 0..1."""
    def g(name: str, axis: int = 0, d: float = 0.0) -> float:
        v = p.get(name)
        if v is None:
            return d
        try:
            x = float(v[axis])
        except (TypeError, ValueError, IndexError):
            return d
        return x if math.isfinite(x) else d

    head = p.get("Angle")
    eo_l = max(0.0, min(1.5, g("Left Eye Open", 0, 1.0)))
    eo_r = max(0.0, min(1.5, g("Right Eye Open", 0, 1.0)))
    m_form = max(-1.0, min(1.0, g("VB Mouth", 0, g("Mouth", 0, 0.0))))
    m_open = max(0.0, min(1.0, max(g("VB Mouth", 1, g("Mouth", 1, 0.0)), 0.9 * g("Jaw Open"))))
    funnel = max(0.0, min(1.0, g("Mouth Funnel")))
    smile_m = max(0.0, m_form)
    # Vowel synthesis (for rigs that only expose a/i/u/e/o): wide open -> a, round -> o, pucker -> u, smile -> i / e
    o_w = m_open * funnel
    a_w = m_open * (1.0 - funnel) * (1.0 - 0.55 * smile_m)
    e_w = m_open * (1.0 - funnel) * 0.55 * smile_m
    u_w = funnel * (1.0 - m_open) * 0.9
    i_w = smile_m * max(0.0, 0.35 - m_open) * 1.4
    tot = a_w + i_w + u_w + e_w + o_w
    if tot > 1.0:
        a_w, i_w, u_w, e_w, o_w = (x / tot for x in (a_w, i_w, u_w, e_w, o_w))
    sm_l = g("Eye L Smile")
    sm_r = g("Eye R Smile")
    breath = max(0.0, min(1.0, g("Breath")))
    arms: dict[str, float] = {}
    for s, side in (("l", "L"), ("r", "R")):
        pose = max(0.0, min(1.0, g(f"Arm {side} Pose")))
        gest = max(-1.0, min(1.0, g(f"Arm {side} Gesture")))
        lift = max(-1.0, min(1.0, g(f"Arm {side} Lift")))
        swing = max(-1.0, min(1.0, g(f"Arm {side} Swing")))
        arms[f"arm_{s}_pose"] = pose
        arms[f"arm_{s}_gesture"] = gest
        arms[f"arm_{s}_swing"] = swing
        # upper arm: raised pose (+ lift) with the wave gesture riding on top; resting arms breathe a little
        arms[f"arm_{s}_raise"] = max(-0.4, min(1.0, pose * (0.55 + 0.45 * max(0.0, lift)) + 0.30 * pose * gest
                                         + 0.18 * swing + 0.05 * (breath - 0.4)))
        arms[f"arm_{s}_elbow"] = max(-1.0, min(1.0, pose * (0.30 + 0.45 * max(0.0, gest)) + 0.10 * swing))
        arms[f"arm_{s}_wrist"] = max(-1.0, min(1.0, 0.5 * g(f"Arm {side} Wrist") + 0.25 * pose * gest))
        arms[f"shoulder_{s}"] = max(-1.0, min(1.0, 0.30 * (breath - 0.35) + 0.45 * pose * max(0.0, lift)))
    arms["shoulder"] = 0.5 * (arms["shoulder_l"] + arms["shoulder_r"])
    return {
        **arms,
        "head_x": _canon_span(head[0] if head else 0.0, 30.0) if head else 0.0,
        "head_y": _canon_span(head[1] if head else 0.0, 30.0) if head else 0.0,
        "head_z": _canon_span(g("Angle Z"), 30.0),
        "body_x": _canon_span(g("Body Angle X"), 10.0),
        "body_y": _canon_span(g("Body Angle Y"), 10.0),
        "body_z": _canon_span(g("Body Angle Z") + 0.35 * g("Body Lean"), 10.0),
        "eye_l_open": eo_l,
        "eye_r_open": eo_r,
        "eye_open": 0.5 * (eo_l + eo_r),
        "eye_smile": max(0.0, min(1.0, max(sm_l, sm_r, 0.6 * g("Eye L Squint")))),
        "eye_x": max(-1.0, min(1.0, g("EyeBall", 0))),
        "eye_y": max(-1.0, min(1.0, g("EyeBall", 1))),
        "brow_l_y": max(-1.0, min(1.0, g("Left Brow Y"))),
        "brow_r_y": max(-1.0, min(1.0, g("Right Brow Y"))),
        "brow_y": max(-1.0, min(1.0, 0.5 * (g("Left Brow Y") + g("Right Brow Y")))),
        "brow_l_angle": max(-1.0, min(1.0, g("Brow L Angle"))),
        "brow_r_angle": max(-1.0, min(1.0, g("Brow R Angle"))),
        "brow_form": max(-1.0, min(1.0, 0.5 * (g("Brow L Form") + g("Brow R Form")))),
        "mouth_open": m_open,
        "mouth_form": m_form,
        "mouth_smile": smile_m,
        "mouth_frown": max(0.0, -m_form),
        "vowel_a": a_w, "vowel_i": i_w, "vowel_u": u_w, "vowel_e": e_w, "vowel_o": o_w,
        "cheek_puff": max(0.0, min(1.0, g("Cheek Puff"))),
        "breath": breath,
        "blush": max(0.0, min(1.0, max(g("Cheek"), g("Anime Blush Lines")))),
        "vfx_sparkle": max(0.0, min(1.0, max(g("Eye Star"), g("VFX Sparkles"), g("Sparkle Eyes")))),
        "vfx_ink": max(0.0, min(1.0, max(g("VFX Sparkles"), g("VFX Music")))),     # Mao: playful paint splashes
        "gloom": max(0.0, min(1.0, g("Gloom"))),
        "vfx_heart": max(0.0, min(1.0, g("Eye Heart"))),
    }


def _axis_fn(spec: Spec, d: float, mn: float, mx: float) -> tuple[str, Callable[[float], float]] | None:
    if spec is None:
        return None
    chan, mode = spec
    lo, hi = (mn, mx) if mn <= mx else (mx, mn)

    def clamp(x: float) -> float:
        return lo if x < lo else hi if x > hi else x

    if mode == "bi":
        up, dn = mx - d, d - mn
        return chan, lambda v: clamp(d + v * up if v >= 0.0 else d + v * dn)
    if mode in ("uni", "inv"):
        far = mx if abs(mx - d) > 1e-9 else mn
        span = far - d
        if mode == "inv":
            return chan, lambda v: clamp(d + (1.0 - min(1.0, v)) * span)
        return chan, lambda v: clamp(d + v * span)
    if mode == "eye":
        ref = d if d > 1e-6 else mx
        return chan, lambda v: clamp(v * ref)
    return None


class UniversalParamMapper:
    """Hikari canonical rig frame -> any puppet's params (fast precompiled table, one dict per frame)."""

    def __init__(self, param_meta: list[dict[str, Any]], model_id: str = "",
                 physics_settings: dict[str, dict[str, Any]] | None = None) -> None:
        self.model_id = model_id
        self.meta = [m for m in param_meta if isinstance(m, dict) and m.get("name")]
        self.defaults: dict[str, tuple[float, float]] = param_defaults_of(self.meta)
        names = set(self.defaults)
        self.passthrough = model_id == HIKARI_ID or all(n in names for n in HIKARI_NATIVE_PARAMS)
        self._table: list[tuple[str, tuple[str, Callable] | None, tuple[str, Callable] | None]] = []
        self.matches: dict[str, str] = {}               # target param -> canonical channel group
        self.channel_of: dict[str, str] = {}            # target param -> canonical channel (inspector "mappedTo")
        self.physics_names: list[str] = []
        self.physics: SpringPhysics | None = None
        self._dyn: dict[str, float] = {}                # stateful channels (arm swing / wrist follow-through)
        self._arm_sp: dict[str, list[float]] = {}
        bound_channels: set[str] = set()
        all_channels: set[str] = set()
        if not self.passthrough:
            phys: list[_PhysParam] = []
            psets = physics_settings or {}
            for m in self.meta:
                rule = match_rule(m["name"])
                if rule is None:
                    arm = arm_rule(m["name"]) if not m.get("isVec2") else None
                    if arm is not None and m.get("bound", True):
                        fx = _axis_fn((arm, "bi"), m["defaults"][0], m["min"][0], m["max"][0])
                        if fx is not None:
                            self._table.append((m["name"], fx, None))
                            self.matches[m["name"]] = "arms"
                            self.channel_of[m["name"]] = arm
                            all_channels.add("arms")
                            bound_channels.add("arms")
                        continue
                    lo, hi, d = m["min"][0], m["max"][0], m["defaults"][0]
                    if (m.get("bound", True) and not m.get("isVec2") and min(lo, hi) < d < max(lo, hi)
                            and (m["name"] in psets or is_physics_param(m["name"]))):
                        phys.append(_PhysParam(m, psets.get(m["name"])))
                    continue
                sx, sy, group = rule
                vec2 = bool(m.get("isVec2"))
                if not vec2 and sx is None:
                    continue
                fx = _axis_fn(sx, m["defaults"][0], m["min"][0], m["max"][0])
                fy = _axis_fn(sy, m["defaults"][1], m["min"][1], m["max"][1]) if vec2 else None
                if fx is None and fy is None:
                    continue
                self._table.append((m["name"], fx, fy))
                self.matches[m["name"]] = group
                self.channel_of[m["name"]] = "/".join(s[0] for s in (sx, sy if vec2 else None) if s is not None)
                all_channels.add(group)
                if m.get("bound", True):
                    bound_channels.add(group)
            if phys:
                self.physics = SpringPhysics(phys)
                self.physics_names = [pp.name for pp in phys]
        else:
            bound_channels = set(CHANNEL_ORDER)
            all_channels = set(CHANNEL_ORDER)
        self.mapped_channels = [c for c in CHANNEL_ORDER if c in bound_channels]
        # declared but every matching param is unbound (e.g. Mao's a/i/u/e/o after a MINT conversion)
        self.unbound_channels = [c for c in CHANNEL_ORDER if c in all_channels and c not in bound_channels]

    @property
    def owns_physics(self) -> bool:
        """True when this mapper drives the puppet's physics params itself (engine drivers must stay off: MINT's
        SimplePhysics nodes hang from static root nodes and would force those params back to rest)."""
        return self.physics is not None

    def step_physics(self, dt: float, canonical: dict[str, float], enabled: bool) -> dict[str, tuple[float, float]]:
        """Advance spring physics + arm follow-through one frame from normalized canonical channels
        (`canonical_channels(...)`). Returns {physics param: value} (empty when the puppet has none)."""
        dt = max(1e-4, min(0.1, _f(dt, 1.0 / 60.0)))
        if "arms" in self.mapped_channels:
            # resting arms sway with body motion + breath (lagging springs); the wrist trails the gesture
            for s in ("l", "r"):
                st = self._arm_sp.setdefault(s, [0.0, 0.0, 0.0, 0.0, None, None])   # swing y, v, wrist y, v, prev bx, prev bz
                bx, bz = _f(canonical.get("body_x")), _f(canonical.get("body_z"))
                if st[4] is None:
                    st[4], st[5] = bx, bz
                vbx = (bx - st[4]) / dt
                vbz = (bz - st[5]) / dt
                st[4], st[5] = bx, bz
                sign = 1.0 if s == "l" else -1.0
                h = dt / 4.0
                for _ in range(4):
                    acc = -4.6 * 4.6 * st[0] - 2.0 * 0.35 * 4.6 * st[1] - 9.0 * sign * vbx - 6.0 * vbz
                    st[1] += acc * h
                    st[0] += st[1] * h
                    g = _f(canonical.get(f"arm_{s}_gesture"))
                    acc = 6.0 * 6.0 * (g - st[2]) - 2.0 * 0.7 * 6.0 * st[3]
                    st[3] += acc * h
                    st[2] += st[3] * h
                if not (math.isfinite(st[0]) and math.isfinite(st[1]) and math.isfinite(st[2]) and math.isfinite(st[3])):
                    st[0] = st[1] = st[2] = st[3] = 0.0
                pose = _f(canonical.get(f"arm_{s}_pose"))
                self._dyn[f"arm_{s}_swing_dyn"] = max(-1.0, min(1.0, st[0]))
                self._dyn[f"arm_{s}_wrist_dyn"] = max(-1.0, min(1.0, 1.1 * (st[2] - _f(canonical.get(f"arm_{s}_gesture")))
                                                               * max(0.25, pose)))
        if self.physics is None:
            return {}
        return self.physics.step(dt, canonical, enabled)

    def step(self, canonical_frame: dict[str, tuple[float, float]], dt: float,
             physics_enabled: bool = True) -> dict[str, tuple[float, float]]:
        """Stateful per-frame output: arm follow-through + spring physics + the stateless mapping."""
        if self.passthrough:
            return self.map_frame(canonical_frame)
        ch = canonical_channels(canonical_frame)
        phys = self.step_physics(dt, ch, physics_enabled)
        out = self._map_channels(ch, self._dyn)
        out.update(phys)
        return out

    def map_frame(self, canonical: dict[str, tuple[float, float]],
                  dyn: dict[str, float] | None = None) -> dict[str, tuple[float, float]]:
        if self.passthrough:
            out = dict(self.defaults)
            for k in out:
                v = canonical.get(k)
                if v is not None:
                    out[k] = v
            return out
        return self._map_channels(canonical_channels(canonical), dyn)

    def _map_channels(self, ch: dict[str, float], dyn: dict[str, float] | None) -> dict[str, tuple[float, float]]:
        if dyn:
            ch = dict(ch)
            for s in ("l", "r"):
                sw = dyn.get(f"arm_{s}_swing_dyn", 0.0)
                wr = dyn.get(f"arm_{s}_wrist_dyn", 0.0)
                ch[f"arm_{s}_raise"] = max(-0.4, min(1.0, ch[f"arm_{s}_raise"] + 0.22 * sw))
                ch[f"arm_{s}_elbow"] = max(-1.0, min(1.0, ch[f"arm_{s}_elbow"] + 0.10 * sw))
                ch[f"arm_{s}_wrist"] = max(-1.0, min(1.0, ch[f"arm_{s}_wrist"] + 0.45 * wr + 0.12 * sw))
        out = dict(self.defaults)                        # author-intended rest pose for everything untouched
        for name, fx, fy in self._table:
            dx, dy = out[name]
            x = fx[1](ch[fx[0]]) if fx is not None else dx
            y = fy[1](ch[fy[0]]) if fy is not None else dy
            out[name] = (x, y)
        return out

    def summary(self) -> dict[str, Any]:
        return {"passthrough": self.passthrough, "mappedChannels": list(self.mapped_channels),
                "unboundChannels": list(self.unbound_channels), "matchedParams": dict(self.matches),
                "physicsCount": len(self.physics_names), "physicsParams": list(self.physics_names)}


# ----------------------------------------------------------------------------- live parameter inspector
CAPABILITY_KEYS = ("head", "body", "eyes", "gaze", "brows", "mouth", "arms", "blush", "breath")
_ARMISH_RE = re.compile(r"腕|手|肩|arm|hand|shoulder|elbow|wrist", re.I)
_TOGGLE_NAME_RE = re.compile(r"表示|show|toggle|enable|(?:^|[^a-z])(?:on|off)(?:$|[^a-z])", re.I)


def is_toggle_param(m: dict[str, Any]) -> bool:
    if m.get("isVec2"):
        return False
    mn, mx, d = float(m["min"][0]), float(m["max"][0]), float(m["defaults"][0])
    if mn == 0.0 and mx == 1.0 and d in (0.0, 1.0):
        return True
    return bool(_TOGGLE_NAME_RE.search(unicodedata.normalize("NFKC", str(m.get("name", "")))))


def extract_physics_settings(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{param name: SimplePhysics node settings} for every physics driver in the puppet (MINT keeps Cubism's)."""
    by_uuid = {p.get("uuid"): str(p.get("name") or "") for p in model.get("param") or [] if isinstance(p, dict)}
    out: dict[str, dict[str, Any]] = {}

    def walk(n: Any, depth: int) -> None:
        if not isinstance(n, dict) or depth > 64:
            return
        if n.get("type") == "SimplePhysics":
            name = by_uuid.get(n.get("param"))
            if name and name not in out:
                out[name] = {k: n.get(k) for k in ("frequency", "angle_damping", "length", "gravity", "output_scale")}
        for c in n.get("children") or []:
            walk(c, depth + 1)

    walk(model.get("nodes"), 0)
    return out


def build_param_inspector(param_meta: list[dict[str, Any]], mapper: "UniversalParamMapper | None",
                          current: dict[str, tuple[float, float]] | None = None,
                          overrides: dict[str, tuple[float, float]] | None = None,
                          native_physics: list[str] | tuple[str, ...] = (),
                          extras_names: tuple[str, ...] | frozenset[str] = ()) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bound params of the live puppet, categorized for the Studio inspector + the capability summary.
    `mapper` None = Hikari's native rig (everything is driven natively; VFX blend shapes count as extras)."""
    cur = current or {}
    ovr = overrides or {}
    nat_phys = set(native_physics)
    phys = set(mapper.physics_names) if mapper is not None else set()
    out: list[dict[str, Any]] = []
    for m in param_meta:
        if not isinstance(m, dict) or not m.get("name") or not m.get("bound", True):
            continue
        name = m["name"]
        mapped: str | None = None
        if mapper is None or mapper.passthrough:
            if name in nat_phys:
                cat = "physics"
            elif name in extras_names:
                cat = "extras"
            elif _ARMISH_RE.search(name):
                cat = "arms"
            else:
                cat = "auto"
            mapped = None if cat == "extras" else "native"
        elif name in phys:
            cat, mapped = "physics", "spring"
        else:
            group = mapper.matches.get(name)
            mapped = mapper.channel_of.get(name)
            if group == "arms" or (group is None and _ARMISH_RE.search(name) and not _TOGGLE_WORDS_RE.search(name)):
                cat = "arms"
            elif group is None or group == "vfx":
                cat = "extras"
            else:
                cat = "auto"
        d = m.get("defaults") or [0.0, 0.0]
        c = cur.get(name, (d[0], d[1]))
        out.append({
            "name": name, "category": cat, "isVec2": bool(m.get("isVec2")), "isToggle": is_toggle_param(m),
            "min": [float(m["min"][0]), float(m["min"][1])], "max": [float(m["max"][0]), float(m["max"][1])],
            "default": [float(d[0]), float(d[1])], "current": [round(float(c[0]), 5), round(float(c[1]), 5)],
            "overridden": name in ovr, "mappedTo": mapped,
        })
    if mapper is None or mapper.passthrough:
        caps: dict[str, Any] = {k: True for k in CAPABILITY_KEYS}
    else:
        caps = {k: (k in mapper.mapped_channels) for k in CAPABILITY_KEYS}
    caps["physicsCount"] = sum(1 for p in out if p["category"] == "physics")
    caps["extrasCount"] = sum(1 for p in out if p["category"] == "extras")
    return out, caps


# ----------------------------------------------------------------------------- inspection + auto framing

@dataclass
class PuppetInfo:
    path: Path
    name: str
    artist: str
    param_meta: list[dict[str, Any]]
    bounds: tuple[float, float, float, float]          # min_x, max_x, min_y, max_y (rest pose, world units)
    face_box: tuple[float, float, float, float] | None
    eye_center: tuple[float, float] | None
    head_top: float
    recommended_camera: dict[str, float]
    mapped_channels: list[str] = field(default_factory=list)
    unbound_channels: list[str] = field(default_factory=list)
    passthrough: bool = False
    physics_settings: dict[str, dict[str, Any]] = field(default_factory=dict)   # SimplePhysics outputs
    physics_params: list[str] = field(default_factory=list)                     # spring-driven by the mapper
    capabilities: dict[str, Any] = field(default_factory=dict)

    @property
    def param_defaults(self) -> dict[str, tuple[float, float]]:
        return param_defaults_of(self.param_meta)

    def make_mapper(self, model_id: str) -> "UniversalParamMapper":
        return UniversalParamMapper(self.param_meta, model_id, self.physics_settings)

    @property
    def bound_param_meta(self) -> list[dict[str, Any]]:
        """Same shape as the Studio's legacy `paramsMeta` (bound params only)."""
        return [{k: m[k] for k in ("name", "isVec2", "min", "max", "defaults")} for m in self.param_meta if m.get("bound")]

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name, "artist": self.artist, "paramCount": len(self.param_meta),
            "boundParamCount": sum(1 for m in self.param_meta if m.get("bound")),
            "mappedChannels": list(self.mapped_channels), "unboundChannels": list(self.unbound_channels),
            "recommendedCamera": dict(self.recommended_camera),
            "bounds": [round(v, 1) for v in self.bounds],
            "capabilities": dict(self.capabilities),
            "physicsCount": int(self.capabilities.get("physicsCount", 0)),
            "mouthMapped": bool(self.capabilities.get("mouth", False)),
        }


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes))


def _params_matching(model: dict[str, Any], groups: tuple[str, ...], axes_ok: Callable[[Spec, Spec], bool] | None = None) -> list[dict]:
    out = []
    for p in model.get("param") or []:
        r = match_rule(str(p.get("name") or ""))
        if r and r[2] in groups and (axes_ok is None or axes_ok(r[0], r[1])):
            out.append(p)
    return out


def recommend_camera(bounds: tuple[float, float, float, float], head_top: float, eye_center: tuple[float, float] | None,
                     face_box: tuple[float, float, float, float] | None,
                     width: int = FRAME_W, height: int = FRAME_H) -> dict[str, float]:
    """Upper-body portrait: crown (incl. hat / ahoge) just below the top edge, eyes on the upper third, chest/waist at
    the bottom. Ratios calibrated on Hikari's hand-tuned camera (scale 1.82, y 355 at 960x1080)."""
    min_x, max_x, min_y, max_y = bounds
    total_h = max(1.0, max_y - min_y)
    if eye_center is not None and eye_center[1] > head_top:
        ex, ey = eye_center
        crown = ey - head_top                           # eye line -> top of hair / hat
        unit = crown
        if face_box is not None:
            span = max(1.0, face_box[1] - face_box[0])  # outer-eye span
            # a tall hat / ahoge inflates the crown: size the body from the face instead (Hikari crown = 1.25 spans)
            unit = max(0.75 * span, min(crown, 1.3 * span))
        # Hikari: crown 169 -> frame 593 world units (= 1080 / 1.82): eyes at ~36 %, crown 7.5 % below the top edge
        bottom = ey + 2.2 * unit
        frame_h = max((bottom - head_top) / 0.925, 0.22 * total_h)
        top = head_top - 0.075 * frame_h
        cy = top + 0.5 * frame_h
        cx = ex
    else:                                                # no face landmarks: frame the top 45 % of the figure
        frame_h = 0.45 * total_h
        top = min_y - 0.04 * frame_h
        cy = top + 0.5 * frame_h
        cx = 0.5 * (min_x + max_x)
    frame_h = min(frame_h, total_h * 1.1)
    scale = height / frame_h
    scale = max(0.2, min(6.0, scale))
    return {"scale": round(scale, 4), "x": round(-cx, 2), "y": round(-cy, 2)}


_CACHE: dict[tuple[str, int, int], PuppetInfo] = {}
_CACHE_LOCK = threading.Lock()


def inspect_puppet(path: Path, model_id: str = "", width: int = FRAME_W, height: int = FRAME_H) -> PuppetInfo:
    """Full inspection (cached by path + mtime + size; an 8 MB puppet takes ~0.2 s the first time)."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    model = read_puppet_json(path)
    meta = model.get("meta") or {}
    name = str(meta.get("name") or "").strip()
    if not name or name.lower() in ("model", "untitled", "puppet"):
        name = path.stem
    authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
    artist = str(meta.get("artist") or ", ".join(str(a) for a in authors) or meta.get("rigger") or "")
    pmeta = extract_param_meta(model)

    parts = _visible_parts(model)
    bounds = _union([pg.box for pg in parts]) or (-500.0, 500.0, -1000.0, 1000.0)

    # Landmarks: eyes = parts deformed by an eye-open param; head = parts deformed by a head-angle param.
    uid_parts = {pg.uuid: pg for pg in parts}
    def bound_nodes(params: list[dict]) -> set[int]:
        s: set[int] = set()
        for p in params:
            for b in p.get("bindings") or []:
                if b.get("param_name") == "deform" and isinstance(b.get("node"), int):
                    s.add(b["node"])
        return s

    eye_params = _params_matching(model, ("eyes",), lambda sx, sy: sx is not None and sx[1] in ("eye", "inv"))
    head_params = _params_matching(model, ("head",))
    eye_nodes = bound_nodes(eye_params) & set(uid_parts)
    head_nodes = bound_nodes(head_params) & set(uid_parts)
    eye_box = _union([uid_parts[u].box for u in eye_nodes])
    eye_center = None if eye_box is None else (0.5 * (eye_box[0] + eye_box[1]), 0.5 * (eye_box[2] + eye_box[3]))
    head_top = bounds[2]
    face_box = None
    if head_nodes:
        hb = [uid_parts[u].box for u in head_nodes]
        if eye_center is not None:
            # ignore head-bound meshes that hang far below the eyes (long hair / twin tails): keep the crown honest
            hb_top = [b for b in hb if b[2] < eye_center[1]]
            head_top = min((b[2] for b in hb_top), default=bounds[2])
            ey = eye_center[1]
            near = [b for b in hb if b[3] > ey and b[2] < ey]
            face_box = _union(near) if near else None
        else:
            head_top = min(b[2] for b in hb)
    cam = recommend_camera(bounds, head_top, eye_center, eye_box, width, height)
    psets = extract_physics_settings(model)
    mapper = UniversalParamMapper(pmeta, model_id, psets)
    _, caps = build_param_inspector(pmeta, mapper, native_physics=list(psets))
    info = PuppetInfo(path=path, name=name, artist=artist, param_meta=pmeta, bounds=bounds, face_box=face_box,
                      eye_center=eye_center, head_top=head_top, recommended_camera=cam,
                      mapped_channels=mapper.mapped_channels, unbound_channels=mapper.unbound_channels,
                      passthrough=mapper.passthrough, physics_settings=psets,
                      physics_params=list(mapper.physics_names), capabilities=caps)
    with _CACHE_LOCK:
        if len(_CACHE) > 32:
            _CACHE.clear()
        _CACHE[key] = info
    return info


# ----------------------------------------------------------------------------- library ids / importer

_ID_BAD = re.compile(r"[^\w\-. ]+", re.UNICODE)
MODEL_ID_RE = re.compile(r"^[\w\-][\w\-. ]{0,63}$", re.UNICODE)
_WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def valid_model_id(model_id: str) -> bool:
    s = str(model_id or "")
    return (bool(MODEL_ID_RE.fullmatch(s)) and ".." not in s and not s.endswith((".", " "))
            and s.lower() not in _WIN_RESERVED)


def safe_stem(filename: str) -> str:
    """A filesystem- and id-safe stem from an uploaded filename (directory parts are ignored)."""
    base = re.split(r"[\\/]", str(filename or ""))[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    stem = unicodedata.normalize("NFKC", stem)
    stem = _ID_BAD.sub("_", stem).strip(" ._")
    stem = re.sub(r"_+", "_", stem)[:48] or "puppet"
    if stem.lower() in _WIN_RESERVED or stem.lower() == HIKARI_ID:
        stem = f"{stem}_model"
    return stem


def unique_dest(library_dir: Path, stem: str, ext: str) -> Path:
    taken = {p.stem.lower() for p in library_dir.glob("*") if p.suffix.lower() in PUPPET_EXTS}
    cand, n = stem, 2
    while cand.lower() in taken or cand.lower() == HIKARI_ID:
        cand = f"{stem}-{n}"
        n += 1
    return library_dir / f"{cand}{ext}"


class ImportError_(ValueError):
    """Bad upload (-> HTTP 400/415)."""


def _check_zip(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ImportError_("Archive has too many files")
    total = 0
    for zi in infos:
        n = zi.filename.replace("\\", "/")
        if n.startswith("/") or re.match(r"^[A-Za-z]:", n) or any(seg == ".." for seg in n.split("/")):
            raise ImportError_(f"Unsafe path in archive: {zi.filename[:80]}")
        if zi.flag_bits & 0x1:
            raise ImportError_("Encrypted archives are not supported")
        total += max(0, int(zi.file_size))
        if total > MAX_ZIP_UNCOMPRESSED:
            raise ImportError_("Archive is too large once extracted")
    return infos


def _safe_extract(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], dest: Path) -> None:
    root = dest.resolve()
    for zi in infos:
        target = (dest / zi.filename.replace("\\", "/")).resolve()
        if not target.is_relative_to(root):
            raise ImportError_("Unsafe path in archive")
        if zi.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)


def convert_live2d_zip(zip_path: Path, stem: str, work_dir: Path, mint_exe: Path = DEFAULT_MINT_CLI) -> Path:
    """Extract a Live2D bundle, run `mint-cli export <moc3> <out_stem> --inp`, return the produced .inp."""
    if not Path(mint_exe).is_file():
        raise ImportError_("MINT converter (mint-cli.exe) is not installed")
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise ImportError_("Not a valid .zip archive") from None
    with zf:
        infos = _check_zip(zf)
        src = work_dir / "src"
        src.mkdir(parents=True, exist_ok=True)
        _safe_extract(zf, infos, src)
    mocs = sorted(src.rglob("*.moc3"), key=lambda p: (len(p.parts), -p.stat().st_size))
    if not mocs:
        raise ImportError_("No .moc3 file found in the archive")
    # prefer the moc3 referenced by a model3.json
    for mj in src.rglob("*.model3.json"):
        try:
            ref = json.loads(mj.read_text(encoding="utf-8", errors="replace")).get("FileReferences", {}).get("Moc")
        except Exception:
            ref = None
        if isinstance(ref, str):
            cand = (mj.parent / ref).resolve()
            if cand.is_file() and cand.is_relative_to(src.resolve()):
                mocs.insert(0, cand)
                break
    moc = mocs[0]
    out_dir = work_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / stem
    try:
        proc = subprocess.run([str(mint_exe), "export", str(moc), str(out_stem), "--inp"], cwd=str(work_dir),
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=MINT_TIMEOUT_SEC,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ImportError_("MINT conversion timed out") from None
    produced = out_stem.with_suffix(".inp")
    if proc.returncode != 0 or not produced.is_file():
        tail = (proc.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()[-3:]
        raise ImportError_("MINT could not convert this model: " + (" | ".join(tail) or f"exit {proc.returncode}"))
    read_puppet_json(produced)                          # validate the conversion
    return produced


def import_puppet_file(upload: Path, original_name: str, library_dir: Path, mint_exe: Path = DEFAULT_MINT_CLI,
                       tmp_root: Path | None = None) -> Path:
    """Validate + install an uploaded .inp / .inx / Live2D .zip into `library_dir`. Returns the installed path.
    Only the clean <name>.inp (or the validated .inp / .inx) is copied; every temp file is removed."""
    upload = Path(upload)
    base = re.split(r"[\\/]", str(original_name or ""))[-1]
    ext = ("." + base.rsplit(".", 1)[-1].lower()) if "." in base else ""
    if ext not in IMPORT_EXTS:
        raise ImportError_("Upload an Inochi2D .inp / .inx or a Live2D .zip bundle")
    if upload.stat().st_size > MAX_IMPORT_BYTES:
        raise ImportError_("File is larger than 150 MB")
    library_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(base)
    if ext in PUPPET_EXTS:
        read_puppet_json(upload)
        dest = unique_dest(library_dir, stem, ext)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(upload, tmp)
        os.replace(tmp, dest)
        return dest
    work = Path(tempfile.mkdtemp(prefix="hikari_mint_", dir=str(tmp_root) if tmp_root else None))
    try:
        produced = convert_live2d_zip(upload, stem, work, mint_exe)
        try:                                             # MINT drops Cubism blend-shape mouths: rebuild if clean
            synthesize_missing_mouth(produced)
        except Exception:
            pass                                         # never fail an import over the optional mouth
        dest = unique_dest(library_dir, stem, ".inp")
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(produced, tmp)
        os.replace(tmp, dest)
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- mouth synthesis (MINT imports)
MOUTH_VOWEL_OPEN = (("あ", 1.0), ("お", 0.8), ("え", 0.55), ("う", 0.32), ("い", 0.25))
_MOUTH_NODE_NAMES = {"口", "mouth", "mouths", "くち"}


def _write_puppet_json(path: Path, model: dict[str, Any]) -> None:
    """Replace the JSON payload of an .inp, keeping the texture / extension sections byte-for-byte (atomic)."""
    path = Path(path)
    raw = path.read_bytes()
    jlen = int.from_bytes(raw[8:12], "big")
    payload = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(raw[:8] + len(payload).to_bytes(4, "big") + payload + raw[12 + jlen:])
    read_puppet_json(tmp)
    os.replace(tmp, path)


def _smoothstep(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def synthesize_missing_mouth(inp_path: Path, write: bool = True) -> dict[str, Any]:
    """Give a converted puppet with NO bound mouth param a clean anime mouth-open, but only when the art allows it.

    Requires the classic layered Cubism mouth: a cavity Part (the mask source of teeth / tongue) hidden behind an
    upper and a lower skin-cover Part that meet at the closed lip line. Opening slides the lower cover's lip band
    down (its lower edge stays put, so the cover keeps hiding the cavity below), lifts the upper cover a little,
    and carries the lower teeth / tongue and lower-lip highlights along - nothing is stretched past the art.
    Deforms are bound to the puppet's empty vowel params (あ 1.0, お 0.8, え 0.55, う 0.32, い 0.25) or a new
    "Mouth Open" param. Returns {"ok": bool, "reason" | "params", ...}; leaves the file untouched unless ok."""
    path = Path(inp_path)
    model = read_puppet_json(path)
    params = [p for p in model.get("param") or [] if isinstance(p, dict)]
    for p in params:
        r = match_rule(str(p.get("name") or ""))
        if r is not None and r[2] == "mouth" and p.get("bindings") and (r[0] or r[1]) and \
                any((s or ("", ""))[0] in ("mouth_open", "vowel_a", "vowel_o") for s in (r[0], r[1])):
            return {"ok": False, "reason": "mouth already bound"}

    def find_mouth(n: Any, depth: int) -> dict | None:
        if not isinstance(n, dict) or depth > 64:
            return None
        if norm_name(n.get("name") or "") in _MOUTH_NODE_NAMES:
            return n
        for c in n.get("children") or []:
            r = find_mouth(c, depth + 1)
            if r is not None:
                return r
        return None

    mouth = find_mouth(model.get("nodes"), 0)
    if mouth is None:
        return {"ok": False, "reason": "no mouth group node"}
    parts: list[dict] = []

    def collect(n: dict, depth: int) -> None:
        for c in n.get("children") or []:
            if isinstance(c, dict) and depth < 8:
                if c.get("type") in ("Part", "AnimatedPart") and c.get("enabled", True) is not False:
                    parts.append(c)
                collect(c, depth + 1)

    collect(mouth, 0)
    if len(parts) < 3:
        return {"ok": False, "reason": "mouth has fewer than 3 art layers (baked single texture)"}
    for c in parts:                                     # deforms are local: all layers must share one space
        tr = c.get("transform") or {}
        if any(abs(_f(v)) > 1e-6 for v in (tr.get("trans") or [0, 0, 0])[:2]) or \
                abs(_f((tr.get("rot") or [0, 0, 0])[2] if len(tr.get("rot") or []) > 2 else 0)) > 1e-6:
            return {"ok": False, "reason": "mouth layers have their own transforms"}
    deform, _ = _rest_state(model)
    geo: dict[int, dict[str, Any]] = {}
    for c in parts:
        verts = (c.get("mesh") or {}).get("verts") or []
        uid = c.get("uuid")
        if not isinstance(uid, int) or len(verts) < 6:
            continue
        offs = deform.get(uid) or []
        xs, ys = [], []
        for k in range(len(verts) // 2):
            ox, oy = (offs[k] if k < len(offs) else (0.0, 0.0))
            xs.append(_f(verts[2 * k]) + ox)
            ys.append(_f(verts[2 * k + 1]) + oy)
        masks = [m.get("source") for m in (c.get("masks") or []) if isinstance(m, dict)]
        geo[uid] = {"node": c, "xs": xs, "ys": ys, "n": len(verts) // 2, "masks": masks,
                    "box": (min(xs), max(xs), min(ys), max(ys)), "z": _f(c.get("zsort"))}
    sources: dict[int, int] = {}
    for uid, g in geo.items():
        for s in g["masks"]:
            if s in geo and s != uid:
                sources[s] = sources.get(s, 0) + 1
    if not sources:
        return {"ok": False, "reason": "no masked mouth cavity (mouth is a single drawn shape)"}
    cav_id = max(sources, key=lambda u: (sources[u], (geo[u]["box"][1] - geo[u]["box"][0]) * (geo[u]["box"][3] - geo[u]["box"][2])))
    cav = geo[cav_id]
    cx0, cx1, cy0, cy1 = cav["box"]
    cw, ch = max(1e-3, cx1 - cx0), max(1e-3, cy1 - cy0)

    def xcover(b: tuple) -> float:
        return max(0.0, min(b[1], cx1) - max(b[0], cx0)) / cw

    lower = upper = None
    for uid, g in geo.items():
        if uid == cav_id or cav_id in g["masks"] or g["z"] >= cav["z"]:      # covers draw in front (lower zsort)
            continue
        b = g["box"]
        if xcover(b) < 0.85:
            continue
        if b[2] > cy0 + 0.15 * ch and b[3] >= cy1 - 0.02 * ch and b[2] < cy1:
            if lower is None or (b[3] - b[2]) > (geo[lower]["box"][3] - geo[lower]["box"][2]):
                lower = uid
        elif b[3] < cy1 - 0.15 * ch and b[2] <= cy0 + 0.02 * ch and b[3] > cy0:
            if upper is None or (b[3] - b[2]) > (geo[upper]["box"][3] - geo[upper]["box"][2]):
                upper = uid
    if lower is None or upper is None:
        return {"ok": False, "reason": "no separate upper / lower lip covers over the mouth cavity"}
    lb, ub = geo[lower]["box"], geo[upper]["box"]
    if lb[2] > ub[3]:
        return {"ok": False, "reason": "lip covers leave a seam (they do not meet)"}
    seam = 0.5 * (lb[2] + ub[3])
    delta = 0.36 * (cy1 - seam)                         # full あ: a third of the hidden cavity below the lips
    if delta <= 1e-3 or delta > 0.6 * (lb[3] - lb[2]):
        return {"ok": False, "reason": "mouth cavity too shallow to open cleanly"}

    # lens-shaped opening: the corners stay pinned at the lip-line ends (the cavity is wider than the lines, so a
    # dropped corner would expose it), the centre opens most
    lx0 = max(lb[0], ub[0], cx0)
    lx1 = min(lb[1], ub[1], cx1)
    lmid, lhalf = 0.5 * (lx0 + lx1), max(1e-3, 0.5 * (lx1 - lx0) * 0.86)

    def xprof(x: float) -> float:
        u = (x - lmid) / lhalf
        return max(0.0, 1.0 - u * u) ** 0.65

    r0 = lb[2] + 0.30 * (lb[3] - lb[2])                 # lower cover: rigid lip band, then compress to its edge

    def w_lower(y: float) -> float:
        return 1.0 if y <= r0 else _smoothstep((lb[3] - y) / max(1e-3, lb[3] - r0))

    r1 = ub[3] - 0.30 * (ub[3] - ub[2])

    def w_upper(y: float) -> float:
        return 1.0 if y >= r1 else _smoothstep((y - ub[2]) / max(1e-3, r1 - ub[2]))

    offsets: dict[int, list[list[float]]] = {}
    moved: dict[str, list[int]] = {"lowerCover": [lower], "upperCover": [upper], "jaw": [], "lowerLipDecor": []}
    for uid, g in geo.items():
        if uid == cav_id:
            continue
        if uid == lower:
            offsets[uid] = [[0.0, delta * w_lower(y) * xprof(x)] for x, y in zip(g["xs"], g["ys"])]
        elif uid == upper:
            offsets[uid] = [[0.0, -0.22 * delta * w_upper(y) * xprof(x)] for x, y in zip(g["xs"], g["ys"])]
        elif cav_id in g["masks"]:
            cyc = 0.5 * (g["box"][2] + g["box"][3])
            if cyc > seam:                                # lower teeth / tongue ride the jaw
                offsets[uid] = [[0.0, 0.55 * delta * xprof(x)] for x in g["xs"]]
                moved["jaw"].append(uid)
        else:
            b = g["box"]
            cyc = 0.5 * (b[2] + b[3])
            if seam - 0.05 * ch < cyc < lb[3] and b[0] >= lb[0] - 1 and b[1] <= lb[1] + 1:
                offsets[uid] = [[0.0, delta * w_lower(y) * xprof(x)] for x, y in zip(g["xs"], g["ys"])]
                moved["lowerLipDecor"].append(uid)

    # target params: empty vowel params, else an unbound open param, else a new "Mouth Open"
    by_name = {str(p.get("name") or ""): p for p in params}
    targets: list[tuple[dict, float]] = []
    for nm, amt in MOUTH_VOWEL_OPEN:
        p = by_name.get(nm)
        if p is not None and not p.get("bindings") and not p.get("is_vec2"):
            targets.append((p, amt))
    if not targets:
        for p in params:
            r = match_rule(str(p.get("name") or ""))
            if r and r[0] and r[0][0] == "mouth_open" and not p.get("bindings") and not p.get("is_vec2"):
                targets.append((p, 1.0))
                break
    if not targets:
        uid_new = max([int(p.get("uuid")) for p in params if isinstance(p.get("uuid"), int)] + [5000]) + 1
        newp = {"name": "Mouth Open", "uuid": uid_new, "is_vec2": False, "min": [0.0, 0.0], "max": [1.0, 1.0],
                "defaults": [0.0, 0.0], "axis_points": [[0.0, 1.0], [0.0, 1.0]], "merge_mode": "Additive",
                "bindings": []}
        model["param"].append(newp)
        targets.append((newp, 1.0))
    for p, amt in targets:
        mn, mx = _pair(p.get("min")), _pair(p.get("max"), (1.0, 1.0))
        if (mn[0], mx[0]) != (0.0, 1.0):
            continue
        p["axis_points"] = [[0.0, 1.0], [0.0, 1.0]]
        binds = []
        for uid, offs in offsets.items():
            zero = [[0.0, 0.0] for _ in offs]
            full = [[round(dx * amt, 4), round(dy * amt, 4)] for dx, dy in offs]
            binds.append({"node": uid, "param_name": "deform", "interpolate_mode": "Linear",
                          "isSet": [[True, True], [True, True]], "values": [[zero, zero], [full, full]]})
        p["bindings"] = binds
    bound = [p["name"] for p, _ in targets if p.get("bindings")]
    if not bound:
        return {"ok": False, "reason": "no usable 0..1 mouth param"}
    if write:
        _write_puppet_json(path, model)
    return {"ok": True, "params": bound, "cavity": cav_id, "delta": round(delta, 2), "seam": round(seam, 2),
            "layers": moved}


def sanitize_model_cameras(value: Any) -> dict[str, dict[str, float]]:
    """stage.savedModelCameras {model_id: {scale, x, y}} - finite + clamped, <= 64 entries."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for mid, cam in list(value.items())[:64]:
        if not isinstance(mid, str) or not valid_model_id(mid) or not isinstance(cam, dict):
            continue
        s, x, y = _f(cam.get("scale"), float("nan")), _f(cam.get("x"), float("nan")), _f(cam.get("y"), float("nan"))
        if not all(math.isfinite(v) for v in (s, x, y)):
            continue
        out[mid] = {"scale": max(0.2, min(6.0, s)), "x": max(-4000.0, min(4000.0, x)), "y": max(-4000.0, min(4000.0, y))}
    return out
