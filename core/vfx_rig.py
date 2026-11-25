"""Hikari v29 rig upgrades used by core/customizer.py on every outfit build.

1. Promoted Midori single-mesh DEFORM raised wave arms
   The rigid upper / forearm / hand wave-arm Parts (+ elbow & wrist pivot nodes) are replaced by ONE continuous mesh
   Part per side, skinned with smooth angle-blend vertex DEFORM bindings. The baking, meshing and skinning code lives
   in tools/build_midori_deform_arms.py (the verified experiment) and is imported from there, so both stay identical.
   The bake only depends on the final wave textures + the wave node layout, so it is cached in memory and on disk
   (assets/hd_parts/wave/_midori_cache/<key>.npz) and outfit switches never re-bake.

2. Mao / Neuro-style expression VFX + blend-shape micro-expressions (10 params)
   VFX Sparkles / Hearts / Sweat / Anger / Shock / Music, Eye Star, Eye Heart, Cheek Puff, Cat Mouth.
   All art is drawn procedurally at 4x supersampling in premultiplied float, downsampled with LANCZOS and stored as
   premultiplied PNGs (the .inp convention). Hikari's head motion is per-mesh DEFORM (the Head node never moves),
   so every new Part gets "follow" deform bindings: VFX follow an affine fit of the face + front-hair deform, eye
   FX follow the iris mesh (barycentric) and copy its blink opacity, the cheek blush follows the face mesh.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.paths import studio_root

ROOT = studio_root()
MIDORI_TOOL = ROOT / "tools" / "build_midori_deform_arms.py"
MIDORI_CACHE_DIR = ROOT / "assets" / "hd_parts" / "wave" / "_midori_cache"
MIDORI_VERSION = "midori-v1"
VFX_CACHE_DIR = ROOT / "assets" / "hd_parts" / "_vfx_cache"
WAVE_TEXTURES = (53, 54, 55, 56, 57, 58)
TEX_NONE = 4294967295

_TOOL: list[Any] = []


def _tool() -> Any:
    """tools/build_midori_deform_arms.py, loaded once as a module (single source of truth for the arm rig)."""
    if not _TOOL:
        spec = importlib.util.spec_from_file_location("hikari_midori_arms", MIDORI_TOOL)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _TOOL.append(mod)
    return _TOOL[0]


def _parse(header: bytes) -> dict[str, Any]:
    return json.loads(header[12:])


def _dump(header: bytes, puppet: dict[str, Any]) -> bytes:
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _index(root: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_id: dict[int, dict[str, Any]] = {}
    parent: dict[int, dict[str, Any]] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        by_id[n["uuid"]] = n
        for c in n.get("children", ()):
            parent[c["uuid"]] = n
            stack.append(c)
    return by_id, parent


# =================================================================================================================
# 1. Midori single-mesh DEFORM raised arms
# =================================================================================================================
def _arm_specs(B: Any) -> dict[str, Callable[[float], tuple[float, float]]]:
    """param value v in [-1, 1] -> (elbow deg, wrist deg) for the R side (L is mirrored by sg)."""
    return {
        "Gesture": lambda v: (-B.GESTURE_ELBOW * v, -B.GESTURE_WRIST * v),
        "Elbow": lambda v: ((-B.ELBOW_NEG) * v if v < 0 else B.ELBOW_POS * v, 0.0),
        "Wrist": lambda v: (0.0, B.WRIST_RANGE * v),
        "Swing": lambda v: (-B.SWING_ELBOW * v, -B.SWING_WRIST * v),
    }


def midori_bake_key(header: bytes, texs: list[tuple[int, bytes]]) -> str | None:
    B = _tool()
    by_id, _ = _index(_parse(header)["nodes"])
    h = hashlib.sha1(MIDORI_VERSION.encode())
    consts = [B.GESTURE_ELBOW, B.GESTURE_WRIST, B.ELBOW_NEG, B.ELBOW_POS, B.WRIST_RANGE, B.SWING_ELBOW, B.SWING_WRIST,
              B.SIL_MARGIN, B.POLY_EPS, B.EDGE_MAX, B.EDGE_MAX_JOINT, B.GRID, B.GRID_ELBOW, B.GRID_WRIST,
              B.ELBOW_BAND, B.WRIST_BAND, B.WRIST_SHIFT, B.ELBOW_REPAIR_R, B.LINE_W]
    h.update(json.dumps(consts).encode())
    for side, L in B.SIDES.items():
        for k in ("upper", "elbow", "forearm", "wrist", "hand"):
            n = by_id.get(L[k])
            if n is None:
                return None
            h.update(json.dumps({kk: vv for kk, vv in n.items() if kk != "children"}, sort_keys=True).encode())
            if n.get("type") == "Part":
                h.update(hashlib.sha1(texs[n["textures"][0]][1]).digest())
    return h.hexdigest()[:20]


def bake_midori_arms(header: bytes, texs: list[tuple[int, bytes]], key: str) -> dict[str, Any]:
    """Bake (or load from the disk cache) both unified wave-arm meshes + their deform offsets."""
    B = _tool()
    path = MIDORI_CACHE_DIR / f"{key}.npz"
    out: dict[str, Any] = {"key": key}
    if path.exists():
        try:
            z = np.load(path, allow_pickle=False)
            for side in B.SIDES:
                d = {k: z[f"{side}_{k}"] for k in ("verts", "uvs", "tris", "posed")}
                d["png"] = z[f"{side}_png"].tobytes()
                d["report"] = json.loads(str(z[f"{side}_report"]))
                out[side] = d
            out["cached"] = True
            return out
        except Exception:
            pass
    by_id, _ = _index(_parse(header)["nodes"])
    specs = _arm_specs(B)
    for side, L in B.SIDES.items():
        sg = L["sg"]
        upper, elbow, forearm = by_id[L["upper"]], by_id[L["elbow"]], by_id[L["forearm"]]
        wrist, hand = by_id[L["wrist"]], by_id[L["hand"]]
        Mu = B.node_matrix(upper)
        Me = B.node_matrix(elbow)
        Mf = Me @ B.node_matrix(forearm)
        Mw = Me @ B.node_matrix(wrist)
        Mh = Mw @ B.node_matrix(hand)
        E, Wp, S = Me[:2, 2].copy(), Mw[:2, 2].copy(), Mu[:2, 2].copy()
        layers = sorted([(upper, Mu), (forearm, Mf), (hand, Mh)], key=lambda t: -float(t[0].get("zsort", 0)))
        canvas, q0, ppu = B.bake_arm(layers, texs)
        canvas, n_stroke = B.repair_elbow_seam(canvas, q0, ppu, E)
        skin = B.ArmSkin(E, Wp, S, canvas[..., 3], q0, ppu)
        verts, tri, uvs, uncovered = B.build_mesh(canvas[..., 3], q0, ppu, skin.zone)
        we_v, ww_v = skin.weights(verts)
        tri = tri[np.argsort((we_v + ww_v)[tri].mean(1), kind="stable")]
        posed = np.zeros((len(specs), len(B.VALUES5), len(verts), 2), np.float32)
        for pi, fn in enumerate(specs.values()):
            for vi, v in enumerate(B.VALUES5):
                e_deg, w_deg = fn(v)
                posed[pi, vi] = skin.pose(verts, sg * e_deg, sg * w_deg) - verts
        flips = {}
        for name, (ev, wv) in {"Gesture-1": specs["Gesture"](-1), "Gesture+1": specs["Gesture"](1),
                               "Elbow-1": specs["Elbow"](-1), "Elbow+1": specs["Elbow"](1),
                               "Wrist-1": specs["Wrist"](-1), "Wrist+1": specs["Wrist"](1)}.items():
            flips[name] = B.tri_stats(skin.pose(verts, sg * ev, sg * wv), tri)["inverted"]
        png = io.BytesIO()
        Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA").save(png, "PNG", compress_level=6)
        out[side] = {"verts": verts.astype(np.float64), "uvs": uvs.astype(np.float64), "tris": tri.astype(np.int32),
                     "posed": posed, "png": png.getvalue(),
                     "report": {"verts": int(len(verts)), "tris": int(len(tri)), "uncovered_px": int(uncovered),
                                "rest_inverted": int(B.tri_stats(verts, tri)["inverted"]), "extreme_inverted": flips,
                                "elbow_stroke_px": int(n_stroke), "tex": f"{canvas.shape[1]}x{canvas.shape[0]}"}}
    try:
        MIDORI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        for side in B.SIDES:
            d = out[side]
            for k in ("verts", "uvs", "tris", "posed"):
                arrays[f"{side}_{k}"] = d[k]
            arrays[f"{side}_png"] = np.frombuffer(d["png"], np.uint8)
            arrays[f"{side}_report"] = np.array(json.dumps(d["report"]))
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except Exception:
        pass
    out["cached"] = False
    return out


def apply_midori_arms(header: bytes, bake: dict[str, Any], tex_ids: dict[str, int]) -> bytes:
    """Swap the rigid raised-arm chain for the unified deform Part on both sides (shoulder roots, their Lift / Body /
    Breath bindings and the seamless resting arms are untouched)."""
    B = _tool()
    puppet = _parse(header)
    by_id, _ = _index(puppet["nodes"])
    params = {p["name"]: p for p in puppet["param"]}
    removed: set[int] = set()
    specs = list(_arm_specs(B))
    for side, L in B.SIDES.items():
        if L["upper"] not in by_id or L["new_uuid"] in by_id:
            continue
        root, upper = by_id[L["root"]], by_id[L["upper"]]
        d = bake[side]
        verts = np.asarray(d["verts"], np.float64)
        part = copy.deepcopy({k: v for k, v in upper.items() if k != "children"})
        part.update(uuid=L["new_uuid"], name=f"{side} arm · raised unified deform mesh (Midori)",
                    zsort=min(float(by_id[L[k]].get("zsort", 0)) for k in ("upper", "forearm", "hand")),
                    transform={"trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
                    textures=[tex_ids[side], TEX_NONE, TEX_NONE],
                    mesh={"verts": np.round(verts.reshape(-1), 4).tolist(),
                          "uvs": np.round(np.asarray(d["uvs"]).reshape(-1), 6).tolist(),
                          "indices": np.asarray(d["tris"]).reshape(-1).astype(int).tolist(), "origin": [0.0, 0.0]})
        part["children"] = []
        root["children"] = [c for c in root.get("children", []) if c["uuid"] not in (L["upper"], L["elbow"])] + [part]
        removed |= {L["upper"], L["elbow"], L["forearm"], L["wrist"], L["hand"]}
        pose = params.get(f"Arm {side} Pose")
        if pose is not None:
            for b in list(pose["bindings"]):
                if b["node"] == L["upper"] and b["param_name"] == "opacity":
                    nb = copy.deepcopy(b)
                    nb["node"] = L["new_uuid"]
                    pose["bindings"].append(nb)
        posed = np.asarray(d["posed"])
        for pi, key in enumerate(specs):
            prm = params.get(f"Arm {side} {key}")
            if prm is None:
                continue
            B.resample_axis(prm, B.AXIS5)
            n1 = len(prm["axis_points"][1])
            vals = []
            for vi in range(len(B.VALUES5)):
                off = np.round(posed[pi, vi].astype(np.float64), 4).tolist()
                vals.append([off for _ in range(n1)])
            prm["bindings"].append({"node": L["new_uuid"], "param_name": "deform", "interpolate_mode": "Linear",
                                    "isSet": [[True] * n1 for _ in B.VALUES5], "values": vals})
    for p in puppet["param"]:
        p["bindings"] = [b for b in p.get("bindings", []) if b["node"] not in removed]
    return _dump(header, puppet)


# =================================================================================================================
# 2. Supersampled premultiplied anime art
# =================================================================================================================
SS = 4


def _hex(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _ss(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rot(pts: np.ndarray, deg: float, about: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(pts, np.float64) - about
    return np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], 1) + about


class Canvas:
    """Premultiplied float canvas in model units (origin = canvas centre, y down), rendered at SS x resolution."""

    def __init__(self, w_units: float, h_units: float, ppu: float) -> None:
        self.wu, self.hu, self.ppu = float(w_units), float(h_units), float(ppu)
        self.W, self.H = int(math.ceil(w_units * ppu)), int(math.ceil(h_units * ppu))
        self.s = ppu * SS
        self.WS, self.HS = self.W * SS, self.H * SS
        self.px = np.zeros((self.HS, self.WS, 4), np.float32)
        ys, xs = np.mgrid[0:self.HS, 0:self.WS].astype(np.float32)
        self.X = (xs + 0.5) / self.s - self.W / ppu / 2.0
        self.Y = (ys + 0.5) / self.s - self.H / ppu / 2.0

    def _to_px(self, pts: np.ndarray) -> list[tuple[float, float]]:
        p = (np.asarray(pts, np.float64) + [self.W / self.ppu / 2.0, self.H / self.ppu / 2.0]) * self.s
        return [(float(x), float(y)) for x, y in p]

    def poly(self, pts: np.ndarray) -> np.ndarray:
        im = Image.new("L", (self.WS, self.HS), 0)
        ImageDraw.Draw(im).polygon(self._to_px(pts), fill=255)
        return np.asarray(im, np.float32) / 255.0

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, deg: float = 0.0) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        pts = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
        return self.poly(_rot(pts, deg) + [cx, cy])

    def stroke(self, pts: np.ndarray, widths: np.ndarray | float) -> np.ndarray:
        """Round-capped variable-width stroke along a polyline (dense discs at SS resolution)."""
        pts = np.asarray(pts, np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(2, int(L[-1] * self.s / 0.7))
        tt = np.linspace(0, L[-1], n)
        px = np.interp(tt, L, pts[:, 0])
        py = np.interp(tt, L, pts[:, 1])
        w = np.interp(tt, L, np.broadcast_to(np.asarray(widths, np.float64), (len(pts),)))
        im = Image.new("L", (self.WS, self.HS), 0)
        dr = ImageDraw.Draw(im)
        for (x, y), ww in zip(self._to_px(np.stack([px, py], 1)), w):
            r = ww * self.s / 2.0
            dr.ellipse((x - r, y - r, x + r, y + r), fill=255)
        return np.asarray(im, np.float32) / 255.0

    def grow(self, mask: np.ndarray, w_units: float) -> np.ndarray:
        """Mask dilated by w_units with an anti-aliased rim (for clean outlines)."""
        dist = cv2.distanceTransform((mask < 0.5).astype(np.uint8), cv2.DIST_L2, 5) / self.s
        return np.maximum(mask, np.clip((w_units - dist) * self.s + 0.5, 0.0, 1.0))

    def shrink(self, mask: np.ndarray, w_units: float) -> np.ndarray:
        dist = cv2.distanceTransform((mask >= 0.5).astype(np.uint8), cv2.DIST_L2, 5) / self.s
        return np.minimum(mask, np.clip((dist - w_units) * self.s + 0.5, 0.0, 1.0))

    def blur(self, mask: np.ndarray, sigma_units: float) -> np.ndarray:
        return cv2.GaussianBlur(mask, (0, 0), max(0.3, sigma_units * self.s))

    def over(self, color: Any, alpha: np.ndarray) -> None:
        a = np.clip(alpha, 0.0, 1.0)[..., None]
        col = np.asarray(color, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        self.px[..., :3] = col * a + self.px[..., :3] * (1.0 - a)
        self.px[..., 3:] = a + self.px[..., 3:] * (1.0 - a)

    def add_glow(self, color: Any, alpha: np.ndarray) -> None:
        """Soft glow placed BEHIND what is already drawn."""
        a = np.clip(alpha, 0.0, 1.0)[..., None] * (1.0 - self.px[..., 3:])
        self.px[..., :3] += np.asarray(color, np.float32) * a
        self.px[..., 3:] += a

    def png(self) -> bytes:
        chans = []
        for c in range(4):
            im = Image.fromarray(np.ascontiguousarray(self.px[..., c]), "F").resize((self.W, self.H), Image.Resampling.LANCZOS)
            chans.append(np.asarray(im, np.float32))
        a = np.clip(chans[3], 0.0, 1.0)
        rgb = np.clip(np.stack(chans[:3], -1), 0.0, None)
        rgb = np.minimum(rgb, a[..., None])                          # stay premultiplied (no LANCZOS halo)
        a8 = np.rint(a * 255.0).astype(np.uint8)
        rgb8 = np.minimum(np.rint(rgb * 255.0), a8[..., None]).astype(np.uint8)
        out = np.dstack([rgb8, a8])
        out[a8 == 0] = 0
        buf = io.BytesIO()
        Image.fromarray(out, "RGBA").save(buf, "PNG", compress_level=6)
        return buf.getvalue()


def _star_pts(rx: float, ry: float, k: float = 2.6, n: int = 720) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([rx * np.sign(c) * np.abs(c) ** k, ry * np.sign(s) * np.abs(s) ** k], 1)


def _heart_pts(w: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], 1)
    pts -= [0.0, (pts[:, 1].max() + pts[:, 1].min()) / 2]
    return pts * (w / 32.0)


def _drop_pts(w: float, h: float, n: int = 400) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.sin(t) * np.sin(t / 2) ** 1.15
    y = -np.cos(t)
    pts = np.stack([x / (2 * np.abs(x).max()) * w, y * h / 2], 1)
    return pts - [0.0, (pts[:, 1].max() + pts[:, 1].min()) / 2]


# ---------------------------------------------------------------- palette (hot pink / neon cyan / pastel gold)
GOLD, GOLD_DEEP = _hex("#ffe9a8"), _hex("#ffb93d")
CYAN, CYAN_DEEP = _hex("#b9f6ff"), _hex("#22c8f0")
WHITE = _hex("#ffffff")


def art_sparkle(size: float, tint: str, bokeh: list[tuple[float, float, float, str]], ppu: float = 9.0) -> tuple[bytes, float, float]:
    """Glowing 4-point anime diamond star (+ bokeh dots in units relative to the star centre)."""
    ext = size * 0.5
    for bx, by, br, _ in bokeh:
        ext = max(ext, abs(bx) + br + 1.5, abs(by) + br + 1.5)
    w = h = 2 * ext + 4.0
    cv = Canvas(w, h, ppu)
    light, deep = (GOLD, GOLD_DEEP) if tint == "gold" else (CYAN, CYAN_DEEP)
    for bx, by, br, bt in bokeh:                                    # bokeh behind the star
        bl, bd = (GOLD, GOLD_DEEP) if bt == "gold" else (CYAN, CYAN_DEEP)
        disc = cv.ellipse(bx, by, br, br)
        cv.add_glow(bd, cv.blur(disc, br * 0.5) * 0.35)
        cv.over(bl * 0.6 + bd * 0.4, cv.blur(disc, 0.12) * 0.42)
        ring = np.clip(disc - cv.shrink(disc, max(0.35, br * 0.16)), 0, 1)
        cv.over(bl, cv.blur(ring, 0.08) * 0.75)
        cv.over(WHITE, cv.ellipse(bx - br * 0.35, by - br * 0.35, br * 0.22, br * 0.22) * 0.55)
    R = size / 2.0
    star = cv.poly(_star_pts(R * 0.78, R, 2.7))
    mini = cv.poly(_star_pts(R * 0.2, R * 0.27, 2.8) + [R * 0.66, -R * 0.58])
    rim = cv.grow(star, 0.45)
    cv.add_glow(deep, cv.blur(cv.grow(star, 0.6), R * 0.22) * 0.9)
    cv.over(deep, rim)
    r = np.hypot(cv.X / (R * 0.78), cv.Y / R)
    fill = WHITE[None, None] * (1 - _ss(0.0, 0.85, r)[..., None]) + light[None, None] * _ss(0.0, 0.85, r)[..., None]
    cv.over(fill, star)
    cv.over(WHITE, cv.blur(cv.ellipse(0, 0, R * 0.13, R * 0.13), R * 0.04))
    cv.over(light, cv.grow(mini, 0.3))
    cv.over(WHITE, mini)
    return cv.png(), w, h


def art_heart(size: float, tilt: float, ppu: float = 9.0) -> tuple[bytes, float, float]:
    w = h = size * 1.45 + 3.0
    cv = Canvas(w, h, ppu)
    hp = _rot(_heart_pts(size), tilt)
    heart = cv.poly(hp)
    lw = max(0.55, size * 0.06)
    out = cv.grow(heart, lw)
    cv.add_glow(_hex("#ff5fae"), cv.blur(out, size * 0.12) * 0.55)
    cv.over(_hex("#8e0c45"), out)
    Yr = _rot(np.stack([cv.X.ravel(), cv.Y.ravel()], 1), -tilt)[:, 1].reshape(cv.X.shape)
    t = _ss(-size * 0.45, size * 0.45, Yr)[..., None]
    fill = _hex("#ff8fcb")[None, None] * (1 - t) + _hex("#e3127a")[None, None] * t
    cv.over(fill, heart)
    inner = heart - cv.shrink(heart, size * 0.07)                  # soft inner rim shade
    cv.over(_hex("#b80d61"), cv.blur(np.clip(inner, 0, 1), size * 0.03) * 0.35 * t[..., 0])
    hl = _rot(np.array([[-size * 0.24, -size * 0.17]]), tilt)[0]
    cv.over(WHITE, cv.blur(cv.ellipse(hl[0], hl[1], size * 0.13, size * 0.075, tilt - 38.0), size * 0.012) * 0.95)
    d2 = _rot(np.array([[-size * 0.07, -size * 0.3]]), tilt)[0]
    cv.over(WHITE, cv.ellipse(d2[0], d2[1], size * 0.035, size * 0.035) * 0.9)
    rl = _rot(np.array([[size * 0.2, size * 0.12]]), tilt)[0]
    cv.over(_hex("#ffc2e2"), cv.blur(cv.ellipse(rl[0], rl[1], size * 0.08, size * 0.035, tilt + 50.0), size * 0.02) * 0.55)
    return cv.png(), w, h


def art_sweat(wd: float, ht: float, tilt: float, ppu: float = 10.0) -> tuple[bytes, float, float]:
    w = h = max(wd, ht) * 1.35 + 3.0
    cv = Canvas(w, h, ppu)
    drop = cv.poly(_rot(_drop_pts(wd, ht), tilt))
    out = cv.grow(drop, 0.65)
    cv.add_glow(_hex("#7fdcff"), cv.blur(out, 1.2) * 0.35)
    cv.over(_hex("#26385f"), out)
    Yr = _rot(np.stack([cv.X.ravel(), cv.Y.ravel()], 1), -tilt)[:, 1].reshape(cv.X.shape)
    t = _ss(-ht * 0.5, ht * 0.5, Yr)[..., None]
    fill = (_hex("#f1fdff")[None, None] * (1 - t) ** 2 + _hex("#9fe0ff")[None, None] * 2 * t * (1 - t)
            + _hex("#3d9ee9")[None, None] * t ** 2)
    cv.over(fill, drop)
    # crisp white crescent highlight inside the round belly + a small top glint
    a = _rot(np.array([[-wd * 0.12, ht * 0.16]]), tilt)[0]
    b = _rot(np.array([[wd * 0.02, ht * 0.08]]), tilt)[0]
    cres = np.clip(cv.ellipse(a[0], a[1], wd * 0.3, ht * 0.25, tilt) - cv.ellipse(b[0], b[1], wd * 0.3, ht * 0.27, tilt), 0, 1)
    cv.over(WHITE, cres * drop * 0.95)
    g = _rot(np.array([[wd * 0.1, -ht * 0.12]]), tilt)[0]
    cv.over(WHITE, cv.ellipse(g[0], g[1], wd * 0.07, ht * 0.07, tilt) * 0.9)
    return cv.png(), w, h


def art_anger(size: float, tilt: float, ppu: float = 9.0) -> tuple[bytes, float, float]:
    """Classic 4-arc cross-popping vein mark."""
    w = h = size * 1.25 + 3.0
    cv = Canvas(w, h, ppu)
    S = size / 2.0
    veins, cores = np.zeros_like(cv.X), np.zeros_like(cv.X)
    for qx, qy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        # each vein is a strongly curved bracket in its own quadrant: bulges toward the centre, ends flare outward
        C = np.array([qx, qy], np.float64) * S * 0.95
        R = S * 0.78
        base = math.atan2(-qy, -qx)
        u = np.linspace(0.0, 1.0, 90)
        ang = base + (u - 0.5) * math.radians(90)
        nrm = np.stack([np.cos(ang), np.sin(ang)], 1)
        mid = C + R * nrm
        thick = S * 0.27 * (0.28 + 0.72 * np.sin(np.pi * u) ** 0.7)
        poly = np.vstack([mid - nrm * thick[:, None] / 2, (mid + nrm * thick[:, None] / 2)[::-1]])
        veins = np.maximum(veins, cv.poly(_rot(poly, tilt)))
        k = slice(12, -12)
        cpoly = np.vstack([mid[k] + nrm[k] * thick[k, None] * 0.02, (mid[k] + nrm[k] * thick[k, None] * 0.2)[::-1]])
        cores = np.maximum(cores, cv.poly(_rot(cpoly, tilt)))
    out = cv.grow(veins, max(0.6, size * 0.035))
    cv.add_glow(_hex("#ff3b4f"), cv.blur(out, size * 0.08) * 0.45)
    cv.over(_hex("#640812"), out)
    r = np.hypot(cv.X, cv.Y) / S
    fill = _hex("#ff2a3d")[None, None] * (1 - _ss(0.4, 1.2, r)[..., None]) + _hex("#d90f2a")[None, None] * _ss(0.4, 1.2, r)[..., None]
    cv.over(fill, veins)
    cv.over(_hex("#ffb0b0"), cv.blur(cores, size * 0.01) * 0.85)
    return cv.png(), w, h


def art_shock(wd: float, ht: float, ppu: float = 6.0) -> tuple[bytes, float, float]:
    """Anime gloom / shock shading: translucent indigo wash with a soft top fade + crisp vertical hatch lines that
    taper and fade out downward. The silhouette narrows toward the bottom like the top of a face."""
    cv = Canvas(wd, ht, ppu)
    X, Y = cv.X, cv.Y
    v = (Y + ht / 2) / ht                                             # 0 top .. 1 bottom
    half = wd * 0.5 * (1.0 - 0.22 * v ** 1.5)
    edge_x = 1.0 - _ss(half - wd * 0.26, half, np.abs(X))
    dome = -ht / 2 + 2.0 + 16.0 * (X / (wd * 0.5)) ** 2              # top edge follows the head's dome
    wash = _ss(dome, dome + ht * 0.3, Y) * (1.0 - _ss(-ht * 0.25, ht * 0.45, Y)) * edge_x * 0.5
    cv.over(_hex("#35296e"), wash)
    rng = np.random.default_rng(29)
    n = 23
    xs = np.linspace(-wd * 0.43, wd * 0.43, n) + rng.uniform(-0.9, 0.9, n)
    lines = np.zeros_like(X)
    for x0 in xs:
        rel = abs(x0) / (wd * 0.5)
        top = -ht / 2 + 2.0 + 16.0 * rel ** 2 + rng.uniform(0.0, 5.0)
        ln = ht * (0.48 + 0.3 * rng.random()) * (1.0 - 0.5 * rel ** 2)
        y = np.linspace(top, top + ln, 32)
        wdt = 1.35 * (1.0 - np.linspace(0, 1, 32) ** 1.4) + 0.15
        m = cv.stroke(np.stack([np.full_like(y, x0), y], 1), wdt)
        fade = (1.0 - _ss(top + ln * 0.25, top + ln, Y)) * _ss(top, top + 9.0, Y)
        lines = np.maximum(lines, m * fade)
    cv.over(_hex("#1c1540"), lines * 0.9 * edge_x)
    return cv.png(), wd, ht


def art_note(kind: str, size: float, tint: str, tilt: float, ppu: float = 9.0) -> tuple[bytes, float, float]:
    """Glossy eighth note (single) or beamed pair (double)."""
    w = h = size * 1.6 + 3.0
    cv = Canvas(w, h, ppu)
    S = size
    fill_a, fill_b, line = ((_hex("#ff8ccc"), _hex("#ff2e93"), _hex("#3a0f33")) if tint == "pink"
                            else (_hex("#9cf6ff"), _hex("#14bfee"), _hex("#0c3550")))
    heads = [(-0.12 * S, 0.30 * S)] if kind == "single" else [(-0.30 * S, 0.33 * S), (0.24 * S, 0.24 * S)]
    hr = (0.17 * S, 0.12 * S)
    shapes = np.zeros_like(cv.X)
    head_masks = []
    tops = []
    for hx, hy in heads:
        hm = cv.poly(_rot(_rot(np.stack([hr[0] * np.cos(np.linspace(0, 2 * np.pi, 120)),
                                         hr[1] * np.sin(np.linspace(0, 2 * np.pi, 120))], 1), -24.0) + [hx, hy], tilt))
        head_masks.append((hm, hx, hy))
        shapes = np.maximum(shapes, hm)
        sx = hx + hr[0] * 0.86
        top = hy - 0.62 * S
        tops.append((sx, top))
        stem = np.array([[sx - 0.035 * S, hy - 0.02 * S], [sx + 0.035 * S, hy - 0.04 * S], [sx + 0.035 * S, top], [sx - 0.035 * S, top]])
        shapes = np.maximum(shapes, cv.poly(_rot(stem, tilt)))
    if kind == "single":
        sx, top = tops[0]
        u = np.linspace(0, 1, 40)
        outer = np.stack([sx + 0.30 * S * np.sin(u * 2.2) * (1 - 0.2 * u), top + 0.42 * S * u ** 1.1], 1)
        inner = np.stack([sx + 0.22 * S * np.sin(u * 2.0) * (1 - 0.3 * u), top + 0.13 * S + 0.25 * S * u], 1)
        shapes = np.maximum(shapes, cv.poly(_rot(np.vstack([[[sx - 0.035 * S, top]], outer, inner[::-1]]), tilt)))
    else:
        (x0, t0), (x1, t1) = tops
        bt = 0.12 * S
        for k in (0.0, 0.2 * S):
            beam = np.array([[x0 - 0.035 * S, t0 + k], [x1 + 0.035 * S, t1 + k], [x1 + 0.035 * S, t1 + k + bt], [x0 - 0.035 * S, t0 + k + bt]])
            shapes = np.maximum(shapes, cv.poly(_rot(beam, tilt)))
    out = cv.grow(shapes, max(0.5, S * 0.045))
    cv.add_glow(fill_b, cv.blur(out, S * 0.1) * 0.55)
    cv.over(line, out)
    Yr = _rot(np.stack([cv.X.ravel(), cv.Y.ravel()], 1), -tilt)[:, 1].reshape(cv.X.shape)
    t = _ss(-0.4 * S, 0.45 * S, Yr)[..., None]
    cv.over(fill_a[None, None] * (1 - t) + fill_b[None, None] * t, shapes)
    for hm, hx, hy in head_masks:
        p = _rot(np.array([[hx - hr[0] * 0.35, hy - hr[1] * 0.35]]), tilt)[0]
        cv.over(WHITE, cv.ellipse(p[0], p[1], hr[0] * 0.38, hr[1] * 0.28, tilt - 24.0) * 0.92)
    return cv.png(), w, h


def art_eye_star(size: float, ppu: float = 18.0) -> tuple[bytes, float, float]:
    w = h = size * 1.5 + 2.0
    cv = Canvas(w, h, ppu)
    R = size / 2.0
    star = cv.poly(_star_pts(R * 0.8, R, 2.4))
    cv.add_glow(_hex("#fff0b3"), cv.blur(cv.grow(star, 0.25), R * 0.28) * 0.95)
    cv.over(_hex("#ffe08a"), cv.grow(star, 0.22))
    r = np.hypot(cv.X / (R * 0.8), cv.Y / R)
    cv.over(WHITE[None, None] * (1 - _ss(0.2, 0.9, r)[..., None]) + _hex("#fff4c8")[None, None] * _ss(0.2, 0.9, r)[..., None], star)
    return cv.png(), w, h


def art_eye_heart(size: float, ppu: float = 18.0) -> tuple[bytes, float, float]:
    w = h = size * 1.5 + 2.0
    cv = Canvas(w, h, ppu)
    heart = cv.poly(_heart_pts(size))
    cv.add_glow(_hex("#ff7fc4"), cv.blur(cv.grow(heart, 0.3), size * 0.14) * 0.95)
    cv.over(_hex("#ff3d9a"), cv.grow(heart, 0.3))
    r = np.hypot(cv.X, cv.Y + size * 0.05) / (size * 0.5)
    cv.over(WHITE[None, None] * (1 - _ss(0.15, 0.95, r)[..., None]) + _hex("#ffb3dc")[None, None] * _ss(0.15, 0.95, r)[..., None], heart)
    cv.over(WHITE, cv.ellipse(-size * 0.2, -size * 0.14, size * 0.09, size * 0.055, -38.0))
    return cv.png(), w, h


def art_cheek_puff(wd: float, ht: float, ppu: float = 8.0) -> tuple[bytes, float, float]:
    w, h = wd + 4.0, ht + 4.0
    cv = Canvas(w, h, ppu)
    r = np.hypot(cv.X / (wd / 2), cv.Y / (ht / 2))
    a = 0.82 * (1.0 - _ss(0.25, 1.0, r))
    col = _hex("#ff5c86")[None, None] * (1 - _ss(0.0, 0.9, r)[..., None]) + _hex("#ff9fb8")[None, None] * _ss(0.0, 0.9, r)[..., None]
    cv.over(col, a)
    # glossy puffed-cheek sheen (upper outer) + three tiny anime blush ticks
    cv.over(WHITE, cv.blur(cv.ellipse(-wd * 0.12, -ht * 0.16, wd * 0.15, ht * 0.11, -18.0), 0.5) * 0.55)
    for k in (-1, 0, 1):
        x0 = k * wd * 0.17 + wd * 0.06
        cv.over(_hex("#e83d6c"), cv.stroke(np.array([[x0 + 1.4, -ht * 0.02], [x0 - 1.0, ht * 0.2]]), [0.55, 0.3]) * 0.55)
    return cv.png(), w, h


def art_cat_mouth(width: float, depth: float, line_rgb: np.ndarray, lw: float, ppu: float = 20.0) -> tuple[bytes, float, float]:
    """':3' / omega cat smile: two soft downward bumps meeting at a small centre peak, slight upturned corners."""
    w, h = width + 4.0, depth + 4.0
    cv = Canvas(w, h, ppu)
    hw = width / 2.0
    u = np.linspace(-1.0, 1.0, 161)
    x = u * hw
    bump = np.abs(np.sin(np.pi * u))                                  # 0 at centre & corners, 1 mid-lobe
    y = -depth * 0.45 + depth * 0.95 * bump ** 0.85
    y += depth * 0.18 * (1 - _ss(0.0, 0.18, np.abs(u)))               # centre peak cusp sits a touch lower
    corner = _ss(0.86, 1.0, np.abs(u))
    y -= depth * 0.32 * corner                                        # little upturned corners
    widths = lw * (0.45 + 0.55 * (1 - _ss(0.75, 1.0, np.abs(u))))
    m = cv.stroke(np.stack([x, y], 1), widths)
    cv.over(line_rgb, m)
    return cv.png(), w, h


# =================================================================================================================
# 3. Deform "follow" maps (new Parts ride Hikari's per-mesh head / eye / face deforms)
# =================================================================================================================
HEAD_FIT_NODES = (1011, 1035)          # ArtMeshFace + ArtMeshFrontHair: affine fit of the head motion
MOUTH_SHAPE_PARAMS = {"VB Mouth", "Mouth", "Mouth X", "Mouth Shrug", "Mouth Funnel", "Jaw Open", "Mouth Spring", "Cheek Puff"}


def _rest(node: dict[str, Any]) -> np.ndarray:
    return np.asarray(node["mesh"]["verts"], np.float64).reshape(-1, 2)


def _bary_weights(rest: np.ndarray, tris: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(P, N) linear interpolation weights of pts inside (or nearest-extrapolated from) a reference mesh."""
    a, b, c = rest[tris[:, 0]], rest[tris[:, 1]], rest[tris[:, 2]]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-6
    W = np.zeros((len(pts), len(rest)))
    for i, p in enumerate(pts):
        d = p - a
        l1 = np.where(ok, (d[:, 0] * v1[:, 1] - v1[:, 0] * d[:, 1]) / np.where(ok, den, 1), -9)
        l2 = np.where(ok, (v0[:, 0] * d[:, 1] - d[:, 0] * v0[:, 1]) / np.where(ok, den, 1), -9)
        l0 = 1 - l1 - l2
        score = np.minimum(np.minimum(l0, l1), l2)
        t = int(np.argmax(score))
        W[i, tris[t, 0]] += l0[t]
        W[i, tris[t, 1]] += l1[t]
        W[i, tris[t, 2]] += l2[t]
    return W


def _deform_bindings(params: list[dict[str, Any]], node: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for p in params:
        for b in p.get("bindings", ()):
            if b["node"] == node and b["param_name"] == "deform":
                out.append((p, b))
    return out


class Follower:
    def __init__(self, puppet: dict[str, Any]) -> None:
        self.params = puppet["param"]
        self.by_id, _ = _index(puppet["nodes"])
        hb = [{p["name"]: (p, b) for p, b in _deform_bindings(self.params, n)} for n in HEAD_FIT_NODES]
        self.head_params = [name for name in hb[0] if all(name in h for h in hb[1:]) and name not in MOUTH_SHAPE_PARAMS]
        self._hb = hb

    def head_affine(self, pts_world: np.ndarray) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Deform bindings that move pts like an affine fit of the face + front-hair head motion."""
        rests = [_rest(self.by_id[n]) for n in HEAD_FIT_NODES]
        R = np.vstack(rests)
        X = np.hstack([R, np.ones((len(R), 1))])
        M = np.hstack([pts_world, np.ones((len(pts_world), 1))]) @ np.linalg.pinv(X)     # (P, N)
        out = []
        for name in self.head_params:
            prm = self._hb[0][name][0]
            V = np.concatenate([np.asarray(h[name][1]["values"], np.float64) for h in self._hb], axis=2)
            out.append((prm, self._binding(np.einsum("pn,ijnc->ijpc", M, V), self._hb[0][name][1])))
        return out

    def mesh_follow(self, ref: int, pts_world: np.ndarray, only: Callable[[str], bool] = lambda n: True):
        node = self.by_id[ref]
        W = _bary_weights(_rest(node), np.asarray(node["mesh"]["indices"], np.int64).reshape(-1, 3), pts_world)
        out = []
        for p, b in _deform_bindings(self.params, ref):
            if not only(p["name"]):
                continue
            V = np.asarray(b["values"], np.float64)
            out.append((p, self._binding(np.einsum("pn,ijnc->ijpc", W, V), b)))
        return out

    @staticmethod
    def _binding(vals: np.ndarray, like: dict[str, Any]) -> dict[str, Any]:
        return {"node": None, "param_name": "deform", "interpolate_mode": like.get("interpolate_mode", "Linear"),
                "isSet": copy.deepcopy(like["isSet"]), "values": np.round(vals, 4).tolist()}


# =================================================================================================================
# 4. VFX / micro-expression spec + rig pass
# =================================================================================================================
VFX_VERSION = "vfx-v2-kuroama"
NODE_BASE = 54000
PARAM_IDS = {"VFX Sparkles": 54100, "VFX Hearts": 54101, "VFX Sweat": 54102, "VFX Anger": 54103, "VFX Shock": 54104,
             "VFX Music": 54105, "Eye Star": 54106, "Eye Heart": 54107, "Cat Mouth": 54109}
IRIS = {"R": 1023, "L": 1024}
EYE_WHITE = {"R": 1014, "L": 1015}
EYE_OPEN = {"R": "Right Eye Open", "L": "Left Eye Open"}
# measured from Hikari-v28-Custom.inp (iris mesh centroids, world units)
IRIS_CENTER = {"R": (-33.7, -443.8), "L": (38.2, -443.9)}
MOUTH_CENTER = (1.5, -395.4)
CHEEK_CENTER = {"R": (-47.0, -407.0), "L": (49.0, -407.0)}
FACE, FACE_GROUP, HEAD = 1011, 102, 100
LIP_PARTS = (40001, 40002, 40003, 40004, 40005, 40006)

# (param, kind, centre xy, art kwargs, zsort)
GLYPHS: list[tuple[str, str, tuple[float, float], dict[str, Any], float]] = [
    ("VFX Shock", "shock", (0.0, -491.0), dict(wd=132.0, ht=78.0), -50.0),
    ("VFX Sweat", "sweat", (90.0, -492.0), dict(wd=21.0, ht=32.0, tilt=-14.0), -52.0),
    ("VFX Anger", "anger", (-80.0, -526.0), dict(size=42.0, tilt=-8.0), -53.0),
    ("VFX Hearts", "heart", (-104.0, -418.0), dict(size=33.0, tilt=-16.0), -54.0),
    ("VFX Hearts", "heart", (108.0, -486.0), dict(size=27.0, tilt=14.0), -54.1),
    ("VFX Hearts", "heart", (-127.0, -490.0), dict(size=18.0, tilt=-6.0), -54.2),
    ("VFX Hearts", "heart", (117.0, -407.0), dict(size=20.0, tilt=20.0), -54.3),
    ("VFX Sparkles", "sparkle", (-100.0, -450.0), dict(size=34.0, tint="gold", bokeh=[(16.0, 17.0, 4.4, "cyan"), (-11.0, -21.0, 2.8, "gold")]), -55.0),
    ("VFX Sparkles", "sparkle", (104.0, -457.0), dict(size=30.0, tint="cyan", bokeh=[(-16.0, 18.0, 4.0, "gold"), (12.0, -18.0, 2.6, "cyan")]), -55.1),
    ("VFX Sparkles", "sparkle", (-120.0, -517.0), dict(size=23.0, tint="cyan", bokeh=[(13.0, 12.0, 3.0, "gold")]), -55.2),
    ("VFX Sparkles", "sparkle", (125.0, -523.0), dict(size=22.0, tint="gold", bokeh=[(-12.0, 13.0, 2.9, "cyan")]), -55.3),
    ("VFX Sparkles", "sparkle", (-90.0, -381.0), dict(size=18.0, tint="gold", bokeh=[(-10.0, -11.0, 2.5, "cyan")]), -55.4),
    ("VFX Sparkles", "sparkle", (94.0, -376.0), dict(size=18.0, tint="cyan", bokeh=[(10.0, -12.0, 2.6, "gold")]), -55.5),
    ("VFX Music", "note", (-118.0, -563.0), dict(kind="single", size=33.0, tint="pink", tilt=-10.0), -56.0),
    ("VFX Music", "note", (118.0, -549.0), dict(kind="double", size=34.0, tint="cyan", tilt=8.0), -56.1),
    ("VFX Music", "note", (124.0, -610.0), dict(kind="single", size=22.0, tint="pink", tilt=14.0), -56.2),
]
# per-param pop-in bindings: prop -> (value at 0, value at 1)
VFX_BIND: dict[str, dict[str, tuple[float, float]]] = {
    "VFX Sparkles": {"opacity": (0.0, 1.0), "transform.s.x": (0.6, 1.05), "transform.s.y": (0.6, 1.05), "transform.t.y": (12.0, -8.0)},
    "VFX Hearts": {"opacity": (0.0, 1.0), "transform.s.x": (0.5, 1.05), "transform.s.y": (0.5, 1.05), "transform.t.y": (18.0, -12.0)},
    "VFX Sweat": {"opacity": (0.0, 1.0), "transform.t.y": (-14.0, 10.0)},
    "VFX Anger": {"opacity": (0.0, 1.0), "transform.s.x": (0.6, 1.12), "transform.s.y": (0.6, 1.12)},
    "VFX Shock": {"opacity": (0.0, 0.92), "transform.t.y": (-10.0, 4.0)},
    "VFX Music": {"opacity": (0.0, 1.0), "transform.t.y": (16.0, -10.0), "transform.r.z": (-0.12, 0.10)},
}
EYE_FX = {"Eye Star": ("eye_star", dict(size=12.0), (-0.6, -1.6)), "Eye Heart": ("eye_heart", dict(size=11.5), (-0.3, -0.6))}
CHEEK_PUFF_PEAK = 11.5                 # model units of outward lower-cheek bulge at Cheek Puff = 1

_ART_CACHE: dict[tuple[Any, ...], tuple[bytes, float, float]] = {}
# the disk cache key includes this module's source so any art tweak invalidates stale PNGs automatically
def _art_src_hash() -> str:
    try:
        p = Path(__file__)
        if p.is_file():
            return hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    except OSError:
        pass
    try:
        st = Path(sys.executable).stat()
        return hashlib.sha1(f"frozen:{VFX_VERSION}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:12]
    except OSError:
        return hashlib.sha1(f"frozen:{VFX_VERSION}".encode()).hexdigest()[:12]


_ART_SRC_HASH = _art_src_hash()


def _art(kind: str, kw: dict[str, Any]) -> tuple[bytes, float, float]:
    key = (VFX_VERSION, _ART_SRC_HASH, kind, json.dumps(kw, sort_keys=True, default=lambda o: o.tolist()))
    if key in _ART_CACHE:
        return _ART_CACHE[key]
    stem = VFX_CACHE_DIR / hashlib.sha1(repr(key).encode()).hexdigest()[:20]
    try:
        meta = json.loads(stem.with_suffix(".json").read_text())
        _ART_CACHE[key] = (stem.with_suffix(".png").read_bytes(), float(meta["w"]), float(meta["h"]))
        return _ART_CACHE[key]
    except Exception:
        pass
    if key not in _ART_CACHE:
        fn = {"sparkle": art_sparkle, "heart": art_heart, "sweat": art_sweat, "anger": art_anger, "shock": art_shock,
              "note": art_note, "eye_star": art_eye_star, "eye_heart": art_eye_heart, "cheek": art_cheek_puff,
              "cat": art_cat_mouth}[kind]
        args = dict(kw)
        if "line_rgb" in args:
            args["line_rgb"] = np.asarray(args["line_rgb"], np.float32)
        _ART_CACHE[key] = fn(**args)
        try:
            VFX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            png, w, h = _ART_CACHE[key]
            stem.with_suffix(".png").write_bytes(png)
            stem.with_suffix(".json").write_text(json.dumps({"w": w, "h": h, "kind": kind}))
        except Exception:
            pass
    return _ART_CACHE[key]


def mouth_line_rgb(lip_png: bytes) -> list[float]:
    """Hikari's mouth lineart colour (darkest half of the opaque upper-lip texels, un-premultiplied)."""
    arr = np.asarray(Image.open(io.BytesIO(lip_png)).convert("RGBA"), np.float32) / 255.0
    a = arr[..., 3]
    m = a > 0.6
    if not np.any(m):
        return [0.47, 0.25, 0.24]
    rgb = arr[m, :3] / a[m, None]
    lum = rgb.mean(1)
    dark = rgb[lum <= np.median(lum)]
    return [round(float(x), 4) for x in np.median(dark, axis=0)]


def build_vfx_textures(lip_png: bytes) -> list[tuple[str, bytes, float, float]]:
    """Ordered (slot, premultiplied png, width units, height units) for every new texture."""
    out = []
    for i, (_, kind, _, kw, _) in enumerate(GLYPHS):
        out.append((f"g{i}",) + _art(kind, kw))
    for name, (kind, kw, _) in EYE_FX.items():
        out.append((name,) + _art(kind, kw))
    out.append(("cheek",) + _art("cheek", dict(wd=30.0, ht=18.0)))
    out.append(("cat",) + _art("cat", dict(width=18.0, depth=4.6, line_rgb=mouth_line_rgb(lip_png), lw=0.72)))
    return out


def _grid_mesh(w: float, h: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = np.linspace(-w / 2, w / 2, nx), np.linspace(-h / 2, h / 2, ny)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx.ravel(), gy.ravel()], 1)
    uvs = np.stack([(gx.ravel() + w / 2) / w, (gy.ravel() + h / 2) / h], 1)
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            tris += [a, a + 1, a + nx, a + 1, a + nx + 1, a + nx]
    return verts, uvs, np.array(tris, np.int64)


def _part(uid: int, name: str, tid: int, center: tuple[float, float], w: float, h: float, zsort: float,
          nx: int = 3, ny: int = 3, masks: list[dict[str, Any]] | None = None) -> tuple[dict[str, Any], np.ndarray]:
    verts, uvs, tris = _grid_mesh(w, h, nx, ny)
    node = {"uuid": uid, "name": name, "type": "Part", "enabled": True, "zsort": zsort,
            "transform": {"trans": [float(center[0]), float(center[1]), 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
            "lockToRoot": False, "textures": [tid, TEX_NONE, TEX_NONE], "blend_mode": "Normal",
            "tint": [1.0, 1.0, 1.0], "screenTint": [0.0, 0.0, 0.0], "mask_threshold": 0.5, "opacity": 1.0,
            "masks": masks or [], "children": [],
            "mesh": {"verts": np.round(verts.reshape(-1), 4).tolist(), "uvs": np.round(uvs.reshape(-1), 6).tolist(),
                     "indices": tris.tolist(), "origin": [0.0, 0.0]}}
    return node, verts + np.asarray(center, np.float64)


def _bind(param: dict[str, Any], node: int, prop: str, values: list[float]) -> None:
    n0, n1 = len(param["axis_points"][0]), len(param["axis_points"][1])
    assert len(values) == n0
    bs = param.setdefault("bindings", [])
    bs[:] = [b for b in bs if not (b["node"] == node and b["param_name"] == prop)]
    bs.append({"node": node, "param_name": prop, "interpolate_mode": "Linear", "isSet": [[True] * n1 for _ in range(n0)],
               "values": [[round(float(v), 5)] * n1 for v in values]})


def _new_param(name: str, uid: int) -> dict[str, Any]:
    return {"name": name, "uuid": uid, "is_vec2": False, "merge_mode": "Additive", "min": [0.0, 0.0], "max": [1.0, 1.0],
            "defaults": [0.0, 0.0], "axis_points": [[0.0, 1.0], [0.0, 1.0]], "bindings": []}


def cheek_puff_offsets(face_rest: np.ndarray) -> np.ndarray:
    """Smooth radial outward bulge of the lower cheeks; chin, eyes, nose and mouth corners stay locked."""
    x, y = face_rest[:, 0], face_rest[:, 1]
    off = np.zeros_like(face_rest)
    lock = (1.0 - _ss(-382.0, -370.0, y)) * _ss(-433.0, -421.0, y)
    for cx, cy in CHEEK_CENTER.values():
        dx, dy = x - cx, y - cy
        w = np.exp(-((dx / 24.0) ** 2 + (dy / 15.0) ** 2)) * lock
        d = np.stack([x - MOUTH_CENTER[0], 0.28 * (y - (-410.0))], 1)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        side = np.sign(cx)
        w = w * (np.sign(x - MOUTH_CENTER[0]) == side)
        # silhouette verts (outermost) get the full push, inner cheek flesh a little less -> rounder, no folds
        off += d * (CHEEK_PUFF_PEAK * w)[:, None]
    return off


def apply_vfx_rig(header: bytes, tid_of: dict[str, int], sizes: dict[str, tuple[float, float]],
                  iris_pngs: dict[str, bytes] | None = None) -> bytes:
    puppet = _parse(header)
    by_id, _ = _index(puppet["nodes"])
    if NODE_BASE in by_id or FACE not in by_id:
        return header
    # pro transforms first: the eye star / heart followers built below then also inherit Pupil Shrink
    apply_pro_transforms(puppet, iris_pngs or {})
    params = puppet.setdefault("param", [])
    by_name = {p["name"]: p for p in params}
    fol = Follower(puppet)

    def param(name: str) -> dict[str, Any]:
        p = by_name.get(name)
        if p is None:
            p = by_name[name] = _new_param(name, PARAM_IDS[name])
            params.append(p)
        return p

    def attach_follow(bindings: list[tuple[dict[str, Any], dict[str, Any]]], uid: int) -> None:
        for prm, b in bindings:
            b["node"] = uid
            prm["bindings"].append(b)

    uid = NODE_BASE
    head = by_id[HEAD]
    # ---- A. expression VFX (Head root, in front of hair / brows / see-through layers)
    for i, (pname, kind, center, kw, z) in enumerate(GLYPHS):
        w, h = sizes[f"g{i}"]
        nx = 5 if kind == "shock" else 3
        node, world = _part(uid, f"{pname} · {kind} {i}", tid_of[f"g{i}"], center, w, h, z, nx, nx)
        head.setdefault("children", []).append(node)
        attach_follow(fol.head_affine(world), uid)
        prm = param(pname)
        for prop, (v0, v1) in VFX_BIND[pname].items():
            _bind(prm, uid, prop, [v0, v1])
        uid += 1
    # ---- B7/B8. eye star / heart: children of the iris Parts, masked by the eye white, follow the iris deform
    #      (Eye Look / bounce / jelly / head) and inherit its blink opacity.
    eye_params = {p["name"]: p for p in params}
    for name, (_, _, (ox, oy)) in EYE_FX.items():
        w, h = sizes[name]
        for side in ("R", "L"):
            iris = by_id[IRIS[side]]
            c = (IRIS_CENTER[side][0] + (ox if side == "R" else -ox), IRIS_CENTER[side][1] + oy)
            z_abs = -33.8
            node, world = _part(uid, f"{name} · {side} iris highlight", tid_of[name], c, w, h,
                                z_abs - float(iris.get("zsort", 0.0)), 3, 3, masks=[{"mode": "Mask", "source": EYE_WHITE[side]}])
            iris.setdefault("children", []).append(node)
            attach_follow(fol.mesh_follow(IRIS[side], world), uid)
            prm = param(name)
            _bind(prm, uid, "opacity", [0.0, 1.0])
            _bind(prm, uid, "transform.s.x", [0.4, 1.0])
            _bind(prm, uid, "transform.s.y", [0.4, 1.0])
            eo = eye_params.get(EYE_OPEN[side])
            if eo is not None:
                for b in list(eo["bindings"]):
                    if b["node"] == IRIS[side] and b["param_name"] == "opacity":
                        nb = copy.deepcopy(b)
                        nb["node"] = uid
                        eo["bindings"].append(nb)
            uid += 1
    # ---- B9. Cheek Puff: smooth radial lower-cheek bulge on the face mesh + rosy puff blush overlay
    face = by_id[FACE]
    cp = by_name.get("Cheek Puff")
    if cp is None:
        cp = by_name["Cheek Puff"] = _new_param("Cheek Puff", 54108)
        params.append(cp)
    n0, n1 = len(cp["axis_points"][0]), len(cp["axis_points"][1])
    if n0 == 2:
        offs = cheek_puff_offsets(_rest(face))
        cp["bindings"] = [b for b in cp["bindings"] if not (b["node"] == FACE and b["param_name"] == "deform")]
        zero = np.zeros_like(offs).round(4).tolist()
        cp["bindings"].append({"node": FACE, "param_name": "deform", "interpolate_mode": "Linear",
                               "isSet": [[True] * n1, [True] * n1],
                               "values": [[zero] * n1, [np.round(offs, 4).tolist()] * n1]})
    fol = Follower(puppet)                      # re-read: the blush must ride the NEW puff deform
    w, h = sizes["cheek"]
    face_group = by_id[FACE_GROUP]
    for side in ("R", "L"):
        node, world = _part(uid, f"Cheek Puff · {side} rosy puff blush", tid_of["cheek"], CHEEK_CENTER[side], w, h, -12.5, 7, 5)
        face_group.setdefault("children", []).append(node)
        attach_follow(fol.mesh_follow(FACE, world), uid)
        _bind(cp, uid, "opacity", [0.0, 0.75])
        uid += 1
    # ---- B10. Cat Mouth ':3' at the neutral mouth (rides the face head motion), lips + cavity fade out
    w, h = sizes["cat"]
    node, world = _part(uid, "Cat Mouth · :3 smile", tid_of["cat"], MOUTH_CENTER, w, h, -20.6, 5, 3)
    puppet["nodes"].setdefault("children", []).append(node)
    attach_follow(fol.mesh_follow(FACE, world, only=lambda n: n in fol.head_params), uid)
    cm = param("Cat Mouth")
    _bind(cm, uid, "opacity", [0.0, 1.0])
    for lip in LIP_PARTS:
        if lip in by_id:
            _bind(cm, lip, "opacity", [1.0, 0.0])
    return _dump(header, puppet)


# =================================================================================================================
# 5. Pro VTuber transforms (Hiyori / Mao style): stage move + lean, pupil shrink, shoulder shrug, and the three
#    face channels that shipped with zero bindings (mouth corners, brow X, eye form).
#    Every vertex offset is a smooth function of the vertex's REST world position, so parts that touch at rest
#    (topwear / resting arms / shoulder lineart, lash / eye white, lips / cavity) move identically at the seam.
# =================================================================================================================
PRO_PARAM_IDS = {"Stage X": 54200, "Stage Y": 54201, "Stage Zoom": 54202, "Pupil Shrink": 54203, "Shoulder Shrug": 54204,
                 "Stage Roll": 54205, "Angle X Cheek Squish": 54206, "Cheek L PY1": 54207, "Cheek R PY1": 54208,
                 "Arm R Fit": 54209, "Arm L Fit": 54210}
PUPPET_ROOT = 1
STAGE_X = 80.0                    # +-units of whole-body sway at Stage X = +-1
STAGE_Y = (45.0, -55.0)           # t.y at Stage Y = -1 (crouch, down) / +1 (hop, UP: Inochi2D y points down)
STAGE_ZOOM = (0.92, 1.10)         # root scale at Stage Zoom = -1 (recoil back) / +1 (lean in)
STAGE_PIVOT_Y = -420.0            # lean pivots on the face so the face stays framed while the body grows / shrinks
STAGE_ROLL = 0.115                # +-radians (~6.6 deg) whole-body pendulum rock at Stage Roll = +-1
STAGE_FOOT_Y = 590.0              # ankle/shoe pivot Y so feet stay planted on the floor while the whole body rocks
PUPIL_SHRINK = 0.68               # iris scale at Pupil Shrink = 1
SHRUG_UP, SHRUG_DOWN = 14.0, 8.0  # shoulder lift (units) at Shoulder Shrug = +1 / drop at -1
SHRUG_TORSO = (1005, 53200, 53201, 51001)   # topwear, both shoulder lineart overlays (same mesh), halter collar
SHRUG_ARMS = (1046, 1000)                   # seamless resting arms
SHRUG_RAISED_ROOTS = (50006, 50015)         # raised wave-arm shoulder roots (rigid: translated by the field value)
IRIS_SETS = {"R": (1023, 1025, 50019), "L": (1024, 1026, 50020)}   # iris, catchlight layer, sparkle-eye FX
MOUTH_PARTS = (40001, 40002, 40003, 40004, 40005, 40006, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045)
MOUTH_HALF_W = 12.5
MOUTH_CORNER = (3.2, 3.2)         # corner lift at +1 / drop at -1 (units)
BROWS = {"R": (1031, 1032), "L": (1033, 1034)}
BROW_SLIDE = 6.0                  # whole-brow horizontal slide at +-1, inner end gets BROW_INNER_EXTRA more
BROW_INNER_EXTRA = 3.0
EYE_FORM_NODES = {"R": {"white": 1014, "lash": (1019, 1020)}, "L": {"white": 1015, "lash": (1021, 1022)}}
EYE_FORM_DROP, EYE_FORM_LIFT = 7.0, 3.0     # upper-lid centre travel at Eye Form = -1 (smug half-lid) / +1 (wide)


def _world_rest(node: dict[str, Any], parent: dict[int, dict[str, Any]]) -> np.ndarray:
    off = np.zeros(2)
    n: dict[str, Any] | None = node
    while n is not None:
        t = n.get("transform", {}).get("trans", [0.0, 0.0, 0.0])
        off += (float(t[0]), float(t[1]))
        n = parent.get(n["uuid"])
    return _rest(node) + off


def _ensure_param(params: list[dict[str, Any]], name: str, lo: float) -> dict[str, Any]:
    for p in params:
        if p["name"] == name:
            return p
    p = _new_param(name, PRO_PARAM_IDS[name])
    if lo < 0:
        p.update(min=[-1.0, 0.0], max=[1.0, 1.0], axis_points=[[0.0, 0.5, 1.0], [0.0, 1.0]])
    params.append(p)
    return p


def _bind_deform(param: dict[str, Any], node: int, offs: list[np.ndarray]) -> None:
    """offs: one (N, 2) offset array per axis point of the (1D) param."""
    n0, n1 = len(param["axis_points"][0]), len(param["axis_points"][1])
    assert len(offs) == n0
    bs = param.setdefault("bindings", [])
    bs[:] = [b for b in bs if not (b["node"] == node and b["param_name"] == "deform")]
    bs.append({"node": node, "param_name": "deform", "interpolate_mode": "Linear", "isSet": [[True] * n1 for _ in range(n0)],
               "values": [[np.round(o, 4).tolist()] * n1 for o in offs]})


def iris_uv_centroid(png: bytes) -> tuple[float, float] | None:
    """Alpha-weighted centre of the iris art in UV space (the pupil sits dead-centre of the iris disc)."""
    try:
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), np.float32)[..., 3] / 255.0
    except Exception:
        return None
    a = np.where(a > 0.5, a, 0.0)
    s = float(a.sum())
    if s < 16.0:
        return None
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return float(((xs + 0.5) * a).sum() / s / a.shape[1]), float(((ys + 0.5) * a).sum() / s / a.shape[0])


def _uv_to_world(node: dict[str, Any], world: np.ndarray, uv: tuple[float, float]) -> np.ndarray:
    uvs = np.asarray(node["mesh"]["uvs"], np.float64).reshape(-1, 2)
    tris = np.asarray(node["mesh"]["indices"], np.int64).reshape(-1, 3)
    W = _bary_weights(uvs, tris, np.array([uv], np.float64))
    return (W @ world)[0]


def shrug_field(pts: np.ndarray, arm: bool) -> np.ndarray:
    """0..1 shoulder-lift weight. Zero across the neck, full on the shoulder caps; on the torso it fades out above
    the bust, on the arms it fades much more slowly so the whole upper arm rides up (the arms sit BEHIND the
    topwear, so the small torso/arm difference near the armpit slides under the top's side edge, never gaps)."""
    x, y = pts[:, 0], pts[:, 1]
    sx = _ss(30.0, 66.0, np.abs(x))
    sy = 1.0 - (_ss(-305.0, -110.0, y) if arm else _ss(-305.0, -232.0, y))
    return sx * sy


def mouth_corner_weight(pts: np.ndarray, side: str) -> np.ndarray:
    """Hikari's LEFT = screen right (+x). Weight 0 at the mouth centre -> 1 at that corner (eased)."""
    u = (pts[:, 0] - MOUTH_CENTER[0]) / MOUTH_HALF_W * (1.0 if side == "L" else -1.0)
    return np.clip(u, 0.0, 1.2) ** 1.6


def apply_pro_transforms(puppet: dict[str, Any], iris_pngs: dict[str, bytes]) -> None:
    by_id, parent = _index(puppet["nodes"])
    params = puppet.setdefault("param", [])

    # ---- 1. Stage X / Y / Zoom on the puppet root: hair, face, body, arms, VFX and physics anchors move as one.
    if puppet["nodes"].get("uuid") == PUPPET_ROOT:
        _bind(_ensure_param(params, "Stage X", -1), PUPPET_ROOT, "transform.t.x", [-STAGE_X, 0.0, STAGE_X])
        _bind(_ensure_param(params, "Stage Y", -1), PUPPET_ROOT, "transform.t.y", [STAGE_Y[0], 0.0, STAGE_Y[1]])
        zp = _ensure_param(params, "Stage Zoom", -1)
        s0, s1 = STAGE_ZOOM
        _bind(zp, PUPPET_ROOT, "transform.s.x", [s0, 1.0, s1])
        _bind(zp, PUPPET_ROOT, "transform.s.y", [s0, 1.0, s1])
        # world = s * p + t; t = pivot * (1 - s) keeps the pivot (face) fixed. Both are linear per segment -> exact.
        _bind(zp, PUPPET_ROOT, "transform.t.y", [STAGE_PIVOT_Y * (1 - s0), 0.0, STAGE_PIVOT_Y * (1 - s1)])
        # Planted-foot whole-body rock: +r.z rotates CW (head right, feet left by -foot_y*sin(r)), so
        # compensating t.x = +foot_y*sin(r), t.y = foot_y*(1-cos(r)) keeps her feet planted while everything rocks.
        rp = _ensure_param(params, "Stage Roll", -1)
        rx = STAGE_FOOT_Y * math.sin(STAGE_ROLL)
        ry = STAGE_FOOT_Y * (1.0 - math.cos(STAGE_ROLL))
        _bind(rp, PUPPET_ROOT, "transform.r.z", [-STAGE_ROLL, 0.0, STAGE_ROLL])
        _bind(rp, PUPPET_ROOT, "transform.t.x", [-rx, 0.0, rx])
        _bind(rp, PUPPET_ROOT, "transform.t.y", [ry, 0.0, ry])

    # ---- 2. Pupil Shrink: vertex scale toward each iris's true art centre (deforms add, so it stays centred
    #         while the gaze / head deforms move the iris). Eye white sits behind -> solid white, no hole.
    ps = _ensure_param(params, "Pupil Shrink", 0)
    for side, nodes in IRIS_SETS.items():
        iris = by_id.get(nodes[0])
        if iris is None:
            continue
        iw = _world_rest(iris, parent)
        uv = iris_uv_centroid(iris_pngs[side]) if side in iris_pngs else None
        c = _uv_to_world(iris, iw, uv) if uv is not None else np.asarray(IRIS_CENTER[side], np.float64)
        for nid in nodes:
            n = by_id.get(nid)
            if n is None or not n.get("mesh", {}).get("verts"):
                continue
            w = _world_rest(n, parent)
            _bind_deform(ps, nid, [np.zeros_like(w), (PUPIL_SHRINK - 1.0) * (w - c)])

    # ---- 3. Shoulder Shrug: one continuous field over topwear + lineart overlays + collar + resting arms,
    #         raised wave-arm roots translated by the field value at their pivot.
    sp = _ensure_param(params, "Shoulder Shrug", -1)
    for nid in SHRUG_TORSO + SHRUG_ARMS:
        n = by_id.get(nid)
        if n is None or not n.get("mesh", {}).get("verts"):
            continue
        w = _world_rest(n, parent)
        f = shrug_field(w, arm=nid in SHRUG_ARMS)
        up = np.stack([np.zeros_like(f), -SHRUG_UP * f], 1)
        down = np.stack([np.zeros_like(f), SHRUG_DOWN * f], 1)
        _bind_deform(sp, nid, [down, np.zeros_like(up), up])
    for nid in SHRUG_RAISED_ROOTS:
        n = by_id.get(nid)
        if n is None:
            continue
        piv = _world_rest({"uuid": nid, "mesh": {"verts": [0.0, 0.0]}, "transform": n.get("transform", {})}, parent)
        f = float(shrug_field(piv, arm=True)[0])
        _bind(sp, nid, "transform.t.y", [SHRUG_DOWN * f, 0.0, -SHRUG_UP * f])

    by_name = {p["name"]: p for p in params}
    # ---- 4a. Left / Right mouth corner: lift (+1) / drop (-1) one half of every mouth layer, eased toward the corner.
    for side, pname in (("L", "Left mouth corner"), ("R", "Right mouth corner")):
        prm = by_name.get(pname)
        if prm is None or len(prm["axis_points"][0]) != 3:
            continue
        sg = 1.0 if side == "L" else -1.0
        for nid in MOUTH_PARTS:
            n = by_id.get(nid)
            if n is None or not n.get("mesh", {}).get("verts"):
                continue
            w = _world_rest(n, parent)
            k = mouth_corner_weight(w, side)
            up = np.stack([sg * 0.7 * k, -MOUTH_CORNER[0] * k], 1)      # smile corners also widen a touch
            down = np.stack([-sg * 0.3 * k, MOUTH_CORNER[1] * k], 1)
            _bind_deform(prm, nid, [down, np.zeros_like(up), up])

    # ---- 4b. Brow L / R X: -1 = pinch inward toward the nose bridge (inner end dips: furrow), +1 = spread outward.
    for side, pname in (("L", "Brow L X"), ("R", "Brow R X")):
        prm = by_name.get(pname)
        if prm is None or len(prm["axis_points"][0]) != 3:
            continue
        out_dir = 1.0 if side == "L" else -1.0                       # outward = away from x = 0
        for nid in BROWS[side]:
            n = by_id.get(nid)
            if n is None or not n.get("mesh", {}).get("verts"):
                continue
            w = _world_rest(n, parent)
            ax = np.abs(w[:, 0])
            inner = 1.0 - _ss(float(ax.min()), float(ax.max()), ax)    # 1 at the inner end, 0 at the tail
            dx = out_dir * (BROW_SLIDE + BROW_INNER_EXTRA * inner)
            pinch = np.stack([-dx, 3.0 * inner], 1)
            spread = np.stack([dx * 0.85, -1.0 * inner], 1)
            _bind_deform(prm, nid, [pinch, np.zeros_like(pinch), spread])

    # ---- 4c. Eye L / R Form: upper-lid arch. Lash moves rigidly by an arch-shaped profile, the eye white's top edge
    #          follows exactly (its bottom stays), the iris is masked by the white so it is clipped cleanly.
    for side, pname in (("L", "Eye L Form"), ("R", "Eye R Form")):
        prm = by_name.get(pname)
        spec = EYE_FORM_NODES[side]
        white = by_id.get(spec["white"])
        if prm is None or white is None or len(prm["axis_points"][0]) != 3:
            continue
        ww = _world_rest(white, parent)
        x0, x1 = float(ww[:, 0].min()), float(ww[:, 0].max())
        cx, hw = (x0 + x1) / 2.0, (x1 - x0) / 2.0
        y_top, y_bot = float(ww[:, 1].min()), float(ww[:, 1].max())

        def emit(nid: int, p: np.ndarray, vy: np.ndarray) -> None:
            # centre travels most -> lid flattens / arches; the corners + lash tail keep ~30% so the tail never spikes
            u = np.clip(np.abs(p[:, 0] - cx) / (hw * 1.35), 0.0, 1.0)
            k = (0.3 + 0.7 * np.cos(u * np.pi / 2.0) ** 1.5) * vy
            down = np.stack([np.zeros_like(k), EYE_FORM_DROP * k], 1)
            up = np.stack([np.zeros_like(k), -EYE_FORM_LIFT * k], 1)
            _bind_deform(prm, nid, [down, np.zeros_like(down), up])

        emit(spec["white"], ww, _ss(y_bot - 4.0, y_top + 6.0, ww[:, 1]))
        for nid in spec["lash"]:
            n = by_id.get(nid)
            if n is not None and n.get("mesh", {}).get("verts"):
                p = _world_rest(n, parent)
                emit(nid, p, np.ones(len(p)))

    # ---- 5. Kuroama v30 Head & Soft-Volume Pass (3D eyeballs, nose tip projection, cheek squish + jiggle, arm-up fit)
    apply_kuroama_v30(puppet, by_id, parent, params)


# =================================================================================================================
# 6. Kuroama v30 Pass: 3D eyeball perspective + nose projection on Angle (11x5), cheek squish on Mouth Shrug +
#    Angle X Cheek Squish, soft L/R cheek physics blend shapes (Cheek L/R PY1), and side-independent Arm R/L Fit.
# =================================================================================================================
KUROAMA_EYE_WRAP = {
    "R": {"sclera": (1014,), "lashes": (1016, 1018, 1019, 1020), "iris": (1023,), "hi": (1025, 50019)},
    "L": {"sclera": (1015,), "lashes": (1017, 1021, 1022), "iris": (1024,), "hi": (1026, 50020)},
}
NOSE_NODE = 1012


def _cheek_squish_weights(face_rest: np.ndarray, side: str | None = None, outer_bias: bool = False) -> np.ndarray:
    """Smooth, broad lower-cheek weight field on ArtMeshFace (1011). Zero on eyes/chin, wide ramp -> zero silhouette pinch."""
    x, y = face_rest[:, 0], face_rest[:, 1]
    lock = (1.0 - _ss(-378.0, -365.0, y)) * _ss(-442.0, -412.0, y)
    sides = (side,) if side in ("R", "L") else ("R", "L")
    total = np.zeros(len(face_rest), np.float64)
    for s in sides:
        cx, cy = CHEEK_CENTER[s]
        dx, dy = x - cx, y - (cy - 1.0)
        w = np.exp(-((dx / 30.0) ** 2 + (dy / 22.0) ** 2)) * lock
        sgn = 1.0 if s == "L" else -1.0
        w *= _ss(4.0, 22.0, (x - MOUTH_CENTER[0]) * sgn)
        if outer_bias:
            w *= _ss(28.0, 58.0, np.abs(x))
        total = np.maximum(total, w)
    return total


def apply_kuroama_v30(puppet: dict[str, Any], by_id: dict[int, dict[str, Any]],
                      parent: dict[int, dict[str, Any]], params: list[dict[str, Any]]) -> None:
    by_name = {p["name"]: p for p in params}

    # ---- 6a. Cheek Squish on Mouth Shrug + 3/4-Angle Extra Outer-Cheek Squish + Cheek L/R PY1 spring shapes
    face = by_id.get(FACE)
    if face is not None and face.get("mesh", {}).get("verts"):
        fw = _world_rest(face, parent)
        sgn_x = np.where(fw[:, 0] >= MOUTH_CENTER[0], 1.0, -1.0)

        # Mouth Shrug -> soft upward + outward lower-cheek squish (keeps existing VB Mouth bindings intact!)
        ms = by_name.get("Mouth Shrug")
        if ms is not None and len(ms["axis_points"][0]) == 2:
            w = _cheek_squish_weights(fw, outer_bias=False)
            squish = np.stack([sgn_x * 2.6 * w, -2.5 * w], 1)
            _bind_deform(ms, FACE, [np.zeros_like(squish), squish])

        # Angle X Cheek Squish -> extra plush outer-cheek curve when smiling/shrugging at 3/4 angles
        ax_cs = _ensure_param(params, "Angle X Cheek Squish", 0)
        w_out = _cheek_squish_weights(fw, outer_bias=True)
        out_squish = np.stack([sgn_x * 3.4 * w_out, -1.6 * w_out], 1)
        _bind_deform(ax_cs, FACE, [np.zeros_like(out_squish), out_squish])

        # Cheek L PY1 / Cheek R PY1 -> soft secondary cheek bounce spring blend shapes (-1 drop .. +1 lift)
        for side, pname in (("L", "Cheek L PY1"), ("R", "Cheek R PY1")):
            cp_y = _ensure_param(params, pname, -1)
            w_s = _cheek_squish_weights(fw, side=side, outer_bias=False)
            sg = 1.0 if side == "L" else -1.0
            drop = np.stack([-sg * 1.2 * w_s, 2.2 * w_s], 1)
            lift = np.stack([sg * 1.8 * w_s, -2.6 * w_s], 1)
            _bind_deform(cp_y, FACE, [drop, np.zeros_like(drop), lift])

    # ---- 6b. 3D Eyeball Perspective + Curved Lash Wrap + 3D Nose Tip Projection across Angle (11x5) yaw stops
    ang = by_name.get("Angle")
    if ang is not None and len(ang.get("axis_points", [])) == 2:
        ax_x = ang["axis_points"][0]
        nx = len(ax_x)
        ang_by_node = {b["node"]: b for b in ang.get("bindings", []) if b.get("param_name") == "deform"}

        for side, spec in KUROAMA_EYE_WRAP.items():
            scl = by_id.get(spec["sclera"][0])
            iris = by_id.get(spec["iris"][0])
            if scl is None or iris is None:
                continue
            sw = _world_rest(scl, parent)
            iw = _world_rest(iris, parent)
            eye_cx = float(0.5 * (sw[:, 0].min() + sw[:, 0].max()))
            eye_hw = max(8.0, float(0.5 * (sw[:, 0].max() - sw[:, 0].min())))
            iris_cx = float(iw[:, 0].mean())
            out_sgn = 1.0 if side == "L" else -1.0  # +1 screen-right for L, -1 screen-left for R

            # Sclera + Lashes: far-eye outer-corner spherical wrap; near-eye subtle convex open perspective
            for nid in spec["sclera"] + spec["lashes"]:
                b = ang_by_node.get(nid)
                nd = by_id.get(nid)
                if b is None or nd is None or not nd.get("mesh", {}).get("verts"):
                    continue
                w = _world_rest(nd, parent)
                V = np.asarray(b["values"], np.float64)
                if V.shape[0] != nx or V.shape[2] != len(w):
                    continue
                dx_c = w[:, 0] - eye_cx
                outer_u = np.clip((dx_c * out_sgn / eye_hw + 0.65) / 1.65, 0.0, 1.0) ** 1.4
                for i, axv in enumerate(ax_x):
                    yn = (float(axv) - 0.5) * 2.0  # [-1..+1]
                    mag = abs(yn) ** 1.35
                    if mag < 1e-4:
                        continue
                    is_far = (side == "R" and yn < 0.0) or (side == "L" and yn > 0.0)
                    if is_far:
                        ddx = -out_sgn * (3.2 * mag) * outer_u - 0.06 * mag * dx_c
                        ddy = (1.3 * mag) * outer_u
                    else:
                        ddx = 0.035 * mag * dx_c
                        ddy = np.zeros_like(dx_c)
                    V[i, :, :, 0] += ddx[None, :]
                    V[i, :, :, 1] += ddy[None, :]
                b["values"] = np.round(V, 4).tolist()

            # Iris + Catchlights: 3D spherical foreshortening + convex cornea-dome highlight parallax
            for is_hi, nids in ((False, spec["iris"]), (True, spec["hi"])):
                for nid in nids:
                    b = ang_by_node.get(nid)
                    nd = by_id.get(nid)
                    if b is None or nd is None or not nd.get("mesh", {}).get("verts"):
                        continue
                    w = _world_rest(nd, parent)
                    V = np.asarray(b["values"], np.float64)
                    if V.shape[0] != nx or V.shape[2] != len(w):
                        continue
                    dx_i = w[:, 0] - iris_cx
                    for i, axv in enumerate(ax_x):
                        yn = (float(axv) - 0.5) * 2.0
                        mag = abs(yn) ** 1.35
                        if mag < 1e-4:
                            continue
                        is_far = (side == "R" and yn < 0.0) or (side == "L" and yn > 0.0)
                        if is_far:
                            ddx = -0.18 * mag * dx_i + (-out_sgn * (1.2 if not is_hi else 2.5) * mag)
                        else:
                            ddx = 0.04 * mag * dx_i + (out_sgn * (0.0 if not is_hi else 1.4) * mag)
                        V[i, :, :, 0] += ddx[None, :]
                    b["values"] = np.round(V, 4).tolist()

        # Nose (1012): 3D silhouette tip projection (tip protrudes in yaw direction relative to bridge)
        nb = ang_by_node.get(NOSE_NODE)
        nnd = by_id.get(NOSE_NODE)
        if nb is not None and nnd is not None and nnd.get("mesh", {}).get("verts"):
            nw = _world_rest(nnd, parent)
            V = np.asarray(nb["values"], np.float64)
            if V.shape[0] == nx and V.shape[2] == len(nw):
                y0, y1 = float(nw[:, 1].min()), float(nw[:, 1].max())
                tip_t = np.clip((nw[:, 1] - y0) / max(1e-3, y1 - y0), 0.0, 1.0) ** 1.2
                for i, axv in enumerate(ax_x):
                    yn = (float(axv) - 0.5) * 2.0
                    mag = abs(yn) ** 1.3
                    if mag < 1e-4:
                        continue
                    sgn = 1.0 if yn > 0.0 else -1.0
                    V[i, :, :, 0] += (sgn * 3.0 * mag * tip_t)[None, :]
                    V[i, :, :, 1] += (-0.7 * mag * (1.0 - tip_t))[None, :]
                nb["values"] = np.round(V, 4).tolist()

    # ---- 6c. Arm R Fit / Arm L Fit (Kuroama Fit-Jacket coupling): single-side topwear & collar lift when raising an arm
    for side, pname in (("R", "Arm R Fit"), ("L", "Arm L Fit")):
        fp = _ensure_param(params, pname, 0)
        sg = 1.0 if side == "L" else -1.0
        for nid in SHRUG_TORSO:
            nd = by_id.get(nid)
            if nd is None or not nd.get("mesh", {}).get("verts"):
                continue
            w = _world_rest(nd, parent)
            f = shrug_field(w, arm=False) * _ss(-10.0, 32.0, w[:, 0] * sg)
            lift = np.stack([sg * 2.4 * f, -6.2 * f], 1)
            _bind_deform(fp, nid, [np.zeros_like(lift), lift])
