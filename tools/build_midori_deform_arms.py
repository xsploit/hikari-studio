"""Build models/Hikari-v29-MidoriDeform.inp: an EXPERIMENTAL copy of Hikari-v28-Custom.inp whose raised wave arms
are ONE continuous mesh Part per side driven by smooth vertex DEFORM bindings (Midori.inx style), instead of
three rigid cut PNGs (upper / forearm / hand) hanging off elbow & wrist pivot nodes.

Production is untouched: this only READS models/Hikari-v28-Custom.inp and WRITES models/Hikari-v29-MidoriDeform.inp.

Pipeline (per side):
  1. Compose the rigid chain at rest (shoulder root -> upper, elbow pivot -> forearm, wrist pivot -> hand), keeping
     the ~20 deg outward forearm wave tilt, and bake the three premultiplied textures into ONE premultiplied canvas
     (back-to-front "over", same z order as the rigid parts) in the shoulder root's local space.
  2. Mesh the silhouette: dilated alpha contour (safety margin so AA never clips) simplified + resampled, an interior
     hex grid, denser grids around the elbow and the wrist collar, Delaunay, triangles kept only inside the dilated
     silhouette. Degenerate triangles removed, all wound the same way, coverage of every visible pixel verified.
  3. Skin it with "angle-blend" rotation skinning (the 2D equivalent of dual-quaternion skinning about a joint
     centre: every vertex rotates about the joint by weight * angle, so its distance to the joint is preserved and
     the elbow never pinches). Weights: smoothstep across the elbow bisector line (wide, fleshy elbow) and across a
     NARROW collar at the wrist crease (palm and fingers get weight 1 = a pure rigid rotation, zero bending).
  4. Bake those rotations into 5-keypoint deform bindings for Arm X Gesture / Elbow / Wrist / Swing (other bindings
     on those params are resampled exactly onto the 5 keypoints), copy the Arm X Pose opacity crossfade onto the
     new Part, and drop the old 3 Parts + 2 pivot nodes (the shoulder root keeps Lift / Body / Breath bindings).

Run:  python tools/build_midori_deform_arms.py
"""
from __future__ import annotations

import copy
import io
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "Hikari-v28-Custom.inp"
DST = ROOT / "models" / "Hikari-v29-MidoriDeform.inp"

# ---------------------------------------------------------------- side layout in v28 (R first, L mirror)
SIDES = {
    "R": dict(root=50006, upper=50001, elbow=50005, forearm=50002, wrist=50004, hand=50003, new_uuid=50031, sg=1.0),
    "L": dict(root=50015, upper=50010, elbow=50014, forearm=50011, wrist=50013, hand=50012, new_uuid=50032, sg=-1.0),
}

# ---------------------------------------------------------------- deform ranges (degrees, R side; L = mirrored)
# r.z > 0 = clockwise on screen. Signs follow the v28 bindings (Gesture -1 -> elbow +, Elbow + = fold).
AXIS5 = [0.0, 0.25, 0.5, 0.75, 1.0]           # normalised axis points for -1, -0.5, 0, +0.5, +1
VALUES5 = [-1.0, -0.5, 0.0, 0.5, 1.0]
GESTURE_ELBOW, GESTURE_WRIST = 20.0, 14.0     # Gesture -1 -> +20 / +14, +1 -> -20 / -14
ELBOW_NEG, ELBOW_POS = -18.0, 28.0            # Elbow -1 -> -18 (open), +1 -> +28 (fold)
WRIST_RANGE = 18.0                            # Wrist -1 -> -18, +1 -> +18
SWING_ELBOW, SWING_WRIST = 6.0, 3.0           # Swing -1 -> +6 / +3, +1 -> -6 / -3

# ---------------------------------------------------------------- mesh / skin tuning (model units)
SIL_MARGIN = 2.6          # silhouette dilation so edge anti-aliasing never clips
POLY_EPS = 0.45           # contour simplification tolerance (< margin, so the hull stays outside the art)
EDGE_MAX = 9.0            # longest outline edge
EDGE_MAX_JOINT = 4.5      # outline edge near the elbow / wrist
GRID = 12.0               # interior hex grid spacing
GRID_ELBOW = 6.0          # elbow band grid spacing
GRID_WRIST = 4.0          # wrist collar grid spacing
ELBOW_BAND = 16.0         # half-width of the elbow smoothstep (distance across the bisector line)
WRIST_BAND = 9.0          # half-width of the wrist collar smoothstep (along the forearm axis)
WRIST_SHIFT = -2.5        # collar centre vs. the narrowest wrist section (toward the elbow keeps the palm heel rigid)


def _deg(d: float) -> float:
    return d * math.pi / 180.0


# ================================================================ INP io
def read_inp(path: Path) -> tuple[bytes, dict, list[tuple[int, bytes]], bytes]:
    raw = path.read_bytes()
    if raw[:8] != b"TRNSRTS\x00":
        raise ValueError("not an Inochi2D puppet")
    end = 12 + int.from_bytes(raw[8:12], "big")
    puppet = json.loads(raw[12:end])
    p = end
    if raw[p:p + 8] != b"TEX_SECT":
        raise ValueError("missing TEX_SECT")
    p += 8
    count = struct.unpack(">I", raw[p:p + 4])[0]
    p += 4
    texs = []
    for _ in range(count):
        ln = struct.unpack(">I", raw[p:p + 4])[0]
        enc = raw[p + 4]
        texs.append((enc, raw[p + 5:p + 5 + ln]))
        p += 5 + ln
    return raw[:8], puppet, texs, raw[p:]          # tail = optional EXT_SECT, kept verbatim


def write_inp(path: Path, magic: bytes, puppet: dict, texs: list[tuple[int, bytes]], tail: bytes) -> None:
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(magic + len(content).to_bytes(4, "big") + content + b"TEX_SECT" + struct.pack(">I", len(texs)))
    for enc, data in texs:
        out += struct.pack(">I", len(data)) + bytes([enc]) + data
    out += tail
    path.write_bytes(bytes(out))


def index_nodes(root: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    by_id, parent = {}, {}
    stack = [root]
    while stack:
        n = stack.pop()
        by_id[n["uuid"]] = n
        for c in n.get("children", []):
            parent[c["uuid"]] = n
            stack.append(c)
    return by_id, parent


# ================================================================ 2D affine helpers (3x3, y down, r.z>0 = clockwise)
def node_matrix(node: dict) -> np.ndarray:
    t = node.get("transform", {})
    tx, ty = (t.get("trans") or [0, 0, 0])[:2]
    rz = (t.get("rot") or [0, 0, 0])[2]
    sx, sy = (t.get("scale") or [1, 1])[:2]
    c, s = math.cos(rz), math.sin(rz)
    return np.array([[c * sx, -s * sy, tx], [s * sx, c * sy, ty], [0, 0, 1]], dtype=np.float64)


def rot_about(pts: np.ndarray, center: np.ndarray, ang: np.ndarray | float) -> np.ndarray:
    """Rotate each point about center by its own angle (radians, clockwise on screen for +)."""
    ang = np.broadcast_to(np.asarray(ang, dtype=np.float64), (len(pts),))
    c, s = np.cos(ang), np.sin(ang)
    d = pts - center
    return center + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=1)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ================================================================ 1. bake the rest pose onto one canvas
def decode_premul(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"), dtype=np.float32) / 255.0


def part_uv_affine(part: dict) -> np.ndarray:
    """Least-squares affine local(x,y) -> uv(u,v) of a Part's (grid) mesh, as 2x3."""
    m = part["mesh"]
    v = np.array(m["verts"], dtype=np.float64).reshape(-1, 2)
    uv = np.array(m["uvs"], dtype=np.float64).reshape(-1, 2)
    A = np.hstack([v, np.ones((len(v), 1))])
    sol, *_ = np.linalg.lstsq(A, uv, rcond=None)
    return sol.T                                                   # 2x3


def bake_arm(parts: list[tuple[dict, np.ndarray]], texs: list[tuple[int, bytes]]):
    """parts: [(Part node, 3x3 part-local -> shoulder-root matrix)] back-to-front.
    Returns premultiplied canvas (H,W,4 float), origin Q0 (shoulder units of pixel edge 0,0) and px/unit."""
    corners, ppu = [], 0.0
    imgs = []
    for part, M in parts:
        v = np.array(part["mesh"]["verts"], dtype=np.float64).reshape(-1, 2)
        lo, hi = v.min(0), v.max(0)
        box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        corners.append((M[:2, :2] @ box.T).T + M[:2, 2])
        img = decode_premul(texs[part["textures"][0]][1])
        imgs.append(img)
        ppu = max(ppu, img.shape[1] / (hi[0] - lo[0]), img.shape[0] / (hi[1] - lo[1]))
    allc = np.vstack(corners)
    pad = SIL_MARGIN + 6.0
    q0 = allc.min(0) - pad
    q1 = allc.max(0) + pad
    W = int(math.ceil((q1[0] - q0[0]) * ppu))
    H = int(math.ceil((q1[1] - q0[1]) * ppu))
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    for (part, M), img in zip(parts, imgs):
        th, tw = img.shape[:2]
        # canvas px index c -> shoulder q = q0 + (c + .5)/ppu -> local l = M^-1 q -> uv -> tex px = uv*size - .5
        C2Q = np.array([[1 / ppu, 0, q0[0] + 0.5 / ppu], [0, 1 / ppu, q0[1] + 0.5 / ppu], [0, 0, 1]])
        UV = np.vstack([part_uv_affine(part), [0, 0, 1]])
        UV2T = np.array([[tw, 0, -0.5], [0, th, -0.5], [0, 0, 1]])
        A = UV2T @ UV @ np.linalg.inv(M) @ C2Q
        layer = cv2.warpAffine(img, A[:2], (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        layer[..., 3] = np.clip(layer[..., 3], 0.0, 1.0)
        layer[..., :3] = np.clip(layer[..., :3], 0.0, layer[..., 3:4])      # cubic overshoot, stay premultiplied
        canvas = layer + canvas * (1.0 - layer[..., 3:4])                   # premultiplied "over"
    return canvas, q0, ppu


LINE_RGB = np.array((150.0, 99.0, 82.0), np.float32) / 255.0   # the wave art's warm-brown outline (customizer.py)
LINE_W = 0.7                                                    # outline width, model units
ELBOW_REPAIR_R = 46.0                                           # repair radius around the elbow pivot


def repair_elbow_seam(canvas: np.ndarray, q0: np.ndarray, ppu: float, E: np.ndarray) -> tuple[np.ndarray, int]:
    """The rigid rig left a small step/notch in the outer-elbow outline (forearm ball-cap meets the upper arm) plus
    faint internal stroke remnants of the cut edges. On the baked canvas: erase internal strokes, smooth both concave
    notches AND convex steps via Gaussian-filtered signed distance field (SDF) inside a feathered disc around the
    elbow, inpaint newly exposed interior skin cleanly, and re-draw one continuous uniform outline."""
    H, W = canvas.shape[:2]
    a = canvas[..., 3].copy()
    straight = np.where(a[..., None] > 1e-4, canvas[..., :3] / np.maximum(a[..., None], 1e-4), 0.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ex, ey = (E - q0) * ppu - 0.5
    r_units = np.hypot(xx - ex, yy - ey) / ppu
    region = r_units < ELBOW_REPAIR_R
    blend = 1.0 - smoothstep(ELBOW_REPAIR_R - 8.0, ELBOW_REPAIR_R, r_units)

    # 1) internal dark strokes: darker than the grey-closed (stroke-free) skin, and > 1.2 units inside the outline
    lum = straight.mean(-1).astype(np.float32)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2.4 * ppu) | 1,) * 2)
    lum_bg = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, kc)
    mask_bin = (a > 0.5).astype(np.uint8)
    d_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5) / ppu
    d_out = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5) / ppu
    stroke = region & (a > 0.96) & (d_in > 1.2) & ((lum_bg - lum) > 0.025)
    stroke = cv2.dilate(stroke.astype(np.uint8), np.ones((3, 3), np.uint8))
    n_stroke = int(stroke.sum())
    if n_stroke:
        s8 = np.round(straight.clip(0, 1) * 255).astype(np.uint8)
        fixed = cv2.inpaint(s8, stroke, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        straight = np.where(stroke[..., None] > 0, fixed, straight)

    # 2) Deterministic outer-elbow C1 cubic Hermite spline bridge across [ey - 6u .. ey + 23u]:
    #    The hand sits to the left of E on the R arm (outer elbow = left edge, side_idx=0) and to the right on the L
    #    arm (outer elbow = right edge, side_idx=-1). Anchoring cleanly above the forearm/upper-arm seam (ey - 6u)
    #    and below it (ey + 23u, well above the elbow tip) eliminates the outer-elbow bump on both R and L arms!
    top_nz = np.flatnonzero(mask_bin[: max(1, H // 3)].any(axis=0))
    x_hand = float(top_nz.mean()) if len(top_nz) else float(W * 0.5)
    side_idx = 0 if x_hand < ex else -1

    y_a = max(6.0 * ppu, ey - 6.0 * ppu)
    y_b = min(float(H - 1) - 6.0 * ppu, ey + 23.0 * ppu)
    y_scan0 = max(0, int(np.floor(y_a - 7.0 * ppu)))
    y_scan1 = min(H - 1, int(np.ceil(y_b + 7.0 * ppu)))
    ys = np.arange(y_scan0, y_scan1 + 1, dtype=np.float64)
    xs = np.full(len(ys), np.nan, dtype=np.float64)
    for k, y in enumerate(ys.astype(int)):
        nz = np.flatnonzero(mask_bin[y])
        if len(nz):
            xs[k] = float(nz[side_idx])

    new_a = a.copy()
    repaired_band = np.zeros((H, W), dtype=bool)
    m_a = np.isfinite(xs) & (ys >= y_a - 6.0 * ppu) & (ys <= y_a)
    m_b = np.isfinite(xs) & (ys >= y_b) & (ys <= y_b + 6.0 * ppu)
    if m_a.sum() >= 5 and m_b.sum() >= 5:
        pa = np.polyfit(ys[m_a], xs[m_a], 1)
        pb = np.polyfit(ys[m_b], xs[m_b], 1)
        xa, dxa = float(np.polyval(pa, y_a)), float(pa[0])
        xb, dxb = float(np.polyval(pb, y_b)), float(pb[0])
        span = max(1.0, y_b - y_a)
        m0, m1 = dxa * span, dxb * span
        yi0 = max(0, int(np.floor(y_a)))
        yi1 = min(H - 1, int(np.ceil(y_b)))
        for y in range(yi0, yi1 + 1):
            t = max(0.0, min(1.0, (float(y) - y_a) / span))
            t2, t3 = t * t, t * t * t
            xt = (2.0 * t3 - 3.0 * t2 + 1.0) * xa + (t3 - 2.0 * t2 + t) * m0 + (-2.0 * t3 + 3.0 * t2) * xb + (t3 - t2) * m1
            x_band = np.abs(xx[y] - xt) < 6.5 * ppu
            a_edge = np.clip(0.5 + (xx[y] - xt if side_idx == 0 else xt - xx[y]), 0.0, 1.0)
            wt = float(smoothstep(y_a, y_a + 2.5 * ppu, float(y)) * (1.0 - smoothstep(y_b - 2.5 * ppu, y_b, float(y))))
            new_a[y, x_band] = a[y, x_band] + (a_edge[x_band] - a[y, x_band]) * wt
            repaired_band[y, x_band] = True

    # 3) Cleanly inpaint skin colour inside the repaired band (removes old dark double-outline + fills grown pixels)
    d_new = cv2.distanceTransform((new_a > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / ppu
    grown = (new_a > 0.02) & (a < 0.90) & repaired_band
    old_dark = (new_a > 0.02) & (d_new < 4.0) & ((lum_bg - lum) > 0.02) & repaired_band
    strip = (grown | old_dark | ((new_a > 0.02) & (d_new < 2.0) & repaired_band)).astype(np.uint8)
    if int(strip.sum()) > 0:
        s8 = np.round(straight.clip(0, 1) * 255).astype(np.uint8)
        skin_fill = cv2.inpaint(s8, strip, max(7, int(round(5.0 * ppu))), cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        fix_w = np.where(strip > 0, 1.0, 0.0).astype(np.float32)
        straight = straight + (skin_fill - straight) * fix_w[..., None]

    # 4) Stroke one crisp, continuous warm-brown anime outline along the exact smooth outer-elbow spline
    lw = np.clip((LINE_W - d_new) / 0.28, 0.0, 1.0) * (new_a > 0.02) * repaired_band
    lum_line = float(LINE_RGB.mean())
    lw = lw * (straight.mean(-1) > lum_line - 0.02)          # never lighten pixels already darker than the line
    straight = straight + (LINE_RGB[None, None, :] - straight) * lw[..., None]

    out = np.empty_like(canvas)
    out[..., 3] = new_a
    out[..., :3] = straight.clip(0, 1) * new_a[..., None]
    return out, n_stroke


# ================================================================ 2. mesh the silhouette
def build_mesh(alpha: np.ndarray, q0: np.ndarray, ppu: float, joint_zone):
    """alpha: canvas alpha (0..1). joint_zone(pts_units) -> 0 (plain) / 1 (elbow band) / 2 (wrist collar).
    Returns verts (N,2 shoulder units), tris (M,3), uvs (N,2)."""
    H, W = alpha.shape
    to_u = lambda px: q0 + (px + 0.5) / ppu                    # pixel centre -> units
    to_px = lambda u: (u - q0) * ppu - 0.5
    vis = (alpha > 0.004).astype(np.uint8)
    r = max(1, int(round(SIL_MARGIN * ppu)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(vis, k)
    dil = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    tol = cv2.dilate(dil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))   # tolerance for edge tests
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cv2.contourArea(c) for c in contours)
    loops = []
    for c in contours:
        if cv2.contourArea(c) < 0.002 * big:
            continue
        poly = cv2.approxPolyDP(c, POLY_EPS * ppu, True)[:, 0, :].astype(np.float64)
        pts = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ua, ub = to_u(a), to_u(b)
            zone = joint_zone(np.array([(ua + ub) / 2]))[0]
            emax = EDGE_MAX_JOINT if zone else EDGE_MAX
            n = max(1, int(math.ceil(np.linalg.norm(ub - ua) / emax)))
            for j in range(n):
                pts.append(ua + (ub - ua) * (j / n))
        loops.append(np.array(pts))
    boundary = np.vstack(loops)
    # interior: hex grids, keep points a safe distance inside the (dilated) outline
    dist = cv2.distanceTransform(dil, cv2.DIST_L2, 5) / ppu
    lo, hi = q0, q0 + np.array([W, H]) / ppu
    interior = []
    for spacing, want in ((GRID, 0), (GRID_ELBOW, 1), (GRID_WRIST, 2)):
        ys = np.arange(lo[1], hi[1], spacing * math.sqrt(3) / 2)
        for row, y in enumerate(ys):
            xs = np.arange(lo[0] + (spacing / 2 if row % 2 else 0.0), hi[0], spacing)
            cand = np.stack([xs, np.full_like(xs, y)], axis=1)
            px = np.round(to_px(cand)).astype(int)
            ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            cand, px = cand[ok], px[ok]
            d = dist[px[:, 1], px[:, 0]]
            z = joint_zone(cand)
            keep = (d > 0.55 * spacing) & (z == want)
            interior.append(cand[keep])
    inner = np.vstack(interior)
    # dedupe interior points where two grid zones meet
    keep = np.ones(len(inner), bool)
    for i, j in sorted(cKDTree(inner).query_pairs(0.6 * GRID_WRIST)):
        if keep[i] and keep[j]:
            keep[j] = False
    inner = inner[keep]
    # Conforming Delaunay: split every outline segment whose diametral circle holds another point, until each
    # outline segment is guaranteed to be a Delaunay edge (so no triangle can bridge a finger gap or the elbow pit).
    for _ in range(40):
        allp = np.vstack(loops + [inner])
        tree = cKDTree(allp)
        changed = False
        new_loops = []
        for lp in loops:
            out = []
            for i in range(len(lp)):
                a, b = lp[i], lp[(i + 1) % len(lp)]
                mid, rad = (a + b) / 2, np.linalg.norm(b - a) / 2
                hit = [j for j in tree.query_ball_point(mid, rad * 0.999)
                       if np.linalg.norm(allp[j] - a) > 1e-6 and np.linalg.norm(allp[j] - b) > 1e-6]
                out.append(a)
                if hit and rad > 0.2:
                    out.append(mid)
                    changed = True
            new_loops.append(np.array(out))
        loops = new_loops
        if not changed:
            break
    verts = np.vstack(loops + [inner])
    tri = Delaunay(verts).simplices
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    cen_px = to_px((a + b + c) / 3).astype(np.float32)
    polys_px = [to_px(lp).astype(np.float32).reshape(-1, 1, 2) for lp in loops]
    ok = np.array([any(cv2.pointPolygonTest(pp, (float(x), float(y)), False) >= 0 for pp in polys_px)
                   for x, y in cen_px])
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    ok &= np.abs(area) > 1e-4
    tri = tri[ok]
    area = area[ok]
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]                      # uniform winding
    used = np.unique(tri)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    verts, tri = verts[used], remap[tri]
    uvs = (verts - q0) * ppu / np.array([W, H])
    # coverage: every visible pixel must sit under a triangle
    cov = np.zeros((H, W), np.uint8)
    pts_px = np.round((verts - q0) * ppu * 4).astype(np.int32)       # 2 bits sub-pixel
    for t in tri:
        cv2.fillConvexPoly(cov, pts_px[t], 1, lineType=cv2.LINE_8, shift=2)
    uncovered = int(((alpha > 0.004) & (cov == 0)).sum())
    return verts, tri, uvs, uncovered


def tri_stats(verts: np.ndarray, tri: np.ndarray) -> dict:
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    angs = []
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q - p, r - p
        angs.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1))))
    return dict(min_area=float(area.min()), inverted=int((area <= 0).sum()), min_angle=float(np.min(angs)))


# ================================================================ 3. skinning rig per side
class ArmSkin:
    def __init__(self, E: np.ndarray, W: np.ndarray, S: np.ndarray, alpha: np.ndarray, q0: np.ndarray, ppu: float):
        self.E = E
        a = (S - E) / np.linalg.norm(S - E)                          # elbow -> shoulder
        b = (W - E) / np.linalg.norm(W - E)                          # elbow -> wrist
        m = (a + b) / np.linalg.norm(a + b)                          # inner bisector
        n = np.array([-m[1], m[0]])
        if n @ b < 0:
            n = -n
        self.a, self.b, self.n = a, b, n
        self.inner_angle = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
        # wrist crease = narrowest silhouette cross-section along the forearm axis near the wrist pivot
        H, Wd = alpha.shape
        vis = alpha > 0.5
        perp = np.array([-b[1], b[0]])
        best = None
        for d in np.arange(-26.0, 12.01, 0.5):
            cen = W + b * d
            ts = np.arange(-45.0, 45.0, 0.25)
            pts = cen + perp * ts[:, None]
            px = np.round((pts - q0) * ppu - 0.5).astype(int)
            okp = (px[:, 0] >= 0) & (px[:, 0] < Wd) & (px[:, 1] >= 0) & (px[:, 1] < H)
            on = np.zeros(len(ts), bool)
            on[okp] = vis[px[okp, 1], px[okp, 0]]
            # contiguous run through the axis
            idx = np.where(on)[0]
            if not len(idx):
                continue
            mid = len(ts) // 2
            runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            run = min(runs, key=lambda r_: 0 if r_[0] <= mid <= r_[-1] else min(abs(r_[0] - mid), abs(r_[-1] - mid)))
            width = (run[-1] - run[0]) * 0.25
            if best is None or width < best[0] - 1e-6:
                best = (width, d, cen + perp * (ts[run[0]] + ts[run[-1]]) / 2)
        self.wrist_width, self.wrist_d, crease = best
        self.C = crease + b * WRIST_SHIFT                                 # collar centre = wrist rotation centre
        self.Wpivot = W

    def weights(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        we = smoothstep(-ELBOW_BAND, ELBOW_BAND, (p - self.E) @ self.n)
        ww = smoothstep(-WRIST_BAND, WRIST_BAND, (p - self.C) @ self.b)
        return we, ww

    def zone(self, p: np.ndarray) -> np.ndarray:
        de = (p - self.E) @ self.n
        re = np.linalg.norm(p - self.E, axis=1)
        dw = (p - self.C) @ self.b
        rw = np.linalg.norm(p - self.C, axis=1)
        z = np.zeros(len(p), int)
        z[(np.abs(de) < ELBOW_BAND * 1.6) & (re < 62.0)] = 1
        z[(np.abs(dw) < WRIST_BAND * 2.2) & (rw < 36.0)] = 2
        return z

    def pose(self, p: np.ndarray, elbow_deg: float, wrist_deg: float) -> np.ndarray:
        we, ww = self.weights(p)
        q = rot_about(p, self.C, ww * _deg(wrist_deg))               # hand about the wrist crease centre
        return rot_about(q, self.E, we * _deg(elbow_deg))            # then everything past the elbow


# ================================================================ 4. params / bindings
def resample_axis(param: dict, new_axis0: list[float]) -> None:
    """Move a 1D param onto new axis points, linearly resampling every binding (exact for linear interp)."""
    old = [float(x) for x in param["axis_points"][0]]
    if old == new_axis0:
        return
    n1 = len(param["axis_points"][1])
    for b in param.get("bindings", []):
        vals = np.array(b["values"], dtype=np.float64)              # (n0, n1, ...) any trailing shape
        flat = vals.reshape(len(old), -1)
        out = np.stack([np.interp(new_axis0, old, flat[:, k]) for k in range(flat.shape[1])], axis=1)
        b["values"] = out.reshape((len(new_axis0),) + vals.shape[1:]).tolist()
        b["isSet"] = [[True] * n1 for _ in new_axis0]
    param["axis_points"][0] = list(new_axis0)


def deform_binding(node_id: int, rest: np.ndarray, posed: list[np.ndarray], n1: int) -> dict:
    vals = []
    for P in posed:
        off = np.round(P - rest, 4).tolist()
        vals.append([off for _ in range(n1)])
    return {"node": node_id, "param_name": "deform", "interpolate_mode": "Linear",
            "isSet": [[True] * n1 for _ in posed], "values": vals}


def main() -> int:
    magic, puppet, texs, tail = read_inp(SRC)
    by_id, parent = index_nodes(puppet["nodes"])
    params = {p["name"]: p for p in puppet["param"]}
    report = {}
    removed: set[int] = set()

    for side, L in SIDES.items():
        sg = L["sg"]
        root = by_id[L["root"]]
        upper, elbow, forearm = by_id[L["upper"]], by_id[L["elbow"]], by_id[L["forearm"]]
        wrist, hand = by_id[L["wrist"]], by_id[L["hand"]]
        Mu = node_matrix(upper)
        Me = node_matrix(elbow)
        Mf = Me @ node_matrix(forearm)
        Mw = Me @ node_matrix(wrist)
        Mh = Mw @ node_matrix(hand)
        E = Me[:2, 2].copy()
        Wp = Mw[:2, 2].copy()
        S = Mu[:2, 2].copy()                                         # shoulder (root origin)

        # back-to-front by zsort (higher zsort = further back)
        layers = sorted([(upper, Mu), (forearm, Mf), (hand, Mh)], key=lambda t: -float(t[0].get("zsort", 0)))
        canvas, q0, ppu = bake_arm(layers, texs)
        canvas, n_stroke = repair_elbow_seam(canvas, q0, ppu, E)
        skin = ArmSkin(E, Wp, S, canvas[..., 3], q0, ppu)
        verts, tri, uvs, uncovered = build_mesh(canvas[..., 3], q0, ppu, skin.zone)
        # Draw order inside the Part = index order: upper arm first, forearm next, hand last. When the elbow folds,
        # forearm flesh then covers the compressed crease exactly like the rigid parts' z order did.
        we_v, ww_v = skin.weights(verts)
        tri = tri[np.argsort((we_v + ww_v)[tri].mean(1), kind="stable")]
        st = tri_stats(verts, tri)

        # texture
        png = io.BytesIO()
        Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGBA").save(png, "PNG", optimize=False, compress_level=6)
        tex_id = len(texs)
        texs.append((0, png.getvalue()))

        # the unified Part (inherits the upper's render settings)
        part = copy.deepcopy({k: v for k, v in upper.items() if k != "children"})
        part.update(uuid=L["new_uuid"], name=f"{side} arm · raised unified deform mesh (Midori)",
                    zsort=min(float(n.get("zsort", 0)) for n in (upper, forearm, hand)),
                    transform={"trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
                    textures=[tex_id, 4294967295, 4294967295],
                    mesh={"verts": np.round(verts.reshape(-1), 4).tolist(),
                          "uvs": np.round(uvs.reshape(-1), 6).tolist(),
                          "indices": tri.reshape(-1).astype(int).tolist(), "origin": [0.0, 0.0]})
        part["children"] = []
        root["children"] = [c for c in root.get("children", []) if c["uuid"] not in (L["upper"], L["elbow"])] + [part]
        removed |= {L["upper"], L["elbow"], L["forearm"], L["wrist"], L["hand"]}

        # ---- bindings
        pose_vals = None
        for b in params[f"Arm {side} Pose"]["bindings"]:
            if b["node"] == L["upper"] and b["param_name"] == "opacity":
                pose_vals = copy.deepcopy(b)
        pose_vals["node"] = L["new_uuid"]
        params[f"Arm {side} Pose"]["bindings"].append(pose_vals)

        specs = {
            # value v in [-1, 1] -> (elbow deg, wrist deg), R side
            "Gesture": lambda v: (-GESTURE_ELBOW * v, -GESTURE_WRIST * v),
            "Elbow": lambda v: ((-ELBOW_NEG) * v if v < 0 else ELBOW_POS * v, 0.0),
            "Wrist": lambda v: (0.0, WRIST_RANGE * v),
            "Swing": lambda v: (-SWING_ELBOW * v, -SWING_WRIST * v),
        }
        ranges = {}
        for key, fn in specs.items():
            prm = params[f"Arm {side} {key}"]
            resample_axis(prm, AXIS5)
            n1 = len(prm["axis_points"][1])
            posed, angs = [], []
            for v in VALUES5:
                e_deg, w_deg = fn(v)
                e_deg, w_deg = sg * e_deg, sg * w_deg
                angs.append((round(e_deg, 2), round(w_deg, 2)))
                posed.append(skin.pose(verts, e_deg, w_deg))
            prm["bindings"].append(deform_binding(L["new_uuid"], verts, posed, n1))
            ranges[key] = angs
        # rigidity / inversion diagnostics at the extremes
        we, ww = skin.weights(verts)
        hand_mask = (ww > 0.999) & (we > 0.999)
        hand_tris = tri[hand_mask[tri].all(1)]
        diag = {}
        for key, (ev, wv) in {"Gesture-1": specs["Gesture"](-1), "Gesture+1": specs["Gesture"](1),
                              "Elbow-1": specs["Elbow"](-1), "Elbow+1": specs["Elbow"](1),
                              "Wrist-1": specs["Wrist"](-1), "Wrist+1": specs["Wrist"](1)}.items():
            P = skin.pose(verts, sg * ev, sg * wv)
            s2 = tri_stats(P, tri)
            e0 = np.linalg.norm(verts[hand_tris[:, 0]] - verts[hand_tris[:, 1]], axis=1)
            e1 = np.linalg.norm(P[hand_tris[:, 0]] - P[hand_tris[:, 1]], axis=1)
            diag[key] = dict(inverted=s2["inverted"], hand_edge_err=float(np.abs(e1 - e0).max()) if len(e0) else 0.0)
        report[side] = dict(verts=len(verts), tris=len(tri), tex=f"{canvas.shape[1]}x{canvas.shape[0]}", ppu=round(ppu, 3),
                            uncovered_px=uncovered, elbow_stroke_px=n_stroke, rest=st, inner_elbow_deg=round(skin.inner_angle, 1),
                            wrist_crease_d=skin.wrist_d, wrist_width=skin.wrist_width, hand_rigid_verts=int(hand_mask.sum()),
                            ranges=ranges, extremes=diag)

    # drop every binding that pointed at a removed node
    for p in puppet["param"]:
        p["bindings"] = [b for b in p.get("bindings", []) if b["node"] not in removed]
    # shape check: values[n0][n1](...) and isSet[n0][n1]
    for p in puppet["param"]:
        n0, n1 = len(p["axis_points"][0]), len(p["axis_points"][1])
        for b in p["bindings"]:
            assert len(b["values"]) == n0 and all(len(r_) == n1 for r_ in b["values"]), (p["name"], b["node"])
            assert len(b["isSet"]) == n0 and all(len(r_) == n1 for r_ in b["isSet"]), (p["name"], b["node"])
    for side, L in SIDES.items():
        nv = len(next(c for c in by_id[L["root"]]["children"] if c["uuid"] == L["new_uuid"])["mesh"]["verts"]) // 2
        for key in ("Gesture", "Elbow", "Wrist", "Swing"):
            b = next(b for b in params[f"Arm {side} {key}"]["bindings"] if b["node"] == L["new_uuid"])
            assert all(len(cell) == nv and all(len(xy) == 2 for xy in cell) for row in b["values"] for cell in row)
    puppet.setdefault("meta", {})["name"] = "Hikari v29 MidoriDeform (experimental)"

    write_inp(DST, magic, puppet, texs, tail)
    for side, rep in report.items():
        print(side, json.dumps(rep))
    print(f"wrote {DST} ({DST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
