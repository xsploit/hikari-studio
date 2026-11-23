"""Non-destructive 4x Anime Lineart Texture Upscaler for Inochi2D (.inp) models.
Because Inochi2D mesh UVs are normalized [0..1], upscaling every small texture in TEX_SECT
up to 1200px gives razor-sharp anime lineart, eyelashes & eye highlights with ZERO blur
and ZERO white halos (premultiplied-alpha safe edge bleed + multi-scale unsharp restoration).
"""
from __future__ import annotations

import io
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import splev, splprep


def vector_smooth_alpha(
    alpha_f32: np.ndarray,
    smooth_px: float = 1.65,
    corner_deg: float = 126.0,
    ss: int = 4,
) -> np.ndarray:
    """Potrace-style corner-preserving cubic B-spline contour vectorization + 4x supersampled rasterization.
    Eliminates low-res pixel stair-stepping on anime body/face/arm/outfit/eyelash silhouettes while
    keeping sharp corners crisp and internal cel-shading 100% untouched."""
    h, w = alpha_f32.shape
    mask = (alpha_f32 >= 0.48).astype(np.uint8) * 255
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(contours) == 0:
        return alpha_f32
    hier = hier[0]
    hi_mask = np.zeros((h * ss, w * ss), dtype=np.uint8)

    def _smooth_contour(pts: np.ndarray) -> np.ndarray:
        n = len(pts)
        if n < 12:
            return pts.astype(np.float32)
        k = max(4, min(10, n // 12))
        p_prev = np.roll(pts, k, axis=0)
        p_next = np.roll(pts, -k, axis=0)
        v1 = p_prev - pts
        v2 = p_next - pts
        n1 = np.linalg.norm(v1, axis=1) + 1e-6
        n2 = np.linalg.norm(v2, axis=1) + 1e-6
        cos_a = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
        ang = np.degrees(np.arccos(cos_a))
        cand = np.where(ang < corner_deg)[0]
        corners: list[int] = []
        if len(cand):
            groups = np.split(cand, np.where(np.diff(cand) > k)[0] + 1)
            for g in groups:
                corners.append(int(g[int(np.argmin(ang[g]))]))
            corners = sorted(set(corners))

        def _fit_seg(seg: np.ndarray, closed: bool) -> np.ndarray:
            m = len(seg)
            if m < 6:
                return seg.astype(np.float32)
            step = max(1, min(3, m // 18))
            sub = seg[::step].astype(np.float64)
            if not closed and not np.array_equal(sub[-1], seg[-1]):
                sub = np.vstack([sub, seg[-1].astype(np.float64)])
            if closed:
                sub = np.vstack([sub, sub[0]])
            if len(sub) < 5:
                return seg.astype(np.float32)
            s_val = len(sub) * (smooth_px ** 2) * 0.45
            try:
                tck, _ = splprep([sub[:, 0], sub[:, 1]], s=s_val, k=3, per=closed)
                u_new = np.linspace(0.0, 1.0, max(len(seg) * 2, 32), endpoint=not closed)
                xn, yn = splev(u_new, tck)
                return np.stack([xn, yn], axis=1).astype(np.float32)
            except Exception:
                return seg.astype(np.float32)

        if not corners:
            return _fit_seg(pts, closed=True)
        out_segs = []
        for idx_c in range(len(corners)):
            c0 = corners[idx_c]
            c1 = corners[(idx_c + 1) % len(corners)]
            seg = pts[c0 : c1 + 1] if c1 > c0 else np.vstack([pts[c0:], pts[: c1 + 1]])
            out_segs.append(_fit_seg(seg, closed=False))
        return np.vstack(out_segs)

    for is_hole in (False, True):
        for i, cnt in enumerate(contours):
            if (hier[i][3] != -1) != is_hole:
                continue
            pts = cnt[:, 0, :]
            if len(pts) < 4:
                continue
            sm = _smooth_contour(pts)
            poly = np.rint(sm * ss).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(hi_mask, [poly], 0 if is_hole else 255, lineType=cv2.LINE_AA)

    return cv2.resize(hi_mask.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def apply_anime_skin_outline(
    rgb: np.ndarray,
    alpha: np.ndarray,
    outline_w: float = 2.6,
    skip_medial_shoulder: str = "",
    top_edges_only: bool = False,
) -> np.ndarray:
    """Add a subtle, crisp warm-brown anime skin outline matching Hikari's neck/chin ink ([150, 104, 95])
    ONLY along exposed peach skin outer silhouettes, leaving fabric/hair/prints and arm-to-torso seams untouched."""
    if outline_w <= 0.1:
        return rgb
    h, w = alpha.shape
    dist_in = cv2.distanceTransform((alpha >= 0.45).astype(np.uint8), cv2.DIST_L2, 5)
    band = np.clip((outline_w - dist_in) / 1.35, 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    is_skin = (r > 165.0) & (r > g + 8.0) & (lum > 150.0)
    skin_w = cv2.GaussianBlur(is_skin.astype(np.float32), (0, 0), 1.5)

    # For Topwear, ONLY outline top-facing shoulder slopes (never vertical left/right seams overlapping arms!)
    if top_edges_only:
        gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=5)
        gmag = np.hypot(gx, gy) + 1e-6
        # Top edge has alpha increasing downward -> gy > 0
        up_facing = np.clip((gy / gmag - 0.45) / 0.35, 0.0, 1.0)
        yy = np.arange(h, dtype=np.float32)[:, None] / float(max(1, h))
        shoulder_band = np.clip((0.36 - yy) / 0.06, 0.0, 1.0)
        band = band * up_facing * shoulder_band

    # Mask out medial upper shoulder socket where arm cap overlaps torso skin
    if skip_medial_shoulder in ("right_arm", "left_arm"):
        yy = np.arange(h, dtype=np.float32)[:, None] / float(max(1, h))
        xx = np.arange(w, dtype=np.float32)[None, :]
        a_bin = (alpha > 0.35).astype(np.float32)
        row_sum = np.maximum(1.0, a_bin.sum(axis=1, keepdims=True))
        row_cx = (a_bin * xx).sum(axis=1, keepdims=True) / row_sum
        top_fade = np.clip((0.30 - yy) / 0.06, 0.0, 1.0)
        if skip_medial_shoulder == "right_arm":
            medial = np.clip((xx - row_cx) / 14.0, 0.0, 1.0)
        else:
            medial = np.clip((row_cx - xx) / 14.0, 0.0, 1.0)
        band = band * (1.0 - top_fade * medial)

    ink = np.array([150.0, 104.0, 95.0], dtype=np.float32)[None, None, :]
    ow = (band * skin_w * 0.65)[..., None]
    return np.clip(rgb * (1.0 - ow) + ink * ow, 0.0, 255.0)


def smooth_premul_silhouette_png(
    png_bytes: bytes,
    smooth_px: float = 1.8,
    corner_deg: float = 126.0,
    clean_dark_fabric_fringe: bool = False,
    symmetrize_shoulders: bool = False,
    skin_outline_w: float = 0.0,
) -> bytes:
    """Apply Potrace-style vector silhouette smoothing to an already-HD premultiplied RGBA PNG."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a0 = arr[..., 3] / 255.0
    rgb = np.zeros_like(arr[..., :3])
    mask = a0 > 0.02
    if not np.any(mask):
        return png_bytes
    rgb[mask] = np.clip(arr[mask, :3] / np.maximum(a0[mask, None], 0.02), 0.0, 255.0)
    h, w = a0.shape

    # Symmetrize outer shoulder trapezius slope (removes sharp chopped right-shoulder corner while keeping prints untouched!)
    if symmetrize_shoulders:
        yy = np.arange(h, dtype=np.float32)[:, None] / float(max(1, h))
        xx = np.arange(w, dtype=np.float32)[None, :] / float(max(1, w))
        t_y = np.clip((yy - 0.12) / 0.04, 0.0, 1.0) * np.clip((0.40 - yy) / 0.05, 0.0, 1.0)
        outer_r = np.clip((xx - 0.60) / 0.06, 0.0, 1.0)
        blend = t_y * outer_r
        a0 = a0 * (1.0 - blend) + np.minimum(a0, a0[:, ::-1]) * blend

    # Optional armpit/outer-bust skin-fringe cleanup on dark tops:
    # Removes 2-3px bright peach cutout remnants sitting right on the outer edge of dark charcoal fabric!
    if clean_dark_fabric_fringe:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        is_dark_cloth = (a0 > 0.65) & (lum < 92.0) & (np.abs(r - b) < 28.0)
        is_peach_edge = (a0 > 0.05) & (r > 135.0) & (r > b + 12.0) & (lum > 115.0)
        dist_in = cv2.distanceTransform((a0 >= 0.45).astype(np.uint8), cv2.DIST_L2, 5)
        near_outer = dist_in <= 5.5
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        near_cloth = cv2.dilate(is_dark_cloth.astype(np.uint8), k9) > 0
        yy = np.arange(h, dtype=np.float32)[:, None] / float(max(1, h))
        bust_band = (yy >= 0.30) & (yy <= 0.78)
        fringe = is_peach_edge & near_outer & near_cloth & bust_band
        if np.any(fringe):
            w_c = cv2.GaussianBlur(is_dark_cloth.astype(np.float32), (0, 0), 5.0)
            cloth_fill = np.stack([
                cv2.GaussianBlur(rgb[..., c] * is_dark_cloth.astype(np.float32), (0, 0), 5.0) / np.maximum(1e-4, w_c)
                for c in range(3)
            ], axis=-1)
            f_soft = cv2.GaussianBlur(fringe.astype(np.float32), (0, 0), 1.0)[..., None]
            rgb = rgb * (1.0 - f_soft) + np.where(w_c[..., None] > 0.01, cloth_fill, rgb) * f_soft

    # Halo-free outward RGB bleed so vector contour AA never pulls black/white garbage
    valid = (a0 > 0.40).astype(np.float32)
    wb = cv2.GaussianBlur(valid, (0, 0), 5.0)
    bled = np.stack([
        cv2.GaussianBlur(rgb[..., c] * valid, (0, 0), 5.0) / np.maximum(1e-4, wb)
        for c in range(3)
    ], axis=-1)
    rgb_full = np.where((a0 > 0.58)[..., None], rgb, np.where(wb[..., None] > 0.01, bled, rgb))

    a1 = vector_smooth_alpha(a0, smooth_px=smooth_px, corner_deg=corner_deg, ss=4)
    if skin_outline_w > 0.1:
        rgb_full = apply_anime_skin_outline(rgb_full, a1, outline_w=skin_outline_w, top_edges_only=symmetrize_shoulders)

    premul = np.zeros_like(arr, dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(rgb_full * a1[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(a1 * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _unpack_inp(path: Path) -> tuple[bytes, list[tuple[int, bytes]], bytes]:
    raw = path.read_bytes()
    assert raw[:8] == b"TRNSRTS\x00", "Invalid INP header"
    json_len = int.from_bytes(raw[8:12], "big")
    header_and_json = raw[: 12 + json_len]
    pos = 12 + json_len
    assert raw[pos : pos + 8] == b"TEX_SECT", "Missing TEX_SECT"
    pos += 8
    tex_count = int.from_bytes(raw[pos : pos + 4], "big")
    pos += 4
    textures: list[tuple[int, bytes]] = []
    for _ in range(tex_count):
        tlen = int.from_bytes(raw[pos : pos + 4], "big")
        pos += 4
        ttype = raw[pos]
        pos += 1
        textures.append((ttype, raw[pos : pos + tlen]))
        pos += tlen
    trailing = raw[pos:]
    return header_and_json, textures, trailing


def _pack_inp(header_and_json: bytes, textures: list[tuple[int, bytes]], trailing: bytes) -> bytes:
    parts = [header_and_json, b"TEX_SECT", struct.pack(">I", len(textures))]
    for ttype, tdata in textures:
        parts.append(struct.pack(">I", len(tdata)))
        parts.append(bytes([ttype]))
        parts.append(tdata)
    parts.append(trailing)
    return b"".join(parts)


def _upscale_premul_png(
    png_bytes: bytes,
    target_scale: int = 4,
    smooth_px: float = 2.6,
    corner_deg: float = 112.0,
    skin_outline_w: float = 0.0,
    skip_medial_shoulder: str = "",
    top_edges_only: bool = False,
) -> bytes:
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = im.size
    m = max(w, h)
    if m >= 1024:
        return png_bytes
    scale = target_scale if m <= 560 else (2 if m <= 800 else 1)
    if scale <= 1:
        return png_bytes

    arr = np.array(im, dtype=np.float32)
    alpha = arr[..., 3] / 255.0

    # Halo-free un-premultiply + edge color bleed into transparent gutter
    straight = np.zeros((h, w, 3), dtype=np.float32)
    mask = alpha > 0.02
    if np.any(mask):
        straight[mask] = np.clip(arr[mask, :3] / np.maximum(alpha[mask, None], 0.02), 0.0, 255.0)
        valid = (alpha > 0.25).astype(np.float32)
        w_b = cv2.GaussianBlur(valid, (0, 0), 2.8)
        bled = np.stack([
            cv2.GaussianBlur(straight[..., c] * valid, (0, 0), 2.8) / np.maximum(1e-4, w_b)
            for c in range(3)
        ], axis=-1)
        straight = np.where((alpha > 0.45)[..., None], straight, np.where(w_b[..., None] > 0.01, bled, 0.0))

    # Crisp progressive Lanczos4 upscale (ZERO bilateral smear!)
    nw, nh = w * scale, h * scale
    up_rgb = cv2.resize(straight, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    up_a = np.clip(cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)

    # Multi-scale anime ink-line restoration (keeps flat skin silky smooth with ZERO dotted seam ringing!)
    blur_fine = cv2.GaussianBlur(up_rgb, (0, 0), 1.15)
    blur_med = cv2.GaussianBlur(up_rgb, (0, 0), 2.4)
    lum = 0.299 * up_rgb[..., 0] + 0.587 * up_rgb[..., 1] + 0.114 * up_rgb[..., 2]
    edge_mag = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3) ** 2
    edge_w = np.clip(np.sqrt(edge_mag) / 56.0, 0.04, 1.0)[..., None]

    sharp = np.clip(
        up_rgb + (up_rgb - blur_fine) * (0.55 * edge_w) + (up_rgb - blur_med) * (0.18 * edge_w),
        0.0, 255.0,
    )

    # Preserve original smooth anti-aliased Lanczos4 alpha (NEVER chop chin/body ink lines or paint fake outlines!)
    up_a = np.clip(up_a, 0.0, 1.0)

    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(sharp * up_a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(up_a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0

    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _reconstruct_iris_hd(png_bytes: bytes, is_left: bool) -> bytes:
    """Reconstruct Hikari's 28x28 iris texture at true razor-sharp 256x256 anime quality,
    preserving her exact silhouette, burgundy rim, plum-to-rose-to-pearl gradient,
    signature ruby-cored oval pupil, radial anime iris fibers, and lid shadow!"""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)
    alpha_src = arr[..., 3] / 255.0

    # Smooth 8x SDF contour of original iris alpha so it aligns 1:1 with the existing mesh UVs!
    S = 256
    nw, nh = S, int(round(S * h / max(1, w)))
    a_up = cv2.resize(alpha_src, (nw, nh), interpolation=cv2.INTER_CUBIC)
    a_blur = cv2.GaussianBlur(a_up, (0, 0), 3.0)
    mask = np.clip((a_blur - 0.44) / 0.14, 0.0, 1.0)

    # Compute exact centroid & radii of the original iris silhouette
    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    w_sum = float(mask.sum()) + 1e-6
    cx = float((xx * mask).sum() / w_sum)
    cy = float((yy * mask).sum() / w_sum)
    rx = max(20.0, float(np.sqrt(((xx - cx) ** 2 * mask).sum() / w_sum) * 2.02))
    ry = max(20.0, float(np.sqrt(((yy - cy) ** 2 * mask).sum() / w_sum) * 2.02))

    # Normalized iris coordinates [-1..1]
    u = (xx - cx) / rx
    v = (yy - cy) / ry
    r = np.hypot(u, v)
    ang = np.arctan2(v, u)

    # 1) Base vertical anime jewel gradient (exact Hikari plum -> ruby -> hot rose -> luminous pearl)
    t_v = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
    col = np.zeros((nh, nw, 3), dtype=np.float32)
    c_top = np.array([76.0, 26.0, 52.0])       # deep wine plum
    c_mid = np.array([196.0, 34.0, 88.0])      # vivid ruby-magenta
    c_low = np.array([255.0, 96.0, 148.0])     # luminous hot rose
    c_gem = np.array([255.0, 224.0, 238.0])    # pearl-pink bottom crescent

    s1 = np.clip(t_v / 0.52, 0.0, 1.0)[..., None]
    s2 = np.clip((t_v - 0.46) / 0.36, 0.0, 1.0)[..., None]
    col = c_top * (1.0 - s1) + c_mid * s1
    col = col * (1.0 - s2) + c_low * s2

    # 2) Delicate high-frequency radial anime iris fibers (crystal-clear detail!)
    fibers = (
        0.55 * np.sin(ang * 28.0)
        + 0.30 * np.sin(ang * 54.0 + 1.3)
        + 0.15 * np.cos(ang * 86.0 - 0.7)
    )
    ring_zone = np.clip(1.0 - np.abs(r - 0.56) / 0.34, 0.0, 1.0)
    col = np.clip(col + (fibers * ring_zone * 16.0)[..., None], 0.0, 255.0)

    # 3) Bottom glowing pearl crescent jewel
    d_gem = np.hypot(u / 0.58, (v - 0.64) / 0.32)
    gem_w = np.clip(1.0 - d_gem, 0.0, 1.0) ** 1.45
    col = col * (1.0 - gem_w[..., None] * 0.92) + c_gem * (gem_w[..., None] * 0.92)

    # 4) Soft magenta inner-top plum highlight arc (just like original tex[23]/tex[24])
    d_hi = np.hypot((u + (0.06 if not is_left else -0.06)) / 0.68, (v + 0.52) / 0.28)
    hi_w = np.clip(1.0 - d_hi, 0.0, 1.0) ** 1.8 * 0.38
    col = col * (1.0 - hi_w[..., None]) + np.array([154.0, 62.0, 114.0]) * hi_w[..., None]

    # 5) Signature Hikari hollow-ring anime oval pupil (deep espresso border + ruby center!)
    d_pup = np.hypot(u / 0.24, (v - 0.03) / 0.34)
    pup_outer = 1.0 - np.clip((d_pup - 0.92) / 0.14, 0.0, 1.0)
    pup_inner = 1.0 - np.clip((d_pup - 0.54) / 0.16, 0.0, 1.0)
    c_pup_ring = np.array([42.0, 11.0, 24.0])
    c_pup_core = np.array([148.0, 24.0, 60.0])
    col = col * (1.0 - pup_outer[..., None]) + c_pup_ring * pup_outer[..., None]
    col = col * (1.0 - pup_inner[..., None] * 0.88) + c_pup_core * (pup_inner[..., None] * 0.88)

    # 6) Crisp dark espresso-burgundy outer iris rim (gives instant pop!)
    rim_w = np.clip((r - 0.78) / 0.16, 0.0, 1.0)
    c_rim = np.array([44.0, 16.0, 28.0])
    col = col * (1.0 - rim_w[..., None] * 0.88) + c_rim * (rim_w[..., None] * 0.88)

    # 7) Upper eyelid cast shadow across top 26% of iris
    lid_sh = 1.0 - np.clip((v + 0.54) / 0.30, 0.0, 1.0)
    col *= (1.0 - 0.42 * lid_sh[..., None])

    # 8) Tiny crisp secondary sparkle glints on the iris sides (like the original rim dots)
    for gx, gy, gr in [(-0.74, 0.28, 0.095), (0.76, 0.34, 0.075)]:
        gd = np.hypot((u - gx) / gr, (v - gy) / gr)
        gw = 1.0 - np.clip((gd - 0.55) / 0.45, 0.0, 1.0)
        col = col * (1.0 - gw[..., None] * 0.92) + np.array([255.0, 244.0, 250.0]) * (gw[..., None] * 0.92)

    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(col * mask[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _reconstruct_catchlight_hd(png_bytes: bytes) -> bytes:
    """Turn blurry 19x15 eye catchlight blobs into crisp vector-sharp glossy anime highlights!"""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)
    alpha_src = arr[..., 3] / 255.0
    scale = 8
    nw, nh = w * scale, h * scale
    a_up = cv2.resize(alpha_src, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # Crisp circular threshold removes pixel blockiness while preserving exact highlight positions!
    a_blur = cv2.GaussianBlur(a_up, (0, 0), 2.8)
    mask = np.clip((a_blur - 0.36) / 0.18, 0.0, 1.0)
    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    # Pure luminous white-pink glossy specular catchlight
    premul[..., 0] = np.clip(np.rint(255.0 * mask), 0, 255).astype(np.uint8)
    premul[..., 1] = np.clip(np.rint(250.0 * mask), 0, 255).astype(np.uint8)
    premul[..., 2] = np.clip(np.rint(254.0 * mask), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(mask * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _upscale_eyelash_or_sclera_hd(png_bytes: bytes, is_sclera: bool = False) -> bytes:
    """8x SDF-smoothed vector-crisp upscale for tiny 40x30 sclera & 48x24 eyelash textures."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)
    alpha = arr[..., 3] / 255.0
    straight = np.zeros((h, w, 3), dtype=np.float32)
    mask = alpha > 0.02
    if np.any(mask):
        straight[mask] = np.clip(arr[mask, :3] / np.maximum(alpha[mask, None], 0.02), 0.0, 255.0)
        valid = (alpha > 0.20).astype(np.float32)
        wb = cv2.GaussianBlur(valid, (0, 0), 2.5)
        bled = np.stack([cv2.GaussianBlur(straight[..., c] * valid, (0, 0), 2.5) / np.maximum(1e-4, wb) for c in range(3)], axis=-1)
        straight = np.where(mask[..., None], straight, np.where(wb[..., None] > 0.01, bled, 0.0))

    scale = 8
    nw, nh = w * scale, h * scale
    up_rgb = cv2.resize(straight, (nw, nh), interpolation=cv2.INTER_CUBIC)
    up_a = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # Potrace-style vector contour smoothing kills ALL 48px wobbles on eyelashes, scleras & brows!
    up_a = vector_smooth_alpha(up_a, smooth_px=2.2 if is_sclera else 1.75, corner_deg=118.0, ss=4)
    if not is_sclera:
        # Deepen and crisp dark mahogany eyelashes
        blur = cv2.GaussianBlur(up_rgb, (0, 0), 1.4)
        up_rgb = np.clip(up_rgb + (up_rgb - blur) * 0.75, 0.0, 255.0)
    else:
        # Smooth crisp studio eye-white shading
        up_rgb = cv2.GaussianBlur(up_rgb, (0, 0), 2.4)

    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(up_rgb * up_a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(up_a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def ensure_hd_model(src_inp: Path, dst_inp: Path, scale: int = 4, force: bool = False) -> Path:
    """Create 4x anime-upscaled crisp HD INP + 256x256 vector-crisp eye reconstruction."""
    if not force and dst_inp.exists() and dst_inp.stat().st_mtime >= src_inp.stat().st_mtime:
        return dst_inp
    dst_inp.parent.mkdir(parents=True, exist_ok=True)
    header_json, textures, trailing = _unpack_inp(src_inp)
    up_textures = []
    for idx, (ttype, tdata) in enumerate(textures):
        if ttype != 0:
            up_textures.append((ttype, tdata))
        elif idx == 23:
            up_textures.append((ttype, _reconstruct_iris_hd(tdata, is_left=False)))
        elif idx == 24:
            up_textures.append((ttype, _reconstruct_iris_hd(tdata, is_left=True)))
        elif idx in (25, 26):
            up_textures.append((ttype, _reconstruct_catchlight_hd(tdata)))
        elif idx in (14, 15):
            up_textures.append((ttype, _upscale_eyelash_or_sclera_hd(tdata, is_sclera=True)))
        elif idx in (16, 17, 19, 21, 27, 29, 31, 33):
            up_textures.append((ttype, _upscale_eyelash_or_sclera_hd(tdata, is_sclera=False)))
        else:
            up_textures.append((ttype, _upscale_premul_png(tdata, target_scale=scale)))
    dst_inp.write_bytes(_pack_inp(header_json, up_textures, trailing))
    return dst_inp
