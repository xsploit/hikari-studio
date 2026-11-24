"""Modular VRM Asset Extractor & Live Character Customizer for Hikari Studio.
- Extracts real high-res EyeIris + EyeHighlight (and other face/hair textures) from any .vrm (GLB 2.0) file.
- Maps the VRM irises + chromatic highlights symmetrically onto Hikari's Inochi2D iris UVs (zero lazy eye!).
- Applies real-time HSV color/brightness/highlight customizations to Eyes, Hair, and Outfit
  and rebuilds Hikari-v28-Custom.inp in <0.6s for hot-reloading into hikari-live-engine.exe!
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.texture_upscaler import (
    _pack_inp,
    _unpack_inp,
    _upscale_eyelash_or_sclera_hd,
    _upscale_premul_png,
    smooth_premul_silhouette_png,
)

from core.wardrobe import SHOE_TEXTURES, TOP_TEXTURE, Wardrobe

DEFAULT_VRM_PATH = Path(r"D:\Waifu Studio\assets\characters\models\a4f45d37bdbb-hikari-refined.vrm")
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_FRONT_HAIR_HD = ASSETS_DIR / "front_hair_lighter_bangs.png"
DEFAULT_BACK_HAIR_HD = ASSETS_DIR / "back_hair_matching.png"
HD_PARTS_DIR = ASSETS_DIR / "hd_parts"
HD_FACE = HD_PARTS_DIR / "face_hd.png"
HD_NECK = HD_PARTS_DIR / "neck_hd.png"
HD_TOPWEAR_REFINED = HD_PARTS_DIR / "topwear_refined_hd.png"
HD_ARM_RIGHT = HD_PARTS_DIR / "arm_right_hd.png"
HD_BOTTOMWEAR = HD_PARTS_DIR / "bottomwear_hd.png"
HD_LEG_RIGHT = HD_PARTS_DIR / "leg_right_hd.png"
HD_WAVE_DIR = HD_PARTS_DIR / "wave"
COLLAR_PREVIEW_INP = ASSETS_DIR / "collar_preview" / "Hikari-v28-CollarPreview.inp"

# Exact verified texture ID groups in Hikari-v28.inp:
TEX_IRIS_R = 23
TEX_IRIS_L = 24
TEX_HI_R = 25
TEX_HI_L = 26
TEX_SCLERA = (14, 15)
TEX_LASH_BROW = (16, 17, 19, 21, 27, 29, 31, 33)
# ALL hair layers: BackHair+Twintails (8), FrontHair+Ahoge (35), FrontHair See-Through (36), Eyebrows (31,32,33,34)
TEX_HAIR = (8, 35, 36, 31, 32, 33, 34)
# Outfit layers (skin-safe masked so chest/belly/leg skin stays untouched!): Bottomwear (4), Topwear (5), Footwear (6,7)
TEX_OUTFIT = (4, 5, 6, 7)

DEFAULT_CUSTOMIZER_CFG: dict[str, Any] = {
    "eyeStyle": "vrm_refined",   # vrm_refined | original_solid_pupil | original_hd | vrm_ruby | vrm_aqua | vrm_amber | vrm_emerald | custom
    "eyeHue": 0.0,               # -180 .. +180 deg
    "eyeSat": 1.05,              # 0.2 .. 1.8
    "eyeVal": 1.08,              # 0.6 .. 1.5
    "eyeHighlight": 1.00,        # 0.0 .. 1.5
    "eyeScale": 1.04,            # 0.80 .. 1.25
    "eyeOffsetX": 0.0,           # -0.15 .. +0.15 (symmetrical convergence)
    "eyeOffsetY": 0.0,           # -0.15 .. +0.15
    "hairColor": "#ff6a9e",      # Direct Hair Color Picker!
    "hairColorStrength": 0.0,    # 0.0 = original Sakura Pink, >0 = direct picked hair color
    "hairHue": 0.0,              # -180 .. +180 deg
    "hairSat": 1.00,             # 0.0 .. 1.6
    "hairVal": 1.00,             # 0.6 .. 1.35
    "bangsMode": "solid",        # solid (full natural outline, zero fade/chop) | seethrough (50% fade over eye) | custom
    "customFrontHairPath": "",
    "outfitStyle": "original",
    "shoeStyle": "original",
    "outfitTint": "#d6284c",     # Recolors dark fabric cleanly while protecting peach skin!
    "outfitTintStrength": 0.0,   # 0.0 .. 1.0 (auto-sets to 0.78 when picking a color in UI)
    "outfitVal": 1.00,           # 0.6 .. 1.8
    "customEyePath": "",
}

EYE_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "vrm_refined":          {"label": "VRM Galaxy Violet (Refined)",      "eyeHue": 0.0,    "eyeSat": 1.05, "eyeVal": 1.08, "eyeHighlight": 1.0},
    "original_solid_pupil": {"label": "Original 2D · Solid Black Pupils", "eyeHue": 0.0,    "eyeSat": 1.05, "eyeVal": 1.02, "eyeHighlight": 1.0},
    "original_hd":          {"label": "Original 2D Art (Upscaled HD)",    "eyeHue": 0.0,    "eyeSat": 1.00, "eyeVal": 1.00, "eyeHighlight": 0.9},
    "vrm_ruby":             {"label": "VRM Ruby Rose",                    "eyeHue": 28.0,   "eyeSat": 1.08, "eyeVal": 1.12, "eyeHighlight": 1.0},
    "vrm_aqua":             {"label": "VRM Cyber Aqua",                   "eyeHue": -128.0, "eyeSat": 1.10, "eyeVal": 1.15, "eyeHighlight": 1.05},
    "vrm_amber":            {"label": "VRM Sunset Gold",                  "eyeHue": 82.0,   "eyeSat": 1.12, "eyeVal": 1.18, "eyeHighlight": 1.0},
    "vrm_emerald":          {"label": "VRM Emerald Mint",                 "eyeHue": -168.0, "eyeSat": 1.05, "eyeVal": 1.12, "eyeHighlight": 1.0},
}


def extract_vrm_eye_assets(vrm_path: Path, out_dir: Path) -> dict[str, Path]:
    """Extract EyeIris and EyeHighlight PNGs from any VRM (GLB 2.0) file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = vrm_path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError("Not a valid GLB/VRM binary file.")
    jlen, _ = struct.unpack_from("<II", data, 12)
    gltf = json.loads(data[20 : 20 + jlen].decode("utf-8"))
    bin_off = 20 + jlen
    blen, _ = struct.unpack_from("<II", data, bin_off)
    bin_data = data[bin_off + 8 : bin_off + 8 + blen]

    texs = gltf.get("textures", [])
    imgs = gltf.get("images", [])
    bvs = gltf.get("bufferViews", [])

    def load_img_by_idx(img_idx: int) -> Image.Image:
        bv = bvs[imgs[img_idx]["bufferView"]]
        off = bv.get("byteOffset", 0)
        return Image.open(io.BytesIO(bin_data[off : off + bv["byteLength"]])).convert("RGBA")

    iris_im = None
    hi_im = None
    for mat in gltf.get("materials", []):
        mname = str(mat.get("name") or "").lower()
        bct = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture", {}).get("index")
        if bct is None:
            continue
        img_idx = texs[bct]["source"]
        if "eyeiris" in mname or ("iris" in mname and iris_im is None):
            iris_im = load_img_by_idx(img_idx)
        elif "eyehighlight" in mname or ("highlight" in mname and "eye" in mname and hi_im is None):
            hi_im = load_img_by_idx(img_idx)

    if iris_im is None:
        raise ValueError("Could not locate EyeIris material in this VRM file.")

    saved: dict[str, Path] = {}
    # Split 1024x512 side-by-side VRM iris texture into clean tight-cropped Left & Right irises
    iw, ih = iris_im.size
    if iw >= ih * 1.6:
        half_r = iris_im.crop((0, 0, iw // 2, ih))
        half_l = iris_im.crop((iw // 2, 0, iw, ih))
    else:
        half_r = iris_im
        half_l = iris_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def crop_pair(iris_half: Image.Image, hi_half: Image.Image | None, tag: str) -> None:
        arr = np.array(iris_half, dtype=np.uint8)
        ys, xs = np.where(arr[..., 3] > 12)
        if len(xs) == 0:
            raise ValueError("Empty VRM iris alpha.")
        # Square centered crop around iris bounding box so pupil is dead-center!
        cx = 0.5 * (xs.min() + xs.max())
        cy = 0.5 * (ys.min() + ys.max())
        rad = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.52
        x0, y0 = int(round(cx - rad)), int(round(cy - rad))
        x1, y1 = int(round(cx + rad)), int(round(cy + rad))
        pad_box = (max(0, x0), max(0, y0), min(iris_half.width, x1), min(iris_half.height, y1))
        ic = iris_half.crop(pad_box).resize((256, 256), Image.Resampling.LANCZOS)
        p_i = out_dir / f"vrm_iris_{tag}.png"
        ic.save(p_i)
        saved[f"iris_{tag}"] = p_i
        if hi_half is not None:
            hc = hi_half.crop(pad_box).resize((256, 256), Image.Resampling.LANCZOS)
            p_h = out_dir / f"vrm_hi_{tag}.png"
            hc.save(p_h)
            saved[f"hi_{tag}"] = p_h

    hi_r = hi_im.crop((0, 0, hi_im.width // 2, hi_im.height)) if (hi_im and hi_im.width >= hi_im.height * 1.6) else hi_im
    hi_l = hi_im.crop((hi_im.width // 2, 0, hi_im.width, hi_im.height)) if (hi_im and hi_im.width >= hi_im.height * 1.6) else (hi_im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if hi_im else None)

    crop_pair(half_r, hi_r, "r")
    crop_pair(half_l, hi_l, "l")
    return saved


def _shift_hsv_rgb(rgb: np.ndarray, hue_deg: float, sat_mul: float, val_mul: float) -> np.ndarray:
    if abs(hue_deg) < 0.5 and abs(sat_mul - 1.0) < 0.01 and abs(val_mul - 1.0) < 0.01:
        return rgb
    u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_deg * 0.5) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_mul, 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_mul, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def _recolor_hair_direct_png(png_bytes: bytes, target_hex: str, strength: float, sat_mul: float, val_mul: float) -> bytes:
    """Recolor anime hair to ANY picked hex color while preserving glossy highlights, shading & dark hair ties!"""
    hx = str(target_hex or "#ff6a9e").strip().lstrip("#")
    if len(hx) != 6:
        return png_bytes
    tr, tg, tb = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    tgt_hsv = cv2.cvtColor(np.array([[[tr, tg, tb]]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    tgt_h, tgt_s, tgt_v = float(tgt_hsv[0]), float(tgt_hsv[1]) / 255.0, float(tgt_hsv[2]) / 255.0

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a = arr[..., 3] / 255.0
    mask = a > 0.01
    rgb = np.zeros_like(arr[..., :3])
    if np.any(mask):
        rgb[mask] = np.clip(arr[mask, :3] / np.maximum(a[mask, None], 0.01), 0.0, 255.0)
        u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Preserve glossy white speculars (low-sat bright pixels) & dark navy hair ties cleanly
        orig_s = hsv[..., 1] / 255.0
        orig_v = hsv[..., 2] / 255.0
        # Target hue replaces pink hair hue, scaled smoothly by target saturation & value
        new_hsv = hsv.copy()
        new_hsv[..., 0] = tgt_h
        # Keep highlights bright white-tinted and shadows rich
        new_hsv[..., 1] = np.clip(orig_s * (0.15 + 1.15 * tgt_s) * sat_mul * 255.0, 0.0, 255.0)
        v_gain = (0.48 + 0.65 * tgt_v) * val_mul
        new_hsv[..., 2] = np.clip(orig_v * v_gain * 255.0, 0.0, 255.0)
        recol = cv2.cvtColor(new_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        rgb = rgb * (1.0 - strength) + recol * strength

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = arr[..., 3].astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _recolor_outfit_skin_safe_png(
    png_bytes: bytes, tint_hex: str, strength: float, val_mul: float, *, preserve_print: bool = False
) -> bytes:
    """Recolor charcoal/black fabric on Topwear/Bottomwear/Footwear while keeping peach skin 100% untouched!"""
    if strength < 0.01 and abs(val_mul - 1.0) < 0.01:
        return png_bytes
    hx = str(tint_hex or "#ffffff").strip().lstrip("#")
    if len(hx) == 6:
        tr, tg, tb = int(hx[0:2], 16) / 255.0, int(hx[2:4], 16) / 255.0, int(hx[4:6], 16) / 255.0
    else:
        tr, tg, tb = 1.0, 1.0, 1.0

    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a = arr[..., 3] / 255.0
    mask = a > 0.02
    rgb = np.zeros_like(arr[..., :3])
    if np.any(mask):
        rgb[mask] = np.clip(arr[mask, :3] / np.maximum(a[mask, None], 0.02), 0.0, 255.0)
        # Detect warm peach skin pixels (high R, warm R > G > B, bright luma) and PROTECT them!
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        is_skin = (r > 158.0) & (r > g + 10.0) & (g > b - 6.0) & (lum > 140.0)
        protected = is_skin
        if preserve_print:
            # Keep the cyan piping, light screen print, and rubber soles legible.
            accent = np.maximum(lum - 105.0, np.minimum(g, b) - r - 16.0)
            protected = protected | (accent >= 37.0)
        cloth_w = cv2.GaussianBlur((mask & ~protected).astype(np.float32), (0, 0), 1.5)
        cloth_w[protected] = 0.0
        cloth_w = cloth_w[..., None]

        # Colorize dark fabric smoothly using luminance-preserving tint
        luma_n = np.clip((lum / 255.0) * val_mul, 0.0, 1.0)[..., None]
        # Lift charcoal floor slightly when tinting brightly so colored tops actually pop!
        lifted = np.clip(luma_n * (1.0 + strength * 0.85) + strength * 0.08, 0.0, 1.0)
        tinted = lifted * np.array([tr, tg, tb], dtype=np.float32)[None, None, :] * 255.0
        scaled_orig = np.clip(rgb * val_mul, 0.0, 255.0)
        recolored = scaled_orig * (1.0 - strength) + tinted * strength
        rgb = np.clip(rgb * (1.0 - cloth_w) + recolored * cloth_w, 0.0, 255.0)

    out = arr.astype(np.uint8)
    out[mask, :3] = np.rint(rgb[mask] * a[mask, None]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _tint_premul_png(png_bytes: bytes, hue_deg: float, sat_mul: float, val_mul: float) -> bytes:
    if abs(hue_deg) < 0.5 and abs(sat_mul - 1.0) < 0.01 and abs(val_mul - 1.0) < 0.01:
        return png_bytes
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a = arr[..., 3] / 255.0
    mask = a > 0.01
    rgb = np.zeros_like(arr[..., :3])
    if np.any(mask):
        rgb[mask] = np.clip(arr[mask, :3] / np.maximum(a[mask, None], 0.01), 0.0, 255.0)
        rgb = _shift_hsv_rgb(rgb, hue_deg, sat_mul, val_mul)
    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = arr[..., 3].astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _build_vrm_mapped_iris(
    orig_iris_png: bytes,
    vrm_iris_path: Path,
    vrm_hi_path: Path | None,
    is_left: bool,
    cfg: dict[str, Any],
) -> bytes:
    """Map the real high-res VRM iris + chromatic highlights symmetrically into Hikari's exact UV box.
    Uses the EXACT geometric center of the original iris alpha so BOTH eyes converge cleanly (zero lazy eye!)."""
    orig = Image.open(io.BytesIO(orig_iris_png)).convert("RGBA")
    ow, oh = orig.size
    orig_a = np.array(orig, dtype=np.float32)[..., 3] / 255.0

    # Target crisp 256px canvas preserving original UV aspect ratio
    S = 256
    nw, nh = S, int(round(S * oh / max(1, ow)))
    a_up = cv2.resize(orig_a, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # Find exact center & size of original iris silhouette in UV space
    ys, xs = np.where(a_up > 0.35)
    cx = 0.5 * (xs.min() + xs.max()) if len(xs) else nw * 0.5
    cy = 0.5 * (ys.min() + ys.max()) if len(ys) else nh * 0.5
    # Slight symmetrical inward offset if user tweaks eyeOffsetX
    sgn = -1.0 if is_left else 1.0
    cx += float(cfg.get("eyeOffsetX", 0.0)) * nw * sgn
    cy += float(cfg.get("eyeOffsetY", 0.0)) * nh
    # Keep iris slightly taller anime oval (matches original 26x28 proportions cleanly!)
    sc = float(cfg.get("eyeScale", 1.04))
    w_box = max(20.0, (xs.max() - xs.min()) * sc) if len(xs) else nw * 0.9
    h_box = max(20.0, (ys.max() - ys.min()) * sc) if len(ys) else nh * 0.9

    # Load VRM iris & apply live HSV customization!
    v_im = Image.open(vrm_iris_path).convert("RGBA")
    v_arr = np.array(v_im, dtype=np.float32)
    v_rgb = _shift_hsv_rgb(
        v_arr[..., :3],
        float(cfg.get("eyeHue", 0.0)),
        float(cfg.get("eyeSat", 1.05)),
        float(cfg.get("eyeVal", 1.08)),
    )
    v_a = v_arr[..., 3:4] / 255.0

    # Composite the real VRM chromatic glossy highlights right on top!
    hi_str = float(cfg.get("eyeHighlight", 1.0))
    if vrm_hi_path and vrm_hi_path.exists() and hi_str > 0.01:
        h_im = Image.open(vrm_hi_path).convert("RGBA").resize(v_im.size, Image.Resampling.LANCZOS)
        h_arr = np.array(h_im, dtype=np.float32)
        ha = np.clip((h_arr[..., 3:4] / 255.0) * hi_str, 0.0, 1.0)
        # Screen/alpha blend highlights crisply over the iris
        v_rgb = np.clip(v_rgb * (1.0 - ha * 0.85) + h_arr[..., :3] * ha * 1.15, 0.0, 255.0)

    # Subtle top eyelid shadow so the VRM iris sits naturally under Hikari's 2D eyelashes
    yy_v = np.linspace(0.0, 1.0, v_im.height, dtype=np.float32)[:, None, None]
    lid = np.clip((0.26 - yy_v) / 0.26, 0.0, 1.0)
    v_rgb *= (1.0 - 0.32 * lid)

    # Crisp subtle unsharp pass on the 256px VRM iris
    blur = cv2.GaussianBlur(v_rgb, (0, 0), 1.0)
    v_rgb = np.clip(v_rgb + (v_rgb - blur) * 0.55, 0.0, 255.0)

    # Place centered into the exact UV canvas
    tw, th = max(8, int(round(w_box))), max(8, int(round(h_box)))
    r_rgb = cv2.resize(v_rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    r_a = np.clip(cv2.resize(v_a[..., 0], (tw, th), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)

    canvas_rgb = np.zeros((nh, nw, 3), dtype=np.float32)
    canvas_a = np.zeros((nh, nw), dtype=np.float32)
    x0 = int(round(cx - tw * 0.5))
    y0 = int(round(cy - th * 0.5))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(tw - sx0, nw - dx0)
    ch = min(th - sy0, nh - dy0)
    if cw > 0 and ch > 0:
        canvas_rgb[dy0 : dy0 + ch, dx0 : dx0 + cw] = r_rgb[sy0 : sy0 + ch, sx0 : sx0 + cw]
        canvas_a[dy0 : dy0 + ch, dx0 : dx0 + cw] = r_a[sy0 : sy0 + ch, sx0 : sx0 + cw]

    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(canvas_rgb * canvas_a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(canvas_a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _build_original_solid_pupil_iris(orig_iris_png: bytes, is_left: bool, cfg: dict[str, Any]) -> bytes:
    """Clean 8x upscaled original 2D ruby-pink iris with a completely filled-in round solid black pupil + crisp catchlight."""
    im = Image.open(io.BytesIO(orig_iris_png)).convert("RGBA")
    w, h = im.size
    arr = np.array(im, dtype=np.float32)
    alpha = arr[..., 3] / 255.0
    straight = np.zeros((h, w, 3), dtype=np.float32)
    m = alpha > 0.02
    if np.any(m):
        straight[m] = np.clip(arr[m, :3] / np.maximum(alpha[m, None], 0.02), 0.0, 255.0)

    S = 256
    nw, nh = S, int(round(S * h / max(1, w)))
    up_rgb = cv2.resize(straight, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    up_a = np.clip((cv2.GaussianBlur(cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_CUBIC), (0, 0), 2.4) - 0.44) / 0.14, 0.0, 1.0)
    up_rgb = _shift_hsv_rgb(up_rgb, float(cfg.get("eyeHue", 0.0)), float(cfg.get("eyeSat", 1.05)), float(cfg.get("eyeVal", 1.02)))

    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    ys, xs = np.where(up_a > 0.35)
    cx = 0.5 * (xs.min() + xs.max()) if len(xs) else nw * 0.5
    cy = 0.5 * (ys.min() + ys.max()) if len(ys) else nh * 0.5
    sgn = -1.0 if is_left else 1.0
    cx += float(cfg.get("eyeOffsetX", 0.0)) * nw * sgn
    cy += float(cfg.get("eyeOffsetY", 0.0)) * nh
    rx = max(20.0, (xs.max() - xs.min()) * 0.5) if len(xs) else nw * 0.45
    ry = max(20.0, (ys.max() - ys.min()) * 0.5) if len(ys) else nh * 0.45

    # Completely filled-in round glossy jet-black anime pupil (replaces hollow oval ring cleanly!)
    d_pup = np.hypot((xx - cx) / (rx * 0.34), (yy - cy) / (ry * 0.36))
    pup_mask = 1.0 - np.clip((d_pup - 0.86) / 0.18, 0.0, 1.0)
    up_rgb = up_rgb * (1.0 - pup_mask[..., None]) + np.array([14.0, 6.0, 14.0]) * pup_mask[..., None]

    # Crisp anime specular catchlight top-left of pupil
    hi_str = float(cfg.get("eyeHighlight", 1.0))
    if hi_str > 0.01:
        for hx, hy, hr in [(-0.34, -0.32, 0.17), (0.34, 0.30, 0.08)]:
            d_h = np.hypot((xx - (cx + hx * rx)) / (rx * hr), (yy - (cy + hy * ry)) / (ry * hr))
            hw = np.clip((1.0 - np.clip((d_h - 0.65) / 0.35, 0.0, 1.0)) * hi_str, 0.0, 1.0)
            up_rgb = up_rgb * (1.0 - hw[..., None] * 0.94) + np.array([255.0, 250.0, 254.0]) * (hw[..., None] * 0.94)

    blur = cv2.GaussianBlur(up_rgb, (0, 0), 1.0)
    up_rgb = np.clip(up_rgb + (up_rgb - blur) * 0.55, 0.0, 255.0)

    premul = np.zeros((nh, nw, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(up_rgb * up_a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(up_a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _transparent_png(size: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _mirror_premul_png_h(png_bytes: bytes, target_size: tuple[int, int] | None = None) -> bytes:
    """Horizontally mirror a premultiplied RGBA PNG cleanly (un-premult -> flip/resize -> re-premult)
    so Left Eye (L) matches Right Eye (R) with 100% mirror symmetry and zero edge halos."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a = arr[..., 3] / 255.0
    rgb = np.zeros_like(arr[..., :3])
    m = a > 0.01
    if np.any(m):
        rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.01), 0.0, 255.0)
    rgb = rgb[:, ::-1, :]
    a = a[:, ::-1]
    if target_size is not None and (rgb.shape[1], rgb.shape[0]) != target_size:
        tw, th = target_size
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LANCZOS4)
        a = np.clip(cv2.resize(a, (tw, th), interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)
    premul = np.zeros((a.shape[0], a.shape[1], 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _pad_mesh_node_to_alpha(node: dict[str, Any], a_arr: np.ndarray, min_steps: int = 2, max_steps: int = 28) -> None:
    """Expand ONLY outer boundary vertices & UVs of an Inochi2D mesh by the exact rest-pose affine Jacobian
    so zero pixels of the texture ever get hard-clipped by tight polygon boundaries.
    Preserves vertex count and every deform/physics binding 100% intact with zero texture shift or folding."""
    m = node.get("mesh")
    if not m:
        return
    verts = np.array(m["verts"], dtype=np.float64).reshape(-1, 2)
    uvs = np.array(m["uvs"], dtype=np.float64).reshape(-1, 2)
    idx = np.array(m["indices"], dtype=np.int32).reshape(-1, 3)

    sx = np.polyfit(uvs[:, 0], verts[:, 0], 1)
    sy = np.polyfit(uvs[:, 1], verts[:, 1], 1)

    def _areas(u_cur: np.ndarray) -> np.ndarray:
        a, b, c = u_cur[idx[:, 0]], u_cur[idx[:, 1]], u_cur[idx[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    areas0 = _areas(uvs)
    sgn = 1.0 if float(np.median(areas0)) >= 0.0 else -1.0

    edge_tris: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for ti, tri in enumerate(idx):
        for k in range(3):
            u, v = int(tri[k]), int(tri[(k + 1) % 3])
            edge_tris.setdefault((min(u, v), max(u, v)), []).append((ti, u, v))

    b_normals: dict[int, np.ndarray] = {}
    for lst in edge_tris.values():
        if len(lst) == 1:
            ti, u, v = lst[0]
            w_idx = [int(x) for x in idx[ti] if int(x) != u and int(x) != v][0]
            e = uvs[v] - uvs[u]
            nrm = np.array([e[1], -e[0]], dtype=np.float64)
            if float(np.dot(nrm, 0.5 * (uvs[u] + uvs[v]) - uvs[w_idx])) < 0.0:
                nrm = -nrm
            ln = float(np.linalg.norm(nrm))
            if ln > 1e-12:
                nrm /= ln
                for vi in (u, v):
                    b_normals[vi] = b_normals.get(vi, np.zeros(2, dtype=np.float64)) + nrm

    inc: dict[int, list[int]] = {}
    for ti, tri in enumerate(idx):
        for vi in tri:
            inc.setdefault(int(vi), []).append(ti)

    h, w = a_arr.shape
    uvs_new = uvs.copy()

    for step in range(max_steps):
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for tri in idx:
            draw.polygon([(float(uvs_new[i, 0] * w), float(uvs_new[i, 1] * h)) for i in tri], fill=255)
        cl = a_arr & (np.array(mask) < 128)
        if not cl.any() and step >= min_steps:
            break
        cy, cx = np.where(cl)
        c_uv = np.stack([cx / float(w), cy / float(h)], axis=1) if len(cx) else np.zeros((0, 2))
        for vi, nsum in b_normals.items():
            ln = float(np.linalg.norm(nsum))
            if ln < 1e-9:
                continue
            nrm = nsum / ln
            if step >= min_steps:
                if len(c_uv) == 0:
                    continue
                dists = np.hypot(c_uv[:, 0] - uvs_new[vi, 0], c_uv[:, 1] - uvs_new[vi, 1])
                if float(dists.min()) > 0.08:
                    continue
                nearest = c_uv[int(np.argmin(dists))]
                to_c = nearest - uvs_new[vi]
                tln = float(np.linalg.norm(to_c))
                if tln > 1e-9:
                    to_c /= tln
                    if float(np.dot(to_c, nrm)) > 0.15:
                        nrm = 0.55 * nrm + 0.45 * to_c
                        nrm /= float(np.linalg.norm(nrm))
            cand = np.clip(uvs_new[vi] + nrm * 0.0028, 0.0, 1.0)
            old = uvs_new[vi].copy()
            uvs_new[vi] = cand
            tis = np.array(inc[vi], dtype=np.int32)
            if float((_areas(uvs_new)[tis] * sgn).min()) <= 1e-7:
                uvs_new[vi] = old

    duv = uvs_new - uvs
    verts_new = verts.copy()
    verts_new[:, 0] += duv[:, 0] * sx[0]
    verts_new[:, 1] += duv[:, 1] * sy[0]
    m["uvs"] = [round(float(x), 6) for x in uvs_new.reshape(-1)]
    m["verts"] = [round(float(x), 4) for x in verts_new.reshape(-1)]


def _pad_all_meshes(header: bytes, raw_textures: list[tuple[int, bytes]]) -> bytes:
    """Pad outer mesh boundaries on hair, arms, torso/topwear, bottomwear, legs & symmetrical eyes
    so zero anti-aliased edge pixels are ever sliced off by tight low-res polygon boundaries."""
    puppet = json.loads(header[12:])
    pending = [puppet["nodes"]]
    nodes: dict[int, dict[str, Any]] = {}
    while pending:
        node = pending.pop()
        nodes[node["uuid"]] = node
        pending.extend(node.get("children", ()))

    def _alpha_from_file(p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert("RGBA"))[..., 3] > 10

    def _alpha_from_raw(tid: int, flip_h: bool = False) -> np.ndarray:
        im = Image.open(io.BytesIO(raw_textures[tid][1])).convert("RGBA")
        a = np.array(im)[..., 3] > 10
        return a[:, ::-1] if flip_h else a

    if 1008 in nodes and DEFAULT_BACK_HAIR_HD.exists():
        _pad_mesh_node_to_alpha(nodes[1008], _alpha_from_file(DEFAULT_BACK_HAIR_HD))
    if 1035 in nodes and DEFAULT_FRONT_HAIR_HD.exists():
        _pad_mesh_node_to_alpha(nodes[1035], _alpha_from_file(DEFAULT_FRONT_HAIR_HD))
    if 1036 in nodes and DEFAULT_FRONT_HAIR_HD.exists():
        _pad_mesh_node_to_alpha(nodes[1036], _alpha_from_file(DEFAULT_FRONT_HAIR_HD))

    # Pad arms, legs, bottomwear, topwear, face & neck against HD parts (or raw fallback) so zero pixels clip!
    for uid, tid, hd_path, flip_h in (
        (1000, 0, HD_ARM_RIGHT, True),
        (1046, 46, HD_ARM_RIGHT, False),
        (1001, 1, HD_LEG_RIGHT, False),
        (1002, 2, HD_LEG_RIGHT, True),
        (1003, 3, HD_NECK, False),
        (1004, 4, HD_BOTTOMWEAR, False),
        (1005, 5, HD_TOPWEAR_REFINED, False),
        (1011, 11, HD_FACE, False),
    ):
        if uid in nodes and tid < len(raw_textures):
            if hd_path.exists():
                a = _alpha_from_file(hd_path)
                if flip_h:
                    a = a[:, ::-1]
            else:
                a = _alpha_from_raw(tid)
            _pad_mesh_node_to_alpha(nodes[uid], a, min_steps=3, max_steps=18)

    # Pad mirrored Left Eye nodes (1015 sclera, 1021 upper lash, 1029 closed eye) so mirrored R tips never clip!
    for uid, tid_r, flip_h in ((1014, 14, False), (1015, 14, True), (1019, 19, False), (1021, 19, True), (1027, 27, False), (1029, 27, True)):
        if uid in nodes and tid_r < len(raw_textures):
            _pad_mesh_node_to_alpha(nodes[uid], _alpha_from_raw(tid_r, flip_h=flip_h), min_steps=2, max_steps=14)

    # Seat ONLY the raised wave shoulder roots (50006, 50015) flush with the trapezius slope (resting arms untouched!)
    for uid, dx, dy in ((50006, 1.8, -3.8), (50015, -1.8, -3.8)):
        if uid in nodes:
            tr = nodes[uid].setdefault("transform", {}).setdefault("trans", [0.0, 0.0, 0.0])
            tr[0] = round(float(tr[0]) + dx, 4)
            tr[1] = round(float(tr[1]) + dy, 4)

    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _sweep_front_bangs(header: bytes) -> bytes:
    """Keep FrontHair mesh geometry 100% untouched (zero triangle folding or crooked hooks);
    only remove the eye DodgeMask & semi-transparent ghost duplicate so both eyes stay crisp."""
    puppet = json.loads(header[12:])
    pending = [puppet["nodes"]]
    nodes = {}
    while pending:
        node = pending.pop()
        nodes[node["uuid"]] = node
        pending.extend(node.get("children", ()))

    front_hair = nodes[1035]
    front_hair["masks"] = [
        mask for mask in front_hair["masks"]
        if not (mask["mode"] == "DodgeMask" and mask["source"] in (1014, 1015))
    ]
    nodes[1036]["enabled"] = False
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _curve_front_bangs_tex(png_bytes: bytes) -> bytes:
    """Smooth 4x pixel-space anime bang curve on ArtMeshFrontHair (tex[35]).
    Gently lifts and curves ONLY the center-right bang lock away from the viewer-right eye
    without chopping flat edges, folding mesh triangles, or touching the right temple lock."""
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr[..., 3] / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    m = a > 0.01
    rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.01), 0.0, 255.0)

    def _ss(lo: float, hi: float, v: np.ndarray) -> np.ndarray:
        t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = w / 928.0, h / 1196.0
    nx, ny = xx / sx, yy / sy

    # Softly remove ONLY the stray detached whisker tip strictly inside nx in [534..632] (never touches right temple lock!)
    stray_tail = _ss(534.0, 562.0, nx) * (1.0 - _ss(612.0, 632.0, nx)) * _ss(690.0, 722.0, ny)
    a = a * (1.0 - stray_tail)

    # Follow the empty transparent V-gap between center bangs and right temple lock -> ZERO edge pinch!
    x_gap = np.interp(ny, [420.0, 500.0, 560.0, 620.0, 680.0, 760.0], [568.0, 582.0, 610.0, 642.0, 682.0, 725.0])
    in_x = _ss(295.0, 415.0, nx) * (1.0 - _ss(x_gap - 10.0, x_gap + 14.0, nx))
    v_t = np.clip((ny - 380.0) / 360.0, 0.0, 1.0)
    v_curve = v_t * v_t * (3.0 - 2.0 * v_t)
    w_flow = in_x * v_curve

    # Gentle, organic anime sweep: 28px left & 52px up — keeps full lush pointed bangs clear of both eyes!
    dx = 28.0 * w_flow * sx
    dy = 52.0 * w_flow * sy

    rgb_out = cv2.remap(rgb, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    a_out = np.clip(cv2.remap(a, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32),
                              interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0), 0.0, 1.0)
    a_out = np.clip((a_out - 0.5) * 1.08 + 0.5, 0.0, 1.0)

    premul = np.zeros((h, w, 4), dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(rgb_out * a_out[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(a_out * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _straight_png_to_premul(img_path: Path, target_size: tuple[int, int] | None = None) -> bytes:
    """Convert any user-drawn or inpainted straight-alpha PNG into clean Inochi2D premultiplied RGBA."""
    im = Image.open(img_path).convert("RGBA")
    if target_size is not None and im.size != target_size:
        im = im.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(im, dtype=np.float32)
    rgb = arr[..., :3].copy()
    a = arr[..., 3] / 255.0
    premul = np.zeros_like(arr, dtype=np.uint8)
    premul[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    premul[..., 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    premul[premul[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(premul).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


_ARM_SKIN_REF: list[np.ndarray] = []


def _arm_skin_reference() -> np.ndarray | None:
    """Median skin tone of the resting arm's shoulder area (what shows through a feathered topwear tab)."""
    if _ARM_SKIN_REF:
        return _ARM_SKIN_REF[0]
    if not HD_ARM_RIGHT.exists():
        return None
    arr = np.array(Image.open(HD_ARM_RIGHT).convert("RGBA"), dtype=np.float32)
    h = arr.shape[0]
    top = arr[int(0.03 * h): int(0.20 * h)]
    inner = cv2.distanceTransform((top[..., 3] > 200).astype(np.uint8), cv2.DIST_L2, 5) > 10
    if not np.any(inner):
        return None
    _ARM_SKIN_REF.append(np.median(top[inner][:, :3], axis=0))
    return _ARM_SKIN_REF[0]


# Topwear UV rows of the vertical shoulder-socket skin tabs (~0.140..0.375, plus the narrow skin tip that runs down
# to the armpit point). The dissolve is at full strength from 0.150 and eases out over 0.360-0.405; near the top
# it is additionally limited per pixel to where the resting arm really lies underneath (never a background hole).
_TAB_Y0, _TAB_IN, _TAB_OUT, _TAB_Y1 = 0.140, 0.150, 0.360, 0.405
_TAB_ARM_MARGIN_PX = 2.0    # the tab only turns transparent >= this far inside the arm silhouette (+3 px ramp)
_TAB_DISSOLVE_PX = 40.0     # cubic alpha dissolve from the cut inward (arm skin covers >= 49 px past the cut)
_TAB_DEFRINGE_PX = 5        # outer px of the cut whose dark / light outline is replaced with skin


# Rest-pose world boxes of the meshes involved (used to map the resting-arm silhouette into topwear texture space):
# ArtMeshTopwear x -92..+93, y -362..-100; right resting arm x -242..-66 / left arm x 65..241, y -323..+121.
_TOP_WORLD_X0, _TOP_WORLD_W, _TOP_WORLD_Y0, _TOP_WORLD_H = -92.0, 185.0, -362.0, 262.0
_ARM_R_WORLD_X0, _ARM_L_WORLD_X0, _ARM_WORLD_W, _ARM_WORLD_Y0, _ARM_WORLD_H = -242.0, 65.0, 176.0, -323.0, 444.0
_CORNER_SPAN_PX = 52.0      # trapezius -> deltoid blend length, from the tab cut toward the neck
_CORNER_MAX_CUT_PX = 12.0   # never carve the top edge deeper than this
_CORNER_LINE_PX = 2.6       # lineart width painted along the carved edge (matches the arm's ~3 px outline)
_ARM_SIL: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
_TEX_NONE = 4294967295


def _arm_silhouette() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(alpha 0..1, distance-inside-silhouette in px, outline rgb) of the HD resting-arm art, cached."""
    if _ARM_SIL:
        return _ARM_SIL[0]
    if not HD_ARM_RIGHT.exists():
        return None
    arr = np.array(Image.open(HD_ARM_RIGHT).convert("RGBA"), dtype=np.float32)
    a = arr[..., 3] / 255.0
    dist = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    edge = (dist >= 1.0) & (dist <= 2.5) & (a > 0.9)
    line = np.median(arr[edge][:, :3], axis=0) if np.any(edge) else np.array([148.0, 96.0, 77.0], np.float32)
    _ARM_SIL.append((a, dist, line))
    return _ARM_SIL[0]


def _arm_in_top_space(h: int, w: int, flip: bool, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Sample the rest-pose resting arm (alpha, distance inside its silhouette in topwear px) on a grid of one
    side's topwear texture columns `xs` x rows `ys` (`flip` = the mirrored right tab / left arm)."""
    sil = _arm_silhouette()
    if sil is None:
        return None
    arm_a, arm_d, _ = sil
    gx, gy = np.meshgrid(xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5)
    wx = ((_TOP_WORLD_X0 + _TOP_WORLD_W) - gx / w * _TOP_WORLD_W) if flip else (gx / w * _TOP_WORLD_W + _TOP_WORLD_X0)
    wy = gy / h * _TOP_WORLD_H + _TOP_WORLD_Y0
    AH, AW = arm_a.shape
    u = (1.0 - (wx - _ARM_L_WORLD_X0) / _ARM_WORLD_W) if flip else ((wx - _ARM_R_WORLD_X0) / _ARM_WORLD_W)
    v = (wy - _ARM_WORLD_Y0) / _ARM_WORLD_H
    inside = (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    axi = np.clip((u * AW).astype(np.int64), 0, AW - 1)
    ayi = np.clip((v * AH).astype(np.int64), 0, AH - 1)
    # arm texels per topwear texel (both ~4 px / world unit at the shipped sizes)
    px_ratio = (w / _TOP_WORLD_W) / (AW / _ARM_WORLD_W)
    alpha = np.where(inside, arm_a[ayi, axi], 0.0)
    dist = np.where(inside, arm_d[ayi, axi] * px_ratio, 0.0)
    return alpha, dist


def _round_tab_top_corner(A: np.ndarray, C: np.ndarray, flip: bool, line_layer: np.ndarray | None = None) -> None:
    """Tuck the flat top-outer corner of a shoulder tab under the arm's rounded deltoid (in place, on one side's
    straight-colour view). Both top edges are measured with sub-pixel precision (column coverage), the carved edge
    starts exactly on the arm's top contour, tangent to it, eases into the topwear's own trapezius line over
    _CORNER_SPAN_PX and then hands back to the texture's native soft edge over a further taper. Only pixels ABOVE
    that curve are removed (never a hole inside the shoulder); the arm's lineart continues along the new edge,
    fading out toward the neck. With `line_layer` the lineart goes into that alpha layer instead of the topwear
    colour, so it can live on an overlay that hides together with the resting arm (raised-arm poses)."""
    sil = _arm_silhouette()
    if sil is None:
        return
    line_rgb = sil[2]
    h, w = A.shape
    sx, sy = w / 740.0, h / 1048.0
    y0, y1 = int(0.10 * h), int(0.24 * h)
    # The tab's vertical cut at the top of the tab zone (leftmost opaque column over rows 0.14-0.17).
    cut_rows = A[int(0.14 * h):int(0.17 * h)]
    cols = np.nonzero((cut_rows > 0.5).any(axis=0))[0]
    if len(cols) == 0:
        return
    x_cut = int(cols[0])
    span, taper = _CORNER_SPAN_PX * sx, 24.0 * sx
    xs = np.arange(max(0, x_cut - int(30 * sx)), min(w, x_cut + int(np.ceil(span + taper + 24 * sx)) + 2))
    ys = np.arange(y0, y1)
    sampled = _arm_in_top_space(h, w, flip, xs, ys)
    if sampled is None:
        return
    arm_a = np.clip(sampled[0], 0.0, 1.0)
    top_a = np.clip(A[y0:y1, xs], 0.0, 1.0)
    big = float(y1 + 100)

    def sub_edge(alpha: np.ndarray) -> np.ndarray:
        """Sub-pixel top edge per column = rows of missing coverage above the solid body (columns that are not
        solid at the bottom of the window have no top edge here)."""
        edge = y0 + np.sum(1.0 - alpha, axis=0)
        return np.where(alpha[-1] > 0.9, edge, big)

    arm_top = sub_edge(arm_a)
    top_top = sub_edge(top_a)
    # Corner = first column (from the cut inward) where the topwear's top edge reaches the arm's top contour:
    # from there on the flat trapezius edge would stand above the rounding deltoid.
    ok = (top_top < big) & (arm_top < big)
    above = np.nonzero((xs >= x_cut) & ok & (top_top <= arm_top + 0.5))[0]
    if len(above) == 0:
        return
    x_c = float(xs[int(above[0])])
    left = np.nonzero((xs >= x_c - 24 * sx) & (xs <= x_c) & (arm_top < big))[0]
    if len(left) < 4:
        return
    k1, k0 = np.polyfit(xs[left].astype(np.float64), arm_top[left], 1)     # deltoid tangent at the corner
    y_arm_c = float(k1 * x_c + k0)
    xf = xs.astype(np.float64)
    t = np.clip((xf - x_c) / span, 0.0, 1.0)
    s_w = t * t * (3.0 - 2.0 * t)
    top_ref = np.where(top_top < big, top_top, y_arm_c)
    y_t = (1.0 - s_w) * (y_arm_c + k1 * (xf - x_c)) + s_w * top_ref
    # At / outside the corner follow the arm itself (and remove everything where neither has a top edge).
    y_t = np.where(xf < x_c, np.where(arm_top < big, np.maximum(arm_top, top_ref), big), y_t)
    y_t = np.where(top_top < big, np.minimum(np.maximum(y_t, top_top), top_top + _CORNER_MAX_CUT_PX * sy), y_t)
    inner = (xf >= x_c) & (y_t < big)
    if inner.sum() > 3:                                                    # smooth out texel raggedness
        y_t[inner] = cv2.GaussianBlur(y_t[inner].reshape(-1, 1).astype(np.float32), (1, 0), 1.2 * sx).ravel()
    tp = np.clip((xf - (x_c + span)) / taper, 0.0, 1.0)                   # hand back to the native soft edge
    tp = tp * tp * (3.0 - 2.0 * tp)
    line_w = (1.0 - s_w) ** 1.5 * np.clip((y_t - top_ref) / 2.0 + 0.35, 0.0, 1.0) * (1.0 - tp)
    # Outside the corner the arm draws its own outline: lineart only along the carved edge (from the corner in),
    # ramping on over 2 px so it joins the arm's outline without a dot.
    line_w = line_w * np.clip((xf - (x_c - 2.0 * sx)) / (2.0 * sx), 0.0, 1.0)
    lw = _CORNER_LINE_PX * sy
    yy = ys.astype(np.float64) + 0.5
    for j, x in enumerate(xs):
        if tp[j] >= 0.999:
            continue
        yt = float(y_t[j])
        cover = np.clip(yy - yt + 0.5, 0.0, 1.0) if yt < big else np.zeros_like(yy)
        A[y0:y1, x] *= cover * (1.0 - tp[j]) + tp[j]
        if line_w[j] > 0.01 and yt < big:
            prof = np.clip(1.0 - np.abs(yy - (yt + 0.5 * lw)) / (0.5 * lw + 0.5), 0.0, 1.0) * line_w[j]
            if line_layer is not None:
                line_layer[y0:y1, x] = np.maximum(line_layer[y0:y1, x], prof * A[y0:y1, x])
            else:
                C[y0:y1, x] = C[y0:y1, x] * (1.0 - prof[:, None]) + line_rgb[None, :] * prof[:, None]


SHOULDER_LINE_NODES = {"R": 53200, "L": 53201}


def _attach_shoulder_line_overlays(header: bytes, tex_r: int, tex_l: int) -> bytes:
    """Mount the two shoulder-corner lineart overlays (see _round_tab_top_corner) as Parts that share
    ArtMeshTopwear's mesh + every one of its deform bindings (so they stay glued to the carved edge in every pose),
    drawn just in front of it, with opacity cross-fading out exactly like the resting arm on Arm R/L Pose."""
    puppet = json.loads(header[12:])
    top = None
    parent = None
    stack = [(puppet["nodes"], None)]
    while stack:
        n, par = stack.pop()
        if n.get("uuid") == 1005:
            top, parent = n, par
            break
        stack.extend((c, n) for c in n.get("children", ()))
    if top is None or parent is None or not top.get("mesh"):
        return header
    kids = parent.setdefault("children", [])
    kids[:] = [c for c in kids if c.get("uuid") not in SHOULDER_LINE_NODES.values()]
    for side, tid in (("R", tex_r), ("L", tex_l)):
        uid = SHOULDER_LINE_NODES[side]
        kids.append({
            "uuid": uid, "name": f"{'Right' if side == 'R' else 'Left'} shoulder corner lineart", "type": "Part",
            "enabled": True, "zsort": round(float(top.get("zsort", -7.0)) - 0.02, 4),
            "transform": json.loads(json.dumps(top.get("transform", {"trans": [0, 0, 0], "rot": [0, 0, 0], "scale": [1, 1]}))),
            "lockToRoot": False, "textures": [tid, _TEX_NONE, _TEX_NONE], "blend_mode": "Normal",
            "tint": [1.0, 1.0, 1.0], "screenTint": [0.0, 0.0, 0.0], "mask_threshold": 0.5, "opacity": 1.0, "masks": [],
            "mesh": json.loads(json.dumps(top["mesh"])), "children": [],
        })
    arm_of = {"R": 1046, "L": 1000}
    for prm in puppet.get("param", []):
        bs = prm.get("bindings", [])
        bs[:] = [b for b in bs if b.get("node") not in SHOULDER_LINE_NODES.values()]
        extra = []
        for b in bs:
            if b.get("node") == 1005 and b.get("param_name") == "deform":
                for uid in SHOULDER_LINE_NODES.values():
                    nb = json.loads(json.dumps(b))
                    nb["node"] = uid
                    extra.append(nb)
        name = str(prm.get("name") or "")
        for side in ("R", "L"):
            if name == f"Arm {side} Pose":
                ob = next((b for b in bs if b.get("node") == arm_of[side] and b.get("param_name") == "opacity"), None)
                if ob is not None:
                    nb = json.loads(json.dumps(ob))
                    nb["node"] = SHOULDER_LINE_NODES[side]
                    extra.append(nb)
        bs.extend(extra)
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _feather_topwear_shoulder_tabs_premul(premul_png: bytes) -> bytes:
    """Topwear texture with both shoulder tabs carved + feathered and the corner lineart painted in (standalone use).
    The customizer build uses _feather_topwear_shoulder_tabs_layers so the lineart can hide with the arms."""
    return _feather_topwear_shoulder_tabs_layers(premul_png, separate_lines=False)[0]


def _feather_topwear_shoulder_tabs_layers(premul_png: bytes, separate_lines: bool = True) -> tuple[bytes, bytes | None, bytes | None]:
    """Melt the left / right vertical shoulder-socket skin tabs of ANY topwear texture (original, hikari-stripe,
    refined-vrm) into the resting arm underneath: works on un-premultiplied RGB + alpha, then re-premultiplies.
    Per row (only where the pixels next to the cut are skin, so halter fabric / cyan piping are never touched):
      1. defringe the outer 5 px of outline along the cut with the row's own interior skin colour;
      2. pull the fading skin toward the arm's tone so there is no colour step;
      3. cubic (smoothstep) alpha dissolve over 40 px from the cut inward, shortened where the skin run is short.
    Before that, the tab's flat top-outer corner is carved into the arm's rounded deltoid (_round_tab_top_corner).
    Returns (topwear, right-arm-side lineart overlay, left-arm-side lineart overlay); overlays are None unless
    `separate_lines`."""
    im = Image.open(io.BytesIO(premul_png)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr[..., 3] / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    m = a > 0.004
    rgb[m] = np.clip(arr[m, :3] / a[m, None], 0.0, 255.0)
    arm_ref = _arm_skin_reference()
    sx = w / 740.0                                   # tuned on the 740 px wide topwear UV layout
    dis_px = _TAB_DISSOLVE_PX * sx
    fringe = max(2, int(round(_TAB_DEFRINGE_PX * sx)))

    def is_skin(px: np.ndarray) -> np.ndarray:
        """Warm peach skin incl. its shaded creases; white piping, grey/black fabric and cyan stripes all fail."""
        r, g, b = px[..., 0], px[..., 1], px[..., 2]
        return (r > 140.0) & (r > b + 16.0) & (r > g + 14.0) & ((0.30 * r + 0.59 * g + 0.11 * b) > 110.0)

    y_lo, y_hi = int(_TAB_Y0 * h), min(h, int(np.ceil(_TAB_Y1 * h)) + 1)
    rows = np.arange(y_lo, y_hi)
    fy = (rows + 0.5) / h
    env = np.clip((fy - _TAB_Y0) / (_TAB_IN - _TAB_Y0), 0.0, 1.0) * np.clip((_TAB_Y1 - fy) / (_TAB_Y1 - _TAB_OUT), 0.0, 1.0)
    env = env * env * (3.0 - 2.0 * env)

    def erode_blur(v: np.ndarray, erode: int, sigma: float) -> np.ndarray:
        """Running minimum (so the effect has already tapered off before a row that can't take it) + Gaussian."""
        if erode > 0:
            pad = np.pad(v, erode, mode="edge")
            v = np.min(np.stack([pad[k:k + len(v)] for k in range(2 * erode + 1)]), axis=0)
        return cv2.GaussianBlur(v.reshape(-1, 1).astype(np.float32), (1, 0), sigma).ravel()

    lines = {False: np.zeros((h, w), np.float32), True: np.zeros((h, w), np.float32)} if separate_lines else None
    for flip in (False, True):                       # left tab, then the mirrored right tab (views, edited in place)
        A = a[:, ::-1] if flip else a
        C = rgb[:, ::-1] if flip else rgb
        # 0. Round the flat top-outer corner of the tab into the arm's deltoid curve (no squared shelf).
        _round_tab_top_corner(A, C, flip, None if lines is None else lines[flip][:, ::-1] if flip else lines[flip])
        # Pass 1: measure every row (cut position, skin run, interior skin colour).
        n_rows = len(rows)
        xe_r = np.zeros(n_rows, dtype=np.int64)
        ok_r = np.zeros(n_rows, dtype=np.float32)
        d_r = np.zeros(n_rows, dtype=np.float32)
        skin_r = np.zeros((n_rows, 3), dtype=np.float32)
        for k, y in enumerate(rows):
            xs = np.nonzero(A[y] > 0.04)[0]
            if len(xs) < 40:
                continue
            xe = int(xs[0])
            probe = C[y, xe + fringe: min(w, xe + fringe + int(130 * sx))]
            skin = is_skin(probe)
            if len(skin) < 8 or int(skin[:8].sum()) < 6:
                continue                             # fabric right at the edge: not a skin-tab row
            # Skin run = up to the first spot where non-skin persists for 3 px (ignores 1-2 px crease lines).
            gone = ~skin
            persist = gone[:-2] & gone[1:-1] & gone[2:]
            run = (int(np.argmax(persist)) if persist.any() else len(skin)) + fringe
            ref_lo = xe + fringe + int(3 * sx)
            ref_hi = max(ref_lo + 1, xe + int(min(run, fringe + 18 * sx)))
            xe_r[k], ok_r[k] = xe, 1.0
            d_r[k] = float(np.clip(run - 5.0 * sx, 0.0, dis_px))
            skin_r[k] = np.median(C[y, ref_lo:ref_hi], axis=0)
        # Smooth across rows: no row-to-row on/off steps, and the fade narrows gradually where fabric closes in.
        strength = erode_blur(ok_r * env, int(round(2 * sx)), 1.5 * sx) * ok_r
        d_s = np.maximum(erode_blur(d_r, int(round(2 * sx)), 2.0 * sx), 3.0 * sx)
        # Per-pixel arm coverage under the dissolve band (1 well inside the arm, 0 outside / on its outline).
        x_hi = min(w, int(xe_r.max() + dis_px + 3)) if ok_r.any() else 0
        arm_cov = None
        if x_hi > 0:
            sampled = _arm_in_top_space(h, w, flip, np.arange(x_hi), rows)
            if sampled is not None:
                arm_cov = np.clip((sampled[1] - _TAB_ARM_MARGIN_PX * sx) / (3.0 * sx), 0.0, 1.0)
        # Pass 2: apply.
        for k, y in enumerate(rows):
            st = float(strength[k])
            if st < 0.01:
                continue
            xe, d_len, row_skin = int(xe_r[k]), float(d_s[k]), skin_r[k]
            target = row_skin if arm_ref is None else 0.45 * row_skin + 0.55 * arm_ref
            n = int(np.ceil(d_len)) + 1
            xx = np.arange(xe, min(w, xe + n))
            d = (xx - xe).astype(np.float32)
            t = np.clip(d / d_len, 0.0, 1.0)
            fade = t * t * (3.0 - 2.0 * t)
            cov = arm_cov[k, xe:xe + len(xx)] if arm_cov is not None else np.ones(len(xx), np.float32)
            stp = st * cov                                   # per-pixel strength: only over real arm skin
            seg = C[y, xe:xe + len(xx)].copy()
            # 1. defringe: the outer px take the row's interior skin colour
            k_def = (np.clip(1.0 - (d - fringe + 1.0) / 2.0, 0.0, 1.0) * stp)[:, None]
            seg = seg * (1.0 - k_def) + row_skin[None, :] * k_def
            # 2. tone match toward the arm where the tab is mostly transparent
            k_tone = (0.6 * (1.0 - fade) * stp)[:, None]
            seg = seg * (1.0 - k_tone) + target[None, :] * k_tone
            C[y, xe:xe + len(xx)] = seg
            # 3. cubic alpha dissolve
            A[y, xe:xe + len(xx)] *= (1.0 - stp) + stp * fade

    def _encode(rgb_s: np.ndarray, alpha: np.ndarray) -> bytes:
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[..., :3] = np.clip(np.rint(rgb_s * alpha[..., None]), 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
        out[out[..., 3] == 0] = 0
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG", compress_level=6)
        return buf.getvalue()

    top_bytes = _encode(rgb, a)
    if lines is None:
        return top_bytes, None, None
    sil = _arm_silhouette()
    line_rgb = sil[2] if sil is not None else np.array([148.0, 96.0, 77.0], np.float32)
    solid = np.broadcast_to(line_rgb[None, None, :], (h, w, 3))
    return top_bytes, _encode(solid, np.clip(lines[False], 0.0, 1.0)), _encode(solid, np.clip(lines[True], 0.0, 1.0))


def _add_neck_back_collar_ring(neck_premul_bytes: bytes) -> bytes:
    """Paint the dark 3D inner back-of-collar ring behind the neck on ArtMeshNeck (tex[3], z=-5.0)
    so the refined-vrm turtleneck collar wraps cleanly around the back of Hikari's neck with zero pink-hair gaps."""
    im = Image.open(io.BytesIO(neck_premul_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr[..., 3] / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    m = a > 0.01
    rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.01), 0.0, 255.0)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / float(max(1, w - 1))
    v = yy / float(max(1, h - 1))

    # ArtMeshNeck world box: vx=[-28.9, 30.8], vy=[-418, -322].
    # Refined collar top tips sit at world x=[-27.4, +27.7] (u=0.025..0.948), vy~-359 (v~0.615).
    # Back collar arches gently UP behind the neck to v~0.585 at center and wraps cleanly to collar tips!
    u_c = (u - 0.486) / 0.462
    in_x = np.clip((1.0 - np.abs(u_c)) / 0.06, 0.0, 1.0)
    v_top = 0.578 + 0.038 * (u_c ** 2)
    in_y = np.clip((v - v_top) / 0.022, 0.0, 1.0)
    collar_a = in_x * in_y

    # Subtle shaded dark charcoal inner turtleneck lining + soft rim highlight at top curve
    rim_band = np.exp(-((v - (v_top + 0.012)) / 0.014) ** 2) * in_x
    collar_rgb = np.zeros((h, w, 3), dtype=np.float32)
    collar_rgb[..., 0] = 28.0 + 18.0 * rim_band
    collar_rgb[..., 1] = 28.0 + 18.0 * rim_band
    collar_rgb[..., 2] = 32.0 + 20.0 * rim_band

    # Subtle soft collar contact shadow on lower neck skin just above the front collar dip
    v_front = 0.655 - 0.035 * (u_c ** 2)
    sh_w = np.clip((v - (v_front - 0.075)) / 0.075, 0.0, 1.0) * np.clip((v_front + 0.05 - v) / 0.04, 0.0, 1.0) * a
    rgb = rgb * (1.0 - 0.14 * sh_w[..., None])

    # Composite neck OVER back-of-collar ring (so the ring shows cleanly behind the neck sides!)
    out_a = a + collar_a * (1.0 - a)
    out_premul = rgb * a[..., None] + collar_rgb * (collar_a * (1.0 - a))[..., None]

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(out_premul), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(out_a * 255.0), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _smooth_raised_upper_arm_cap(arm_premul_bytes: bytes, is_left: bool) -> bytes:
    """Round off the flat y=0 top shoulder-border cut on raised wave arms (tex[53]/tex[54])
    so waving blends smoothly into the shoulder with zero flat horizontal ledge."""
    im = Image.open(io.BytesIO(arm_premul_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr[..., 3] / 255.0
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    m = a > 0.01
    rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.01), 0.0, 255.0)

    # Gently feather the top 26px border where the original v27 arm hit y=0 flat!
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (1.0 - xx / float(max(1, w - 1))) if is_left else (xx / float(max(1, w - 1)))
    # Top-right shoulder cap sits at u in [0.70..1.00], y in [0..34]
    cap_y = 24.0 * np.clip((u - 0.72) / 0.28, 0.0, 1.0) ** 1.6
    top_fade = np.clip((yy - cap_y) / 14.0, 0.0, 1.0)
    a = a * np.where(yy < 48.0, top_fade, 1.0)

    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _upgrade_spring_physics(header: bytes) -> bytes:
    """Upgrade Hikari's rig to flagship Live2D-style SpringPendulum physics:
    1) Wires anatomical 2-cup Gaussian squash-and-stretch bust deform bindings onto Chest Y (5094) & Chest X (5095)
       on ArtMeshTopwear (1005), keeping collar, shoulders & waist 100% locked!
    2) Adds a moving Chest Physics Anchor (52000) + dual XY/YX SpringPendulum drivers (52001, 52002).
    3) Upgrades existing hair, twintail, ahoge, skirt, top-fabric, eye-jelly & arm drivers to juicy SpringPendulum."""
    puppet = json.loads(header[12:])
    nodes: dict[int, dict[str, Any]] = {}
    stack = [puppet["nodes"]]
    while stack:
        n = stack.pop()
        nodes[n["uuid"]] = n
        stack.extend(n.get("children", ()))

    # 1) Upgrade all 31 existing SimplePhysics drivers from stiff Pendulum -> juicy SpringPendulum!
    for n in nodes.values():
        if n.get("type") != "SimplePhysics":
            continue
        nm = str(n.get("name") or "").lower()
        os_x, os_y = n.get("output_scale", [1.0, 0.0])
        if "ahoge" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.16
            n["length_damping"] = 0.20
            n["frequency"] = max(1.75, float(n.get("frequency", 1.55)))
            n["output_scale"] = [round(float(os_x) * 1.45, 4), round(float(os_y) * 1.45, 4)]
        elif "back hair" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.22
            n["length_damping"] = 0.26
            n["output_scale"] = [round(float(os_x) * 1.38, 4), round(float(os_y) * 1.38, 4)]
        elif "front hair" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.24
            n["length_damping"] = 0.28
            n["output_scale"] = [round(float(os_x) * 1.32, 4), round(float(os_y) * 1.32, 4)]
        elif "skirt" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.20
            n["length_damping"] = 0.26
            n["output_scale"] = [round(float(os_x) * 1.40, 4), round(float(os_y) * 1.40, 4)]
        elif "top fabric" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.28
            n["length_damping"] = 0.28
            n["output_scale"] = [round(float(os_x) * 1.35, 4), round(float(os_y) * 1.35, 4)]
        elif "eye jelly" in nm or "eyelbounce" in nm or "eyerbounce" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.78
            n["length_damping"] = 0.78
            n["output_scale"] = [round(float(os_x) * 0.35, 4), round(float(os_y) * 0.35, 4)]
        elif "arm swing" in nm:
            n["angle_damping"] = 0.30
            n["output_scale"] = [round(float(os_x) * 1.28, 4), round(float(os_y) * 1.28, 4)]
        elif "brow" in nm or "mouth" in nm:
            n["model_type"] = "SpringPendulum"
            n["angle_damping"] = 0.36
            n["length_damping"] = 0.36
            n["output_scale"] = [round(float(os_x) * 1.25, 4), round(float(os_y) * 1.25, 4)]

    # 2) Wire anatomical 2-cup Gaussian squash-and-stretch deform bindings onto Chest Y (5094) & Chest X (5095)
    top_node = nodes.get(1005)
    body_node = nodes.get(104)
    if top_node and top_node.get("mesh") and body_node is not None:
        uvs = np.array(top_node["mesh"]["uvs"], dtype=np.float64).reshape(-1, 2)
        n_verts = len(uvs)
        u = uvs[:, 0]
        v = uvs[:, 1]

        # Symmetrical 2-cup Gaussian bust weights (collar/shoulders v<0.31 and waist v>0.79 strictly 0.0!)
        w_l = np.exp(-(((u - 0.30) / 0.23) ** 2 + ((v - 0.56) / 0.15) ** 2))
        w_r = np.exp(-(((u - 0.70) / 0.23) ** 2 + ((v - 0.56) / 0.15) ** 2))
        w_raw = np.clip(np.maximum(w_l, w_r) + 0.35 * np.minimum(w_l, w_r), 0.0, 1.0)
        # Hard guard: zero on collar/shoulders (v < 0.31) and waist/belly (v > 0.78)
        v_guard = np.clip((v - 0.31) / 0.08, 0.0, 1.0) * np.clip((0.79 - v) / 0.08, 0.0, 1.0)
        w = w_raw * v_guard
        side_sgn = np.where(u < 0.5, -1.0, 1.0)

        # Chest Y (5094): -1.0 = drop + soft outward squash, 0.0 = rest, +1.0 = lift + subtle perk
        def_y_neg = [[round(float(2.8 * side_sgn[i] * w[i]), 3), round(float(10.5 * w[i]), 3)] for i in range(n_verts)]
        def_zero = [[0.0, 0.0] for _ in range(n_verts)]
        def_y_pos = [[round(float(-2.2 * side_sgn[i] * w[i]), 3), round(float(-9.5 * w[i]), 3)] for i in range(n_verts)]

        # Chest X (5095): -1.0 = sway left, 0.0 = rest, +1.0 = sway right (with subtle natural arc)
        def_x_neg = [[round(float(-8.5 * w[i]), 3), round(float(1.6 * w[i]), 3)] for i in range(n_verts)]
        def_x_pos = [[round(float(8.5 * w[i]), 3), round(float(1.6 * w[i]), 3)] for i in range(n_verts)]

        is_set = [[True, True], [True, True], [True, True]]
        for param in puppet.get("param", []):
            pid = param.get("uuid")
            if pid == 5094:
                param["bindings"] = [
                    b for b in param.get("bindings", []) if b.get("node") != 1005
                ] + [{
                    "interpolate_mode": "Linear",
                    "isSet": is_set,
                    "node": 1005,
                    "param_name": "deform",
                    "values": [
                        [def_y_neg, def_y_neg],
                        [def_zero, def_zero],
                        [def_y_pos, def_y_pos],
                    ],
                }]
            elif pid == 5095:
                param["bindings"] = [
                    b for b in param.get("bindings", []) if b.get("node") != 1005
                ] + [{
                    "interpolate_mode": "Linear",
                    "isSet": is_set,
                    "node": 1005,
                    "param_name": "deform",
                    "values": [
                        [def_x_neg, def_x_neg],
                        [def_zero, def_zero],
                        [def_x_pos, def_x_pos],
                    ],
                }]

        # 3) Attach moving Chest Physics Anchor (52000) + dual XY/YX SpringPendulum drivers (52001, 52002)
        children = body_node.setdefault("children", [])
        children[:] = [c for c in children if c.get("uuid") != 52000]
        chest_anchor = {
            "uuid": 52000,
            "name": "Physics: Chest Spring Anchor",
            "type": "Node",
            "enabled": True,
            "zsort": 0.0,
            "lockToRoot": False,
            "transform": {"trans": [0.0, -210.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
            "children": [
                {
                    "uuid": 52001,
                    "name": "Chest Spring X -> Chest X",
                    "type": "SimplePhysics",
                    "enabled": True,
                    "zsort": 0.0,
                    "lockToRoot": False,
                    "transform": {"trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
                    "param": 5095,
                    "model_type": "SpringPendulum",
                    "map_mode": "XY",
                    "gravity": 1.0,
                    "length": 26.0,
                    "frequency": 2.35,
                    "angle_damping": 0.22,
                    "length_damping": 0.22,
                    "output_scale": [4.2, 0.0],
                    "local_only": False,
                    "children": [],
                },
                {
                    "uuid": 52002,
                    "name": "Chest Spring Y -> Chest Y",
                    "type": "SimplePhysics",
                    "enabled": True,
                    "zsort": 0.0,
                    "lockToRoot": False,
                    "transform": {"trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]},
                    "param": 5094,
                    "model_type": "SpringPendulum",
                    "map_mode": "YX",
                    "gravity": 1.0,
                    "length": 26.0,
                    "frequency": 2.45,
                    "angle_damping": 0.22,
                    "length_damping": 0.20,
                    "output_scale": [5.4, 0.0],
                    "local_only": False,
                    "children": [],
                },
            ],
        }
        children.append(chest_anchor)

        # Drive Chest Physics Anchor (52000) in world X & Y from Body Angle X/Y/Z, Body Lean, and Breath!
        for param in puppet.get("param", []):
            pname = str(param.get("name") or "")
            bs = param.setdefault("bindings", [])
            bs[:] = [b for b in bs if b.get("node") != 52000]
            if pname == "Body Angle X":
                bs.append({"interpolate_mode": "Linear", "isSet": is_set, "node": 52000, "param_name": "transform.t.x", "values": [[-28.0, -28.0], [0.0, 0.0], [28.0, 28.0]]})
            elif pname == "Body Angle Z":
                bs.append({"interpolate_mode": "Linear", "isSet": is_set, "node": 52000, "param_name": "transform.t.x", "values": [[-22.0, -22.0], [0.0, 0.0], [22.0, 22.0]]})
            elif pname == "Body Lean":
                bs.append({"interpolate_mode": "Linear", "isSet": is_set, "node": 52000, "param_name": "transform.t.x", "values": [[-20.0, -20.0], [0.0, 0.0], [20.0, 20.0]]})
            elif pname == "Body Angle Y":
                bs.append({"interpolate_mode": "Linear", "isSet": is_set, "node": 52000, "param_name": "transform.t.y", "values": [[32.0, 32.0], [0.0, 0.0], [-32.0, -32.0]]})
            elif pname == "Breath":
                nx = len(param.get("axis_points", [[0.0, 1.0]])[0])
                ny = len(param.get("axis_points", [[0.0, 1.0], [0.0, 1.0]])[1])
                b_vals = [[round(-18.0 * (ix / max(1, nx - 1)), 3) for _ in range(ny)] for ix in range(nx)]
                b_set = [[True for _ in range(ny)] for _ in range(nx)]
                bs.append({"interpolate_mode": "Linear", "isSet": b_set, "node": 52000, "param_name": "transform.t.y", "values": b_vals})

    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _clean_wave_arm_premul(img_path: Path, is_upper: bool = False, is_left: bool = False) -> bytes:
    """Load Codex HD wave arm PNG cleanly while defringing any stray dark 2px boundary specks
    so skin edges stay crisp and clean against both dark rooms and shoulders."""
    im = Image.open(img_path).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    rgb = arr[..., :3].copy()
    a = arr[..., 3] / 255.0

    # Replace isolated dark black-matte boundary specks (lum < 95 right on alpha edge) with local peach skin!
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dist_in = cv2.distanceTransform((a >= 0.30).astype(np.uint8), cv2.DIST_L2, 5)
    is_peach = (a > 0.70) & (rgb[..., 0] > 180.0) & (rgb[..., 0] > rgb[..., 2] + 15.0)
    dark_speck = (a > 0.03) & (dist_in <= 4.5) & (lum < 110.0)
    if np.any(dark_speck) and np.any(is_peach):
        wp = cv2.GaussianBlur(is_peach.astype(np.float32), (0, 0), 4.0)
        pfill = np.stack([
            cv2.GaussianBlur(rgb[..., c] * is_peach.astype(np.float32), (0, 0), 4.0) / np.maximum(1e-4, wp)
            for c in range(3)
        ], axis=-1)
        m_sp = (dark_speck & (wp > 0.02))[..., None]
        # Keep natural warm-brown hand outline on hands (only defringe upper & forearm black crust!)
        rgb = np.where(m_sp, pfill * 0.88 + rgb * 0.12, rgb)

    # Gently round off any flat y=0 top shoulder border on upper wave arm
    # AND boost medial shoulder opacity underneath topwear's feather zone so 0% dark background bleeds through!
    if is_upper:
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (1.0 - xx / float(max(1, w - 1))) if is_left else (xx / float(max(1, w - 1)))
        # Keep medial shoulder (u in [0.76..0.94], y in [40..420]) solid opaque under topwear's 28px feather!
        medial_band = np.clip((u - 0.74) / 0.06, 0.0, 1.0) * np.clip((0.97 - u) / 0.04, 0.0, 1.0)
        v_band = np.clip((yy - 28.0) / 24.0, 0.0, 1.0) * np.clip((480.0 - yy) / 80.0, 0.0, 1.0)
        boost = (a > 0.12).astype(np.float32) * medial_band * v_band
        a = np.clip(a + boost * (1.0 - a), 0.0, 1.0)

        # Smooth rounded deltoid arc across the flat y=0 top border (u in [0.58..1.0])
        cap_y = 14.0 * np.clip((u - 0.62) / 0.38, 0.0, 1.0) ** 1.5
        top_fade = np.clip((yy - cap_y) / 12.0, 0.0, 1.0)
        a = a * np.where(yy < 36.0, top_fade, 1.0)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * a[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


# Ball-joint shaping for the raised wave arms (applied to the clean HD wave textures at build time; source art untouched).
# Each joint end is trimmed to a disc centred on its pivot, so the part keeps the same outline at any rotation and
# its corners can never swing out past the neighbouring part. Past the pivot line the trim radius grows by
# WAVE_BALL_SLOPE units per unit, so the disc blends into the straight limb with no step or dent.
#   wrist 16 = half the forearm/hand width at the wrist pivot
#   elbow 28 = the closest the upper arm's outer contour comes to the elbow pivot below it
WAVE_WRIST_BALL_R = 16.0
WAVE_ELBOW_BALL_R = 28.0
WAVE_BALL_SLOPE = 1.5
WAVE_BALL_FEATHER = 0.8
WAVE_LINE_RGB = (150.0, 99.0, 82.0)   # the wave art's warm-brown outline (measured from its edges)
WAVE_LINE_W = 0.7                      # outline width in model units (~5.5 texture px)
WaveTrim = dict[str, Any]


def _wave_joint_masks(header: bytes) -> dict[int, list[WaveTrim]]:
    """Per wave-arm texture id: the joint trims to apply (pivots read from the rig, in that part's uv space)."""
    puppet = json.loads(header[12:])
    nodes: dict[int, dict[str, Any]] = {}
    stack = [puppet["nodes"]]
    while stack:
        n = stack.pop()
        nodes[n["uuid"]] = n
        stack.extend(n.get("children", ()))
    out: dict[int, list[WaveTrim]] = {}

    def trans(nid: int) -> tuple[float, float]:
        t = nodes.get(nid, {}).get("transform", {}).get("trans", [0.0, 0.0])
        return float(t[0]), float(t[1])

    def add(part_id: int, pivot: tuple[float, float], **trim: Any) -> None:
        part = nodes.get(part_id)
        if not part or not part.get("mesh"):
            return
        m = part["mesh"]
        verts = np.array(m["verts"], dtype=np.float64).reshape(-1, 2)
        uvs = np.array(m["uvs"], dtype=np.float64).reshape(-1, 2)
        ax = np.polyfit(verts[:, 0], uvs[:, 0], 1)
        ay = np.polyfit(verts[:, 1], uvs[:, 1], 1)
        trim.update(cu=float(np.polyval(ax, pivot[0])), cv=float(np.polyval(ay, pivot[1])),
                    su=float(ax[0]), sv=float(ay[0]))
        out.setdefault(int(part["textures"][0]), []).append(trim)

    for upper, forearm, hand, elbow, wrist, inner in ((50001, 50002, 50003, 50005, 50004, +1.0),
                                                       (50010, 50011, 50012, 50014, 50013, -1.0)):
        if wrist not in nodes or elbow not in nodes:
            continue
        # dir = which side of the pivot is the joint end (texture y points down)
        add(forearm, trans(wrist), kind="disc", r=0.0, dir=-1.0, start=1.5, line=True, center_row=True)   # distal cap, under the hand
        # The hand's own wrist base stays untouched (full original outline): with the near-rigid wrist its base
        # matches the forearm width and only tilts <= ~11.5 deg, and any cut there doubles the outline at the seam.
        add(forearm, (0.0, 0.0), kind="disc", r=WAVE_ELBOW_BALL_R, dir=+1.0, start=3.0)     # elbow cap, over the upper arm
        # Outer elbow of the upper arm (outer side only, y -14..+12 around the pivot): it bulges 29-32 units out,
        # so when the forearm swung in it showed as a corner past the forearm's elbow disc. Trim it to an arc just
        # outside the forearm's (+0.5) and re-draw the outline on it, since this arc is the visible silhouette.
        add(upper, trans(elbow), kind="bulge", r=WAVE_ELBOW_BALL_R + 0.5, outer=-inner, y_lo=-14.0, y_hi=12.0,
            line=True)
        # The art fades the forearm's inner (crease) edge over ~5 units and kinks it outward ~3.5 units at the
        # crease (y -52..-30) so it melts under the upper arm; when the elbow opens both swing into view as a ghost
        # wedge + jog. Make that edge crisp and straight (the line the art follows above and below: 15.5 @ y-62 ->
        # 25.2 @ y-22, elbow-pivot units).
        add(forearm, (0.0, 0.0), kind="crisp", side=inner, y_from=-60.0, line=(15.5, -62.0, 0.2425))
        # Same soft fade on the upper arm's crease edge (elbow-pivot y -52..-10): crisp it so the crease apex is a
        # clean V instead of a fuzzy spike when the elbow opens.
        add(upper, trans(elbow), kind="crisp", side=0.0, y_from=-58.0, y_to=-6.0)
    return out


def _round_wave_joint_premul(premul_png: bytes, trims: list[WaveTrim]) -> bytes:
    """Apply ball-joint trims (and crease-edge hardening) to a premultiplied wave-arm texture."""
    im = np.array(Image.open(io.BytesIO(premul_png)).convert("RGBA"), dtype=np.float32)
    h, w = im.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a0 = im[..., 3] / 255.0
    a = a0.copy()
    line_w = np.zeros((h, w), dtype=np.float32)
    d_in = cv2.distanceTransform((a0 > 0.5).astype(np.uint8), cv2.DIST_L2, 5)   # px inside the original outline

    def add_line(e: np.ndarray, wgt: np.ndarray | float = 1.0, edge_only: bool = False, px_per_unit: float = 1.0) -> None:
        """Outline colour along a new trimmed edge (e = distance inside it, model units), including its
        half-transparent fringe so no pale skin pixels show where the cut crosses the original outline.
        edge_only: just where the cut meets the original outline (for parts drawn OVER another part)."""
        lw = np.clip((WAVE_LINE_W - e) / 0.25, 0.0, 1.0) * (e > -WAVE_BALL_FEATHER) * (a0 > 0.02)
        if edge_only:
            lw = lw * np.clip((2.0 - d_in / px_per_unit) / 0.6, 0.0, 1.0)
        np.maximum(line_w, lw * wgt, out=line_w)

    for tr in trims:
        ux = (xx / w - tr["cu"]) / tr["su"]          # model units relative to the pivot
        uy = (yy / h - tr["cv"]) / tr["sv"]
        if tr["kind"] == "disc" and tr.get("center_row"):
            # Wrist: the pivot sits ~0.7 units off the limb's centreline, so centre the disc on the limb itself
            # (measured on the pivot row) with r = its half-width + r. Wrist bends stay <= ~11.5 deg, so the
            # off-pivot centre drifts < 0.2 units - far less than the dent a pivot-centred disc leaves.
            row = int(np.clip(round(tr["cv"] * h), 0, h - 1))
            xs = np.where(a0[row] > 0.5)[0]
            if len(xs):
                lo, hi = (xs[0] / w - tr["cu"]) / tr["su"], (xs[-1] / w - tr["cu"]) / tr["su"]
                ux = ux - 0.5 * (lo + hi)
                tr = {**tr, "r": 0.5 * abs(hi - lo) + tr["r"]}
        if tr["kind"] == "disc":
            t = tr["dir"] * uy                        # distance past the pivot line, toward the joint end
            r_eff = tr["r"] + WAVE_BALL_SLOPE * np.maximum(0.0, tr.get("start", 0.0) - t)
            d = np.sqrt(ux * ux + uy * uy)
            a = a * np.clip((r_eff - d) / WAVE_BALL_FEATHER + 0.5, 0.0, 1.0)
            if tr.get("line"):
                add_line(r_eff - d, edge_only=tr["line"] == "edge", px_per_unit=tr["su"] * w)
        elif tr["kind"] == "fringe":
            row = int(np.clip(round(tr["cv"] * h), 0, h - 1))
            xs = np.where(a0[row] > 0.5)[0]
            if not len(xs):
                continue
            lo, hi = (xs[0] / w - tr["cu"]) / tr["su"], (xs[-1] / w - tr["cu"]) / tr["su"]
            off = np.abs(ux - 0.5 * (lo + hi)) - (0.5 * (hi - lo) - tr["edge_units"])
            near_side = np.clip(off / 0.5, 0.0, 1.0)              # only the two side corners, not the bottom
            soft = (a0 > 0.02) & (a0 < 0.97)
            wgt = np.clip((uy - tr["y_from"]) / 1.0, 0.0, 1.0) * near_side * soft
            np.maximum(line_w, wgt.astype(np.float32), out=line_w)
        elif tr["kind"] == "bulge":
            r_eff = tr["r"] + WAVE_BALL_SLOPE * (np.maximum(0.0, uy - tr["y_hi"]) + np.maximum(0.0, tr["y_lo"] - uy))
            d = np.sqrt(ux * ux + uy * uy)
            wgt = np.clip(tr["outer"] * ux / 4.0, 0.0, 1.0)
            a = a * (1.0 - wgt * (1.0 - np.clip((r_eff - d) / WAVE_BALL_FEATHER + 0.5, 0.0, 1.0)))
            if tr.get("line"):
                add_line(r_eff - d, wgt)
        else:
            wgt = np.clip((uy - tr["y_from"]) / 4.0, 0.0, 1.0) * np.clip((tr.get("y_to", 1e9) - uy) / 4.0, 0.0, 1.0)
            if tr["side"]:
                wgt = wgt * np.clip(tr["side"] * ux / 2.0, 0.0, 1.0)
            crisp = np.clip((a - 0.5) / 0.12 + 0.5, 0.0, 1.0)
            crisp = crisp * crisp * (3.0 - 2.0 * crisp)
            if "line" in tr:
                x0, y0, slope = tr["line"]
                crisp = crisp * np.clip((x0 + slope * (uy - y0) - tr["side"] * ux) / WAVE_BALL_FEATHER + 0.5, 0.0, 1.0)
            a = a * (1.0 - wgt) + np.minimum(a, crisp) * wgt
    rgb = im[..., :3] / np.maximum(a0, 1e-4)[..., None]          # straight colour
    lw3 = line_w[..., None]
    rgb = rgb * (1.0 - lw3) + np.array(WAVE_LINE_RGB, dtype=np.float32) * lw3
    im[..., :3] = rgb * a[..., None]
    im[..., 3] = a * 255.0
    out = np.clip(np.rint(im), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _load_wraparound_collar_asset() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], bytes] | None:
    """Extract the rigged wraparound halter collar node (51001), its deform bindings, and tex[61]
    from assets/collar_preview/Hikari-v28-CollarPreview.inp."""
    if not COLLAR_PREVIEW_INP.exists():
        return None
    h_col, t_col, _ = _unpack_inp(COLLAR_PREVIEW_INP)
    if len(t_col) <= 61:
        return None
    p_col = json.loads(h_col[12:])
    collar_node: dict[str, Any] | None = None
    stack = [p_col["nodes"]]
    while stack:
        n = stack.pop()
        if n.get("uuid") == 51001:
            collar_node = n
            break
        stack.extend(n.get("children", ()))
    if collar_node is None:
        return None
    bindings_by_param: dict[str, list[dict[str, Any]]] = {}
    for param in p_col.get("param", []):
        bs = [b for b in param.get("bindings", []) if b.get("node") == 51001]
        if bs:
            bindings_by_param[str(param["name"])] = bs
    return collar_node, bindings_by_param, t_col[61][1]


def _build_iris_emissive_png(iris_premul_bytes: bytes) -> bytes:
    """Build a crisp GPU Emissive map from any iris texture (VRM Galaxy, Solid Black Pupil, Ruby, Aqua, etc.).
    Makes catchlights + vivid inner iris glow rings shine softly while keeping black pupils velvet-dark."""
    im = Image.open(io.BytesIO(iris_premul_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    a = arr[..., 3] / 255.0
    rgb = np.zeros_like(arr[..., :3])
    m = a > 0.02
    if np.any(m):
        rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.02), 0.0, 255.0)
    mx = np.max(rgb, axis=-1) / 255.0
    mn = np.min(rgb, axis=-1) / 255.0
    sat = np.where(mx > 0.02, (mx - mn) / np.maximum(mx, 0.02), 0.0)
    # Catchlights (bright white-pink) + vivid inner iris glow (high sat + mid-high brightness); dark pupils stay 0!
    hi_white = np.clip((mn - 0.68) / 0.25, 0.0, 1.0)
    hi_vivid = np.clip((mx - 0.42) / 0.42, 0.0, 1.0) * np.clip((sat - 0.32) / 0.35, 0.0, 1.0)
    glow = np.clip(hi_white * 0.92 + hi_vivid * 0.46, 0.0, 1.0) * a
    # Soft bloom spread inside the iris for rich studio luminescence
    glow_soft = cv2.GaussianBlur(glow, (0, 0), 3.0)
    em_w = np.clip(glow * 0.78 + glow_soft * 0.40, 0.0, 1.0) * a
    em_rgb = np.clip(rgb * em_w[..., None] * 0.85, 0.0, 255.0)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(em_rgb), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(em_w * 255.0), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _build_topwear_cyber_emissive_png(top_premul_bytes: bytes) -> bytes:
    """Build a crisp GPU Emissive map for ONLY the cyber-tech LED/circuit accents on the lower refined-vrm top
    (y > 0.56, bright white/red/cyan print glyphs — 0% skin, 0% black fabric!)."""
    im = Image.open(io.BytesIO(top_premul_bytes)).convert("RGBA")
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]
    a = arr[..., 3] / 255.0
    rgb = np.zeros_like(arr[..., :3])
    m = a > 0.1
    if np.any(m):
        rgb[m] = np.clip(arr[m, :3] / np.maximum(a[m, None], 0.1), 0.0, 255.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    yy = np.arange(h, dtype=np.float32)[:, None] / float(max(1, h))
    xx = np.arange(w, dtype=np.float32)[None, :] / float(max(1, w))
    # Strictly inside the lower cyber-print band (y in [0.56..0.81], x in [0.08..0.92]) — never waist skin!
    in_band = (yy >= 0.56) & (yy <= 0.805) & (xx >= 0.08) & (xx <= 0.92)
    # Exclude peach skin explicitly
    is_skin = (r > 180.0) & (r > g + 10.0) & (g > b - 6.0) & (lum > 165.0)
    # Detect bright white/silver tech rings/lines AND vivid red LED indicators
    is_led_red = (r > 135.0) & (r > g + 55.0) & (r > b + 55.0)
    is_led_white = (lum > 155.0) & (np.abs(r - b) < 28.0) & ~is_skin
    led = (in_band & (is_led_red | is_led_white) & (a > 0.5)).astype(np.float32)

    em_rgb = np.zeros_like(rgb)
    # Tint red LEDs vivid neon crimson & white circuits crisp cool-cyan cyber glow!
    em_rgb[..., 0] = np.where(is_led_red, 255.0, 170.0)
    em_rgb[..., 1] = np.where(is_led_red, 45.0, 232.0)
    em_rgb[..., 2] = np.where(is_led_red, 72.0, 255.0)
    glow = np.clip(led * 0.90 + cv2.GaussianBlur(led, (0, 0), 2.2) * 0.45, 0.0, 1.0) * in_band.astype(np.float32)

    out = np.zeros_like(arr, dtype=np.uint8)
    out[..., :3] = np.clip(np.rint(em_rgb * glow[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.rint(glow * 255.0), 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _attach_emissive_maps(header: bytes, part_emissive_tids: dict[int, int]) -> bytes:
    """Attach GPU Emissive texture indices onto Part nodes and wire a live 'Emissive Glow' parameter (52010)
    driving emissionStrength [0..2] on every glowing part."""
    if not part_emissive_tids:
        return header
    puppet = json.loads(header[12:])
    stack = [puppet["nodes"]]
    while stack:
        n = stack.pop()
        uid = n.get("uuid")
        if uid in part_emissive_tids:
            ts = list(n.get("textures") or [4294967295, 4294967295, 4294967295])
            while len(ts) < 3:
                ts.append(4294967295)
            ts[1] = part_emissive_tids[uid]
            n["textures"] = ts
            n["emissionStrength"] = 1.0
        stack.extend(n.get("children", ()))

    # Add/replace 'Emissive Glow' parameter (uuid 52010, range [0.0 .. 2.0], default 1.0)
    params = puppet.setdefault("param", [])
    params[:] = [p for p in params if p.get("uuid") != 52010]
    bindings = []
    for uid in sorted(part_emissive_tids.keys()):
        # Subtle eye glow vs crisp cyber LED glow
        hi_val = 0.95 if uid in (1023, 1024) else 0.78
        bindings.append({
            "interpolate_mode": "Linear",
            "isSet": [[True, True], [True, True]],
            "node": uid,
            "param_name": "emissionStrength",
            "values": [[0.0, 0.0], [hi_val * 2.0, hi_val * 2.0]],
        })
    params.append({
        "axis_points": [[0.0, 1.0], [0.0, 1.0]],
        "bindings": bindings,
        "defaults": [1.0, 0.0],
        "is_vec2": False,
        "max": [2.0, 1.0],
        "merge_mode": "Additive",
        "min": [0.0, 0.0],
        "name": "Emissive Glow",
        "uuid": 52010,
    })
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


def _attach_wraparound_collar(
    header: bytes,
    collar_node: dict[str, Any],
    bindings_by_param: dict[str, list[dict[str, Any]]],
    tex_index: int,
) -> bytes:
    """Attach wraparound collar mesh node 51001 under Body (104) with all deform bindings cleanly."""
    puppet = json.loads(header[12:])
    node_copy = json.loads(json.dumps(collar_node))
    node_copy["textures"] = [tex_index, 4294967295, 4294967295]
    stack = [puppet["nodes"]]
    while stack:
        n = stack.pop()
        if n.get("uuid") == 104:
            children = n.setdefault("children", [])
            children[:] = [c for c in children if c.get("uuid") != 51001]
            children.append(node_copy)
            break
        stack.extend(n.get("children", ()))
    for param in puppet.get("param", []):
        pname = str(param.get("name") or "")
        if pname in bindings_by_param:
            bs = param.setdefault("bindings", [])
            bs[:] = [b for b in bs if b.get("node") != 51001]
            bs.extend(json.loads(json.dumps(bindings_by_param[pname])))
    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


# ---------------------------------------------------------------------------------------------------------------
# Rig tuning post-pass: wider (safe) raised wave-arm ranges + Mouth Spring anchor. The original seamless resting
# arms (#1046 / #1000, deformed with the torso) are left untouched.
# ---------------------------------------------------------------------------------------------------------------
WAVE_FOREARM_OUT = 20.0   # deg the raised forearm rests tilted outward from vertical
RAISED_ARM_PARAMS = {"Arm R Elbow": 53110, "Arm R Wrist": 53111, "Arm L Elbow": 53112, "Arm L Wrist": 53113}


def _deg(v: float) -> float:
    return float(np.deg2rad(v))


def _bind_1d(param: dict[str, Any], node: int, prop: str, v_neg: float, v_zero: float, v_pos: float) -> None:
    """Replace/insert a binding on a 3-point (-1, 0, +1) 1D parameter."""
    bs = param.setdefault("bindings", [])
    bs[:] = [b for b in bs if not (b.get("node") == node and b.get("param_name") == prop)]
    bs.append({
        "node": node, "param_name": prop, "interpolate_mode": "Linear",
        "isSet": [[True, True], [True, True], [True, True]],
        "values": [[round(v_neg, 5), round(v_neg, 5)], [round(v_zero, 5), round(v_zero, 5)], [round(v_pos, 5), round(v_pos, 5)]],
    })


def _new_param_1d(name: str, uid: int) -> dict[str, Any]:
    return {"name": name, "uuid": uid, "is_vec2": False, "merge_mode": "Additive", "min": [-1.0, 0.0],
            "max": [1.0, 1.0], "defaults": [0.0, 0.0], "axis_points": [[0.0, 0.5, 1.0], [0.0, 1.0]], "bindings": []}


def _tune_arm_and_mouth_rig(header: bytes) -> bytes:
    """- Raised wave arms: lock Arm R/L Lift tightly to -5.5..+6.5 deg so shoulders NEVER separate or notch on
      any emote/slider. The forearm rests tilted WAVE_FOREARM_OUT deg outward (a wave, not hands-up) and the
      waving comes from the elbow: Gesture elbow +-16, Elbow slider -14..+26 (+ = fold). The hand art already
      contains the wrist, so it rides almost rigidly on the forearm (Gesture +-4, Wrist slider +-6, Swing +-1.5
      deg) - any real wrist bend reads as a broken hand. Stacked extremes (elbow -50..+32 incl. rest tilt)
      stay inside the wave art's rounded ball joints (see _wave_joint_masks).
    - Mouth Spring physics anchor follows the live VB Mouth / Jaw Open (the legacy 'Mouth' param is never driven)."""
    puppet = json.loads(header[12:])
    node_by_id: dict[int, dict[str, Any]] = {}
    stack = [puppet["nodes"]]
    while stack:
        n = stack.pop()
        node_by_id[n["uuid"]] = n
        stack.extend(n.get("children", ()))
    nodes = set(node_by_id)
    params = puppet.setdefault("param", [])
    by_name = {str(p.get("name")): p for p in params}

    for side, sh, el, wr in (("R", 50006, 50005, 50004), ("L", 50015, 50014, 50013)):
        if el not in nodes:
            continue
        sg = 1.0 if side == "R" else -1.0
        lift = by_name.get(f"Arm {side} Lift")
        if lift is not None:
            _bind_1d(lift, sh, "transform.r.z", sg * _deg(-5.5), 0.0, sg * _deg(6.5))
        # Rest pose: forearm (and the hand riding on it) tilted outward. r.z > 0 = clockwise on screen, and the
        # right arm is on screen-left, so outward is negative for R.
        rot = node_by_id[el].setdefault("transform", {}).setdefault("rot", [0.0, 0.0, 0.0])
        rot[2] = sg * _deg(-WAVE_FOREARM_OUT)
        gest = by_name.get(f"Arm {side} Gesture")
        if gest is not None:
            _bind_1d(gest, el, "transform.r.z", sg * _deg(16.0), 0.0, sg * _deg(-16.0))
            _bind_1d(gest, wr, "transform.r.z", sg * _deg(4.0), 0.0, sg * _deg(-4.0))
        swing = by_name.get(f"Arm {side} Swing")
        if swing is not None:
            _bind_1d(swing, wr, "transform.r.z", sg * _deg(1.5), 0.0, sg * _deg(-1.5))
        for pname, nid, v_neg, v_pos in ((f"Arm {side} Elbow", el, sg * _deg(-14.0), sg * _deg(26.0)),   # + = fold
                                         (f"Arm {side} Wrist", wr, sg * _deg(-6.0), sg * _deg(6.0))):
            prm = by_name.get(pname)
            if prm is None:
                prm = _new_param_1d(pname, RAISED_ARM_PARAMS[pname])
                params.append(prm)
                by_name[pname] = prm
            _bind_1d(prm, nid, "transform.r.z", v_neg, 0.0, v_pos)

    if 21200 in nodes:
        vb = by_name.get("VB Mouth")
        if vb is not None:
            xs_ax, ys_ax = vb["axis_points"][0], vb["axis_points"][1]
            bs = vb.setdefault("bindings", [])
            bs[:] = [b for b in bs if b.get("node") != 21200]
            bs.append({"node": 21200, "param_name": "transform.t.x", "interpolate_mode": "Linear",
                       "isSet": [[True] * len(ys_ax) for _ in xs_ax],
                       "values": [[round(52.0 * float(yv), 4) for yv in ys_ax] for _ in xs_ax]})
        jaw = by_name.get("Jaw Open")
        if jaw is not None:
            bs = jaw.setdefault("bindings", [])
            bs[:] = [b for b in bs if b.get("node") != 21200]
            bs.append({"node": 21200, "param_name": "transform.t.y", "interpolate_mode": "Linear",
                       "isSet": [[True, True], [True, True]], "values": [[0.0, 0.0], [24.0, 24.0]]})

    content = json.dumps(puppet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return header[:8] + len(content).to_bytes(4, "big") + content


class CharacterCustomizer:
    """Caches base 4x-upscaled textures in RAM so applying ANY eye/hair/outfit tweak rebuilds the INP in ~0.25s!"""
    def __init__(self, src_inp: Path, cache_dir: Path, vrm_assets_dir: Path) -> None:
        self.src_inp = src_inp
        self.cache_dir = cache_dir
        self.vrm_assets_dir = vrm_assets_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vrm_assets_dir.mkdir(parents=True, exist_ok=True)

        header, self._raw_textures, self._trailing = _unpack_inp(src_inp)
        header = _pad_all_meshes(header, self._raw_textures)
        header = _upgrade_spring_physics(header)
        header = _tune_arm_and_mouth_rig(header)
        self._header_seethrough = header
        self._header_solid = _sweep_front_bangs(header)
        self._header_json = self._header_solid
        self._wave_joints = _wave_joint_masks(header)
        self.wardrobe_seethrough = Wardrobe(self._header_seethrough)
        self.wardrobe = Wardrobe(self._header_solid)
        self._base_hd_textures: list[tuple[int, bytes]] | None = None
        self._premul_cache: dict[Path, tuple[float, bytes]] = {}
        self._smooth_wardrobe_cache: dict[str, bytes] = {}
        self._wraparound_collar = _load_wraparound_collar_asset()
        self.vrm_eyes: dict[str, Path] = {}
        self._ensure_default_vrm_eyes()

    def _load_premul_cached(self, img_path: Path) -> bytes:
        mtime = img_path.stat().st_mtime
        cached = self._premul_cache.get(img_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        data = _straight_png_to_premul(img_path)
        self._premul_cache[img_path] = (mtime, data)
        return data

    def _cached_header(self, tag: tuple[Any, ...], header: bytes, build: Any) -> bytes:
        """Header post-passes (collar / emissive) re-parse the whole rig JSON (~0.18 s each). Their result only
        depends on the input header + tag, so reuse it; returning the same bytes object keeps hashing cheap."""
        cache: dict[tuple[Any, ...], bytes] = self.__dict__.setdefault("_header_cache", {})
        key = tag + (len(header), hash(header))
        out = cache.get(key)
        if out is None:
            if len(cache) > 12:
                cache.clear()
            out = cache[key] = build(header)
        return out

    def _feathered_top(self, premul_png: bytes) -> tuple[bytes, bytes | None, bytes | None]:
        """Cached shoulder-tab feather + corner carve for whichever topwear texture is active:
        (topwear, right-shoulder lineart overlay, left-shoulder lineart overlay)."""
        cache: dict[tuple[Any, ...], tuple[bytes, bytes | None, bytes | None]] = self.__dict__.setdefault("_tab_cache", {})
        key = ("tabs", len(premul_png), hash(premul_png))
        out = cache.get(key)
        if out is None:
            if len(cache) > 8:
                cache.clear()
            out = _feather_topwear_shoulder_tabs_layers(premul_png)
            cache[key] = out
        return out

    def _ensure_default_vrm_eyes(self) -> None:
        ir = self.vrm_assets_dir / "vrm_iris_r.png"
        il = self.vrm_assets_dir / "vrm_iris_l.png"
        hr = self.vrm_assets_dir / "vrm_hi_r.png"
        hl = self.vrm_assets_dir / "vrm_hi_l.png"
        if ir.exists() and il.exists():
            self.vrm_eyes = {"iris_r": ir, "iris_l": il}
            if hr.exists():
                self.vrm_eyes["hi_r"] = hr
            if hl.exists():
                self.vrm_eyes["hi_l"] = hl
            return
        if DEFAULT_VRM_PATH.exists():
            self.vrm_eyes = extract_vrm_eye_assets(DEFAULT_VRM_PATH, self.vrm_assets_dir)

    def import_vrm(self, vrm_path: Path) -> dict[str, str]:
        self.vrm_eyes = extract_vrm_eye_assets(vrm_path, self.vrm_assets_dir)
        return {k: str(v) for k, v in self.vrm_eyes.items()}

    def _ensure_base_hd(self) -> list[tuple[int, bytes]]:
        if self._base_hd_textures is not None:
            return self._base_hd_textures
        base_file = self.cache_dir / "Hikari-v28-BaseHD-v5.inp"
        if base_file.exists() and base_file.stat().st_mtime >= self.src_inp.stat().st_mtime:
            _, texs, _ = _unpack_inp(base_file)
            self._base_hd_textures = texs
            return texs

        out = []
        for idx, (ttype, tdata) in enumerate(self._raw_textures):
            if ttype != 0:
                out.append((ttype, tdata))
            elif idx in TEX_SCLERA:
                out.append((ttype, _upscale_eyelash_or_sclera_hd(tdata, is_sclera=True)))
            elif idx in TEX_LASH_BROW:
                out.append((ttype, _upscale_eyelash_or_sclera_hd(tdata, is_sclera=False)))
            else:
                out.append((ttype, _upscale_premul_png(tdata, target_scale=4)))

        # Enforce 100% mirror-symmetrical eyes & brows (clean vector-smoothed Right Eye R -> Left Eye L!)
        for src_r, dst_list in (
            (14, (15,)),         # Sclera R -> L
            (16, (17, 18)),      # Lower lash R -> L
            (19, (21, 22)),      # Upper lash + double-eyelid crease R -> L
            (27, (29, 30)),      # Closed eye R -> L
            (31, (33, 34)),      # Eyebrow R -> L
        ):
            r_png = out[src_r][1]
            for dst_l in dst_list:
                if dst_l < len(out) and out[dst_l][0] == 0:
                    ref_size = Image.open(io.BytesIO(out[dst_l][1])).size
                    out[dst_l] = (0, _mirror_premul_png_h(r_png, target_size=ref_size))

        base_file.write_bytes(_pack_inp(self._header_json, out, self._trailing))
        self._base_hd_textures = out
        return out

    def build_custom_model(self, cfg: dict[str, Any], out_inp: Path) -> Path:
        base_texs = self._ensure_base_hd()
        out_texs = list(base_texs)

        style = str(cfg.get("eyeStyle") or "vrm_refined")
        custom_eye = Path(str(cfg.get("customEyePath") or "")) if cfg.get("customEyePath") else None

        # 1) Eyes (VRM Refined / VRM Color Variants / Custom PNG / Clean Original HD)
        if style == "custom" and custom_eye and custom_eye.exists():
            out_texs[TEX_IRIS_R] = (0, _build_vrm_mapped_iris(self._raw_textures[TEX_IRIS_R][1], custom_eye, None, False, cfg))
            out_texs[TEX_IRIS_L] = (0, _build_vrm_mapped_iris(self._raw_textures[TEX_IRIS_L][1], custom_eye, None, True, cfg))
            # Hide old blurry 19x15 catchlight layer when custom/VRM iris carries its own crisp highlights!
            out_texs[TEX_HI_R] = (0, _transparent_png())
            out_texs[TEX_HI_L] = (0, _transparent_png())
        elif style == "original_solid_pupil":
            out_texs[TEX_IRIS_R] = (0, _build_original_solid_pupil_iris(self._raw_textures[TEX_IRIS_R][1], False, cfg))
            out_texs[TEX_IRIS_L] = (0, _build_original_solid_pupil_iris(self._raw_textures[TEX_IRIS_L][1], True, cfg))
            out_texs[TEX_HI_R] = (0, _transparent_png())
            out_texs[TEX_HI_L] = (0, _transparent_png())
        elif style != "original_hd" and "iris_r" in self.vrm_eyes and "iris_l" in self.vrm_eyes:
            out_texs[TEX_IRIS_R] = (0, _build_vrm_mapped_iris(
                self._raw_textures[TEX_IRIS_R][1], self.vrm_eyes["iris_r"], self.vrm_eyes.get("hi_r"), False, cfg
            ))
            out_texs[TEX_IRIS_L] = (0, _build_vrm_mapped_iris(
                self._raw_textures[TEX_IRIS_L][1], self.vrm_eyes["iris_l"], self.vrm_eyes.get("hi_l"), True, cfg
            ))
            # Clear the old 19x15 blurry catchlight blotches since VRM highlights are crisp & baked cleanly!
            out_texs[TEX_HI_R] = (0, _transparent_png())
            out_texs[TEX_HI_L] = (0, _transparent_png())
        else:
            # Clean original 2D eye art upscaled + HSV tint (NO snake pupil!)
            out_texs[TEX_IRIS_R] = (0, _tint_premul_png(base_texs[TEX_IRIS_R][1], float(cfg.get("eyeHue", 0)), float(cfg.get("eyeSat", 1)), float(cfg.get("eyeVal", 1))))
            out_texs[TEX_IRIS_L] = (0, _tint_premul_png(base_texs[TEX_IRIS_L][1], float(cfg.get("eyeHue", 0)), float(cfg.get("eyeSat", 1)), float(cfg.get("eyeVal", 1))))

        # Remove the tiny floating smudge dash on the lower inner eyelid (ArtMeshEyelashR2/L2: tex 16, 17, 18)
        for tid_low in (16, 17, 18):
            if tid_low < len(out_texs):
                out_texs[tid_low] = (0, _transparent_png())

        # 1.5) HD Front & Back Hair (clean natural-outline bangs + matching HD twintails, or custom PNG)
        bangs_mode = str(cfg.get("bangsMode") or "solid")
        custom_fh = Path(str(cfg.get("customFrontHairPath") or "")) if cfg.get("customFrontHairPath") else None
        if DEFAULT_BACK_HAIR_HD.exists():
            out_texs[8] = (0, self._load_premul_cached(DEFAULT_BACK_HAIR_HD))
        if bangs_mode == "custom" and custom_fh and custom_fh.exists():
            out_texs[35] = (0, self._load_premul_cached(custom_fh))
        elif DEFAULT_FRONT_HAIR_HD.exists():
            hd_front = self._load_premul_cached(DEFAULT_FRONT_HAIR_HD)
            out_texs[35] = (0, hd_front)
            if bangs_mode == "seethrough":
                out_texs[36] = (0, hd_front)

        # 1.6) HD Body Parts (face, neck, mirrored symmetrical arms/legs, pleated skirt) — full native crispness!
        if HD_FACE.exists():
            out_texs[11] = (0, self._load_premul_cached(HD_FACE))
        if HD_NECK.exists():
            out_texs[3] = (0, self._load_premul_cached(HD_NECK))
        if HD_ARM_RIGHT.exists():
            arm_r_bytes = self._load_premul_cached(HD_ARM_RIGHT)
            out_texs[46] = (0, arm_r_bytes)
            arm_l_key = HD_ARM_RIGHT.with_name("_arm_left_mirror")
            mtime = HD_ARM_RIGHT.stat().st_mtime
            cached_l = self._premul_cache.get(arm_l_key)
            if cached_l is None or cached_l[0] != mtime:
                arm_l_bytes = _mirror_premul_png_h(arm_r_bytes)
                self._premul_cache[arm_l_key] = (mtime, arm_l_bytes)
            else:
                arm_l_bytes = cached_l[1]
            out_texs[0] = (0, arm_l_bytes)
        if HD_BOTTOMWEAR.exists():
            out_texs[4] = (0, self._load_premul_cached(HD_BOTTOMWEAR))
        if HD_LEG_RIGHT.exists():
            leg_r_bytes = self._load_premul_cached(HD_LEG_RIGHT)
            out_texs[1] = (0, leg_r_bytes)
            leg_l_key = HD_LEG_RIGHT.with_name("_leg_left_mirror")
            mtime = HD_LEG_RIGHT.stat().st_mtime
            cached_ll = self._premul_cache.get(leg_l_key)
            if cached_ll is None or cached_ll[0] != mtime:
                leg_l_bytes = _mirror_premul_png_h(leg_r_bytes)
                self._premul_cache[leg_l_key] = (mtime, leg_l_bytes)
            else:
                leg_l_bytes = cached_ll[1]
            out_texs[2] = (0, leg_l_bytes)

        # 2) Hair Color Customization (Direct Color Picker OR Hue/Sat/Val sliders across 100% of hair layers!)
        h_col = str(cfg.get("hairColor") or "#ff6a9e")
        h_cstr = float(cfg.get("hairColorStrength", 0.0))
        h_hue = float(cfg.get("hairHue", 0.0))
        h_sat = float(cfg.get("hairSat", 1.0))
        h_val = float(cfg.get("hairVal", 1.0))
        # Recolored hair is cached per (layer, source, settings): rebuilds that only touch eyes / outfit / rig skip it.
        hair_cache: dict[tuple[Any, ...], bytes] = self.__dict__.setdefault("_hair_cache", {})
        if h_cstr >= 0.01:
            hair_mode: tuple[Any, ...] | None = ("direct", h_col, round(h_cstr, 4), round(h_sat, 4), round(h_val, 4))
        elif abs(h_hue) >= 0.5 or abs(h_sat - 1.0) >= 0.01 or abs(h_val - 1.0) >= 0.01:
            hair_mode = ("hsv", round(h_hue, 3), round(h_sat, 4), round(h_val, 4))
        else:
            hair_mode = None
        if hair_mode is not None:
            if len(hair_cache) > 4 * len(TEX_HAIR):
                hair_cache.clear()
            for tid in TEX_HAIR:
                if tid < len(out_texs) and out_texs[tid][0] == 0:
                    src_png = out_texs[tid][1]
                    hkey = (tid, len(src_png), hash(src_png)) + hair_mode
                    painted_hair = hair_cache.get(hkey)
                    if painted_hair is None:
                        if hair_mode[0] == "direct":
                            painted_hair = _recolor_hair_direct_png(src_png, h_col, h_cstr, h_sat, h_val)
                        else:
                            painted_hair = _tint_premul_png(src_png, h_hue, h_sat, h_val)
                        hair_cache[hkey] = painted_hair
                    out_texs[tid] = (0, painted_hair)

        # Select original or alternate artwork verbatim (no procedural cutout hacking!) before applying fabric tint.
        top_style = cfg.get("outfitStyle", "original")
        is_high_top = cfg.get("shoeStyle") == "d3-high-top"
        if top_style == "hikari-stripe":
            out_texs[TOP_TEXTURE] = (0, self.wardrobe.stripe_top)
        elif top_style == "refined-vrm":
            if HD_TOPWEAR_REFINED.exists():
                out_texs[TOP_TEXTURE] = (0, self._load_premul_cached(HD_TOPWEAR_REFINED))
            else:
                out_texs[TOP_TEXTURE] = (0, self.wardrobe.refined_top)
            if self._wraparound_collar is None and out_texs[3][0] == 0:
                out_texs[3] = (0, _add_neck_back_collar_ring(out_texs[3][1]))
        # Load clean HD wave arms (tex 53..58) with edge-crust defringe when present!
        if len(out_texs) > 58:
            wave_map = (
                (53, HD_WAVE_DIR / "wave-right-upper-hd.png", True, False, True),
                (54, HD_WAVE_DIR / "wave-left-upper-hd.png", True, True, True),
                (55, HD_WAVE_DIR / "wave-right-forearm-hd.png", False, False, True),
                (56, HD_WAVE_DIR / "wave-left-forearm-hd.png", False, True, True),
                (57, HD_WAVE_DIR / "wave-right-hand-hd.png", False, False, False),
                (58, HD_WAVE_DIR / "wave-left-hand-hd.png", False, True, False),
            )
            for tid_w, wpath, is_up, is_l, defringe in wave_map:
                if wpath.exists():
                    ckey = f"wave_{tid_w}_{wpath.stat().st_mtime}"
                    if ckey not in self._smooth_wardrobe_cache:
                        if defringe:
                            self._smooth_wardrobe_cache[ckey] = _clean_wave_arm_premul(wpath, is_upper=is_up, is_left=is_l)
                        else:
                            self._smooth_wardrobe_cache[ckey] = _straight_png_to_premul(wpath)
                    jkey = f"{ckey}_joint_v14_{WAVE_WRIST_BALL_R}_{WAVE_ELBOW_BALL_R}_{WAVE_BALL_SLOPE}"
                    trims = self._wave_joints.get(tid_w)
                    if trims and jkey not in self._smooth_wardrobe_cache:
                        self._smooth_wardrobe_cache[jkey] = _round_wave_joint_premul(self._smooth_wardrobe_cache[ckey], trims)
                    out_texs[tid_w] = (0, self._smooth_wardrobe_cache[jkey if trims else ckey])
                elif tid_w in (53, 54):
                    ckey = f"raised_{tid_w}"
                    if ckey not in self._smooth_wardrobe_cache:
                        self._smooth_wardrobe_cache[ckey] = _smooth_raised_upper_arm_cap(out_texs[tid_w][1], is_left=is_l)
                    out_texs[tid_w] = (0, self._smooth_wardrobe_cache[ckey])
        # Seamless shoulders on every outfit: melt the vertical shoulder-socket skin tabs into the arm (pre-tint).
        shoulder_lines: tuple[bytes, bytes] | None = None
        if out_texs[TOP_TEXTURE][0] == 0:
            top_png, line_r, line_l = self._feathered_top(out_texs[TOP_TEXTURE][1])
            out_texs[TOP_TEXTURE] = (0, top_png)
            if line_r is not None and line_l is not None:
                shoulder_lines = (line_r, line_l)
        if is_high_top:
            for texture_index, paint in zip(SHOE_TEXTURES, self.wardrobe.high_tops, strict=True):
                out_texs[texture_index] = (0, paint)
        o_tint = str(cfg.get("outfitTint") or "#ffffff")
        o_str = float(cfg.get("outfitTintStrength", 0.0))
        o_val = float(cfg.get("outfitVal", 1.0))
        if o_str >= 0.01 or abs(o_val - 1.0) >= 0.01:
            for tid in TEX_OUTFIT:
                if tid < len(out_texs) and out_texs[tid][0] == 0:
                    preserve_print = (top_style != "original" and tid == TOP_TEXTURE) or (
                        is_high_top and tid in SHOE_TEXTURES
                    )
                    painted = _recolor_outfit_skin_safe_png(
                        out_texs[tid][1], o_tint, o_str, o_val, preserve_print=preserve_print,
                    )
                    out_texs[tid] = (0, painted)

        use_seethrough = (bangs_mode == "seethrough")
        active_wardrobe = self.wardrobe_seethrough if use_seethrough else self.wardrobe
        base_header = self._header_seethrough if use_seethrough else self._header_solid
        header = active_wardrobe.raised_shoe_header if is_high_top else base_header

        # Attach real rigged 3D wraparound collar mesh (51001) behind neck for refined-vrm halter!
        if top_style == "refined-vrm" and self._wraparound_collar is not None:
            collar_node, collar_binds, collar_tex = self._wraparound_collar
            if o_str >= 0.01 or abs(o_val - 1.0) >= 0.01:
                collar_tex = _recolor_outfit_skin_safe_png(collar_tex, o_tint, o_str, o_val, preserve_print=False)
            collar_tid = len(out_texs)
            out_texs.append((0, collar_tex))
            header = self._cached_header(("collar", collar_tid), header,
                                         lambda h: _attach_wraparound_collar(h, collar_node, collar_binds, collar_tid))

        # Attach real GPU Emissive Glow maps (Galaxy / Solid-Pupil Eyes + Cyber Halter Top LEDs!)
        emissive_map: dict[int, int] = {}
        em_r = _build_iris_emissive_png(out_texs[TEX_IRIS_R][1])
        em_l = _build_iris_emissive_png(out_texs[TEX_IRIS_L][1])
        emissive_map[1023] = len(out_texs)
        out_texs.append((0, em_r))
        emissive_map[1024] = len(out_texs)
        out_texs.append((0, em_l))
        if top_style == "refined-vrm":
            top_png = out_texs[TOP_TEXTURE][1]
            em_cache: dict[tuple[Any, ...], bytes] = self.__dict__.setdefault("_em_top_cache", {})
            em_key = (len(top_png), hash(top_png))
            em_top = em_cache.get(em_key)
            if em_top is None:
                if len(em_cache) > 6:
                    em_cache.clear()
                em_top = em_cache[em_key] = _build_topwear_cyber_emissive_png(top_png)
            emissive_map[1005] = len(out_texs)
            out_texs.append((0, em_top))
        header = self._cached_header(("emissive", tuple(sorted(emissive_map.items()))), header,
                                     lambda h: _attach_emissive_maps(h, emissive_map))

        # Shoulder-corner lineart overlays (glued to ArtMeshTopwear, fade out with the resting arms).
        if shoulder_lines is not None:
            tid_r = len(out_texs)
            out_texs.append((0, shoulder_lines[0]))
            tid_l = len(out_texs)
            out_texs.append((0, shoulder_lines[1]))
            header = self._cached_header(("shoulder-lines", tid_r, tid_l), header,
                                         lambda h: _attach_shoulder_line_overlays(h, tid_r, tid_l))

        tmp = out_inp.with_suffix(".tmp")
        tmp.write_bytes(_pack_inp(header, out_texs, self._trailing))
        tmp.replace(out_inp)
        return out_inp
