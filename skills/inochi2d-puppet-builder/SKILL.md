---
name: inochi2d-puppet-builder
description: End-to-end playbook for creating, converting (from Live2D .moc3 / PSD2Live / VRM / AI art), upgrading, rigging, and visually verifying 60 FPS 2.5D Inochi2D (.inp) VTuber puppets with AI coding agents (Codex, Claude, Antigravity, Cursor). Makes going from zero to a working Hikari Studio puppet as easy as possible.
---

# Inochi2D (`.inp`) AI Puppet Builder Skill

Use this skill whenever the user wants to **create a new 2.5D VTuber puppet**, **convert an existing Live2D (`.moc3`) or VRM model into Inochi2D (`.inp`)**, or **upgrade textures, eyes, mouth, physics, or gesture arms** for Hikari Studio.

---

## 0. Pick the Easiest Starting Path First

Ask the user what they already have (or inspect their workspace) and pick the **lowest-effort path**:

| What the User Has | Recommended Path | Time |
| :--- | :--- | :---: |
| **Wants a custom character fast** (or just wants to swap eyes, hair color, bangs, outfit, shoes) | **Path A — Reskin Existing `.inp` Rig:** Start from `models/Hikari-v28-BaseHD-v5.inp` and use `core/customizer.py` + `core/wardrobe.py` (supports `.vrm` eye extraction, skin-safe outfit tinting, custom hair/top PNG swaps, `<0.6s` hot-reload). | ~5 min |
| **Has a Live2D model (`.moc3` / `.model3.json`) or single AI anime portrait** *(exact pipeline we used for Hikari!)* | **Path B — Live2D → Inochi2D via MINT + Upgrade Pass:** Auto-rig the portrait into Live2D (PSD2Live / Live2D Cubism) if needed → convert `.moc3` to `.inp` with **MINT (`mint-cli.exe`)** → restore dropped physics → upgrade to VBridger mouth, 3-bone rigid arms & 2× HD premultiplied textures. | 30–60 min |
| **Building from scratch in Inochi Creator** (layered PSD) | **Path C — Direct Inochi Creator → Programmatic Polish:** Cut layers in PSD → import into Inochi Creator (`.inx`), export `.inp`, then run Phases 3–6 below in Python for HD upscaling, 3-bone arms, and automated visual QA. | 1–2 hrs |

---

## 1. The `.inp` Binary Format (Why AI Agents Can Edit Puppets Directly)

An Inochi2D `.inp` file is a simple, open binary container — **never guess or hack binary offsets**; unpack and repack cleanly in Python:

```text
[8 bytes]  Magic header:  b"TRNSRTS\0"
[4 bytes]  Big-endian uint32: JSON scene-graph length (L)
[L bytes]  UTF-8 JSON scene graph (meta, nodes, meshes, UVs, param, bindings, automation)
[8 bytes]  Section tag:   b"TEX_SECT"
[4 bytes]  Big-endian uint32: Texture count (N)
For i in 0..N-1:
  [4 bytes] Big-endian uint32: Encoded texture byte length (T_i)
  [1 byte]  Encoding flag (0 = PNG, 1 = TGA, 2 = BC7)
  [T_i B]   Raw encoded image bytes
[rest]     Optional trailing EXT_SECT metadata (preserve or strip cleanly)
```

```python
import io, json
from pathlib import Path
import numpy as np
from PIL import Image

def unpack_inp(path: Path) -> tuple[dict, list[tuple[int, bytes]], bytes]:
    raw = path.read_bytes()
    assert raw[:8] == b"TRNSRTS\0", "Invalid .inp header"
    end = 12 + int.from_bytes(raw[8:12], "big")
    model = json.loads(raw[12:end])
    assert raw[end:end + 8] == b"TEX_SECT", "Missing TEX_SECT"
    count = int.from_bytes(raw[end + 8:end + 12], "big")
    cur = end + 12
    textures = []
    for _ in range(count):
        ln = int.from_bytes(raw[cur:cur + 4], "big")
        enc = raw[cur + 4]
        textures.append((enc, raw[cur + 5:cur + 5 + ln]))
        cur += 5 + ln
    return model, textures, raw[cur:]

def pack_inp(model: dict, textures: list[tuple[int, bytes]], trailing: bytes = b"") -> bytes:
    payload = json.dumps(model, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    out = [b"TRNSRTS\0", len(payload).to_bytes(4, "big"), payload, b"TEX_SECT", len(textures).to_bytes(4, "big")]
    for enc, data in textures:
        out.extend([len(data).to_bytes(4, "big"), bytes([enc]), data])
    if trailing:
        out.append(trailing)
    return b"".join(out)

def premul_png(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    """Inochi2D OpenGL REQUIRES premultiplied alpha PNGs (rgb * alpha)."""
    a = np.clip(np.nan_to_num(alpha, nan=0.0), 0.0, 1.0)
    c = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 255.0) * a[..., None]
    rgba = np.zeros((*a.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(np.rint(c), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
```

> **#1 RULE — PREMULTIPLIED ALPHA:** Inochi2D stores **premultiplied RGBA** in `TEX_SECT`. Before *any* color shift, upscale, or crop, un-premultiply (`rgb / np.maximum(alpha, 1e-6)`). Before writing back into `.inp`, re-premultiply (`rgb * alpha`). Skipping this creates dark halos around hair and eyelashes!

---

## 2. Path B Playbook: Converting Live2D (`.moc3`) → Flagship Inochi2D (`.inp`)

This is the exact, battle-tested pipeline we ran to build Hikari (all real gotchas included so you don't hit them twice):

### Step 1 — Start With Clean Front-Facing Art (If Generating First)
- Generate a symmetrical, front-facing anime character in **neutral T/A-pose**, flat cel lighting, plain solid background, eyes open, neutral mouth.
- If using an auto-rigger (PSD2Live / Live2D Cubism / Image2Live2D), export the complete Live2D folder (`*.moc3`, `*.model3.json`, `*.cdi3.json`, `*.physics3.json`, textures).

### Step 2 — Convert Live2D `.moc3` to `.inp` via MINT
Use `mint-cli` (from MINT — Live2D-to-Inochi2D converter):

```bash
# CRITICAL GOTCHA #1: mint-cli requires the .model3.json stem to EXACTLY match the .moc3 stem!
# Copy into a temp folder and rename <name>.model3.json -> <moc_stem>.model3.json first.
mint-cli.exe export path/to/model.moc3 path/to/output --inp
```

### Step 3 — Rescue What MINT Drops (Critical Post-Conversion Fixes!)
1. **Physics Deformations (`Front hair`, `Back hair`, `Skirt`):** MINT frequently strips Cubism physics deform bindings, leaving hair/clothes stiff.
   - *Fix:* Sample Cubism physics meshes before conversion (`cubism-physics-samples.json`) OR inject smooth parametric deform bindings into `model["param"]` with matching `min`/`defaults`/`max` and `axis_points` for hair sway, ahoge bounce, skirt swing, eye bounce, brow spring, mouth spring, and arm swing.
2. **Head-Turn Mouth Detachment:** Check that the `Angle` (Head X/Y) parameter still has `deform` bindings for every mouth part UUID (`mouth_ids <= posed`), or the mouth will float off the face during head turns!
3. **Closed-Eye Artifacts:** Auto-rigged Live2D models often leave stray iris pixels or jagged lash lines when `EyeOpen == 0`. Clean eyelash/sclera alpha mats and calm eyelid bounce springs (`damping >= 0.78`, `overshoot <= ±0.22`).

---

## 3. Flagship Quality Upgrades (Apply to Any `.inp`)

### Upgrade 1 — Non-Destructive 2× HD Lineart Upscale & Silhouette Smoothing
- Un-premultiply → run edge-preserving 2× upscale (Lanczos + bilateral denoise + unsharp lineart restoration + smooth silhouette alpha matting) → re-premultiply.
- Reference implementation: `core/texture_upscaler.py` (`_upscale_premul_png`, `smooth_premul_silhouette_png`).

### Upgrade 2 — Symmetrical VRM Eye Transfer (Zero Lazy-Eye!)
If the character has a 3D `.vrm` (GLB 2.0) model:
1. Parse GLB (`glTF` header → JSON chunk → BIN chunk → materials matching `eyeiris` / `eyehighlight`, see `core/customizer.py:extract_vrm_eye_assets`).
2. Tight-crop non-zero alpha iris bounds and center **symmetrically** onto both Left and Right Inochi2D iris canvases (mirroring secondary highlights). This completely eliminates cross-eye / lazy-eye drift.

### Upgrade 3 — Replace Sprite-Swapped Mouths with a 6-Layer Stencil VBridger Mouth
Flat mouth sprite swapping looks choppy. Replace it with an Inochi2D **OpenGL stencil-masked geometric mouth**:
- 6 layers: inner mouth cavity (stencil `Mask`), tongue, upper teeth, lower teeth, smooth 11-point Bezier-like upper/lower lip curves, and outer skin blend.
- Bind to the standard 8-parameter VTS/VBridger space (`jawOpen`, `mouthOpen`, `mouthForm`, `mouthFunnel`, `lipsPress`, `mouthShrug`, `mouthX`, `tongueOut`) so `modules/lipsync.py` (12-MFCC + IPA F1/F2 Formants at `0.28 ms/frame`) drives smooth anime vowels (`A, I, U, E, O, M/B/P, F/V`).

### Upgrade 4 — True 3-Bone Rigid Toggle Arms (No Spaghetti Fingers!)
Never deform a single arm mesh through a 120° wave — fingers turn into taffy. Build a **3-bone rigid Node hierarchy** (see `assets/hd_parts/wave/`):

```text
Shoulder Node          (rigid translation anchored to torso + lift rotation)
├── Upper Arm Part     (feathered deltoid tucked behind topwear + rounded elbow cap)
└── Elbow Node         (rigid transform.r.z rotation for wave & secondary spring swing)
    ├── Forearm Part   (rounded elbow-to-wrist overlap cap)
    └── Wrist Node     (rigid transform.r.z wrist flick + spring follow-through)
        └── Hand Part  (100% rigid — zero finger stretching at any angle!)
```

- Generate one waving arm reference matching the puppet's exact lighting & camera projection.
- Segment into `upper`, `forearm`, and `hand` with feathered circular joint caps centered on verified `(shoulder, elbow, wrist)` pixel pivots, mirror for left/right, and cross-fade `opacity` against the resting arm during `greet_wave` / `cheer_bounce`.

### Upgrade 5 — Skin-Safe HSV Wardrobe & Hair Recoloring
When recoloring outfits on shared body/clothing textures (`core/customizer.py`, `core/wardrobe.py`), mask out the peach skin hue/saturation band before shifting fabric colors so neck, collarbone, and midriff skin stay untouched.

---

## 4. Mandatory Visual Verification Loop (Never Ship Blind!)

After **every** `.inp` script edit, render through the real native GPU engine (`engine/hikari-live-engine.exe` or `server.py` WebSocket capture) and **visually inspect** this stress grid:

1. **Neutral front** (check overall framing, halo-free hair/eyelash edges, centered symmetrical gaze)
2. **Head Yaw `±15°` & Pitch `±12°`** (verify mouth, eyes, nose, and bangs stay locked to the face)
3. **Full blink (`eyeOpen = 0`) & Sparkle Eyes** (no stray iris bleed, clean closed lash curve)
4. **Vowel Grid (`A, I, U, E, O, M/B/P`)** (teeth/tongue strictly clipped inside mouth stencil mask)
5. **Wave Arm Peak (`greet_wave`)** (seamless shoulder tuck, rounded elbow/wrist caps, crisp rigid fingers)
6. **Shader ON and Shader OFF** (verifies `GL_COLOR_ATTACHMENT0` albedo blit when GLSL post-shader is bypassed)

---

## 5. Fast Troubleshooting Table

| Symptom | Root Cause | Fix |
| :--- | :--- | :--- |
| `mint-cli` fails to find `.model3.json` | `.model3.json` filename stem differs from `.moc3` | Copy to temp dir & rename to `<moc_stem>.model3.json` |
| Hair/skirt froze after MINT conversion | MINT dropped Cubism physics deform bindings | Restore/synthesize hair, ahoge, skirt & spring deform bindings in `param` |
| Mouth floats off face during head turn | `Angle` param lost `deform` bindings for mouth UUIDs | Assert `mouth_ids <= posed` on `Angle` param after editing |
| Black/dark halos on pink hair or lashes | Straight RGBA packed instead of premultiplied alpha | Multiply `rgb *= alpha` before PNG encode (`premul_png`) |
| Cross-eyed / lazy-eye gaze | Uncentered iris crop or asymmetric L/R UV offset | Center tight iris alpha bbox identically on both L/R canvases |
| Waving hand stretches like rubber | Single mesh deformed >45° | Use 3-bone rigid Node hierarchy (`Shoulder → Elbow → Wrist`) |
| Outfit tint turns skin red/purple | Global HSV shift on combined body+top texture | Mask out skin HSV band before recoloring fabric |
| Shader OFF shows black / neon frame | Blitting `GL_COLOR_ATTACHMENT1` instead of `0` | Read `GL_COLOR_ATTACHMENT0` when shader is disabled |
| Blink looks overly jiggly / jelly | Under-damped eyelid bounce spring | Set damping `>= 0.78`, clamp overshoot `<= ±0.22` |
