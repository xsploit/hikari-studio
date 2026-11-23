---
name: inochi2d-puppet-builder
description: Build, curate, upscale, rig, and visually verify flagship 2.5D Inochi2D (.inp) VTuber puppets using AI coding agents (Codex, Claude, Antigravity) and image generation. Covers .inp binary packing/unpacking, premultiplied alpha matting, 3-bone rigid toggle arms, 6-layer stencil-masked VBridger mouths, VRM GLB eye extraction, and headless visual QA.
---

# Inochi2D (`.inp`) AI Puppet Builder Skill

Use this skill whenever the user wants to create, upgrade, debug, or re-rig a **2.5D Inochi2D (`.inp`)** VTuber puppet with AI-assisted art generation and programmatic rigging.

## 1. Core Inochi2D `.inp` Binary Wire Format

An Inochi2D `.inp` puppet is a compact, deterministic binary container — **no proprietary GUI required** to inspect, patch, upgrade textures, or assemble full rigs programmatically:

```
[8 bytes]  Magic header:  b"TRNSRTS\0"
[4 bytes]  Big-endian uint32: JSON scene-graph payload length (L)
[L bytes]  UTF-8 JSON scene graph (nodes, meshes, UVs, params, bindings, physics)
[8 bytes]  Section tag:   b"TEX_SECT"
[4 bytes]  Big-endian uint32: Texture slot count (N)
For i in 0..N-1:
  [4 bytes] Big-endian uint32: Encoded texture byte length (T_i)
  [1 byte]  Encoding flag (0 = PNG, 1 = TGA, 2 = BC7)
  [T_i B]   Raw encoded image bytes
[rest]     Optional trailing EXT_SECT metadata (preserve verbatim!)
```

### Reference Unpack / Pack Helpers (Python)

```python
import json
from pathlib import Path

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
    chunks = [b"TRNSRTS\0", len(payload).to_bytes(4, "big"), payload, b"TEX_SECT", len(textures).to_bytes(4, "big")]
    for enc, data in textures:
        chunks.extend([len(data).to_bytes(4, "big"), bytes([enc]), data])
    if trailing:
        chunks.append(trailing)
    return b"".join(chunks)
```

> **CRITICAL ALPHA RULE:** Inochi2D expects **premultiplied alpha** (`rgb_out = rgb * alpha`). Writing straight RGBA into `TEX_SECT` produces dark/black fringes around hair, eyelashes, and arms! Always un-premultiply (`rgb / max(alpha, eps)`) before any color/HSV/upscale edit, and re-premultiply (`rgb * alpha`) before saving PNG bytes back into `.inp`.

---

## 2. Step-by-Step AI Curation & Rigging Pipeline

Follow these phases in order. **Never skip Phase 6 (visual verification)** — inspecting rendered frames catches 95% of Z-order, UV, and alpha defects immediately.

### Phase 1: Base Character & Layered Part Separation
1. Start from a clean, symmetrical, front-facing character reference (neutral T-pose / A-pose, flat cel/anime lighting, solid chroma or alpha background) OR extract reference textures from an existing `.vrm` (GLB 2.0) model.
2. Decompose into clean anatomical & depth-sorted `Part` layers (back-to-front Z-sort):
   - Back hair / twin-tails
   - Legs, bottomwear, torso/topwear, neck, resting arms
   - Face base, blush overlay (`///`), ears
   - Sclera (masked), left/right irises + specular highlights (stencil-clipped to sclera), upper/lower eyelashes, eyelids, eyebrows
   - 6-layer geometric mouth (inner cavity, tongue, upper teeth, lower teeth, lip lines, skin mask)
   - Front hair / bangs (solid or subtle see-through pass over eyes), ahoge
   - Toggle gesture arms (hidden by default at `opacity = 0` until triggered)

### Phase 2: Non-Destructive HD Texture Upscaling & Silhouette Matting
- Never upscale premultiplied PNGs directly with naive bicubic filters — halos will bleed inward.
- Un-premultiply → run edge-preserving 2× anime lineart upscale (Lanczos + bilateral denoise + unsharp lineart restoration + smooth silhouette alpha matting) → re-premultiply.
- Keep reference implementation in `core/texture_upscaler.py` (`_upscale_premul_png`, `smooth_premul_silhouette_png`).

### Phase 3: Symmetrical VRM Eye Extraction & Zero-Lazy-Eye Mapping
When upgrading eyes from a `.vrm` (GLB 2.0) or generated high-res eye sheet:
1. Parse GLB chunks (`glTF` header → JSON chunk → BIN chunk → `materials` matching `eyeiris` / `eyehighlight`).
2. Compute the tight non-zero alpha bounding box of the iris, center it symmetrically onto both Left and Right Inochi2D iris texture canvases, and mirror the secondary highlight cleanly so gaze tracking has **zero cross-eye or lazy-eye drift**.
3. Bind separate `Eye::LookX/Y` parameter ranges with clamped inner/outer travel so irises never clip past the sclera stencil mask.

### Phase 4: True 3-Bone Rigid Toggle Arms (No Spaghetti Mesh Deformation!)
Deforming a single mesh arm through a 120° wave stretches fingers into taffy. Instead, build a **true 3-bone rigid node hierarchy** (see `assets/hd_parts/wave/` and `core/rig_controller.py`):

```text
Shoulder Node          (rigid translation anchored to torso + lift rotation)
├── Upper Arm Part     (feathered deltoid tucked behind topwear + rounded elbow cap)
└── Elbow Node         (rigid transform.r.z rotation for wave & secondary spring swing)
    ├── Forearm Part   (rounded elbow-to-wrist overlap cap)
    └── Wrist Node     (rigid transform.r.z wrist flick + spring follow-through)
        └── Hand Part  (100% rigid — zero finger distortion at any angle!)
```

- Generate the waving pose once with an image generator matched to the puppet's exact lighting & camera projection.
- Segment into `upper`, `forearm`, and `hand` with feathered circular joint caps centered directly on the verified `(shoulder, elbow, wrist)` pixel pivots.
- Mirror cleanly for left/right symmetry and cross-fade `opacity` against the resting arm during `greet_wave` / `cheer_bounce`.

### Phase 5: 6-Layer VBridger Mouth & 8-Parameter Lip-Sync Binding
Bind the mouth rig to the standard VTS/VBridger 8-parameter space so any lip-sync engine (MFCC, IPA Formants, Hybrid, or audio amplitude) drives expressive anime phonemes (`A, I, U, E, O, M/B/P, F/V`):
- `jawOpen`, `mouthOpen`, `mouthForm` (smile ↔ frown), `mouthFunnel` (pucker for `O`/`U`), `lipsPress` (`M/B/P`), `mouthShrug`, `mouthX`, `tongueOut`.
- Use an Inochi2D `Mask` on the inner mouth cavity so teeth and tongue stay strictly inside the lip silhouette at every opening angle.

### Phase 6: Automated Visual Verification Loop (MANDATORY)
Never trust JSON edits blind. After every `.inp` build:
1. Render the puppet through the real GPU engine (`engine/hikari-live-engine.exe` or `server.py` WebSocket frame capture) under **multiple stress poses**:
   - Neutral front, Head Yaw `±15°`, Head Pitch `±12°`
   - Full blink (`eyeOpen = 0`), Wide sparkle eyes
   - Vowel grid (`A, I, U, E, O, M/B/P`)
   - Full arm wave (`greet_wave` peak frame)
   - Shader ON **and** Shader OFF (`GL_COLOR_ATTACHMENT0` sanity check!)
2. Visually inspect every captured frame for:
   - Black/white premultiplied alpha fringes
   - Detached neck/collar or elbow seams
   - Iris clipping outside sclera or mismatched gaze convergence
   - Z-order pops (bangs behind eyes, waving arm clipping in front of face)
3. Fix the root cause in the builder script (never hand-edit binary blobs) and re-verify until every pose is artifact-free.

---

## 3. Common Pitfalls & Fast Fixes

| Symptom | Root Cause | Fix |
| :--- | :--- | :--- |
| Dark halo around pink hair / eyelashes | Straight RGBA packed instead of premultiplied alpha | Multiply `rgb *= alpha` before PNG encode (`premul_png`) |
| Eyes look slightly cross-eyed or lazy | Asymmetric iris UV bbox or uncentered VRM crop | Center iris centroid symmetrically on both L/R canvases |
| Waving fingers stretch or bend unnaturally | Single-mesh deformer rotated >45° | Switch to 3-bone rigid Node hierarchy (`Shoulder → Elbow → Wrist`) |
| Outfit recolor turns skin red/pink | Global HSV shift on combined body/outfit texture | Mask out peach skin hue/saturation band before tinting fabric |
| Shader OFF shows black or neon silhouette | Blitting `GL_COLOR_ATTACHMENT1` (emissive) instead of `0` | Read `GL_COLOR_ATTACHMENT0` (albedo) when post-shader is disabled |
| Blink feels jittery or overly bouncy | Under-damped eyelid spring | Set eye-bounce spring damping `>= 0.78` and clamp overshoot `<= ±0.22` |
