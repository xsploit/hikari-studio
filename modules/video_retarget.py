"""Video-to-VTuber Motion & Expression Retargeting Lab.
Extracts 60 FPS head pose (Yaw/Pitch/Roll), body sway, eye blinks, mouth openness,
and arm energy from ANY uploaded anime or real-human video clip using OpenCV optical flow + FFmpeg audio!
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np


def _smooth_1d(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if len(arr) < 5 or sigma <= 0:
        return arr.copy()
    k = max(3, int(round(sigma * 6)) | 1)
    x = np.arange(k) - k // 2
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(np.pad(arr, k // 2, mode="edge"), ker, mode="valid")


def extract_motion_from_video(video_path: Path, out_dir: Path, name: str | None = None) -> dict:
    """Analyze a video clip and save a 60 FPS motion clip JSON (+ optional extracted WAV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = (name or video_path.stem).strip().replace(" ", "_")
    safe_id = "".join(c for c in clip_id if c.isalnum() or c in ("-", "_"))[:48] or f"clip_{int(time.time())}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(max(total_src, 1), src_fps * 30.0))  # up to 30s per motion clip

    # Optional face detector for real/anime face lock (falls back gracefully to upper-center ROI)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_det = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.exists() else None

    frames_gray = []
    face_boxes = []
    W_AN = 256
    H_AN = 256

    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        # Center-crop square or letterbox-friendly resize for consistent optical flow
        small = cv2.resize(frame, (W_AN, H_AN), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)

        if face_det is not None and idx % 4 == 0:
            faces = face_det.detectMultiScale(gray, 1.12, 4, minSize=(42, 42))
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                face_boxes.append((fx, fy, fw, fh))
        idx += 1
    cap.release()

    n = len(frames_gray)
    if n < 4:
        raise RuntimeError("Video too short (need at least 4 frames).")

    # Compute stable face ROI (detected median or upper-center anime portrait region)
    if face_boxes:
        bx = int(np.median([b[0] for b in face_boxes]))
        by = int(np.median([b[1] for b in face_boxes]))
        bw = int(np.median([b[2] for b in face_boxes]))
        bh = int(np.median([b[3] for b in face_boxes]))
    else:
        bx, by, bw, bh = int(W_AN * 0.26), int(H_AN * 0.14), int(W_AN * 0.48), int(H_AN * 0.44)

    fx0, fx1 = max(0, bx), min(W_AN, bx + bw)
    fy0, fy1 = max(0, by), min(H_AN, by + bh)

    # Eye band (upper-mid face) and mouth band (lower face)
    ey0, ey1 = fy0 + int((fy1 - fy0) * 0.28), fy0 + int((fy1 - fy0) * 0.54)
    my0, my1 = fy0 + int((fy1 - fy0) * 0.62), fy0 + int((fy1 - fy0) * 0.90)
    mx0, mx1 = fx0 + int((fx1 - fx0) * 0.22), fx0 + int((fx1 - fx0) * 0.78)

    dx_head = np.zeros(n, dtype=np.float64)
    dy_head = np.zeros(n, dtype=np.float64)
    rot_head = np.zeros(n, dtype=np.float64)
    dx_body = np.zeros(n, dtype=np.float64)
    eye_dark = np.zeros(n, dtype=np.float64)
    mouth_act = np.zeros(n, dtype=np.float64)
    arm_l_act = np.zeros(n, dtype=np.float64)
    arm_r_act = np.zeros(n, dtype=np.float64)

    prev = frames_gray[0]
    cx_f = 0.5 * (fx0 + fx1)
    cy_f = 0.5 * (fy0 + fy1)
    yy_f, xx_f = np.mgrid[fy0:fy1, fx0:fx1]
    rx_f = (xx_f - cx_f) / max(1.0, bw * 0.5)
    ry_f = (yy_f - cy_f) / max(1.0, bh * 0.5)
    r2_f = rx_f * rx_f + ry_f * ry_f + 0.15

    for i in range(1, n):
        cur = frames_gray[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 2, 13, 2, 5, 1.1, 0
        )
        # Head translation + angular curl inside face ROI
        f_fx = flow[fy0:fy1, fx0:fx1, 0]
        f_fy = flow[fy0:fy1, fx0:fx1, 1]
        dx_head[i] = float(np.median(f_fx))
        dy_head[i] = float(np.median(f_fy))
        # Tangential rotation estimate (rx*fy - ry*fx)
        rot_head[i] = float(np.mean((rx_f * f_fy - ry_f * f_fx) / r2_f))

        # Lower-body horizontal sway
        body_fx = flow[fy1:, :, 0]
        dx_body[i] = float(np.median(body_fx)) if body_fx.size else 0.0

        # Eye vertical-gradient drop detects blinks!
        eye_patch = cur[ey0:ey1, fx0:fx1].astype(np.float64)
        eye_dark[i] = float(np.mean(np.abs(cv2.Sobel(eye_patch, cv2.CV_64F, 0, 1, ksize=3))))

        # Mouth vertical contrast + local motion detects speech flaps
        m_patch = cur[my0:my1, mx0:mx1].astype(np.float64)
        mouth_act[i] = float(np.std(m_patch)) + 3.5 * float(np.mean(np.abs(flow[my0:my1, mx0:mx1, 1])))

        # Left / right side arm motion energy
        arm_r_act[i] = float(np.mean(np.hypot(flow[fy0:, :max(16, fx0), 0], flow[fy0:, :max(16, fx0), 1])))
        arm_l_act[i] = float(np.mean(np.hypot(flow[fy0:, min(W_AN - 16, fx1):, 0], flow[fy0:, min(W_AN - 16, fx1):, 1])))
        prev = cur

    # Leaky integration of optical flow velocities -> smooth zero-drift pose angles!
    def leaky_integrate(vel: np.ndarray, decay: float = 0.94) -> np.ndarray:
        out = np.zeros_like(vel)
        acc = 0.0
        for i in range(len(vel)):
            acc = acc * decay + vel[i]
            out[i] = acc
        out -= np.median(out)
        return out

    ang_x = _smooth_1d(leaky_integrate(dx_head, 0.94), 2.2)
    ang_y = _smooth_1d(leaky_integrate(-dy_head, 0.93), 2.2)
    ang_z = _smooth_1d(leaky_integrate(rot_head, 0.92), 2.5)
    body_x = _smooth_1d(leaky_integrate(dx_body, 0.94), 3.0)

    # Normalize safely into Hikari v28's natural ±22° head / ±9° body envelope
    def norm_clip(a: np.ndarray, gain: float, lim: float) -> np.ndarray:
        p95 = max(1e-4, float(np.percentile(np.abs(a), 95)))
        scale = min(gain, lim / p95)
        return np.clip(a * scale, -lim, lim)

    ang_x = norm_clip(ang_x, 3.2, 22.0)
    ang_y = norm_clip(ang_y, 2.8, 16.0)
    ang_z = norm_clip(ang_z, 9.0, 14.0)
    body_x = norm_clip(body_x, 1.5, 8.5)
    body_z = np.clip(ang_z * 0.45, -7.0, 7.0)

    # Eye blink detection (dips below 82% of rolling eye contrast baseline)
    eye_dark[0] = eye_dark[1]
    eye_base = max(1.0, float(np.percentile(eye_dark, 75)))
    eye_ratio = np.clip(eye_dark / eye_base, 0.0, 1.2)
    eye_open = np.clip((eye_ratio - 0.58) / 0.30, 0.0, 1.0)
    eye_open = _smooth_1d(eye_open, 1.0)

    # Mouth openness curve
    mouth_act[0] = mouth_act[1]
    m_lo = float(np.percentile(mouth_act, 15))
    m_hi = max(m_lo + 1.0, float(np.percentile(mouth_act, 96)))
    mouth_open = _smooth_1d(np.clip((mouth_act - m_lo) / (m_hi - m_lo), 0.0, 1.0), 1.2)

    # Arm wave gesture curves
    arm_r = _smooth_1d(np.clip(arm_r_act / max(0.8, float(np.percentile(arm_r_act, 95))), 0.0, 1.0), 3.0)
    arm_l = _smooth_1d(np.clip(arm_l_act / max(0.8, float(np.percentile(arm_l_act, 95))), 0.0, 1.0), 3.0)

    # Resample to clean 60 FPS!
    dur = n / float(src_fps)
    n60 = max(6, int(round(dur * 60.0)))
    t_src = np.linspace(0.0, dur, n)
    t_60 = np.linspace(0.0, dur, n60)

    def rs(a: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in np.interp(t_60, t_src, a)]

    # Extract audio track if present via FFmpeg so playing the motion clip ALSO lip-syncs!
    wav_path = out_dir / f"{safe_id}.wav"
    has_audio = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
             "-t", "30", "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(wav_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
        )
        has_audio = (r.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 2000)
        if not has_audio and wav_path.exists():
            wav_path.unlink(missing_ok=True)
    except Exception:
        has_audio = False

    clip_data = {
        "id": safe_id,
        "name": name or video_path.stem,
        "fps": 60,
        "frames": n60,
        "duration": round(dur, 2),
        "hasAudio": has_audio,
        "audioFile": wav_path.name if has_audio else "",
        "tracks": {
            "AngleX": rs(ang_x),
            "AngleY": rs(ang_y),
            "AngleZ": rs(ang_z),
            "BodyX": rs(body_x),
            "BodyZ": rs(body_z),
            "EyeOpen": rs(eye_open),
            "MouthOpen": rs(mouth_open),
            "ArmR": rs(arm_r),
            "ArmL": rs(arm_l),
        },
    }
    (out_dir / f"{safe_id}.json").write_text(json.dumps(clip_data), encoding="utf-8")
    return {k: v for k, v in clip_data.items() if k != "tracks"}
