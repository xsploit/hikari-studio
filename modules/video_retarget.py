"""Video-to-VTuber general-motion retargeting (Motion Lab).

Extracts a drift-free 60 FPS *general motion* clip from any anime VTuber or real-human video:
head yaw / pitch / roll, body sway / roll, whole-body Stage X / Y / Zoom (hops, sways, lean-ins), independent
left / right eye open (winks + blinks), eye gaze, brows, shoulder shrug and clean arm raises + wave sway.

Mouth / lip-sync is deliberately NOT extracted: Hikari Studio's live lip-sync engines own the mouth.

Pipeline (100% offline, OpenCV + NumPy only):
  1. Face seed: anime dual-eye-blob lock (two dark non-skin blobs with skin bridge + cheeks below), else an
     OpenCV Haar frontal face (real humans), else a static upper-centre box (degraded, flagged in metadata).
  2. Face lock: a similarity transform (x, y, scale, roll) from the *reference* frame to every frame. Each frame
     is predicted with pyramidal KLT from the previous frame, then refined directly against the reference patch
     (warp the frame back into reference space, KLT ref->warped, RANSAC). Every estimate is relative to the
     reference, so nothing accumulates: zero drift. A dual-eye re-detect re-locks after occlusion / failure.
  3. Torso lock (same machinery on the chest) -> body roll, neck nod / turn and the shoulder-band shrug.
  4. All face measurements (eyes, gaze, brows, cheek asymmetry) run on the frame warped back into reference
     space, so head motion never fakes a blink.
  5. Arms: a skin model learned from the subject's own cheeks, masked outside the head / hair oval and the chest
     column, minus static skin-coloured background -> hand elevation above the shoulder line + horizontal sway.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

ANALYSIS_W = 640
MAX_SECONDS = 30.0

LEGACY_TRACKS = ("AngleX", "AngleY", "AngleZ", "BodyX", "BodyZ", "EyeOpen", "ArmR", "ArmL")
RICH_TRACKS = ("EyeLOpen", "EyeROpen", "EyeLookX", "EyeLookY", "BrowLY", "BrowRY",
               "StageX", "StageY", "StageZoom", "ShoulderShrug", "ArmRGesture", "ArmLGesture")
TRACK_KEYS = LEGACY_TRACKS + RICH_TRACKS

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


# ------------------------------------------------------------------------------------------------ helpers
def _smooth_1d(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < 5 or sigma <= 0:
        return arr.copy()
    k = max(3, int(round(sigma * 6)) | 1)
    x = np.arange(k) - k // 2
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(np.pad(arr, k // 2, mode="edge"), ker, mode="valid")


def _baseline(a: np.ndarray, window: float) -> np.ndarray:
    """Slow running median (window in samples) for high-passing perspective / lean drift."""
    a = np.asarray(a, dtype=np.float64)
    w = max(3, int(round(window)) | 1)
    if len(a) < 5:
        return np.full_like(a, float(np.median(a)) if len(a) else 0.0)
    pad = np.pad(a, w // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    return _smooth_1d(np.median(win, axis=1), w / 6.0)


def _frac(m: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    h, w = m.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(m[y0:y1, x0:x1].mean())


def _generic_skin(bgr: np.ndarray) -> np.ndarray:
    """Broad anime / human skin prior in HSV (only used to seed; the tracker then learns the subject's skin)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    m = (((h <= 22) | (h >= 172)) & (s >= 18) & (s <= 135) & (v >= 140)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]), float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """2x3 affine A o B (apply B first)."""
    A3 = np.vstack([A, [0.0, 0.0, 1.0]])
    B3 = np.vstack([B, [0.0, 0.0, 1.0]])
    return (A3 @ B3)[:2]


def _sim_from_pairs(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Exact similarity transform from two point pairs (src -> dst)."""
    (x0, y0), (x1, y1) = src
    (u0, v0), (u1, v1) = dst
    zs = complex(x1 - x0, y1 - y0)
    zd = complex(u1 - u0, v1 - v0)
    if abs(zs) < 1e-6:
        return np.array([[1.0, 0, u0 - x0], [0, 1.0, v0 - y0]])
    r = zd / zs
    a, b = r.real, r.imag
    tx = u0 - (a * x0 - b * y0)
    ty = v0 - (b * x0 + a * y0)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def _sim_params(M: np.ndarray) -> tuple[float, float]:
    """(scale, image-space rotation in degrees; positive = clockwise on screen)."""
    return math.hypot(M[0, 0], M[1, 0]), math.degrees(math.atan2(M[1, 0], M[0, 0]))


# ------------------------------------------------------------------------------------------------ face seeds
def detect_anime_eye_pair(bgr: np.ndarray) -> dict[str, Any] | None:
    """Anime / stylised face lock: two dark non-skin blobs side by side, skin bridge between, skin cheeks below.

    Returns {"eyes": [(cx, cy, w, h) screen-left, screen-right], "score": float} or None."""
    h, w = bgr.shape[:2]
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    sk = _generic_skin(bgr)
    eye = ((sk == 0) & (v < 175)).astype(np.uint8)
    eye = cv2.morphologyEx(eye, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(eye, 8)
    s = w / float(ANALYSIS_W)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, a = (int(t) for t in st[i])
        if a < 15 * s * s or bw < 4 * s or bh < 3 * s or bw > w * 0.09 or bh > w * 0.09:
            continue
        below = _frac(sk, x, y + bh, x + bw, y + bh + max(3, bh * 0.7))
        if below < 0.5:
            continue
        cands.append((x + bw / 2.0, y + bh / 2.0, float(bw), float(bh), float(a), below))
    best = None
    for A in cands:
        for B in cands:
            if A[0] >= B[0]:
                continue
            dx = B[0] - A[0]
            dy = abs(B[1] - A[1])
            ew = (A[2] + B[2]) / 2.0
            eh = (A[3] + B[3]) / 2.0
            if dy > 0.3 * dx or dx < 1.2 * ew or dx > 4.5 * ew:
                continue
            ar = max(A[4], B[4]) / max(1.0, min(A[4], B[4]))
            if ar > 3.0:
                continue
            bridge = _frac(sk, A[0] + A[2] * 0.5, min(A[1], B[1]), B[0] - B[2] * 0.5, max(A[1], B[1]) + eh * 0.6)
            cheek = _frac(sk, A[0] - ew * 0.2, max(A[1], B[1]) + eh * 0.8, B[0] + ew * 0.2,
                          max(A[1], B[1]) + eh * 0.8 + dx * 0.5)
            if bridge < 0.45 or cheek < 0.55:
                continue
            sc = bridge * cheek * min(A[5], B[5]) / math.sqrt(ar) * (1.0 - dy / dx) * math.sqrt(ew * eh)
            if best is None or sc > best[0]:
                best = (sc, A, B)
    if best is None:
        return None
    return {"eyes": [best[1][:4], best[2][:4]], "score": float(best[0])}


def detect_human_face(bgr: np.ndarray) -> dict[str, Any] | None:
    """Real-human seed: OpenCV Haar frontal face (+ Haar eyes when visible, else anthropometric eye spots)."""
    try:
        base = Path(cv2.data.haarcascades)
        face_det = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        eye_det = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
    except Exception:
        return None
    if face_det.empty():
        return None
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    w = bgr.shape[1]
    faces = face_det.detectMultiScale(gray, 1.1, 5, minSize=(int(w * 0.06), int(w * 0.06)))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = (int(t) for t in max(faces, key=lambda b: b[2] * b[3]))
    ew, eh = fw * 0.22, fw * 0.12
    eyes = [(fx + fw * 0.30, fy + fh * 0.40, ew, eh), (fx + fw * 0.70, fy + fh * 0.40, ew, eh)]
    if not eye_det.empty():
        roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        found = eye_det.detectMultiScale(roi, 1.1, 6, minSize=(int(fw * 0.12), int(fw * 0.12)))
        if len(found) >= 2:
            found = sorted(found, key=lambda b: -b[2] * b[3])[:2]
            found = sorted(found, key=lambda b: b[0])
            pair = [(fx + ex + ewd / 2.0, fy + ey + ehd / 2.0, ewd * 0.8, ehd * 0.5) for ex, ey, ewd, ehd in found]
            if pair[1][0] - pair[0][0] > fw * 0.25:
                eyes = pair
    return {"eyes": eyes, "score": 1.0, "box": (fx, fy, fw, fh)}


# ------------------------------------------------------------------------------------------------ lock
class SimilarityLock:
    """Drift-free similarity (x, y, scale, roll) lock of a reference patch.

    predict: KLT previous frame -> current; refine: warp the current frame back into reference space and match it
    against the reference itself, so the estimate never integrates error."""

    def __init__(self, ref_gray: np.ndarray, rect: tuple[float, float, float, float],
                 exclude: tuple[float, float, float, float] | None = None, max_pts: int = 160) -> None:
        self.ref = ref_gray
        h, w = ref_gray.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        mask = np.zeros_like(ref_gray)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        if exclude is not None:
            ex0, ey0, ex1, ey1 = (int(round(v)) for v in exclude)
            mask[max(0, ey0):max(0, ey1), max(0, ex0):max(0, ex1)] = 0
        span = max(8.0, min(x1 - x0, y1 - y0))
        pts = cv2.goodFeaturesToTrack(ref_gray, max_pts, 0.01, max(2.0, span / 28.0), mask=mask, blockSize=5)
        self.pts = pts.reshape(-1, 2).astype(np.float32) if pts is not None else np.zeros((0, 2), np.float32)
        self.ok = len(self.pts) >= 8
        self.last_inliers = 0
        self.last_residual = 0.0

    def _fit(self, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray | None, int]:
        if len(src) < 6:
            return None, 0
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                             maxIters=800, confidence=0.995)
        if M is None or inl is None:
            return None, 0
        return M.astype(np.float64), int(inl.sum())

    def refine(self, cur_gray: np.ndarray, M_guess: np.ndarray) -> tuple[np.ndarray, int]:
        """One reference-relative correction of M_guess (ref -> cur)."""
        h, w = self.ref.shape[:2]
        warped = cv2.warpAffine(cur_gray, M_guess, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.ref, warped, self.pts, None, **_LK)
        if p1 is None:
            return M_guess, 0
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(warped, self.ref, p1, None, **_LK)
        good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
        if p0b is not None:
            good &= np.linalg.norm(p0b - self.pts, axis=1) < 1.0
        C, n_in = self._fit(self.pts[good], p1[good])
        if C is None:
            return M_guess, 0
        sc, _rot = _sim_params(C)
        if not (0.8 < sc < 1.25) or abs(C[0, 2]) > w * 0.25 or abs(C[1, 2]) > h * 0.25:
            return M_guess, 0
        res = np.linalg.norm(p1[good] - (self.pts[good] @ C[:, :2].T + C[:, 2]), axis=1)
        self.last_residual = float(np.median(res)) if len(res) else 0.0
        return _compose(M_guess, C), n_in

    def track(self, prev_gray: np.ndarray, cur_gray: np.ndarray, M_prev: np.ndarray) -> tuple[np.ndarray, bool]:
        P_prev = (self.pts @ M_prev[:, :2].T + M_prev[:, 2]).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, P_prev, None, **_LK)
        M_pred = M_prev
        if p1 is not None:
            good = st.reshape(-1) == 1
            Mp, _n = self._fit(self.pts[good], p1[good])
            if Mp is not None:
                M_pred = Mp
        M, n_in = self.refine(cur_gray, M_pred)
        if n_in >= 8:
            M2, n2 = self.refine(cur_gray, M)          # second pass: sub-pixel convergence
            if n2 >= n_in * 0.8:
                M, n_in = M2, n2
        self.last_inliers = n_in
        ok = n_in >= max(8, int(0.2 * len(self.pts)))
        return (M if ok else M_pred), ok


# ------------------------------------------------------------------------------------------------ skin model
class SkinModel:
    """Subject skin colour learned from the reference cheeks (Lab mean / spread)."""

    def __init__(self, bgr: np.ndarray, region: tuple[float, float, float, float]) -> None:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in region)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        sel = np.zeros((h, w), bool)
        sel[y0:y1, x0:x1] = True
        prior = _generic_skin(bgr).astype(bool)
        pick = sel & prior
        if pick.sum() < 30:
            pick = sel
        px = lab[pick.reshape(-1)]
        if len(px) < 10:
            px = lab
        self.mean = np.median(px, axis=0)
        mad = np.median(np.abs(px - self.mean), axis=0) * 1.4826
        self.std = np.maximum(mad, np.array([9.0, 4.0, 4.0], np.float32))

    def mask(self, bgr: np.ndarray, k: float = 3.2) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = ((lab - self.mean) / self.std)
        d2 = d[..., 0] ** 2 * 0.35 + d[..., 1] ** 2 + d[..., 2] ** 2       # forgiving on shading (L)
        return (d2 < k * k).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ analysis
def _open_video(video_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    return cap, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)


def analyze_video(video_path: Path, max_seconds: float = MAX_SECONDS) -> dict[str, Any]:
    """Track a clip. Returns per-source-frame tracks, overlay geometry and tracker metrics."""
    cap, fps, _total = _open_video(Path(video_path))
    max_frames = int(round(fps * max_seconds))
    frames_small: list[np.ndarray] = []        # 1/4-res masks for the arm pass (static background)
    hand_cands: list[list[tuple[float, float, float]]] = []   # per frame: (hand x, hand top y, sharpness)
    edge_ref = 1.0                             # the subject's own edge sharpness (seed face)
    geo: list[dict[str, Any] | None] = []

    mode = "fallback_center"
    seed = None
    face_lock: SimilarityLock | None = None
    torso_lock: SimilarityLock | None = None
    shoulder_pts = np.zeros((0, 2), np.float32)
    skin: SkinModel | None = None
    M_f = M_t = None
    prev_gray = None
    ref_bgr = None
    relocks = 0
    lost = 0
    inl_hist: list[int] = []
    res_hist: list[float] = []

    raw: dict[str, list[float]] = {k: [] for k in (
        "eyeL", "eyeR", "lookLx", "lookLy", "lookRx", "lookRy", "browL", "browR", "asym", "cx", "cy", "scale",
        "roll", "tcx", "tcy", "tscale", "troll", "neck_dx", "neck_dy", "shrug", "locked")}

    idx = 0
    AH = None
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        AH = int(round(h0 * ANALYSIS_W / float(w0)))
        bgr = cv2.resize(frame, (ANALYSIS_W, AH), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ---- seed (first frames until a face is found; then the reference never changes)
        if seed is None and (idx < fps * 3.0 or idx % 6 == 0):
            det = detect_anime_eye_pair(bgr)
            m = "anime_klt"
            if det is None:
                det = detect_human_face(bgr)
                m = "human_klt"
            if det is not None:
                (lx, ly, lw, lh), (rx, ry, rw, rh) = det["eyes"]
                d = math.hypot(rx - lx, ry - ly)
                ew = max((lw + rw) / 2.0, d * 0.25)
                eh = max((lh + rh) / 2.0, ew * 0.45)
                mid = ((lx + rx) / 2.0, (ly + ry) / 2.0)
                seed = {"eyes": [(lx, ly), (rx, ry)], "d": d, "ew": ew, "eh": eh, "mid": mid, "frame": idx}
                mode = m
                ref_bgr = bgr.copy()
                gm0 = _grad_mag(gray)
                fy0_, fy1_ = max(0, int(mid[1] - d)), max(1, int(mid[1] + 1.5 * d))
                fx0_, fx1_ = max(0, int(mid[0] - 1.2 * d)), max(1, int(mid[0] + 1.2 * d))
                edge_ref = max(1.0, float(np.percentile(gm0[fy0_:fy1_, fx0_:fx1_], 95)))
                face_rect = (mid[0] - 1.25 * d, mid[1] - 1.0 * d, mid[0] + 1.25 * d, mid[1] + 1.55 * d)
                face_lock = SimilarityLock(gray, face_rect)
                skin = SkinModel(bgr, (mid[0] - 0.75 * d, mid[1] + 0.35 * d, mid[0] + 0.75 * d, mid[1] + 0.75 * d))
                torso_rect = (mid[0] - 1.6 * d, mid[1] + 2.2 * d, mid[0] + 1.6 * d, mid[1] + 4.4 * d)
                if torso_rect[1] < AH - 0.6 * d:
                    torso_lock = SimilarityLock(gray, torso_rect)
                    if not torso_lock.ok:
                        torso_lock = None
                # shoulder band points (outside the head / neck column), tracked in torso space
                sm = np.zeros_like(gray)
                y0, y1 = int(mid[1] + 1.45 * d), int(mid[1] + 2.4 * d)
                for sx0, sx1 in ((mid[0] - 2.2 * d, mid[0] - 0.65 * d), (mid[0] + 0.65 * d, mid[0] + 2.2 * d)):
                    sm[max(0, y0):max(0, y1), max(0, int(sx0)):max(0, int(sx1))] = 255
                sp = cv2.goodFeaturesToTrack(gray, 80, 0.01, max(2.0, d / 10.0), mask=sm, blockSize=5)
                shoulder_pts = sp.reshape(-1, 2).astype(np.float32) if sp is not None else shoulder_pts
                M_f = np.array([[1.0, 0, 0], [0, 1.0, 0]])
                M_t = M_f.copy()
                prev_gray = gray

        if seed is None or face_lock is None:
            for k in raw:
                raw[k].append(float("nan"))
            geo.append(None)
            frames_small.append(np.zeros((AH // 4, ANALYSIS_W // 4), np.uint8))
            hand_cands.append([])
            idx += 1
            continue

        d, ew, eh = seed["d"], seed["ew"], seed["eh"]
        mx, my = seed["mid"]

        # ---- face lock
        if idx != seed["frame"]:
            M_new, okf = face_lock.track(prev_gray, gray, M_f)
            cur_inl = face_lock.last_inliers if okf else 0
            if (not okf) or (idx % 15 == 0):           # dual-eye re-lock (gross failure guard)
                det = detect_anime_eye_pair(bgr) if mode == "anime_klt" else detect_human_face(bgr)
                if det is not None:
                    dst = np.array([det["eyes"][0][:2], det["eyes"][1][:2]])
                    pred = np.array([_apply(M_new, *seed["eyes"][0]), _apply(M_new, *seed["eyes"][1])])
                    err = float(np.max(np.linalg.norm(pred - dst, axis=1)))
                    if (not okf) or err > 0.3 * d * _sim_params(M_new)[0]:
                        M_e = _sim_from_pairs(np.array(seed["eyes"]), dst)
                        M_r, n_r = face_lock.refine(gray, M_e)
                        if n_r >= 8:
                            M_r2, n_r2 = face_lock.refine(gray, M_r)
                            if n_r2 >= n_r:
                                M_r, n_r = M_r2, n_r2
                        # only a re-lock the reference itself confirms better than the running lock may win
                        if n_r >= max(12, int(0.2 * len(face_lock.pts))) and n_r > 1.15 * cur_inl:
                            M_new, okf = M_r, True
                            face_lock.last_inliers = n_r
                            relocks += 1
                face_lock.last_inliers = max(face_lock.last_inliers, cur_inl)
            if not okf:
                lost += 1
            M_f = M_new
            if torso_lock is not None:
                M_t, _okt = torso_lock.track(prev_gray, gray, M_t)   # miss -> KLT prediction, re-refined next frame
        inl_hist.append(face_lock.last_inliers if idx != seed["frame"] else len(face_lock.pts))
        res_hist.append(face_lock.last_residual)
        prev_gray = gray

        scale, rot = _sim_params(M_f)
        fcx, fcy = _apply(M_f, mx, my + 0.3 * d)
        raw["cx"].append(fcx)
        raw["cy"].append(fcy)
        raw["scale"].append(scale)
        raw["roll"].append(rot)
        raw["locked"].append(1.0)

        # ---- canonical (reference-space) face measurements
        canon = cv2.warpAffine(bgr, M_f, (ANALYSIS_W, AH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
        c_hsv_v = cv2.cvtColor(canon, cv2.COLOR_BGR2HSV)[..., 2]
        c_gray = cv2.cvtColor(canon, cv2.COLOR_BGR2GRAY)
        c_skin = skin.mask(canon)
        eye_vals = []
        for (ex, ey) in seed["eyes"]:
            x0, x1 = int(ex - 0.85 * ew), int(ex + 0.85 * ew)
            y0, y1 = int(ey - 1.05 * eh), int(ey + 1.05 * eh)
            x0, y0 = max(0, x0), max(0, y0)
            vroi = c_hsv_v[y0:y1, x0:x1]
            sroi = c_skin[y0:y1, x0:x1]
            if vroi.size == 0:
                eye_vals.append((0.0, float("nan"), float("nan"), float("nan"), float("nan")))
                continue
            m = ((sroi == 0) & (vroi < 175))
            area = float(m.sum()) / m.size
            rows = float((m.mean(axis=1) >= 0.22).sum()) / max(1, m.shape[0])
            open_raw = (rows ** 0.7) * (area ** 0.3)
            lx = ly = ecx = float("nan")
            if m.sum() >= 6:
                my_, mx_ = np.nonzero(m)
                ecx = float(mx_.mean()) + x0                       # whole-eye centre (lashes + white + iris)
                ecy = float(my_.mean()) + y0
                thr = np.percentile(vroi[m], 35)
                dm = m & (vroi <= thr)
                yy, xx = np.nonzero(dm)
                if len(xx) >= 3:
                    # iris relative to the eye's own centre: a head turn moves both, so it never fakes a glance
                    lx = (float(xx.mean()) + x0 - ecx) / (0.35 * ew)
                    ly = (float(yy.mean()) + y0 - ecy) / (0.35 * eh)
            # brow band: strongest horizontal-edge row centroid above the eye
            by0, by1 = int(ey - 2.1 * eh), int(ey - 0.85 * eh)
            band = c_gray[max(0, by0):max(0, by1), x0:x1].astype(np.float32)
            brow = float("nan")
            if band.shape[0] >= 4 and band.shape[1] >= 4:
                gy = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1) ** 2
                if gy.sum() > 1e-6:
                    brow = float((gy * np.arange(len(gy))).sum() / gy.sum()) + max(0, by0) - ey
            eye_vals.append((open_raw, lx, ly, brow, ecx))
        # screen-left eye = the character's RIGHT eye (anatomical, no mirroring)
        (r_open, r_lx, r_ly, r_brow, r_ecx), (l_open, l_lx, l_ly, l_brow, l_ecx) = eye_vals
        raw["eyeR"].append(r_open)
        raw["eyeL"].append(l_open)
        raw["lookRx"].append(r_lx)
        raw["lookRy"].append(r_ly)
        raw["lookLx"].append(l_lx)
        raw["lookLy"].append(l_ly)
        raw["browR"].append(r_brow)
        raw["browL"].append(l_brow)

        # yaw cue: inner features (eye centres) vs the face-skin silhouette centroid in reference space. Turning
        # toward screen-right moves the features right of the cheek silhouette, whichever part the lock follows.
        cy0, cy1 = max(0, int(my + 0.2 * d)), max(1, int(my + 0.6 * d))
        xl0, xr1 = max(0, int(mx - 1.3 * d)), max(1, int(mx + 1.3 * d))
        band = c_skin[cy0:cy1, xl0:xr1].astype(np.float64)
        yaw_f = float("nan")
        if band.sum() > 10:
            skin_cx = float((band.sum(axis=0) * (np.arange(band.shape[1]) + xl0)).sum() / band.sum())
            eyes_cx = np.nanmean([l_ecx, r_ecx]) if np.isfinite([l_ecx, r_ecx]).any() else mx
            yaw_f = (float(eyes_cx) - skin_cx) / d
        raw["asym"].append(yaw_f)

        # ---- torso: body roll, neck nod / turn, shoulder shrug
        if torso_lock is not None:
            ts, tr = _sim_params(M_t)
            tcx, tcy = _apply(M_t, mx, my + 3.3 * d)
            px, py = _apply(M_t, mx, my + 0.3 * d)      # where the face would be if only the torso moved
            raw["tcx"].append(tcx)
            raw["tcy"].append(tcy)
            raw["tscale"].append(ts)
            raw["troll"].append(tr)
            raw["neck_dx"].append((fcx - px) / (d * ts))
            raw["neck_dy"].append((fcy - py) / (d * ts))
            shr = float("nan")
            if len(shoulder_pts) >= 6:
                tw = cv2.warpAffine(gray, M_t, (ANALYSIS_W, AH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(torso_lock.ref, tw, shoulder_pts, None, **_LK)
                if p1 is not None:
                    g = st.reshape(-1) == 1
                    side_l = shoulder_pts[:, 0] < mx
                    dys = []
                    for sel in (g & side_l, g & ~side_l):
                        if sel.sum() >= 3:
                            dys.append(-float(np.median(p1[sel, 1] - shoulder_pts[sel, 1])) / d)
                    if len(dys) == 2:
                        # a shrug is bilateral: only the motion both shoulders agree on counts
                        shr = math.copysign(min(abs(dys[0]), abs(dys[1])), dys[0]) if dys[0] * dys[1] > 0 else 0.0
            raw["shrug"].append(shr)
        else:
            for k in ("tcx", "tcy", "tscale", "troll", "neck_dx", "neck_dy", "shrug"):
                raw[k].append(float("nan"))

        # ---- arm mask (full frame): skin outside head / hair oval and outside the chest column
        sk_full = skin.mask(bgr, 3.0)
        hcx, hcy = _apply(M_f, mx, my + 0.1 * d)
        cv2.ellipse(sk_full, (int(hcx), int(hcy)), (int(1.35 * d * scale), int(1.75 * d * scale)),
                    -rot, 0, 360, 0, -1)
        ncx, ncy = _apply(M_f, mx, my + 1.2 * d)
        cv2.rectangle(sk_full, (int(ncx - 0.75 * d * scale), int(ncy)), (int(ncx + 0.75 * d * scale), AH), 0, -1)
        frames_small.append(cv2.resize(sk_full, (ANALYSIS_W // 4, AH // 4), interpolation=cv2.INTER_AREA))
        # hand candidates: skin blobs with the subject's crisp outline (blurred / bokeh background lamps fail)
        cands = []
        nn, lab, st, cen = cv2.connectedComponentsWithStats(sk_full, 8)
        min_a, max_a = (0.16 * d * scale) ** 2, (1.4 * d * scale) ** 2
        gm = None
        for j in range(1, nn):
            x, y, bw, bh, a = (int(t) for t in st[j])
            if a < min_a or a > max_a:
                continue
            if gm is None:
                gm = _grad_mag(gray)
            x0, y0 = max(0, x - 3), max(0, y - 3)
            comp = (lab[y0:y + bh + 3, x0:x + bw + 3] == j).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((5, 5), np.uint8)) - cv2.erode(comp, np.ones((3, 3), np.uint8))
            vals = gm[y0:y + bh + 3, x0:x + bw + 3][ring > 0]
            sharp = float(np.percentile(vals, 50)) / edge_ref if vals.size else 0.0
            cands.append((float(cen[j][0]), float(y), sharp))
        hand_cands.append(cands)

        geo.append({
            "M": M_f.tolist(),
            "eyes": [_apply(M_f, *seed["eyes"][0]), _apply(M_f, *seed["eyes"][1])],
            "face": [_apply(M_f, mx + ax * d, my + ay * d) for ax, ay in ((-1.1, -0.9), (1.1, -0.9), (1.1, 1.45), (-1.1, 1.45))],
            "shoulderY": _apply(M_f, mx, my + 1.5 * d)[1],
            "scale": scale,
        })
        idx += 1
    cap.release()

    n = len(geo)
    if n < 4:
        raise RuntimeError("Video too short (need at least 4 frames).")
    first = next((i for i, g in enumerate(geo) if g is not None), None)
    if first is None:
        return _fallback_center(n, fps)

    def arr(k: str) -> np.ndarray:
        a = np.asarray(raw[k], dtype=np.float64)
        ok = np.isfinite(a)
        if not ok.any():
            return np.zeros(n)
        idxs = np.arange(n)
        return np.interp(idxs, idxs[ok], a[ok])            # hold / backfill gaps (pre-lock frames, misses)

    d = seed["d"]
    sig = fps / 30.0

    # ---- eyes: openness vs the clip's own open-eye level; adaptive closed level (no blinks -> no false blinks)
    def eye_track(a: np.ndarray) -> np.ndarray:
        hi = max(1e-4, float(np.percentile(a, 90)))
        r = a / hi
        lo = min(0.55, max(0.2, float(np.percentile(r, 2))))
        top = max(lo + 0.15, 0.85)
        return np.clip((r - lo) / (top - lo), 0.0, 1.0)

    eye_l = _smooth_1d(eye_track(arr("eyeL")), 0.6 * sig)
    eye_r = _smooth_1d(eye_track(arr("eyeR")), 0.6 * sig)

    # gaze: only trust samples while the eye is open; hold otherwise
    def gaze(a_l: str, a_r: str, el: np.ndarray, er: np.ndarray) -> np.ndarray:
        gl = np.asarray(raw[a_l], np.float64)
        gr = np.asarray(raw[a_r], np.float64)
        gl[~(el > 0.6)] = np.nan
        gr[~(er > 0.6)] = np.nan
        g = np.nanmean(np.vstack([gl, gr]), axis=0) if np.isfinite(gl).any() or np.isfinite(gr).any() else np.zeros(n)
        ok = np.isfinite(g)
        if not ok.any():
            return np.zeros(n)
        g = np.interp(np.arange(n), np.arange(n)[ok], g[ok])
        g = g - np.median(g)
        return np.clip(_smooth_1d(g, 2.0 * sig) * 1.3, -1.0, 1.0)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        look_x = gaze("lookLx", "lookRx", eye_l, eye_r)
        look_y = -gaze("lookLy", "lookRy", eye_l, eye_r)      # +Y = looking up

    def brow_track(k: str) -> np.ndarray:
        b = arr(k)
        b = -(b - np.median(b)) / max(1.0, 0.35 * seed["eh"])  # up = positive
        b = np.sign(b) * np.maximum(0.0, np.abs(b) - 0.15)       # dead zone against edge noise
        return np.clip(_smooth_1d(b, 2.5 * sig), -1.0, 1.0)

    brow_l = brow_track("browL")
    brow_r = brow_track("browR")

    # ---- head pose
    roll = arr("roll")
    ang_z = np.clip(-_smooth_1d(roll - np.median(roll), 1.2 * sig), -25.0, 25.0)   # +Z = head top to screen-left
    has_torso = np.isfinite(np.asarray(raw["tcx"], np.float64)).any()
    asym = arr("asym")
    yaw = (asym - np.median(asym)) * 95.0
    if has_torso:
        ndx = arr("neck_dx")
        ndy = arr("neck_dy")
        yaw = 0.7 * yaw + 0.3 * (ndx - np.median(ndx)) * 90.0
        # neck nod vs the torso; slow drift (lean-in perspective) is mostly removed, held nods survive at 35%
        ndy_hp = ndy - _baseline(ndy, 3.0 * fps)
        pitch = -(ndy_hp + 0.35 * (ndy - ndy_hp - np.median(ndy))) * 55.0
    else:
        cy = arr("cy")
        pitch = -(cy - np.median(cy)) / d * 20.0
    ang_x = np.clip(_smooth_1d(yaw, 1.5 * sig), -30.0, 30.0)
    ang_y = np.clip(_smooth_1d(pitch, 1.5 * sig), -24.0, 24.0)

    # ---- stage: whole-body position / scale relative to the clip's median rest pose
    cx, cy, sc = arr("cx"), arr("cy"), arr("scale")
    if has_torso:
        bx = 0.5 * (cx + arr("tcx"))
        by = 0.5 * (cy + arr("tcy"))
    else:
        bx, by = cx, cy
    stage_x = np.clip(_smooth_1d((bx - np.median(bx)) / (2.5 * d), 1.0 * sig), -1.0, 1.0)
    stage_y = np.clip(_smooth_1d(-(by - np.median(by)) / (2.0 * d), 1.0 * sig), -1.0, 1.0)
    stage_zoom = np.clip(_smooth_1d(np.log(sc / np.median(sc)) / math.log(1.6), 1.0 * sig), -1.0, 1.0)

    # ---- body
    body_x = np.clip(0.35 * ang_x, -10.0, 10.0)
    if has_torso:
        tr = arr("troll")
        body_z = np.clip(-_smooth_1d(tr - np.median(tr), 2.0 * sig) + 0.3 * ang_z, -10.0, 10.0)
        shr = arr("shrug")
        shr = (shr - _baseline(shr, 2.0 * fps)) / 0.15      # a shrug is a transient: high-pass out lean drift
        shr = np.sign(shr) * np.maximum(0.0, np.abs(shr) - 0.2)
        shrug = np.clip(_smooth_1d(shr, 2.5 * sig), -1.0, 1.0)
    else:
        body_z = np.clip(0.45 * ang_z, -7.0, 7.0)
        shrug = np.zeros(n)

    # ---- arms (post pass): remove static skin-coloured background, then hand elevation per side
    stack = np.stack([m if m.shape == frames_small[first].shape else np.zeros_like(frames_small[first])
                      for m in frames_small]).astype(np.float32)
    static = (stack > 0.5).mean(axis=0) > 0.75
    arm_r_raw = np.zeros(n)
    arm_l_raw = np.zeros(n)
    hand_r_x = np.full(n, np.nan)
    hand_l_x = np.full(n, np.nan)
    hands_geo: list[list[tuple[float, float]]] = [[] for _ in range(n)]
    for i in range(n):
        g = geo[i] if geo[i] is not None else geo[first]
        fsc = g["scale"]
        face_x = 0.5 * (g["eyes"][0][0] + g["eyes"][1][0])
        sh_y = g["shoulderY"]
        best = {"R": None, "L": None}
        for hx, top, sharp in hand_cands[i]:
            if sharp < 0.6:                               # soft-edged blob: out-of-focus background
                continue
            sy_, sx_ = min(static.shape[0] - 1, int(top / 4.0) + 1), min(static.shape[1] - 1, int(hx / 4.0))
            if static[sy_, sx_]:                          # skin-coloured background that never moves
                continue
            lat = abs(hx - face_x) / (d * fsc)
            if lat > 2.8 or top < sh_y - 2.6 * d * fsc:   # outside arm reach: background, never a hand
                continue
            side = "R" if hx < face_x else "L"            # screen-left = the character's right arm
            if best[side] is None or top < best[side][1]:
                best[side] = (hx, top, lat)
        for side, out, hx_arr in (("R", arm_r_raw, hand_r_x), ("L", arm_l_raw, hand_l_x)):
            b = best[side]
            if b is None:
                continue
            elev = (sh_y + 1.0 * d * fsc - b[1]) / (1.2 * d * fsc)
            # lateral gate: a hand resting on the chest / chin is not a raised wave arm
            out[i] = elev * min(1.0, max(0.0, (b[2] - 1.35) / 0.45))
            hx_arr[i] = (b[0] - face_x) / (d * fsc)
            hands_geo[i].append((b[0], b[1]))
    # a raise is a held gesture: brief pops (a quick wave) are held ~0.35 s each side so the rig's raised arm lands
    hold = max(1, int(round(0.35 * fps)))

    def held(a: np.ndarray) -> np.ndarray:
        a = np.clip(a, 0.0, 1.0)
        pad = np.pad(a, hold, mode="edge")
        return np.lib.stride_tricks.sliding_window_view(pad, 2 * hold + 1).max(axis=1)

    arm_r = np.clip(_smooth_1d(held(arm_r_raw), 2.0 * sig), 0.0, 1.0)
    arm_l = np.clip(_smooth_1d(held(arm_l_raw), 2.0 * sig), 0.0, 1.0)

    def gesture(hx: np.ndarray, raise_: np.ndarray) -> np.ndarray:
        up = (raise_ > 0.35) & np.isfinite(hx)
        out = np.zeros(n)
        if up.sum() >= 3:
            ref = float(np.median(hx[up]))
            out[up] = (hx[up] - ref) / 0.6
        return np.clip(_smooth_1d(out, 1.5 * sig), -1.0, 1.0) * np.clip((raise_ - 0.25) / 0.2, 0.0, 1.0)

    arm_r_g = gesture(hand_r_x, arm_r)
    arm_l_g = -gesture(hand_l_x, arm_l)     # mirrored so +gesture = outward wave on both arms

    for i in range(n):
        if geo[i] is not None:
            geo[i]["hands"] = hands_geo[i]

    tracks = {
        "AngleX": ang_x, "AngleY": ang_y, "AngleZ": ang_z, "BodyX": body_x, "BodyZ": body_z,
        "EyeOpen": 0.5 * (eye_l + eye_r), "ArmR": arm_r, "ArmL": arm_l,
        "EyeLOpen": eye_l, "EyeROpen": eye_r, "EyeLookX": look_x, "EyeLookY": look_y,
        "BrowLY": brow_l, "BrowRY": brow_r, "StageX": stage_x, "StageY": stage_y, "StageZoom": stage_zoom,
        "ShoulderShrug": shrug, "ArmRGesture": arm_r_g, "ArmLGesture": arm_l_g,
    }
    locked = np.isfinite(np.asarray(raw["locked"], np.float64))
    metrics = {
        "frames": n, "fps": fps, "lockRate": round(float(locked.mean()), 4), "firstLockFrame": int(first),
        "relocks": relocks, "lostFrames": lost, "torso": bool(has_torso),
        "meanInliers": round(float(np.mean(inl_hist)) if inl_hist else 0.0, 1),
        "medianRefResidualPx": round(float(np.median(res_hist)) if res_hist else 0.0, 3),
        "eyeDistPx": round(float(d), 1),
    }
    return {"fps": fps, "frames": n, "tracks": tracks, "geometry": geo, "mode": mode, "metrics": metrics,
            "analysisSize": (ANALYSIS_W, AH)}


def _fallback_center(n: int, fps: float) -> dict[str, Any]:
    """No face anywhere: emit a neutral clip (eyes open, rest pose) rather than inventing motion from noise."""
    z = np.zeros(n)
    tracks = {k: z.copy() for k in TRACK_KEYS}
    for k in ("EyeOpen", "EyeLOpen", "EyeROpen"):
        tracks[k] = np.ones(n)
    return {"fps": fps, "frames": n, "tracks": tracks, "geometry": [None] * n, "mode": "fallback_center",
            "metrics": {"frames": n, "fps": fps, "lockRate": 0.0, "firstLockFrame": -1}, "analysisSize": None}


def extract_motion_from_video(video_path: Path, out_dir: Path, name: str | None = None) -> dict:
    """Analyze a video clip and save a 60 FPS general-motion clip JSON (no audio / mouth: live lip-sync owns it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = (name or Path(video_path).stem).strip().replace(" ", "_")
    safe_id = "".join(c for c in clip_id if c.isalnum() or c in ("-", "_"))[:48] or f"clip_{int(time.time())}"

    res = analyze_video(Path(video_path))
    n = res["frames"]
    fps = res["fps"]
    dur = n / float(fps)
    n60 = max(6, int(round(dur * 60.0)))
    t_src = np.arange(n) / float(fps)
    t_60 = np.arange(n60) / 60.0

    def rs(a: np.ndarray) -> list[float]:
        return [round(float(v), 4) for v in np.interp(t_60, t_src, a)]

    tracks = {k: rs(res["tracks"][k]) for k in TRACK_KEYS}
    clip_data = {
        "id": safe_id,
        "name": name or Path(video_path).stem,
        "fps": 60,
        "frames": n60,
        "duration": round(dur, 2),
        "hasAudio": False,
        "audioFile": "",
        "trackerMode": res["mode"],
        "trackCount": len(tracks),
        "trackerMetrics": res["metrics"],
        "tracks": tracks,
    }
    (out_dir / f"{safe_id}.json").write_text(json.dumps(clip_data), encoding="utf-8")
    return {k: v for k, v in clip_data.items() if k != "tracks"}
