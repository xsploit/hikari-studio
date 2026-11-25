"""Real-Time 60 FPS Multi-Engine Lip-Sync Suite -> VBridger 6-Axis Mouth + Editable Visemes.

Engines (config["lipsync"]["engine"]):
  hybrid         (default) 12-MFCC uLipSync scores fused with IPA F1/F2 formant scores, plus an
                 onset / bilabial-closure (M/B/P) and fricative (F/V) detector.
  ulipsync_mfcc  Pure 12-MFCC uLipSync / wLipSync pipeline (mel filterbank -> dB -> DCT ->
                 standardization -> cosine / L1 / L2 phoneme scoring).
  ipa_formant    F1/F2 spectral-centroid female-voice vowel triangle (the original Hikari engine).
  amplitude      Classic VTube Studio-style volume mouth flap + spectral-tilt smile/round.

Sources: live microphone (sounddevice 48 kHz), TTS / clip playback synced to the browser audio
clock, Rhubarb cue files (dialogue.cues.json), and Fish Audio full-duplex 48 kHz PCM.

MFCC algorithm notice: the MFCC analysis and scoring math is a NumPy port of the MIT-licensed
uLipSync (Copyright (c) 2021 hecomi) and wLipSync (Copyright (c) 2024 Noeri Huisman) algorithms,
as documented in Waifu Studio's LIPSYNC-NOTICES.txt. No upstream calibration data, audio or avatar
assets are included: the default calibration is synthesized at start-up from formant-filtered
harmonic vowel spectra.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import copy
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
import wave

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

DEFAULT_VISEMES: dict[str, dict[str, float]] = {
    "X": {"label": "Rest / Closed",  "open": 0.00, "form":  0.08, "funnel": 0.00, "x":  0.00, "shrug": 0.00, "tongue": 0.00},
    "A": {"label": "M / B / P Seal", "open": 0.00, "form": -0.14, "funnel": 0.00, "x":  0.00, "shrug": 0.22, "tongue": 0.00},
    "B": {"label": "K / S / T / EE", "open": 0.24, "form":  0.42, "funnel": 0.00, "x":  0.06, "shrug": 0.06, "tongue": 0.00},
    "C": {"label": "EH / AE",        "open": 0.54, "form":  0.34, "funnel": 0.06, "x":  0.03, "shrug": 0.00, "tongue": 0.08},
    "D": {"label": "AA / AH Wide",   "open": 0.92, "form":  0.14, "funnel": 0.14, "x":  0.00, "shrug": 0.00, "tongue": 0.22},
    "E": {"label": "AO / OH Round",  "open": 0.64, "form": -0.52, "funnel": 0.72, "x": -0.02, "shrug": 0.00, "tongue": 0.08},
    "F": {"label": "UW / OO Pucker", "open": 0.30, "form": -0.86, "funnel": 1.00, "x": -0.04, "shrug": 0.08, "tongue": 0.00},
    "G": {"label": "F / V Lip Tuck", "open": 0.16, "form":  0.12, "funnel": 0.18, "x":  0.00, "shrug": 0.78, "tongue": 0.00},
    "H": {"label": "L / TH Tongue",  "open": 0.44, "form":  0.18, "funnel": 0.08, "x":  0.02, "shrug": 0.04, "tongue": 0.72},
}

LIPSYNC_ENGINES = ("hybrid", "ulipsync_mfcc", "ipa_formant", "amplitude")
MFCC_SCORING = ("cosine", "l1", "l2")

# VTube Studio-style output mapping per bound Inochi2D mouth channel (applied in core.rig_controller).
#   t = clamp((v - inMin) / (inMax - inMin)) -> signed/unsigned power `curve` -> lerp(outMin, outMax)
#   -> exponential `smooth` (fraction of the previous value kept per 60 FPS frame; 0 = raw).
DEFAULT_PARAM_MAP: dict[str, dict[str, Any]] = {
    "jawOpen":     {"enabled": True, "inMin": 0.02, "inMax": 0.95, "outMin": 0.0,  "outMax": 1.0,  "smooth": 0.14, "curve": 0.92},
    "mouthOpen":   {"enabled": True, "inMin": 0.02, "inMax": 0.95, "outMin": 0.0,  "outMax": 1.0,  "smooth": 0.12, "curve": 0.95},
    "mouthForm":   {"enabled": True, "inMin": -1.0, "inMax": 1.0,  "outMin": -1.0, "outMax": 1.0,  "smooth": 0.16, "curve": 1.0},
    "mouthFunnel": {"enabled": True, "inMin": 0.05, "inMax": 0.95, "outMin": 0.0,  "outMax": 1.0,  "smooth": 0.14, "curve": 0.90},
    "lipsPress":   {"enabled": True, "inMin": -1.0, "inMax": 1.0,  "outMin": -1.0, "outMax": 1.0,  "smooth": 0.10, "curve": 1.0},
    "mouthShrug":  {"enabled": True, "inMin": 0.05, "inMax": 0.95, "outMin": 0.0,  "outMax": 1.0,  "smooth": 0.15, "curve": 1.0},
    "mouthX":      {"enabled": True, "inMin": -1.0, "inMax": 1.0,  "outMin": -0.6, "outMax": 0.6,  "smooth": 0.20, "curve": 1.0},
    "tongueOut":   {"enabled": True, "inMin": 0.15, "inMax": 0.95, "outMin": 0.0,  "outMax": 0.85, "smooth": 0.18, "curve": 1.1},
}

DEFAULT_LIPSYNC_CFG: dict[str, Any] = {
    "gain": 1.45,
    "noiseGate": 0.012,
    "attackOmega": 30.0,
    "dampingZeta": 0.78,
    "smileBias": 0.14,
    "roundness": 1.12,
    "jawBounce": 1.02,
    "lipSealSharpness": 1.18,
    "microFlutter": 0.22,
    "speechToHeadNod": 0.80,
    "speechToBrows": 0.72,
    "speechToArmGesture": 0.85,
    "holdViseme": "",
    "micEnabled": False,
    # --- Multi-engine suite ---
    "engine": "hybrid",              # hybrid | ulipsync_mfcc | ipa_formant | amplitude
    "mfccScoring": "cosine",         # cosine | l1 | l2
    "coarticulation": 0.22,          # softmax temperature (0.08 sharp .. 0.60 blended)
    "paramMap": copy.deepcopy(DEFAULT_PARAM_MAP),
    "mfccCalibration": {},           # {"D": [[12 floats], ...], ...} live voice calibration (max 16 per viseme)
}


class Spring2nd:
    """Critically/under-damped 2nd-order spring for liquid-smooth anime motion."""
    def __init__(self, value: float = 0.0) -> None:
        self.y = float(value)
        self.v = 0.0

    def step(self, target: float, dt: float, omega: float, zeta: float) -> float:
        dt = max(1e-4, min(0.05, dt))
        # Sub-step twice for stability at high omega
        h = dt * 0.5
        for _ in range(2):
            acc = omega * omega * (target - self.y) - 2.0 * zeta * omega * self.v
            self.v += acc * h
            self.y += self.v * h
        return self.y


# ===========================================================================
# 12-MFCC uLipSync / wLipSync analysis (NumPy port of the algorithm)
# ===========================================================================

MFCC_NUM = 12
MFCC_TARGET_RATE = 16_000
MFCC_SAMPLE_COUNT = 1024           # 64 ms analysis frame at the 16 kHz profile rate
MFCC_MEL_CHANNELS = 30
MFCC_MAX_SAMPLES_PER_PHONEME = 16
_PRE_EMPHASIS = 0.97
_LOW_PASS_RANGE_HZ = 500.0

# uLipSync phoneme -> Hikari viseme code
MFCC_PHONEMES: dict[str, str] = {"A": "D", "I": "B", "U": "F", "E": "C", "O": "E", "N": "A", "FV": "G"}
VISEME_TO_PHONEME: dict[str, str] = {v: k for k, v in MFCC_PHONEMES.items()}
# Friendly calibration labels (UI pills / ARPAbet-ish) -> viseme code. Single letters are viseme codes.
FRIENDLY_VISEMES: dict[str, str] = {
    "AA": "D", "AH": "D", "EE": "B", "IY": "B", "II": "B", "OO": "F", "UW": "F", "UU": "F",
    "OH": "E", "AO": "E", "EH": "C", "AE": "C", "FV": "G", "MBP": "A", "MB": "A", "N": "A",
}


def resolve_viseme(label: str) -> str:
    """Accept a viseme code (D/B/F/C/E/G/A) or a friendly vowel label (AA/EE/OO/OH/EH/FV/MBP)."""
    code = str(label or "").strip().upper()
    if code in VISEME_TO_PHONEME:
        return code
    return FRIENDLY_VISEMES.get(code, code)
MFCC_VISEMES: tuple[str, ...] = tuple(MFCC_PHONEMES.values())


def _mel_filter_bank_matrix(n: int, sample_rate: float, mel_div: int) -> np.ndarray:
    """Algorithm.MelFilterBank as a dense (mel_div, n) weight matrix over |FFT| bins."""
    def to_mel(hz: float) -> float:
        return 1127.0 * math.log(hz / 700.0 + 1.0)

    def to_hz(mel: float) -> float:
        return 700.0 * (math.exp(mel / 1127.0) - 1.0)

    f_max = sample_rate / 2.0
    mel_max = to_mel(f_max)
    n_max = n // 2
    df = f_max / n_max
    d_mel = mel_max / (mel_div + 1)
    mat = np.zeros((mel_div, n), dtype=np.float64)
    for b in range(mel_div):
        f_begin = to_hz(d_mel * b)
        f_center = to_hz(d_mel * (b + 1))
        f_end = to_hz(d_mel * (b + 2))
        i_begin = math.ceil(f_begin / df)
        i_center = round(f_center / df)          # Python round() is round-half-even, as upstream
        i_end = min(math.floor(f_end / df), n_max)
        for i in range(max(0, i_begin + 1), i_end + 1):
            f = df * i
            a = (f - f_begin) / (f_center - f_begin) if i < i_center else (f_end - f) / (f_end - f_center)
            mat[b, i] = a / ((f_end - f_begin) * 0.5)
    return mat


class _MfccPlan:
    """Per-input-sample-rate tables (low-pass FIR, decimation indices, FFT/mel/DCT matrices)."""

    def __init__(self, rate: int) -> None:
        self.rate = int(rate)
        n = MFCC_SAMPLE_COUNT
        ratio = self.rate / MFCC_TARGET_RATE
        self.input_len = int(math.ceil(n * ratio))
        # DownSample: integer ratios skip, fractional ratios take floor(ratio * j) (as upstream).
        self.idx = np.minimum(np.floor(ratio * np.arange(n)).astype(np.int64), self.input_len - 1)
        # Algorithm.LowPassFilter windowless sinc taps. uLipSync adds the filtered signal to the
        # original; that behaviour is kept so templates and live frames share one pipeline.
        cutoff = (MFCC_TARGET_RATE / 2 - _LOW_PASS_RANGE_HZ) / self.rate
        rng = _LOW_PASS_RANGE_HZ / self.rate
        taps = int(round(3.1 / rng))
        if (taps + 1) % 2 == 0:
            taps += 1
        x = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
        ang = 2.0 * math.pi * cutoff * x
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(np.abs(ang) < 1e-12, 2.0 * cutoff, 2.0 * cutoff * np.sin(ang) / ang)
        self.taps = taps
        self.nfft = 1 << int(math.ceil(math.log2(self.input_len + taps - 1)))
        self.lp_fft = np.fft.rfft(b, self.nfft)
        self.hamming = 0.54 - 0.46 * np.cos(2.0 * math.pi * np.arange(n) / (n - 1))
        self.mel = _mel_filter_bank_matrix(n, float(MFCC_TARGET_RATE), MFCC_MEL_CHANNELS)
        a = math.pi / MFCC_MEL_CHANNELS
        j = np.arange(MFCC_MEL_CHANNELS) + 0.5
        self.dct = np.stack([np.cos(j * i * a) for i in range(1, MFCC_NUM + 1)])   # cepstrum[1..=12]


class MfccAnalyzer:
    """Stateless 12-MFCC extractor; plans are cached per input sample rate."""

    def __init__(self) -> None:
        self._plans: dict[int, _MfccPlan] = {}

    def plan(self, rate: int) -> _MfccPlan | None:
        rate = int(rate)
        if rate < MFCC_TARGET_RATE or rate > 384_000:
            return None          # below the profile rate uLipSync would analyse an unresampled frame
        p = self._plans.get(rate)
        if p is None:
            p = self._plans[rate] = _MfccPlan(rate)
        return p

    def window_len(self, rate: int) -> int:
        p = self.plan(rate)
        return p.input_len if p else 0

    def compute(self, frame: np.ndarray, rate: int) -> np.ndarray | None:
        """12 MFCCs of the newest `window_len(rate)` samples, or None for silence / bad input."""
        p = self.plan(rate)
        if p is None:
            return None
        x = np.asarray(frame, dtype=np.float64)
        if len(x) < p.input_len:
            x = np.pad(x, (p.input_len - len(x), 0))
        x = x[-p.input_len:]
        if not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) <= 0.0:
            return None
        # LowPassFilter (causal FIR via FFT convolution) + DownSample, signal + filtered as upstream
        lp = np.fft.irfft(np.fft.rfft(x, p.nfft) * p.lp_fft, p.nfft)[: p.input_len]
        data = x[p.idx] + lp[p.idx]
        # PreEmphasis(0.97)
        data[1:] = data[1:] - _PRE_EMPHASIS * data[:-1]
        # HammingWindow + Normalize(1)
        data *= p.hamming
        mx = float(np.max(np.abs(data)))
        if mx < 1e-30:
            return None
        data /= mx
        spectrum = np.abs(np.fft.fft(data))
        mel = p.mel @ spectrum
        with np.errstate(divide="ignore"):
            mel_db = 10.0 * np.log10(mel)
        if not np.all(np.isfinite(mel_db)):
            return None
        mfcc = p.dct @ mel_db
        return mfcc if np.all(np.isfinite(mfcc)) else None


class MfccProfile:
    """uLipSync profile: averaged phoneme templates + (optional) standardization statistics."""

    def __init__(self, samples: dict[str, list[np.ndarray]], use_standardization: bool = True) -> None:
        self.names = [k for k in MFCC_PHONEMES if samples.get(k)]
        if not self.names:
            raise ValueError("MFCC profile needs at least one phoneme")
        allv = np.array([v for k in self.names for v in samples[k]], dtype=np.float64)
        if use_standardization:
            self.means = allv.mean(axis=0)
            self.stds = np.sqrt(((allv - self.means) ** 2).mean(axis=0))
            self.stds = np.where(self.stds > 1e-9, self.stds, 1.0)
        else:
            self.means = np.zeros(MFCC_NUM)
            self.stds = np.ones(MFCC_NUM)
        self.templates = np.array([np.mean(samples[k], axis=0) for k in self.names])
        self.norm = (self.templates - self.means) / self.stds

    def score(self, mfcc: np.ndarray, method: str = "cosine") -> dict[str, float] | None:
        """LipSyncJob.CalcScores: normalized per-phoneme scores keyed by uLipSync phoneme name."""
        x = (np.asarray(mfcc, dtype=np.float64) - self.means) / self.stds
        n = float(MFCC_NUM)
        if method == "l1":
            d = np.abs(self.norm - x).sum(axis=1) / n
            s = np.power(10.0, -d)
        elif method == "l2":
            d = ((self.norm - x) ** 2).sum(axis=1)
            s = np.power(10.0, -np.sqrt(d / n))
        else:
            xn = float(np.linalg.norm(x))
            yn = np.linalg.norm(self.norm, axis=1)
            cos = (self.norm @ x) / np.maximum(xn * yn, 1e-12)
            s = np.power(np.maximum(cos, 0.0), 100.0)
        tot = float(s.sum())
        if not (tot > 0.0 and math.isfinite(tot)):
            return None
        return {k: float(v / tot) for k, v in zip(self.names, s)}


# ---------------------------------------------------------------------------
# Default female-anime calibration, synthesized (no recorded audio involved)
# ---------------------------------------------------------------------------

# (F1, F2, F3) in Hz, bright young female voice. U sits between Japanese /ɯ/ and English /u/.
_VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "A": (880.0, 1420.0, 2950.0),
    "I": (330.0, 2780.0, 3350.0),
    "U": (370.0, 1050.0, 2650.0),
    "E": (580.0, 2300.0, 3000.0),
    "O": (540.0, 930.0, 2850.0),
}
_FORMANT_BW = (90.0, 110.0, 170.0)


def _resonator(f: np.ndarray, fc: float, bw: float) -> np.ndarray:
    return (fc * fc) / np.sqrt((fc * fc - f * f) ** 2 + (bw * f) ** 2 + 1e-9)


def synthesize_phoneme(phoneme: str, rate: int = 48_000, length: int = 4096, f0: float = 230.0,
                       formant_scale: float = 1.0, seed: int = 0, noise: float = 0.004) -> np.ndarray:
    """Formant-filtered harmonic vowel / nasal / fricative test signal (float32, peak ~0.3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64) / rate
    if phoneme == "FV":
        spec = rng.normal(size=length // 2 + 1) + 1j * rng.normal(size=length // 2 + 1)
        f = np.fft.rfftfreq(length, 1.0 / rate)
        env = (f / 2600.0) ** 4 / (1.0 + (f / 2600.0) ** 4) / (1.0 + (f / 9000.0) ** 6)
        sig = np.fft.irfft(spec * env, length)
    else:
        k = np.arange(1, int(min(7800.0, rate / 2 - 200.0) // f0) + 1)
        fk = k * f0 * (1.0 + 0.002 * rng.normal(size=len(k)))
        src = 1.0 / fk                                        # glottal -12 dB/oct + lip radiation +6
        if phoneme == "N":
            env = (_resonator(fk, 270.0 * formant_scale, 60.0) * 1.6
                   * np.sqrt((1250.0 ** 2 - fk ** 2) ** 2 + (200.0 * fk) ** 2) / 1250.0 ** 2
                   * (0.25 + 0.6 * _resonator(fk, 2250.0 * formant_scale, 320.0))
                   / (1.0 + (fk / 650.0) ** 2))
        else:
            env = np.ones_like(fk)
            for fc, bw in zip(_VOWEL_FORMANTS[phoneme], _FORMANT_BW):
                env *= _resonator(fk, fc * formant_scale, bw)
            env *= _resonator(fk, 4200.0, 260.0) * _resonator(fk, 4900.0, 320.0)
        amp = src * env
        ph = rng.uniform(0.0, 2.0 * math.pi, size=len(k))
        sig = (amp[:, None] * np.sin(2.0 * math.pi * fk[:, None] * t[None, :] + ph[:, None])).sum(axis=0)
    sig = sig / (np.max(np.abs(sig)) + 1e-12) * 0.3
    sig = sig + noise * rng.normal(size=length)
    return sig.astype(np.float32)


_DEFAULT_CALIBRATION: dict[str, list[np.ndarray]] | None = None
_DEFAULT_LOCK = threading.Lock()


def default_calibration(analyzer: MfccAnalyzer | None = None) -> dict[str, list[np.ndarray]]:
    """Synthesized default female-anime calibration: 15 MFCC vectors per phoneme (cached)."""
    global _DEFAULT_CALIBRATION
    with _DEFAULT_LOCK:
        if _DEFAULT_CALIBRATION is not None:
            return _DEFAULT_CALIBRATION
        an = analyzer or MfccAnalyzer()
        rate = 48_000
        out: dict[str, list[np.ndarray]] = {}
        seed = 1
        for ph in MFCC_PHONEMES:
            vecs = []
            for f0 in (185.0, 215.0, 245.0, 275.0, 310.0):
                for scale in (0.95, 1.0, 1.05):
                    seed += 1
                    sig = synthesize_phoneme(ph, rate, an.window_len(rate) + 64, f0=f0, formant_scale=scale, seed=seed)
                    v = an.compute(sig, rate)
                    if v is not None:
                        vecs.append(v)
            out[ph] = vecs
        _DEFAULT_CALIBRATION = out
        return out


# ===========================================================================
# Causal prosody (Super Star / EchoAvatar 2026 style prosody-to-body features)
# ===========================================================================

PROSODY_FIELDS = ("pitchNorm", "pitchSlope", "stressOnset", "speechRate", "emphasis")
PROSODY_WINDOW_SEC = 0.043          # causal analysis window: the newest ~43 ms, never any future sample
_PROSODY_RATE = 12_000              # decimated pitch-analysis rate
_PROSODY_F0 = (80.0, 620.0)
@lru_cache(maxsize=16)
def _prosody_window(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


class CausalProsody:
    """Strictly causal, zero-lookahead prosody tracker (one call per 60 FPS frame, well under 0.5 ms).

    ``update(frame, sr, dt)`` takes the NEWEST samples only (the window ends "now") and returns:
      pitchNorm   -1..+1  F0 in semitones relative to a running speaker median (+ = above her usual pitch)
      pitchSlope  -1..+1  smoothed derivative of pitchNorm (+ = rising / question intonation)
      stressOnset  0..1   positive energy-derivative impulse (stressed syllable attack), decays in ~90 ms
      speechRate   0..1   smoothed syllable-onset rate (1.0 ~= 7 syllables / s)
      emphasis     0..1   sustained loud + expressive phrase emphasis
    Every field is exactly 0.0 while silent (backwards compatible defaults)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pitch_norm = 0.0
        self.pitch_slope = 0.0
        self.stress_onset = 0.0
        self.speech_rate = 0.0
        self.emphasis = 0.0
        self.f0_hz = 0.0
        self.median_st: float | None = None
        self._voiced_time = 0.0
        self._unvoiced_for = 1.0
        self._pn_raw = 0.0
        self._prev_pn: float | None = None
        self._st_hist: deque[float] = deque(maxlen=3)
        self._fast = 0.0
        self._slow = 0.0
        self._silent_for = 1.0
        self._t = 0.0
        self._last_syl = -1.0
        self._syl_times: deque[float] = deque(maxlen=32)
        self.cpu_ms = 0.0

    def values(self) -> dict[str, float]:
        return {"pitchNorm": self.pitch_norm, "pitchSlope": self.pitch_slope, "stressOnset": self.stress_onset,
                "speechRate": self.speech_rate, "emphasis": self.emphasis}

    @staticmethod
    def estimate_f0(frame: np.ndarray, sr: int) -> float | None:
        """Causal autocorrelation F0 (Hz) of the newest window, decimated to ~12 kHz; None when unvoiced."""
        q = max(1, int(round(sr / _PROSODY_RATE)))
        x = np.asarray(frame, dtype=np.float32)
        if q > 1:
            n = (len(x) // q) * q
            x = x[len(x) - n:].reshape(-1, q).mean(axis=1)      # box anti-alias + decimate
        rate = sr / q
        n = len(x)
        if n < 128:
            return None
        x = x - float(x.mean())
        x = x * _prosody_window(n)
        nfft = 1 << int(math.ceil(math.log2(2 * n)))
        f = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(f.real * f.real + f.imag * f.imag, nfft)[: n // 2 + 2]
        if ac[0] <= 1e-9:
            return None
        ac = ac / ac[0]
        lo = max(2, int(rate / _PROSODY_F0[1]))
        hi = min(len(ac) - 2, int(rate / _PROSODY_F0[0]))
        if hi <= lo + 2:
            return None
        seg = ac[lo:hi]
        best = float(seg.max())
        if best < 0.40:
            return None
        mid = seg[1:-1]
        peaks = np.nonzero((mid >= seg[:-2]) & (mid >= seg[2:]) & (mid >= 0.90 * best))[0]
        k = int(peaks[0] + 1 + lo) if len(peaks) else int(np.argmax(seg) + lo)
        a, b, c = float(ac[k - 1]), float(ac[k]), float(ac[k + 1])
        den = a - 2.0 * b + c
        lag = k + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)
        return rate / max(1.0, lag)

    def update(self, frame: np.ndarray | None, sr: int, dt: float, gate: float = 0.008,
               gain: float = 1.0) -> dict[str, float]:
        t0 = time.perf_counter()
        dt = max(1e-4, min(0.1, float(dt)))
        self._t += dt
        level = 0.0
        f0 = None
        if frame is not None and sr > 0 and len(frame) >= 64:
            w = np.asarray(frame, dtype=np.float32)
            tail = w[-min(len(w), max(64, int(0.021 * sr))):]
            rms = float(np.sqrt(np.mean(tail * tail) + 1e-12))
            level = min(1.0, max(0.0, (rms - gate) * gain * 4.2))
            if level > 0.04:
                try:
                    f0 = self.estimate_f0(w, sr)
                except Exception:
                    f0 = None

        # --- energy envelopes -> stressed-syllable onsets + speech rate
        self._fast += (level - self._fast) * (1.0 - math.exp(-dt / 0.015))
        self._slow += (level - self._slow) * (1.0 - math.exp(-dt / 0.14))
        impulse = max(0.0, min(1.0, 2.4 * (self._fast - self._slow) - 0.03))
        self.stress_onset = max(impulse, self.stress_onset * math.exp(-dt / 0.09))
        if impulse > 0.28 and self._t - self._last_syl > 0.11:
            self._last_syl = self._t
            self._syl_times.append(self._t)
        while self._syl_times and self._t - self._syl_times[0] > 2.5:
            self._syl_times.popleft()
        rate_t = min(1.0, (len(self._syl_times) / 2.5) / 7.0)
        self.speech_rate += (rate_t - self.speech_rate) * (1.0 - math.exp(-dt / 0.35))

        # --- pitch relative to a running speaker median (semitones)
        if f0 is not None:
            self.f0_hz = f0
            st = 12.0 * math.log2(f0 / 100.0)
            if self._unvoiced_for > 0.2:
                self._st_hist.clear()
                self._prev_pn = None
            if self._st_hist and abs(st - self._st_hist[-1]) > 7.0:     # octave error / glitch: hold
                st = self._st_hist[-1]
            self._st_hist.append(st)
            st_m = sorted(self._st_hist)[len(self._st_hist) // 2]
            if self.median_st is None:
                self.median_st = st_m
            else:
                speed = 3.0 if self._voiced_time < 1.5 else 1.2          # semitones / s (stochastic median)
                d = st_m - self.median_st
                self.median_st += math.copysign(min(abs(d), speed * dt), d)
            self._voiced_time += dt
            self._unvoiced_for = 0.0
            pn_t = max(-1.0, min(1.0, (st_m - self.median_st) / 6.0))
            self._pn_raw += (pn_t - self._pn_raw) * (1.0 - math.exp(-dt / 0.05))
            if self._prev_pn is not None:
                dpn = (self._pn_raw - self._prev_pn) / dt
                sl_t = max(-1.0, min(1.0, dpn * 0.6))
                self.pitch_slope += (sl_t - self.pitch_slope) * (1.0 - math.exp(-dt / 0.15))
            self._prev_pn = self._pn_raw
        else:
            self._unvoiced_for += dt
            self.pitch_slope *= math.exp(-dt / 0.2)
            if self._unvoiced_for > 0.2:
                self._pn_raw *= math.exp(-dt / 0.15)
        self.pitch_norm = self._pn_raw

        # --- sustained loud + expressive emphasis
        loud = max(0.0, min(1.0, (self._slow - 0.28) / 0.5))
        expr = min(1.0, 0.45 + 1.2 * abs(self.pitch_norm) + 1.5 * abs(self.pitch_slope))
        e_t = loud * expr
        tau = 0.5 if e_t > self.emphasis else 0.35
        self.emphasis += (e_t - self.emphasis) * (1.0 - math.exp(-dt / tau))

        # --- exact zeros in silence
        if level <= 0.0:
            self._silent_for += dt
        else:
            self._silent_for = 0.0
        if self._silent_for > 0.6:                          # speech ended: the syllable window empties
            self._syl_times.clear()
            for name in ("pitch_norm", "pitch_slope", "stress_onset", "speech_rate", "emphasis", "_pn_raw"):
                if abs(getattr(self, name)) < 2e-3 or self._silent_for > 1.0:
                    setattr(self, name, 0.0)
        if self.median_st is None:
            self.pitch_norm = self.pitch_slope = 0.0
        self.cpu_ms = (time.perf_counter() - t0) * 1000.0
        return self.values()


class UserVoiceActivity:
    """Tiny user-voice VAD for visual backchanneling: feed mic / call RMS, get speech -> micro-pause pulses.

    A pulse fires ONCE when the user has talked >= 0.4 s and then paused for 0.25 s (well inside a 0.25-0.6 s
    micro-pause, before a full turn end). Consumers compare ``pulses`` with their last seen count: no polling."""

    PAUSE_FIRE_SEC = 0.25
    MIN_SPEECH_SEC = 0.4

    def __init__(self) -> None:
        self.energy = 0.0
        self.floor = 0.004
        self.active = False
        self.speech_since = 0.0
        self.speech_len = 0.0
        self.silent_since = 0.0
        self.pulses = 0
        self.last_pulse_t = 0.0
        self._armed = False
        self._last_t: float | None = None

    def feed(self, rms: float, t: float) -> bool:
        """Returns True when this sample fired a micro-pause pulse."""
        try:
            rms = float(rms)
        except (TypeError, ValueError):
            rms = 0.0
        rms = max(0.0, rms) if math.isfinite(rms) else 0.0
        dt = 0.016 if self._last_t is None else max(1e-4, min(0.25, t - self._last_t))
        self._last_t = t
        self.energy += (rms - self.energy) * (1.0 - math.exp(-dt / 0.04))
        if not self.active:                                   # noise floor follows the quiet level only
            k = 0.5 if self.energy < self.floor else 0.02
            self.floor += (self.energy - self.floor) * min(1.0, k * dt * 10.0)
        self.floor = max(0.002, min(0.05, self.floor))
        on_th = max(0.012, self.floor * 3.0)
        off_th = on_th * 0.6
        if not self.active and self.energy > on_th:
            self.note(True, t)
        elif self.active and self.energy < off_th:
            self.note(False, t)
        return self.tick(t)

    def note(self, active: bool, t: float) -> None:
        if active and not self.active:
            self.active = True
            self.speech_since = t
        elif not active and self.active:
            self.active = False
            self.speech_len = t - self.speech_since
            self.silent_since = t
            self._armed = self.speech_len >= self.MIN_SPEECH_SEC

    def tick(self, t: float) -> bool:
        if self._armed and not self.active and t - self.silent_since >= self.PAUSE_FIRE_SEC:
            self._armed = False
            self.pulses += 1
            self.last_pulse_t = t
            return True
        return False

    def state(self, t: float) -> dict[str, Any]:
        return {"active": self.active, "energy": round(self.energy, 5), "pulses": self.pulses,
                "pauseFor": round(t - self.silent_since, 3) if not self.active else 0.0,
                "speakingFor": round(t - self.speech_since, 3) if self.active else 0.0}


# ===========================================================================
# Live engine
# ===========================================================================

class LiveLipSyncEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.visemes: dict[str, dict[str, float]] = copy.deepcopy(DEFAULT_VISEMES)
        self.cfg: dict[str, Any] = copy.deepcopy(DEFAULT_LIPSYNC_CFG)

        # Springs for the 6 VBridger mouth axes + lip press + speech energy
        self.sp_open = Spring2nd(0.0)
        self.sp_form = Spring2nd(0.08)
        self.sp_funnel = Spring2nd(0.0)
        self.sp_x = Spring2nd(0.0)
        self.sp_shrug = Spring2nd(0.0)
        self.sp_tongue = Spring2nd(0.0)
        self.sp_press = Spring2nd(0.0)
        self.sp_energy = Spring2nd(0.0)
        self.sp_stress = Spring2nd(0.0)

        # Audio playback track (PCM float32 mono @ sr)
        self._pcm: np.ndarray | None = None
        self._sr: int = 48000
        self._play_start: float = 0.0
        self._playing: bool = False
        self._cues: list[dict] | None = None
        self._track_id: str = ""
        self._duration: float = 0.0

        # Live microphone ring buffer
        self._mic_stream = None
        self._mic_ring = np.zeros(4096, dtype=np.float32)
        self._mic_pos = 0
        self._mic_sr = 48000

        # Live Fish Audio Full-Duplex PCM subscriber (GET http://127.0.0.1:8766/pcm)
        self._fish_ring = np.zeros(4096, dtype=np.float32)
        self._fish_pos = 0
        self._fish_last_rx = 0.0
        self._fish_connected = False
        self._fish_stop = threading.Event()
        self._fish_thread: threading.Thread | None = None

        # Track previous frame level for crisp syllable-onset & bilabial M/B/P closure detection
        self._prev_level: float = 0.0
        self._prev_code: str = "X"
        self._closure_hold: float = 0.0

        # 12-MFCC uLipSync analyzer + profile (default synthesized calibration + user calibration)
        self.mfcc = MfccAnalyzer()
        self._profile: MfccProfile | None = None
        self._profile_err: str = ""
        self._recent_mfcc: deque[tuple[float, np.ndarray]] = deque(maxlen=48)
        self.last_mfcc: np.ndarray | None = None
        self.rebuild_profile()

        # Live telemetry for HUD
        self.active_viseme: str = "X"
        self.active_source: str = "idle"
        self.live_rms: float = 0.0
        self.live_bands: list[float] = [0.0] * 12
        self.live_scores: dict[str, float] = {}
        self.live_onset: float = 0.0
        self.frame_ms: float = 0.0
        # Prosody: per-frame F0 / brightness from analyze_frame, smoothed in step() (rig: head beats, brows, chin)
        self._frame_pitch: float | None = None       # normalized F0 of the current frame, None when unvoiced
        self._frame_bright: float | None = None
        self._pitch_s: float = 0.42
        self._bright_s: float = 0.5
        self._voiced_s: float = 0.0
        self._unvoiced_for: float = 1.0
        self.live_f0_hz: float = 0.0
        self.live_axes: dict[str, float] = {
            "open": 0.0, "form": 0.08, "funnel": 0.0, "x": 0.0, "shrug": 0.0, "tongue": 0.0,
            "press": 0.0, "energy": 0.0, "stress": 0.0, "onset": 0.0,
            "pitch": 0.42, "voiced": 0.0, "brightness": 0.5,
            "pitchNorm": 0.0, "pitchSlope": 0.0, "stressOnset": 0.0, "speechRate": 0.0, "emphasis": 0.0,
        }
        # Causal prosody-to-body features (zero lookahead) + user-voice VAD for visual backchanneling
        self.prosody = CausalProsody()
        self._prosody_snapshot = self._prosody_status_snapshot()
        self.user_voice = UserVoiceActivity()

    # ------------------------------------------------------------------ config
    def update_config(self, patch: dict) -> None:
        rebuild = False
        with self.lock:
            for k, v in patch.items():
                if k not in self.cfg:
                    continue
                if k == "paramMap" and isinstance(v, dict):
                    pm = self.cfg.setdefault("paramMap", copy.deepcopy(DEFAULT_PARAM_MAP))
                    for ch, vals in v.items():
                        if isinstance(vals, dict):
                            pm.setdefault(ch, {}).update(vals)
                elif k == "mfccCalibration":
                    self.cfg[k] = copy.deepcopy(v) if isinstance(v, dict) else {}
                    rebuild = True
                else:
                    self.cfg[k] = v
            want_mic = bool(self.cfg.get("micEnabled", False))
        if rebuild:
            self.rebuild_profile()
        self.set_mic_enabled(want_mic)

    def update_visemes(self, patch: dict) -> None:
        with self.lock:
            for code, vals in patch.items():
                if code in self.visemes and isinstance(vals, dict):
                    for axis in ("open", "form", "funnel", "x", "shrug", "tongue"):
                        if axis in vals:
                            self.visemes[code][axis] = float(vals[axis])

    def reset_visemes(self) -> None:
        with self.lock:
            self.visemes = copy.deepcopy(DEFAULT_VISEMES)

    @property
    def engine(self) -> str:
        e = str(self.cfg.get("engine") or "hybrid").lower()
        return e if e in LIPSYNC_ENGINES else "hybrid"

    def _coart(self) -> float:
        try:
            return max(0.08, min(0.60, float(self.cfg.get("coarticulation", 0.22))))
        except (TypeError, ValueError):
            return 0.22

    # ------------------------------------------------------------- calibration
    def rebuild_profile(self) -> None:
        """Default synthesized templates, with any user-calibrated phonemes replacing their default."""
        base = default_calibration(self.mfcc)
        user = self.cfg.get("mfccCalibration") or {}
        samples: dict[str, list[np.ndarray]] = {}
        for ph, vecs in base.items():
            code = MFCC_PHONEMES[ph]
            mine = []
            for v in (user.get(code) or [])[-MFCC_MAX_SAMPLES_PER_PHONEME:]:
                arr = np.asarray(v, dtype=np.float64)
                if arr.shape == (MFCC_NUM,) and np.all(np.isfinite(arr)):
                    mine.append(arr)
            samples[ph] = mine if mine else list(vecs)
        try:
            prof = MfccProfile(samples, use_standardization=True)
            err = ""
        except Exception as exc:
            prof, err = None, str(exc)
        with self.lock:
            self._profile = prof
            self._profile_err = err

    def calibrated_visemes(self) -> dict[str, int]:
        user = self.cfg.get("mfccCalibration") or {}
        return {code: len(user.get(code) or []) for code in MFCC_VISEMES if user.get(code)}

    def calibrate(self, viseme: str, window_sec: float = 0.6) -> dict[str, Any]:
        """Capture the live voiced MFCCs of the last `window_sec` into the user profile.
        `viseme` may be a viseme code (D/B/F/C/E/G/A) or a friendly label (AA/EE/OO/OH/EH/FV/MBP)."""
        code = resolve_viseme(viseme)
        if code not in VISEME_TO_PHONEME:
            return {"ok": False, "error": f"Unknown phoneme '{viseme}'. Use one of {list(MFCC_VISEMES)} "
                                          f"or {sorted(FRIENDLY_VISEMES)}."}
        now = time.perf_counter()
        with self.lock:
            vecs = [v for (t, v) in self._recent_mfcc if now - t <= window_sec]
        if not vecs:
            frame, rate, _src = self._current_window(long=True)
            if frame is not None:
                rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
                if rms > float(self.cfg.get("noiseGate", 0.012)):
                    v = self.mfcc.compute(frame, rate)
                    if v is not None:
                        vecs = [v]
        if not vecs:
            return {"ok": False, "error": "No voiced audio right now. Hold the sound (mic, TTS or Fish call) and try again."}
        with self.lock:
            cal = self.cfg.setdefault("mfccCalibration", {})
            lst = list(cal.get(code) or [])
            # Keep a few frames per capture, newest first-class; cap at 16 per viseme (uLipSync style)
            step = max(1, len(vecs) // 4)
            for v in vecs[::step][:4]:
                lst.append([round(float(x), 5) for x in v])
            cal[code] = lst[-MFCC_MAX_SAMPLES_PER_PHONEME:]
            count = len(cal[code])
            snapshot = copy.deepcopy(cal)
        self.rebuild_profile()
        return {"ok": True, "phoneme": code, "uLipSyncPhoneme": VISEME_TO_PHONEME[code], "samples": count,
                "framesCaptured": len(vecs), "calibration": snapshot}

    def reset_calibration(self) -> dict[str, Any]:
        with self.lock:
            self.cfg["mfccCalibration"] = {}
        self.rebuild_profile()
        return {"ok": True, "calibration": {}}

    # ------------------------------------------------------------------ inputs
    def set_mic_enabled(self, enabled: bool) -> bool:
        if sd is None:
            with self.lock:
                self.cfg["micEnabled"] = False
            return False
        if enabled and self._mic_stream is None:
            try:
                def cb(indata, frames, time_info, status):
                    mono = indata[:, 0].astype(np.float32)
                    n = len(mono)
                    with self.lock:
                        if n >= len(self._mic_ring):
                            self._mic_ring[:] = mono[-len(self._mic_ring):]
                            self._mic_pos = 0
                        else:
                            end = self._mic_pos + n
                            if end <= len(self._mic_ring):
                                self._mic_ring[self._mic_pos:end] = mono
                            else:
                                first = len(self._mic_ring) - self._mic_pos
                                self._mic_ring[self._mic_pos:] = mono[:first]
                                self._mic_ring[:n - first] = mono[first:]
                            self._mic_pos = end % len(self._mic_ring)

                self._mic_stream = sd.InputStream(
                    channels=1, samplerate=self._mic_sr, blocksize=512, callback=cb
                )
                self._mic_stream.start()
                with self.lock:
                    self.cfg["micEnabled"] = True
                return True
            except Exception:
                self._mic_stream = None
                with self.lock:
                    self.cfg["micEnabled"] = False
                return False
        elif not enabled and self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
            with self.lock:
                self.cfg["micEnabled"] = False
        return bool(self.cfg.get("micEnabled", False))

    def playback_generation(self) -> int:
        with self.lock:
            return int(getattr(self, "_play_gen", 0))

    def load_wav_and_play(self, wav_path: Path, cues_path: Path | None = None, track_id: str = "",
                          expect_gen: int | None = None) -> float:
        with wave.open(str(wav_path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)

        cues = None
        if cues_path and cues_path.exists():
            try:
                data = json.loads(cues_path.read_text(encoding="utf-8"))
                cues = data.get("mouthCues")
            except Exception:
                cues = None

        dur = len(pcm) / float(sr)
        with self.lock:
            cur_gen = int(getattr(self, "_play_gen", 0))
            if expect_gen is not None and cur_gen != expect_gen:
                return 0.0                       # interrupted while decoding in worker thread: do not resurrect!
            self._play_gen = cur_gen + 1
            self._pcm = pcm
            self._sr = sr
            self._cues = cues
            self._play_start = time.perf_counter()
            self._playing = True
            self._track_id = track_id or wav_path.name
            self._duration = dur
            self.cfg["holdViseme"] = ""
        return dur

    def sync_playback_time(self, elapsed_sec: float) -> None:
        with self.lock:
            if self._playing and self._pcm is not None:
                self._play_start = time.perf_counter() - max(0.0, float(elapsed_sec))

    def stop_playback(self) -> None:
        with self.lock:
            self._play_gen = int(getattr(self, "_play_gen", 0)) + 1
            self._playing = False
            self._pcm = None
            self._cues = None
            self._track_id = ""
            self._duration = 0.0

    def ingest_fish_pcm(self, raw_s16le: bytes) -> None:
        """Zero-latency direct ingestion of 48 kHz mono s16le PCM blocks from built-in Fish WebRTC call."""
        n_even = len(raw_s16le) & ~1
        if n_even == 0:
            return
        samples = np.frombuffer(raw_s16le[:n_even], dtype="<i2").astype(np.float32) / 32768.0
        n = len(samples)
        with self.lock:
            if n >= len(self._fish_ring):
                self._fish_ring[:] = samples[-len(self._fish_ring):]
                self._fish_pos = 0
            else:
                end = self._fish_pos + n
                if end <= len(self._fish_ring):
                    self._fish_ring[self._fish_pos:end] = samples
                else:
                    first = len(self._fish_ring) - self._fish_pos
                    self._fish_ring[self._fish_pos:] = samples[:first]
                    self._fish_ring[:n - first] = samples[first:]
                self._fish_pos = end % len(self._fish_ring)
            self._fish_last_rx = time.perf_counter()
            self._fish_connected = True

    def start_fish_pcm_bridge(self, url: str = "http://127.0.0.1:8766/pcm") -> None:
        if self._fish_thread is not None and self._fish_thread.is_alive():
            return
        self._fish_stop.clear()

        def worker():
            import httpx
            while not self._fish_stop.is_set():
                try:
                    with httpx.stream("GET", url, timeout=httpx.Timeout(connect=2.0, read=None, write=2.0, pool=2.0)) as r:
                        if r.status_code != 200:
                            self._fish_connected = False
                            self._fish_stop.wait(2.5)
                            continue
                        self._fish_connected = True
                        carry = b""
                        for chunk in r.iter_bytes(chunk_size=1920):
                            if self._fish_stop.is_set():
                                break
                            if not chunk:
                                continue
                            buf = carry + chunk
                            n_even = len(buf) & ~1
                            carry = buf[n_even:]
                            if n_even == 0:
                                continue
                            self.ingest_fish_pcm(buf[:n_even])
                except Exception:
                    self._fish_connected = False
                    self._fish_stop.wait(2.5)
            self._fish_connected = False

        self._fish_thread = threading.Thread(target=worker, daemon=True)
        self._fish_thread.start()

    def stop_fish_pcm_bridge(self) -> None:
        self._fish_stop.set()

    @property
    def fish_duplex_connected(self) -> bool:
        return self._fish_connected

    @property
    def fish_duplex_active(self) -> bool:
        return (time.perf_counter() - self._fish_last_rx) < 0.35

    def is_speaking(self) -> bool:
        with self.lock:
            return (
                self._playing
                or (time.perf_counter() - self._fish_last_rx) < 0.35
                or (bool(self.cfg.get("micEnabled")) and self.live_rms > float(self.cfg["noiseGate"]))
            )

    def _current_window(self, long: bool = False) -> tuple[np.ndarray | None, int, str]:
        """Newest analysis window of the active source (long = MFCC-sized window)."""
        with self.lock:
            fish_active = (time.perf_counter() - self._fish_last_rx) < 0.35
            if fish_active:
                ring, pos, rate, src = self._fish_ring.copy(), self._fish_pos, 48000, "fish"
            elif self._playing and self._pcm is not None:
                pcm, rate = self._pcm, self._sr
                idx = int((time.perf_counter() - self._play_start) * rate)
                n = self.mfcc.window_len(rate) if long else 1024
                n = n or 1024
                i0 = max(0, idx - n // 2)
                chunk = pcm[i0:i0 + n]
                return (np.pad(chunk, (0, n - len(chunk))) if len(chunk) < n else chunk.copy()), rate, "playback"
            elif bool(self.cfg.get("micEnabled")):
                ring, pos, rate, src = self._mic_ring.copy(), self._mic_pos, self._mic_sr, "mic"
            else:
                return None, 0, "idle"
        n = (self.mfcc.window_len(rate) if long else 1024) or 1024
        return np.roll(ring, -pos)[-n:], rate, src

    # ------------------------------------------------------------- analysis
    @staticmethod
    def _softmax(scores: dict[str, float], temp: float) -> dict[str, float]:
        if not scores:
            return {}
        mx = max(scores.values())
        ex = {k: math.exp((v - mx) / temp) for k, v in scores.items()}
        tot = sum(ex.values()) + 1e-12
        return {k: v / tot for k, v in ex.items()}

    @staticmethod
    def _sharpen(probs: dict[str, float], temp: float) -> dict[str, float]:
        """Re-temper already-normalized probabilities; identity at the default 0.22."""
        if not probs:
            return {}
        power = 0.22 / max(0.08, temp)
        ex = {k: max(0.0, v) ** power for k, v in probs.items()}
        tot = sum(ex.values())
        if tot <= 0.0:
            return dict(probs)
        return {k: v / tot for k, v in ex.items()}

    # Pitch range used for normalization: 120 Hz -> 0.0, 560 Hz -> 1.0 (bright young female voice sits ~0.35-0.7).
    _F0_LO, _F0_HI = 120.0, 560.0

    def _estimate_pitch(self, win: np.ndarray, sr: int) -> tuple[float | None, float]:
        """Autocorrelation F0 of a Hann-windowed frame -> (normalized pitch or None if unvoiced, f0 Hz).
        Picks the FIRST strong peak (>= 93 % of the best) so octave-down errors are avoided."""
        n = len(win)
        if n < 256 or sr < 8000:
            return None, 0.0
        f = np.fft.rfft(win, 2 * n)
        ac = np.fft.irfft(f.real * f.real + f.imag * f.imag)[: n // 2 + 1]
        if ac[0] <= 1e-9:
            return None, 0.0
        ac = ac / ac[0]
        lo = max(2, int(sr / 700.0))
        hi = min(len(ac) - 2, int(sr / 95.0))
        if hi <= lo + 2:
            return None, 0.0
        seg = ac[lo:hi]
        best = float(seg.max())
        if best < 0.32:
            return None, 0.0
        # local maxima above 93 % of the best peak; take the shortest lag
        mid = seg[1:-1]
        peaks = np.nonzero((mid >= seg[:-2]) & (mid >= seg[2:]) & (mid >= 0.93 * best))[0]
        k = int(peaks[0] + 1 + lo) if len(peaks) else int(np.argmax(seg) + lo)
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        den = a - 2.0 * b + c
        lag = k + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)
        f0 = sr / max(1.0, lag)
        pn = math.log2(max(1.0, f0) / self._F0_LO) / math.log2(self._F0_HI / self._F0_LO)
        return max(0.0, min(1.0, pn)), f0

    def analyze_frame(self, chunk: np.ndarray, sr: int, long_chunk: np.ndarray | None = None,
                      engine: str | None = None) -> tuple[str, float, list[float], dict[str, float], dict[str, float]]:
        """One analysis frame -> (viseme code, rms, 12 HUD bands, target axes, viseme probabilities)."""
        cfg = self.cfg
        engine = engine or self.engine
        temp = self._coart()
        chunk = np.asarray(chunk, dtype=np.float32)
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        gate = float(cfg["noiseGate"])
        gain = float(cfg["gain"])
        level = max(0.0, (rms - gate) * gain * 3.8)
        level = min(1.0, level ** 0.72) if level > 0 else 0.0

        # 12-band FFT spectrum for live HUD + formant vowel classification
        n = len(chunk)
        win = chunk * np.hanning(n)
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)

        edges = [80, 180, 320, 500, 750, 1050, 1450, 2000, 2700, 3600, 5000, 7000, 10000]
        bands = []
        for i in range(12):
            m = (freqs >= edges[i]) & (freqs < edges[i + 1])
            val = float(np.mean(spec[m])) if np.any(m) else 0.0
            bands.append(min(1.0, val * gain * 0.09))

        flux = level - self._prev_level
        self._prev_level = level
        # Syllable onset / plosive release (drives Mouth Spring in the rig)
        onset = max(0.0, min(1.0, (flux - 0.05) * 3.2))
        if self._prev_code == "A" and level > 0.22:
            onset = max(onset, 0.75)                     # bilabial release: "p"/"b" pop
        self.live_onset = onset

        if level < 0.04:
            self._prev_code = "X"
            return "X", rms, bands, dict(self.visemes["X"]), {"X": 1.0}

        power = spec * spec
        # Prosody features for the rig (cheap: one 2n FFT): F0 + spectral brightness (centroid 300 Hz..6 kHz)
        try:
            self._frame_pitch, self.live_f0_hz = self._estimate_pitch(win, sr)
            bm = (freqs >= 300.0) & (freqs < 6000.0)
            ps = float(np.sum(power[bm]))
            if ps > 1e-10:
                cen = float(np.sum(freqs[bm] * power[bm]) / ps)
                self._frame_bright = max(0.0, min(1.0, (cen - 700.0) / 2600.0))
        except Exception:                      # never let prosody break lip-sync
            self._frame_pitch = None

        def band_e(f0: float, f1: float) -> float:
            m = (freqs >= f0) & (freqs < f1)
            return float(np.mean(spec[m])) if np.any(m) else 0.0

        e_low = band_e(120, 420)
        e_f1 = band_e(420, 1000)
        e_mid = band_e(1000, 1700)
        e_f2 = band_e(1700, 3200)
        e_hi = band_e(3200, 8000)
        tot = e_low + e_f1 + e_mid + e_f2 + e_hi + 1e-6
        r_low = e_low / tot
        r_hi = e_hi / tot
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.float32)))))
        voiced = max(0.0, min(1.0, (1.0 - r_hi * 2.2) * (1.0 - max(0.0, zcr - 0.16) * 3.5)))

        # ---- amplitude engine: classic VTube Studio volume flap + spectral tilt -------------
        if engine == "amplitude":
            # Volume opens the mouth; two spectral-tilt ratios shape it (no phoneme model):
            #   brightness (1.8-3.5 kHz vs 0.25-1.1 kHz): EE/EH bright -> smile, OO/OH dark -> round
            #   openness   (0.6-1.1 kHz vs 0.2-0.6 kHz):  high F1 (AA) -> wide jaw
            def mag(f0: float, f1: float) -> float:
                m = (freqs >= f0) & (freqs < f1)
                return float(np.mean(spec[m])) + 1e-9 if np.any(m) else 1e-9

            bright = math.log10(mag(1800, 3500) / mag(250, 1100))
            opened = math.log10(mag(600, 1100) / mag(200, 600))
            bright_n = max(0.0, min(1.0, (bright + 2.0) / 1.3))
            open_n = max(0.0, min(1.0, (opened + 1.3) / 1.6))
            hiss = max(0.0, min(1.0, (bright - 0.2) / 0.8))
            vol = min(1.0, level ** 0.85 * 1.08)
            open_ = vol * (0.40 + 0.60 * open_n) * (1.0 - 0.7 * hiss)
            tgt = {
                "open": open_,
                "form": (2.0 * bright_n - 1.0) * 0.75 * (1.0 - hiss) * (0.45 + 0.55 * level),
                "funnel": (1.0 - bright_n) * (1.0 - 0.5 * open_n) * (1.0 - hiss) * min(1.0, 0.3 + vol),
                "x": 0.0,
                "shrug": 0.72 * hiss,
                "tongue": 0.15 * open_ if open_n > 0.8 else 0.0,
            }
            if hiss > 0.5:
                code = "G"
            elif open_n > 0.8:
                code = "D"
            elif bright_n > 0.6:
                code = "B" if open_n < 0.35 else "C"
            else:
                code = "F" if open_n < 0.45 else "E"
            self._prev_code = code
            return code, rms, bands, tgt, {code: 1.0}

        # ---- IPA F1/F2 formant scores --------------------------------------------------------
        def spectral_centroid(f0: float, f1: float, fallback: float) -> float:
            m = (freqs >= f0) & (freqs < f1)
            if not np.any(m):
                return fallback
            w = power[m]
            s = float(np.sum(w))
            if s < 1e-10:
                return fallback
            return float(np.sum(freqs[m] * w) / s)

        f1_hz = spectral_centroid(220.0, 790.0, 540.0)
        f2_hz = spectral_centroid(max(750.0, f1_hz + 240.0), 3000.0, 1550.0)

        # Gaussian log-likelihood scores in normalized (F1, F2) female-voice vowel space:
        # D (AA/AH): F1~820, F2~1320 | C (EH/AE): F1~660, F2~2050 | B (EE/IY): F1~340, F2~2550
        # E (AO/OH): F1~580, F2~980  | F (UW/OO): F1~350, F2~880  | H (L/TH):  F1~490, F2~1480
        def v_score(f1_t: float, f2_t: float, s1: float = 175.0, s2: float = 380.0) -> float:
            d2 = ((f1_hz - f1_t) / s1) ** 2 + ((f2_hz - f2_t) / s2) ** 2
            return math.exp(-0.5 * d2)

        closure = max(0.0, (0.16 - level) * 5.0) + max(0.0, -flux * 3.0) * (1.0 if level < 0.28 else 0.0)
        fricative = max(0.0, (r_hi - 0.20) * 3.2 + (zcr - 0.16) * 2.8) * max(0.2, 1.0 - r_low * 2.5) * (1.05 - 0.45 * level)
        sc = {
            "D": v_score(810.0, 1340.0, 190.0, 360.0) * (0.55 + 0.75 * level) * voiced,
            "C": v_score(650.0, 2060.0, 170.0, 380.0) * (0.55 + 0.55 * level) * voiced,
            "B": (v_score(340.0, 2550.0, 150.0, 420.0) * voiced + max(0.0, (r_hi - 0.34) * 1.4 + (zcr - 0.28) * 1.1) * 0.25),
            "E": v_score(580.0, 1000.0, 155.0, 270.0) * (0.50 + 0.65 * level) * voiced,
            "F": v_score(350.0, 880.0, 145.0, 250.0) * (1.05 - 0.10 * level) * voiced,
            "H": v_score(490.0, 1500.0, 140.0, 270.0) * 0.72 * voiced,
            "G": fricative,
            "A": closure,
        }

        probs: dict[str, float]
        mfcc_probs: dict[str, float] | None = None
        if engine in ("hybrid", "ulipsync_mfcc"):
            mfcc_probs = self._mfcc_probs(long_chunk if long_chunk is not None else chunk, sr)

        if engine == "ulipsync_mfcc" and mfcc_probs:
            probs = self._sharpen(mfcc_probs, temp)
        elif engine == "hybrid" and mfcc_probs:
            p_f = self._softmax(sc, temp)
            keys = set(p_f) | set(mfcc_probs)
            fused = {k: 0.62 * mfcc_probs.get(k, 0.0) + 0.38 * p_f.get(k, 0.0) for k in keys}
            # Consonant detector: crisp M/B/P closure and F/V tuck override vowel blur
            if closure > 0.35:
                fused["A"] = fused.get("A", 0.0) + min(0.9, closure * 0.8)
            if fricative > 0.45 and voiced < 0.55:
                fused["G"] = fused.get("G", 0.0) + min(0.6, fricative * 0.5)
            s = sum(fused.values()) or 1.0
            probs = self._sharpen({k: v / s for k, v in fused.items()}, temp)
        else:
            # ipa_formant, or MFCC unavailable (input rate < 16 kHz / degenerate frame)
            probs = self._softmax(sc, temp)

        code = max(probs, key=probs.get)
        onset_punch = max(0.0, min(0.24, flux * 1.5))
        dyn = min(1.14, 0.26 + 0.78 * level + onset_punch)
        blended = {"open": 0.0, "form": 0.0, "funnel": 0.0, "x": 0.0, "shrug": 0.0, "tongue": 0.0}
        for k, p_k in probs.items():
            vk = self.visemes.get(k)
            if vk is None:
                continue
            for ax in blended:
                blended[ax] += vk[ax] * p_k

        tgt = {
            "open": blended["open"] * dyn,
            "form": blended["form"] * (0.45 + 0.55 * level),
            "funnel": blended["funnel"] * dyn,
            "x": blended["x"] * level,
            "shrug": blended["shrug"] * (0.42 + 0.58 * level),
            "tongue": blended["tongue"] * dyn,
        }
        self._prev_code = code
        return code, rms, bands, tgt, probs

    # Backwards-compatible name used by earlier plugins
    def _classify_frame_spectrum(self, chunk: np.ndarray, sr: int) -> tuple[str, float, list[float], dict[str, float]]:
        code, rms, bands, tgt, _p = self.analyze_frame(chunk, sr, engine="ipa_formant")
        return code, rms, bands, tgt

    def _mfcc_probs(self, frame: np.ndarray, sr: int) -> dict[str, float] | None:
        prof = self._profile
        if prof is None:
            return None
        v = self.mfcc.compute(frame, sr)
        if v is None:
            return None
        self.last_mfcc = v
        self._recent_mfcc.append((time.perf_counter(), v))
        method = str(self.cfg.get("mfccScoring") or "cosine").lower()
        scores = prof.score(v, method if method in MFCC_SCORING else "cosine")
        if not scores:
            return None
        return {MFCC_PHONEMES[ph]: s for ph, s in scores.items()}

    # ------------------------------------------------------------------ frame
    def step(self, dt: float, t_now: float) -> dict[str, float]:
        t_start = time.perf_counter()
        with self.lock:
            cfg = dict(self.cfg)
            visemes = self.visemes
            hold = str(cfg.get("holdViseme") or "")
            playing = self._playing
            pcm = self._pcm
            sr = self._sr
            cues = self._cues
            play_start = self._play_start
            mic_on = bool(cfg.get("micEnabled"))
            mic_ring = self._mic_ring.copy() if mic_on else None
            mic_pos = self._mic_pos
            fish_active = (time.perf_counter() - self._fish_last_rx) < 0.35
            fish_ring = self._fish_ring.copy() if fish_active else None
            fish_pos = self._fish_pos

        engine = self.engine
        need_long = engine in ("hybrid", "ulipsync_mfcc")
        code = "X"
        rms = 0.0
        bands = [0.0] * 12
        tgt = dict(visemes["X"])
        probs: dict[str, float] = {"X": 1.0}
        energy_raw = 0.0
        source = "idle"
        self.live_onset = 0.0
        self._frame_pitch = None
        self._frame_bright = None
        pros_frame: np.ndarray | None = None          # newest samples only (window ends "now"): strictly causal
        pros_sr = 0

        if hold and hold in visemes:
            code = hold
            tgt = dict(visemes[hold])
            energy_raw = max(0.25, tgt["open"])
            probs = {hold: 1.0}
            source = "hold"
        elif fish_active and fish_ring is not None:
            # Live Fish Audio Drama 3 / Live 3 remote 48 kHz PCM stream
            ordered = np.roll(fish_ring, -fish_pos)
            pros_frame, pros_sr = ordered[-int(PROSODY_WINDOW_SEC * 48000):], 48000
            long_n = self.mfcc.window_len(48000)
            code, rms, bands, tgt, probs = self.analyze_frame(
                ordered[-1024:], 48000, ordered[-long_n:] if need_long and long_n else None, engine)
            energy_raw = min(1.0, max(0.0, (rms - float(cfg["noiseGate"])) * float(cfg["gain"]) * 4.4))
            source = "fish"
        elif playing and pcm is not None:
            elapsed = time.perf_counter() - play_start
            if elapsed >= len(pcm) / float(sr):
                with self.lock:
                    self._playing = False
            else:
                idx = int(elapsed * sr)
                pros_frame, pros_sr = pcm[max(0, idx - int(PROSODY_WINDOW_SEC * sr)):idx], sr
                win_n = 1024
                i0 = max(0, idx - win_n // 2)
                chunk = pcm[i0:i0 + win_n]
                if len(chunk) < win_n:
                    chunk = np.pad(chunk, (0, win_n - len(chunk)))
                long_chunk = None
                long_n = self.mfcc.window_len(sr) if need_long else 0
                if long_n:
                    j0 = max(0, idx - long_n // 2)
                    long_chunk = pcm[j0:j0 + long_n]
                    if len(long_chunk) < long_n:
                        long_chunk = np.pad(long_chunk, (0, long_n - len(long_chunk)))
                spec_code, rms, bands, spec_tgt, spec_probs = self.analyze_frame(chunk, sr, long_chunk, engine)
                energy_raw = min(1.0, max(0.0, (rms - float(cfg["noiseGate"])) * float(cfg["gain"]) * 4.2))
                source = "playback"
                if cues:
                    # Exact Rhubarb cue lookup with smooth cross-cue coarticulation!
                    cue_code = "X"
                    for c in cues:
                        if c["start"] <= elapsed < c["end"]:
                            cue_code = c["value"]
                            break
                    code = cue_code if cue_code in visemes else spec_code
                    v = visemes[code]
                    mod = 0.55 + 0.45 * min(1.0, energy_raw * 1.4)
                    if code in ("X", "A"):
                        mod = 1.0
                    tgt = {
                        "open": v["open"] * mod,
                        "form": v["form"],
                        "funnel": v["funnel"] * mod,
                        "x": v["x"],
                        "shrug": v["shrug"],
                        "tongue": v["tongue"] * mod,
                    }
                    probs = {code: 1.0}
                    source = "rhubarb"
                else:
                    code = spec_code
                    tgt = spec_tgt
                    probs = spec_probs
        elif mic_on and mic_ring is not None:
            ordered = np.roll(mic_ring, -mic_pos)
            pros_frame, pros_sr = ordered[-int(PROSODY_WINDOW_SEC * self._mic_sr):], self._mic_sr
            long_n = self.mfcc.window_len(self._mic_sr)
            code, rms, bands, tgt, probs = self.analyze_frame(
                ordered[-1024:], self._mic_sr, ordered[-long_n:] if need_long and long_n else None, engine)
            energy_raw = min(1.0, max(0.0, (rms - float(cfg["noiseGate"])) * float(cfg["gain"]) * 4.2))
            source = "mic"

        # Apply user live shaping sliders!
        smile_bias = float(cfg["smileBias"])
        roundness = float(cfg["roundness"])
        jaw_bounce = float(cfg["jawBounce"])
        lip_seal = float(cfg["lipSealSharpness"])
        flutter = float(cfg["microFlutter"]) * energy_raw * 0.045 * math.sin(t_now * 28.0)

        target_open = max(0.0, min(1.0, tgt["open"] * jaw_bounce + flutter))
        if code == "A":
            target_open = 0.0
        target_form = max(-1.0, min(1.0, tgt["form"] + smile_bias))
        target_funnel = max(0.0, min(1.0, tgt["funnel"] * roundness))
        target_x = max(-1.0, min(1.0, tgt["x"]))
        target_shrug = max(0.0, min(1.0, tgt["shrug"] * lip_seal))
        target_tongue = max(0.0, min(1.0, tgt["tongue"]))

        omega = max(6.0, min(55.0, float(cfg["attackOmega"])))
        zeta = max(0.35, min(1.4, float(cfg["dampingZeta"])))
        # Snappy upward vowel punch + razor-clean lip seal closure on M/B/P!
        if code in ("A", "X") and target_open < self.sp_open.y:
            om_open = omega * 1.40 * lip_seal
        elif target_open > self.sp_open.y:
            om_open = omega * 1.16
        else:
            om_open = omega * 0.92

        # Crisp bilabial lip-press (-0.85 on M/B/P, -0.45 on F/V, +open on wide vowels!)
        target_press = -0.85 if code == "A" else (-0.45 if code == "G" else (target_open * 0.75))
        o = max(0.0, min(1.0, self.sp_open.step(target_open, dt, om_open, zeta)))
        f = max(-1.0, min(1.0, self.sp_form.step(target_form, dt, omega * 0.85, zeta)))
        fn = max(0.0, min(1.0, self.sp_funnel.step(target_funnel, dt, omega * 0.90, zeta)))
        mx = max(-1.0, min(1.0, self.sp_x.step(target_x, dt, omega * 0.75, zeta)))
        sh = max(0.0, min(1.0, self.sp_shrug.step(target_shrug, dt, omega * 1.15, zeta)))
        tg = max(0.0, min(1.0, self.sp_tongue.step(target_tongue, dt, omega * 0.95, zeta)))
        pr = max(-1.0, min(1.0, self.sp_press.step(target_press, dt, omega * 1.10, zeta)))
        en = max(0.0, min(1.0, self.sp_energy.step(energy_raw, dt, 14.0, 0.85)))
        st = max(0.0, min(1.0, self.sp_stress.step(max(0.0, energy_raw - en) * 2.5, dt, 18.0, 0.75)))

        # Smoothed prosody: pitch glides (~60 ms) while voiced, holds through short consonant gaps and eases back to
        # a neutral 0.42 after ~0.4 s of silence; brightness likewise.
        dtc = max(1e-4, min(0.1, dt))
        fp = self._frame_pitch
        if fp is not None and energy_raw > 0.02:
            self._unvoiced_for = 0.0
            self._pitch_s += (fp - self._pitch_s) * (1.0 - math.exp(-dtc / 0.06))
            self._voiced_s += (1.0 - self._voiced_s) * (1.0 - math.exp(-dtc / 0.05))
        else:
            self._unvoiced_for += dtc
            self._voiced_s += (0.0 - self._voiced_s) * (1.0 - math.exp(-dtc / 0.12))
            if self._unvoiced_for > 0.4:
                self._pitch_s += (0.42 - self._pitch_s) * (1.0 - math.exp(-dtc / 0.6))
        fb = self._frame_bright
        if fb is not None:
            self._bright_s += (fb - self._bright_s) * (1.0 - math.exp(-dtc / 0.08))
        elif self._unvoiced_for > 0.4:
            self._bright_s += (0.5 - self._bright_s) * (1.0 - math.exp(-dtc / 0.6))

        try:
            pros = self.prosody.update(pros_frame, pros_sr, dt, float(cfg["noiseGate"]), float(cfg["gain"]))
        except Exception:                                  # never let prosody break lip-sync
            pros = {k: 0.0 for k in PROSODY_FIELDS}
        self._prosody_snapshot = self._prosody_status_snapshot()

        self.active_viseme = code
        self.active_source = source
        self.live_rms = rms
        self.live_bands = bands
        self.live_scores = {k: round(v, 4) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:6]}
        self.live_axes = {
            "open": o, "form": f, "funnel": fn, "x": mx, "shrug": sh, "tongue": tg,
            "press": pr, "energy": en, "stress": st, "onset": self.live_onset,
            "pitch": max(0.0, min(1.0, self._pitch_s)), "voiced": max(0.0, min(1.0, self._voiced_s)),
            "brightness": max(0.0, min(1.0, self._bright_s)),
            **pros,
        }
        ms = (time.perf_counter() - t_start) * 1000.0
        self.frame_ms = ms if self.frame_ms == 0.0 else self.frame_ms * 0.9 + ms * 0.1
        return self.live_axes

    def sample(self) -> dict[str, float]:
        """Latest frame dict (same as the last step() result): mouth axes + energy / pitch + the 5 prosody fields."""
        return dict(self.live_axes)

    def prosody_status(self) -> dict[str, Any]:
        return dict(self._prosody_snapshot)

    def _prosody_status_snapshot(self) -> dict[str, Any]:
        pr = self.prosody
        return {**{k: round(v, 4) for k, v in pr.values().items()}, "f0Hz": round(pr.f0_hz, 1),
                "medianHz": round(100.0 * 2.0 ** (pr.median_st / 12.0), 1) if pr.median_st is not None else 0.0,
                "cpuMs": round(pr.cpu_ms, 4)}

    # --------------------------------------------------- user voice (backchannel)
    def feed_user_energy(self, rms: float, t: float | None = None) -> bool:
        """Live mic / call RMS of the USER (not her voice). True when a speech -> micro-pause pulse fired."""
        with self.lock:
            return self.user_voice.feed(rms, time.perf_counter() if t is None else t)

    def note_user_speech(self, active: bool, t: float | None = None) -> None:
        """Explicit user VAD edge (e.g. from the duplex call's own VAD)."""
        with self.lock:
            now = time.perf_counter() if t is None else t
            self.user_voice.note(bool(active), now)
            self.user_voice.tick(now)

    def user_voice_state(self) -> dict[str, Any]:
        with self.lock:
            now = time.perf_counter()
            self.user_voice.tick(now)
            return self.user_voice.state(now)

    # ----------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engines": list(LIPSYNC_ENGINES),
            "mfccScoring": str(self.cfg.get("mfccScoring") or "cosine"),
            "coarticulation": self._coart(),
            "activeViseme": self.active_viseme,
            "activeLabel": self.visemes.get(self.active_viseme, {}).get("label", ""),
            "source": self.active_source,
            "rms": round(self.live_rms, 5),
            "f0Hz": round(self.live_f0_hz, 1),
            "scores": dict(self.live_scores),
            "onset": round(self.live_onset, 3),
            "axes": {k: round(v, 4) for k, v in self.live_axes.items()},
            "frameMs": round(self.frame_ms, 3),
            "prosody": self.prosody_status(),
            "mfccReady": self._profile is not None,
            "mfccError": self._profile_err,
            "lastMfcc": [round(float(x), 3) for x in self.last_mfcc] if self.last_mfcc is not None else None,
            "calibratedPhonemes": self.calibrated_visemes(),
            "phonemeMap": dict(MFCC_PHONEMES),
        }
