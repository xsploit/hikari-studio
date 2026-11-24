"""Thread-safe IPC bridge to the native DLang OpenGL 3.2 + GLSL 330 engine (hikari-live-engine.exe).
Streams premultiplied RGBA frames from GPU and encodes ultra-fast dual RGB+Alpha JPEGs
for 60 FPS WebSocket delivery with zero chroma halos and full transparency support.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import struct
import subprocess
import threading
import time
from typing import Any

import cv2
import numpy as np

# Cap OpenCV's internal worker pool (default spawns 30-60+ idle threads on high-core CPUs!)
try:
    cv2.setNumThreads(2)
except Exception:
    pass


def _hex_to_rgb01(hx: str, fallback: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    s = str(hx or "").strip().lstrip("#")
    if len(s) == 6:
        try:
            return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
        except Exception:
            pass
    return fallback


START_DEADLINE_SEC = 60.0      # READY handshake (model build + GL init)
FRAME_DEADLINE_SEC = 5.0       # a single frame read longer than this = wedged engine -> kill it
STOP_LOCK_WAIT_SEC = 1.0


class NativeEngineBridge:
    def __init__(self, exe_path: Path, sdl_path: Path, model_path: Path, width: int = 960, height: int = 1080) -> None:
        self.exe_path = exe_path
        self.sdl_path = sdl_path
        self.model_path = model_path
        self.width = width
        self.height = height
        self._frame_bytes = width * height * 4
        # Pre-allocated reusable frame buffers: eliminates ~250 MB/sec of per-frame GC churn!
        self._raw_buf = bytearray(self._frame_bytes)
        self._raw_view = memoryview(self._raw_buf)
        self._rgba_view = np.frombuffer(self._raw_buf, dtype=np.uint8).reshape(height, width, 4)
        self._bgr_buf = np.empty((height, width, 3), dtype=np.uint8)
        self._alpha_buf = np.empty((height, width), dtype=np.uint8)
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._closing = threading.Event()
        self._frame_started: float | None = None          # set while a frame round-trip is in flight
        self.stderr_tail: deque[str] = deque(maxlen=50)
        self.kills = 0
        # Set once here and replaced only by a successful start(): an old engine's exiting stderr reader
        # can never wipe the new engine's parameter list.
        self.param_names: list[str] = []
        self.param_index: dict[str, int] = {}
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="engine-watchdog")
        self._watchdog.start()

    # ------------------------------------------------------------------ robustness
    def _watchdog_loop(self) -> None:
        """Kill a wedged engine so a blocked pipe read returns (EOF) and the lock is released."""
        while True:
            time.sleep(0.5)
            started = self._frame_started
            proc = self._proc
            if started is not None and proc is not None and (time.perf_counter() - started) > FRAME_DEADLINE_SEC:
                self._kill(proc)

    def _kill(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                self.kills += 1
        except Exception:
            pass

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep reading stderr after READY so a chatty engine can never block on a full pipe."""
        try:
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, b""):
                self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    def start(self) -> None:
        with self._lock:
            if self._closing.is_set():
                raise RuntimeError("hikari-live-engine is shutting down")
            if self._proc is not None and self._proc.poll() is None:
                return
            self._proc = subprocess.Popen(
                [str(self.exe_path), str(self.sdl_path), str(self.model_path),
                 "--width", str(self.width), "--height", str(self.height)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            assert self._proc.stderr is not None
            proc = self._proc
            box: list[bytes] = []
            reader = threading.Thread(target=lambda: box.append(proc.stderr.readline()), daemon=True)
            reader.start()
            reader.join(START_DEADLINE_SEC)
            if reader.is_alive() or not box:
                self._kill(proc)
                self._proc = None
                raise RuntimeError(f"hikari-live-engine did not report READY within {START_DEADLINE_SEC:.0f} s")
            ready = box[0].decode("utf-8", errors="replace").strip()
            if not ready.startswith("READY "):
                self._kill(proc)
                self._proc = None
                raise RuntimeError(f"hikari-live-engine failed to start: {ready[:200]}")
            threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True, name="engine-stderr").start()
            parts = ready.split(" ", 2)
            self.param_names = parts[2].split("|") if len(parts) == 3 else []
            self.param_index = {name: i for i, name in enumerate(self.param_names)}

    def stop(self) -> None:
        """Clean shutdown that can never wedge: graceful QUIT if the lock is free, otherwise kill the
        process first (unblocking any stalled frame read) and then reclaim the lock with a deadline."""
        self._closing.set()
        if self._lock.acquire(timeout=STOP_LOCK_WAIT_SEC):
            try:
                self._stop_locked()
            finally:
                self._lock.release()
            return
        self._kill(self._proc)
        if self._lock.acquire(timeout=3.0):
            try:
                self._stop_locked()
            finally:
                self._lock.release()

    def _stop_locked(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin:
                    proc.stdin.write(b"QUIT\n")
                    proc.stdin.flush()
                proc.wait(timeout=2.0)
            except Exception:
                self._kill(proc)
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            self._proc = None

    def reload_model(self, candidate: Path) -> None:
        """Load a candidate before replacing the last working puppet on disk."""
        with self._lock:
            previous = self.model_path
            self._stop_locked()
            self.model_path = candidate
            try:
                self.start()
                candidate.replace(previous)
            except Exception:
                self._stop_locked()
                self.model_path = previous
                self.start()
                raise
            self.model_path = previous

    def switch_model(self, target: Path) -> None:
        """Live-switch to another puppet file. Nothing on disk is replaced (unlike reload_model). If `target` fails to
        start, the previous puppet is restarted and the error re-raised; param_names only change on success."""
        target = Path(target)
        if not target.is_file():
            raise FileNotFoundError(f"Puppet not found: {target.name}")
        with self._lock:
            if self._closing.is_set():
                raise RuntimeError("hikari-live-engine is shutting down")
            previous = self.model_path
            self._stop_locked()
            self.model_path = target
            try:
                self.start()
            except Exception:
                self._stop_locked()
                self.model_path = previous
                try:
                    self.start()
                except Exception as exc:                  # previous engine also failed: report the original error
                    self.stderr_tail.append(f"[switch_model] rollback to {previous.name} failed: {exc}")
                raise

    def render_packet(
        self,
        dt: float,
        camera: dict[str, Any],
        shader: dict[str, Any],
        physics_enabled: bool,
        param_map: dict[str, tuple[float, float]],
        jpeg_quality: int = 92,
    ) -> bytes:
        """Render one GPU frame and return binary packet: [u32 rgb_len][rgb_jpg][alpha_jpg]."""
        if self._closing.is_set():
            raise RuntimeError("hikari-live-engine is shutting down")
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self.start()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout

            kr, kg, kb = _hex_to_rgb01(shader.get("keyColor", "#ffb8e0"), (1.0, 0.72, 0.88))
            fr, fg, fb = _hex_to_rgb01(shader.get("fillColor", "#8ce0ff"), (0.55, 0.88, 1.0))
            tr, tg, tb = _hex_to_rgb01(shader.get("tint", "#ffffff"), (1.0, 1.0, 1.0))
            ar, ag, ab = _hex_to_rgb01(shader.get("ambientColor", "#ffffff"), (1.0, 1.0, 1.0))

            header = (
                f"F {dt:.6f} "
                f"{float(camera.get('scale', 1.62)):.4f} {float(camera.get('x', 0.0)):.4f} {float(camera.get('y', 365.0)):.4f} "
                f"{1 if physics_enabled else 0} {1 if shader.get('enabled', True) else 0} "
                f"{float(shader.get('rimStrength', 0.95)):.4f} {float(shader.get('rimWidth', 1.0)):.4f} "
                f"{kr:.4f} {kg:.4f} {kb:.4f} {fr:.4f} {fg:.4f} {fb:.4f} "
                f"{float(shader.get('keyAngle', 135.0)):.2f} "
                f"{float(shader.get('bloomStrength', 0.65)):.4f} {float(shader.get('bloomThresh', 0.90)):.4f} "
                f"{float(shader.get('vibrance', 1.06)):.4f} {float(shader.get('contrast', 1.04)):.4f} "
                f"{float(shader.get('brightness', 1.00)):.4f} {float(shader.get('warmth', 0.0)):.4f} "
                f"{tr:.4f} {tg:.4f} {tb:.4f} "
                f"{ar:.4f} {ag:.4f} {ab:.4f} {float(shader.get('ambientWrap', 0.25)):.4f} "
                f"{float(shader.get('chromaAb', 0.0)):.4f} {float(shader.get('sharpness', 0.72)):.4f}"
            )

            vals = []
            for name in self.param_names:
                vx, vy = param_map.get(name, (0.0, 0.0))
                vals.append(f"{vx:.5f} {vy:.5f}")
            cmd = f"{header} | {' '.join(vals)}\n".encode("ascii")

            self._frame_started = time.perf_counter()     # watchdog deadline for this round-trip
            try:
                self._proc.stdin.write(cmd)
                self._proc.stdin.flush()

                # Read exact frame bytes straight into pre-allocated buffer (zero 4 MB/frame allocations!)
                got = 0
                total = self._frame_bytes
                while got < total:
                    n = self._proc.stdout.readinto(self._raw_view[got:])
                    if not n:
                        raise RuntimeError("hikari-live-engine closed stdout unexpectedly")
                    got += n
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise RuntimeError(f"hikari-live-engine pipe failed: {exc.__class__.__name__}") from None
            finally:
                self._frame_started = None

            q = max(75, min(98, int(jpeg_quality)))
            rgb_flags = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
                rgb_flags.extend([int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)])
            cv2.cvtColor(self._rgba_view, cv2.COLOR_RGBA2BGR, dst=self._bgr_buf)
            np.copyto(self._alpha_buf, self._rgba_view[..., 3])
            ok1, j_rgb = cv2.imencode(".jpg", self._bgr_buf, rgb_flags)
            # Lossless fast PNG for single-channel alpha eliminates 8x8 JPEG edge mosquito artifacts completely!
            ok2, p_a = cv2.imencode(".png", self._alpha_buf, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            if not (ok1 and ok2):
                raise RuntimeError("Frame encoding failed")
            rgb_b = j_rgb.tobytes()
            a_b = p_a.tobytes()
            return struct.pack(">I", len(rgb_b)) + rgb_b + a_b
