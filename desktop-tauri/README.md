# Hikari Studio — Desktop Shell (Rust + Tauri v2)

A small native window around Hikari Studio. It starts the Python backend if
needed, shows the stage, and gives you tray controls. Nothing in `server.py`
or `web/` is changed.

## Run

```
launch-desktop.cmd          # builds once (~2 min), then starts
launch-desktop.cmd build    # rebuild after changing the Rust code
```

Or by hand: `cd src-tauri && cargo build --release`, then run
`src-tauri\target\release\hikari-desktop.exe`. No Node or npm needed.

## What it does

1. Opens a dark (`#0e0e11`) window right away — no white flash.
2. Checks `http://127.0.0.1:8800/api/state`.
   - **Already up** (you started `server.py` yourself): it just shows the
     stage and never stops that server.
   - **Not up**: it starts `server.py` with no console window, shows a
     "Starting… Ns" counter, and switches to `http://127.0.0.1:8800/` as soon
     as `/api/state` answers.
3. If the backend it started crashes, it goes back to the splash and restarts
   it (up to 3 times). Backend output goes to `logs\backend.log`.
4. When you close the window or pick **Quit**, the windows hide right away
   and a backend the shell started gets `POST /api/shutdown`, so it can end
   Fish calls and save settings. If it hasn't exited after 2.5 s (or the
   request fails), it and `hikari-live-engine.exe` are killed. A backend you
   started yourself is never touched. They live in a Windows Job Object
   (`KILL_ON_JOB_CLOSE`), so they also die if the shell crashes or is killed
   from Task Manager. No leftover processes.

## Tray menu

| Item | Does |
|---|---|
| Show / Hide Hikari | Hide to tray / bring back (left-click the tray icon does the same) |
| Always on Top | Keep the main window above other windows |
| Open OBS Transparent Window | Frameless, see-through, always-on-top window with `/obs`. Drag it with the left mouse button; close with Alt+F4 |
| Reload Stage | Reload the stage (and the OBS window if open) |
| Quit | Close everything |

## Settings (environment variables)

| Variable | Default |
|---|---|
| `PYTHON` | Newest `%LOCALAPPDATA%\Programs\Python\Python3xx\python.exe`, else `python` on PATH |
| `HIKARI_STUDIO_DIR` | First folder above the exe that contains `server.py` |

## Why it's light

- One Rust process (~28 MB) plus the WebView2 it needs; idle CPU is ~0.
- All waiting (health checks, watching the backend) runs on Tokio tasks.
  While the backend runs, the shell waits on its exit signal — no polling.
- The health check is a raw 12-byte HTTP read: no HTTP client library.
- Dependencies: `tauri` (default features off), `tokio`, `windows-sys`.

## Files

```
desktop-tauri/
  launch-desktop.cmd
  dist/index.html                splash page shown before the stage loads
  src-tauri/
    Cargo.toml  build.rs  tauri.conf.json
    capabilities/obs-drag.json   only lets the OBS page start a window drag
    icons/
    src/main.rs                  window, tray, backend supervisor
    src/job.rs                   Windows Job Object (+ test: cargo test)
```
