# Hikari Studio — Modular 60 FPS AI VTuber Platform

## Architecture
- `engine/` — Native DLang OpenGL 3.2 + GLSL 330 real-time Inochi2D renderer (`hikari-live-engine.exe`) with hot-tweakable GPU Contrast-Adaptive Sharpening (CAS), directional dual rim-lighting, selective specular bloom, room ambient light-wrap, and color grading.
- `core/` — Clean separation of concerns:
  - `texture_upscaler.py` — Non-destructive 2× anime lineart texture upscaler (`Hikari-v28-HD.inp`).
  - `events.py` — Priority-ordered sync/async `HookRegistry` with error isolation & microsecond telemetry.
  - `plugins.py` — Auto-discovering `PluginManager` loading drop-in `plugins/*.py`.
  - `rig_controller.py` — 60 FPS rig compositor (breath, blink, mouse look-at, expressions, 3-bone rigid toggle arms, lip-sync, video motion blend, raw overrides, plugin hooks).
  - `engine_bridge.py` — Thread-safe IPC bridge streaming premultiplied RGB+Alpha frames.
  - `config.py` — Presets & atomic `settings.json` persistence.
- `modules/` — Swappable capability modules (`lipsync.py`, `video_retarget.py`, `ai_companion.py`).
- `plugins/` — Drop-in plugins (`auto_director.py`, `head_pat_physics.py`, `webhook_bridge.py`).
- `web/` — Modular ES-module frontend (`index.html`, `obs.html`, `css/studio.css`, `js/panels/*.js`).

## Authoring a Plugin
Drop a `.py` file into `plugins/` subclassing `core.plugins.StudioPlugin`:

```python
from core.plugins import StudioPlugin

class MyPlugin(StudioPlugin):
    id = "my_plugin"
    name = "My Custom Plugin"
    version = "1.0.0"
    description = "Reacts to live rig frames or chat."

    def register(self, hooks, ctx):
        hooks.register(self.id, "on_frame_params", self.on_frame, priority=80)

    def on_frame(self, ctx, frame):
        # frame["params"], frame["camera"], frame["energy"], frame["dt"]
        pass
```

## External Automation / Webhook Hook API
Trigger expressions, rigid arm waves, shader presets, or live speech from OBS, Stream Deck, or external agents:
```bash
curl -X POST http://127.0.0.1:8800/api/hook/trigger \
  -H "Content-Type: application/json" \
  -d '{"action":"wave","duration":3.5}'
```
Supported actions: `wave`, `cheer`, `expression`, `shader_preset`, `say`.
