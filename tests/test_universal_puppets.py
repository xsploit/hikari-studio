"""Verification: Universal Inochi2D puppets (core/puppet_adapter.py + engine / rig / config / server wiring).

  * inspect_puppet: magic validation, param_meta, rest-pose bounds, recommended upper-body camera (Hikari matches
    her hand-tuned 1.82 / 355 framing; Hiyori / Mao are framed head + hat to chest)
  * UniversalParamMapper: every frame starts from the puppet's defaults (eyes open), canonical -> real ranges for
    1D + vec2 params, Japanese MINT / English Live2D / official Inochi2D names, Hikari passthrough, mapped_channels
  * RigController.reload_puppet_meta: Hikari output unchanged, library puppets get mapped frames + their own overrides
  * config: stage.activeModel / savedModelCameras defaults + sanitizing (survive a reload)
  * importer: .inp copy, zip-slip / bad extension / oversize rejected, Live2D .zip -> MINT -> .inp (if MINT is installed)
  * NativeEngineBridge.switch_model on the real engine: switch, never touches files, rolls back on a bad puppet
  * GET /api/models, POST /api/models/select, POST /api/models/import, DELETE /api/models/{id}, customizer while a
    library puppet is live - real server.py handlers on a random loopback port (never 8800), temp settings + library
  * v2: universal spring physics (Hiyori 36 / Mao 15 physics params, chain cascade, stability, settle when off),
    arm / shoulder / wrist mapping + emote arm motion, Live Parameter Inspector (categories, toggles, overrides on any
    puppet incl. Hikari, cleared on switch), 1-click mouth flap test through the real lip-sync path, Mao's clean
    synthesized mouth (synthesize_missing_mouth), GET/POST /api/models/params + POST /api/models/test-mouth

    python tests/test_universal_puppets.py        (or: python -m pytest tests/test_universal_puppets.py)
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from aiohttp import FormData  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import core.puppet_adapter as pa  # noqa: E402
from core.config import ConfigStore, default_config, sanitize_section  # noqa: E402
from core.events import HookRegistry  # noqa: E402
from core.rig_controller import RigController  # noqa: E402

MODELS = HERE / "models"
HIKARI = MODELS / "Hikari-v28-Custom.inp"
HIYORI = MODELS / "library" / "Hiyori.inp"
MAO = MODELS / "library" / "Mao.inp"
ENGINE_EXE = HERE / "engine" / "hikari-live-engine.exe"
SDL_DLL = Path("D:/Hikari-Rig-Tools/Inochi-Creator-0.8.6/SDL2.dll")
LIVE2D_SAMPLE = Path(r"C:\Users\SUBSECT\Documents\Codex\2026-09-06\i-need-a-sort-of-wiztree-4\outputs\HIKKY-Live2D")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail if not ok else ""))
    print(("  PASS " if ok else "  FAIL ") + name + ("" if ok else f"   -> {detail[:300]}"), flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class FakeLipSync:
    """Silent lip-sync source (the rig only reads step())."""

    def step(self, dt: float, t_now: float) -> dict:
        return {"open": 0.0, "form": 0.0, "funnel": 0.0, "x": 0.0, "shrug": 0.0, "tongue": 0.0, "energy": 0.0,
                "stress": 0.0, "onset": 0.0, "pitch": 0.5, "voiced": 0.0, "press": 0.0}


def hikari_defaults() -> dict:
    from server import _read_param_meta
    return _read_param_meta(HIKARI)[0]


def new_rig() -> RigController:
    return RigController(hikari_defaults(), FakeLipSync(), HookRegistry())


def run(rig: RigController, cfg: dict, frames: int = 30) -> dict:
    p = {}
    for _ in range(frames):
        p, _, _ = rig.step(1.0 / 60.0, cfg, None)
    return p


def quiet_cfg(**rig) -> dict:
    cfg = default_config()
    cfg["rig"].update({"autoBlink": False, "autoGaze": False, "mouseLookAt": False, "autoVfx": False, **rig})
    return cfg


# ------------------------------------------------------------------------------------------------ inspection
def test_inspect(tmp: Path) -> None:
    bad = tmp / "bad.inp"
    bad.write_bytes(b"NOTANINP" + b"\0" * 64)
    try:
        pa.read_puppet_json(bad)
        check("inspect. bad magic rejected", False, "no error")
    except pa.PuppetFormatError:
        check("inspect. bad magic rejected", True)
    trunc = tmp / "trunc.inp"
    trunc.write_bytes(pa.INP_MAGIC + (10_000).to_bytes(4, "big") + b"{}")
    try:
        pa.read_puppet_json(trunc)
        check("inspect. truncated payload rejected", False)
    except pa.PuppetFormatError:
        check("inspect. truncated payload rejected", True)

    hk = pa.inspect_puppet(HIKARI, "hikari")
    cam = hk.recommended_camera
    check("inspect. Hikari auto camera ~ her hand-tuned framing (1.82 / 355)",
          abs(cam["scale"] - 1.82) / 1.82 < 0.08 and abs(cam["y"] - 355) < 45 and abs(cam["x"]) < 20, str(cam))
    check("inspect. Hikari mapper = passthrough", hk.passthrough and "mouth" in hk.mapped_channels)

    for path, mid, n_params in ((HIYORI, "Hiyori", 67), (MAO, "Mao", 130)):
        info = pa.inspect_puppet(path, mid)
        c = info.recommended_camera
        check(f"inspect. {mid} param_meta ({n_params} params, vec2 + ranges)",
              len(info.param_meta) == n_params and any(m["isVec2"] and m["name"] == "角度" and m["max"][0] == 30.0
                                                       for m in info.param_meta), str(len(info.param_meta)))
        check(f"inspect. {mid} name from file stem (MINT meta name is 'model')", info.name == mid, info.name)
        # where the eyes / crown land in the 960x1080 frame
        ey_px = 540 + c["scale"] * (info.eye_center[1] + c["y"])
        ex_px = 480 + c["scale"] * (info.eye_center[0] + c["x"])
        top_px = 540 + c["scale"] * (info.head_top + c["y"])
        check(f"inspect. {mid} upper-body framing: crown visible, eyes upper third, centred",
              0 < top_px < 160 and 250 < ey_px < 520 and abs(ex_px - 480) < 20,
              f"top {top_px:.0f} eyes {ey_px:.0f},{ex_px:.0f} cam {c}")
        frame_world_h = 1080 / c["scale"]
        body_h = info.bounds[3] - info.bounds[2]
        check(f"inspect. {mid} not zoomed to the knees: frame shows < 75 % of the figure",
              frame_world_h < 0.75 * body_h, f"{frame_world_h:.0f} / {body_h:.0f}")
        check(f"inspect. {mid} camera within config limits",
              0.2 <= c["scale"] <= 6.0 and abs(c["x"]) <= 4000 and abs(c["y"]) <= 4000, str(c))
    check("inspect. cache hit returns same object", pa.inspect_puppet(HIYORI, "Hiyori") is pa.inspect_puppet(HIYORI, "Hiyori"))


# ------------------------------------------------------------------------------------------------ mapper
def test_mapper() -> None:
    hd = hikari_defaults()
    hiyori = pa.inspect_puppet(HIYORI, "Hiyori")
    mao = pa.inspect_puppet(MAO, "Mao")
    m = pa.UniversalParamMapper(hiyori.param_meta, "Hiyori")
    out = m.map_frame(dict(hd))
    check("mapper. Hiyori frame has every puppet param", set(out) == set(hiyori.param_defaults), str(set(hiyori.param_defaults) ^ set(out)))
    check("mapper. rest frame: eyes OPEN at the author default 1.0",
          out["左目\u3000開閉"] == (1.0, 0.0) and out["右目\u3000開閉"] == (1.0, 0.0), str(out["左目\u3000開閉"]))
    check("mapper. untouched params keep author defaults (あし 1.0, 左手B 回転 10.0)",
          out["あし"] == (1.0, 0.0) and out["左手B 回転"] == (10.0, 0.0))
    check("mapper. rest mouth = default form 1.0, closed", out["口\u3000変形 / 口\u3000開閉"] == (1.0, 0.0), str(out["口\u3000変形 / 口\u3000開閉"]))

    f = dict(hd)
    f.update({"Angle": (15.0, -12.0), "Angle Z": (30.0, 0.0), "Body Angle X": (5.0, 0.0), "Body Angle Z": (-10.0, 0.0),
              "EyeBall": (0.5, -1.0), "Left Eye Open": (0.0, 0.0), "Right Eye Open": (1.2, 0.0),
              "VB Mouth": (-0.5, 0.85), "Breath": (0.6, 0.0), "Cheek": (1.0, 0.0), "Left Brow Y": (0.4, 0.0)})
    o = m.map_frame(f)
    check("mapper. Head X/Y -> 角度 vec2 in -30..30 (degrees kept)", o["角度"] == (15.0, -12.0), str(o["角度"]))
    check("mapper. Head Z -> 角度 Z", abs(o["角度 Z"][0] - 30.0) < 1e-6)
    check("mapper. Body X/Z -> 体の回転 X/Z in -10..10",
          abs(o["体の回転\u3000X"][0] - 5.0) < 1e-6 and abs(o["体の回転\u3000Z"][0] + 10.0) < 1e-6)
    check("mapper. gaze -> 目玉 vec2", o["目玉"] == (0.5, -1.0))
    check("mapper. blink 0 -> 左目 0, wide 1.2 -> 右目 1.2 (clamped to max)",
          o["左目\u3000開閉"][0] == 0.0 and abs(o["右目\u3000開閉"][0] - 1.2) < 1e-4, f"{o['左目　開閉']} {o['右目　開閉']}")
    mo = o["口\u3000変形 / 口\u3000開閉"]
    check("mapper. Mouth Open 0.85 -> 口開閉 .y 0.85, form -0.5 -> -0.5 (default 1, min -2)",
          abs(mo[1] - 0.85) < 1e-6 and abs(mo[0] - (-0.5)) < 1e-6, str(mo))
    check("mapper. breath / blush / brow", abs(o["呼吸"][0] - 0.6) < 1e-6 and abs(o["照れ"][0] - 1.0) < 1e-6
          and abs(o["左眉\u3000上下"][0] - 0.4) < 1e-6)
    check("mapper. Hiyori mapped channels",
          m.mapped_channels == ["head", "body", "eyes", "gaze", "brows", "mouth", "arms", "breath", "blush"], str(m.mapped_channels))

    mm = pa.UniversalParamMapper(mao.param_meta, "Mao")
    f = dict(hd)
    f.update({"VB Mouth": (0.1, 0.85), "Jaw Open": (0.85, 0.0), "Cheek Puff": (0.7, 0.0), "Eye Star": (1.0, 0.0),
              "VFX Sparkles": (0.5, 0.0)})
    o = mm.map_frame(f)
    check("mapper. Mao talking -> あ vowel dominant", o["あ"][0] > 0.6 and o["あ"][0] > max(o[v][0] for v in "いうえお"),
          str({v: o[v] for v in "あいうえお"}))
    f2 = dict(hd)
    f2.update({"VB Mouth": (-0.6, 0.6), "Mouth Funnel": (1.0, 0.0)})
    o2 = mm.map_frame(f2)
    check("mapper. Mao round mouth -> お", o2["お"][0] > 0.5 and o2["お"][0] > o2["あ"][0], str({v: o2[v] for v in "あいうえお"}))
    check("mapper. Cheek Puff -> むくれ口, Eye Star -> 目 エフェクト, Sparkles -> 顔インク",
          abs(o["むくれ口"][0] - 0.7) < 1e-6 and o["目\u3000エフェクト"][0] == 1.0 and abs(o["顔インク\u3000表示"][0] - 0.5) < 1e-6)
    check("mapper. Mao 上がり口 keeps its default 1 at rest", mm.map_frame(dict(hd))["上がり口"] == (1.0, 0.0))
    syn = [m_["name"] for m_ in mao.param_meta if m_["name"] in "あいうえお" and m_["bound"]]
    check("mapper. Mao: synthesized vowel mouth bound (あいうえお) -> mouth mapped, nothing unbound",
          "mouth" in mm.mapped_channels and not mm.unbound_channels and "vfx" in mm.mapped_channels and len(syn) == 5,
          f"{mm.mapped_channels} / {mm.unbound_channels} / {syn}")

    # English Live2D + official Inochi2D names (synthetic puppets)
    def meta(name, mn, mx, d, vec2=False):
        return {"name": name, "isVec2": vec2, "min": [mn, mn if vec2 else 0.0], "max": [mx, mx if vec2 else 1.0],
                "defaults": [d, d if vec2 else 0.0], "bound": True}
    en = [meta("ParamAngleX", -30, 30, 0), meta("ParamAngleY", -30, 30, 0), meta("ParamAngleZ", -30, 30, 0),
          meta("ParamEyeLOpen", 0, 1, 1), meta("ParamEyeROpen", 0, 1, 1), meta("ParamEyeBallX", -1, 1, 0),
          meta("ParamMouthOpenY", 0, 1, 0), meta("ParamMouthForm", -1, 1, 0), meta("ParamBodyAngleX", -10, 10, 0),
          meta("ParamBreath", 0, 1, 0), meta("ParamCheek", 0, 1, 0), meta("ParamBrowLY", -1, 1, 0), meta("ParamHairFront", -1, 1, 0)]
    me = pa.UniversalParamMapper(en, "EnglishRig")
    f = dict(hd)
    f.update({"Angle": (-15.0, 6.0), "VB Mouth": (0.4, 0.7), "Left Eye Open": (0.5, 0.0), "Right Eye Open": (1.0, 0.0)})
    o = me.map_frame(f)
    check("mapper. English Live2D names (ParamAngleX / ParamEyeLOpen / ParamMouthOpenY ...)",
          o["ParamAngleX"][0] == -15.0 and o["ParamAngleY"][0] == 6.0 and o["ParamEyeLOpen"][0] == 0.5
          and o["ParamEyeROpen"][0] == 1.0 and abs(o["ParamMouthOpenY"][0] - 0.7) < 1e-6 and abs(o["ParamMouthForm"][0] - 0.4) < 1e-6
          and o["ParamHairFront"] == (0.0, 0.0), str(o))
    off = [meta("Head:: Yaw-Pitch", -1, 1, 0, True), meta("Head:: Roll", -1, 1, 0), meta("Eye:: Left:: Blink", 0, 1, 0),
           meta("Eye:: Right:: Blink", 0, 1, 0), meta("Eye:: Left:: XY", -1, 1, 0, True), meta("Mouth:: Open", 0, 1, 0),
           meta("Body:: Yaw-Pitch", -1, 1, 0, True), meta("Breath", 0, 1, 0)]
    mi = pa.UniversalParamMapper(off, "OfficialRig")
    f = dict(hd)
    f.update({"Angle": (15.0, -30.0), "Left Eye Open": (0.0, 0.0), "Right Eye Open": (1.0, 0.0), "VB Mouth": (0.0, 0.5),
              "EyeBall": (0.3, 0.2), "Body Angle X": (10.0, 0.0)})
    o = mi.map_frame(f)
    check("mapper. official Inochi2D names (Head:: Yaw-Pitch vec2, Eye:: Left:: Blink inverted, Mouth:: Open)",
          o["Head:: Yaw-Pitch"] == (0.5, -1.0) and o["Eye:: Left:: Blink"][0] == 1.0 and o["Eye:: Right:: Blink"][0] == 0.0
          and o["Mouth:: Open"][0] == 0.5 and o["Eye:: Left:: XY"] == (0.3, 0.2) and o["Body:: Yaw-Pitch"][0] == 1.0, str(o))
    check("mapper. official rig channels", set(mi.mapped_channels) >= {"head", "eyes", "gaze", "mouth", "body", "breath"},
          str(mi.mapped_channels))

    hk = pa.inspect_puppet(HIKARI, "")                  # Hikari-native params, any id -> passthrough
    mh = pa.UniversalParamMapper(hk.param_meta, "SomeHikariVariant")
    f = dict(hd)
    f["Angle"] = (12.0, 3.0)
    check("mapper. Hikari-native puppet = passthrough (identical values)", mh.passthrough and mh.map_frame(f) == f)

    nan = dict(hd)
    nan.update({"Angle": (float("nan"), float("inf")), "Left Eye Open": (float("nan"), 0.0)})
    o = m.map_frame(nan)
    check("mapper. non-finite canonical values never reach the puppet",
          all(math.isfinite(v) for xy in o.values() for v in xy) and o["左目\u3000開閉"][0] == 1.0, str(o["角度"]))


# ------------------------------------------------------------------------------------------------ rig controller
def test_rig() -> None:
    cfg = quiet_cfg()
    a = new_rig()
    pa_ = run(a, cfg, 20)
    check("rig. Hikari active: output keys = Hikari params (zero regression)", set(pa_) == set(a.defaults))
    s = a.reload_puppet_meta(None, "hikari")
    check("rig. reload hikari -> no mapper", a._mapper is None and s["passthrough"] is True)

    hiyori = pa.inspect_puppet(HIYORI, "Hiyori")
    s = a.reload_puppet_meta(hiyori.param_meta, "Hiyori")
    p = run(a, cfg, 20)
    check("rig. Hiyori active: output keys = Hiyori params", set(p) == set(hiyori.param_defaults), str(len(p)))
    check("rig. Hiyori active: eyes open, breathing moves", p["左目\u3000開閉"][0] > 0.95 and "呼吸" in p)
    check("rig. canonical frame still kept for telemetry", "Angle" in a.last_canonical and "VB Mouth" in a.last_canonical)
    check("rig. reload summary exposes mapped channels", "mouth" in s["mappedChannels"] and s["passthrough"] is False, str(s))

    cfg2 = quiet_cfg(overrides={"VB Mouth": [0.0, 0.85], "Jaw Open": [0.85, 0.0], "左手": [0.7, 0.0]})
    p = run(a, cfg2, 5)
    check("rig. canonical override -> mapped (Mouth Open 0.85)", abs(p["口\u3000変形 / 口\u3000開閉"][1] - 0.85) < 1e-6,
          str(p["口\u3000変形 / 口\u3000開閉"]))
    check("rig. puppet-native raw override applied after mapping (左手 0.7)", abs(p["左手"][0] - 0.7) < 1e-6, str(p["左手"]))
    a.reload_puppet_meta(None, "hikari")
    p = run(a, cfg, 3)
    check("rig. switch back to hikari restores native output", set(p) == set(a.defaults) and "角度" not in p)


# ------------------------------------------------------------------------------------------------ config
def test_config(tmp: Path) -> None:
    d = default_config()
    check("config. stage.activeModel default 'hikari' + savedModelCameras {}",
          d["stage"]["activeModel"] == "hikari" and d["stage"]["savedModelCameras"] == {})
    st = tmp / "settings.json"
    st.write_text(json.dumps({"stage": {"activeModel": "Hiyori", "savedModelCameras": {
        "hikari": {"scale": 1.9, "x": 1, "y": 350}, "Mao": {"scale": "nan", "x": 0, "y": 0},
        "../evil": {"scale": 1, "x": 0, "y": 0}, "Hiyori": {"scale": 99, "x": -9999, "y": 5}}}}), encoding="utf-8")
    cs = ConfigStore(st)
    smc = cs.data["stage"]["savedModelCameras"]
    check("config. saved cameras survive a reload (sanitized, clamped, bad ids / NaN dropped)",
          cs.data["stage"]["activeModel"] == "Hiyori" and smc.get("hikari") == {"scale": 1.9, "x": 1.0, "y": 350.0}
          and "Mao" not in smc and "../evil" not in smc and smc["Hiyori"] == {"scale": 6.0, "x": -4000.0, "y": 5.0}, str(smc))
    v = sanitize_section("stage", {"activeModel": "../../x"})
    check("config. invalid activeModel id dropped", "activeModel" not in v, str(v))


# ------------------------------------------------------------------------------------------------ importer
def test_importer(tmp: Path) -> None:
    lib = tmp / "lib"
    lib.mkdir()
    src = tmp / "upload.part"
    shutil.copyfile(HIYORI, src)
    dest = pa.import_puppet_file(src, "C:\\fake\\dir\\My Puppet.inp", lib)
    check("import. .inp installed with a safe name (directories ignored)", dest == lib / "My Puppet.inp" and dest.is_file(), str(dest))
    dest2 = pa.import_puppet_file(src, "My Puppet.inp", lib)
    check("import. name clash -> unique name, nothing overwritten", dest2.name == "My Puppet-2.inp", dest2.name)
    dest3 = pa.import_puppet_file(src, "hikari.inx", lib)
    check("import. reserved id 'hikari' renamed", dest3.stem.lower() != "hikari" and dest3.suffix == ".inx", dest3.name)
    for name, payload, why in (("evil.exe", b"MZ", "bad extension"), ("fake.inp", b"not a puppet", "bad magic")):
        f = tmp / "x.part"
        f.write_bytes(payload)
        try:
            pa.import_puppet_file(f, name, lib)
            check(f"import. {why} rejected", False)
        except (pa.ImportError_, pa.PuppetFormatError):
            check(f"import. {why} rejected", True)
    zs = tmp / "slip.zip"
    with zipfile.ZipFile(zs, "w") as zf:
        zf.writestr("model/model.moc3", b"MOC3")
        zf.writestr("../../escape.txt", b"pwned")
    try:
        pa.import_puppet_file(zs, "slip.zip", lib, tmp_root=tmp)
        check("import. zip-slip rejected", False)
    except pa.ImportError_ as exc:
        check("import. zip-slip rejected", "Unsafe" in str(exc), str(exc))
    check("import. zip-slip wrote nothing outside", not (tmp.parent / "escape.txt").exists() and not (tmp / "escape.txt").exists())
    za = tmp / "abs.zip"
    with zipfile.ZipFile(za, "w") as zf:
        zf.writestr("C:/Windows/evil.moc3", b"x")
    try:
        pa.import_puppet_file(za, "abs.zip", lib, tmp_root=tmp)
        check("import. absolute path in zip rejected", False)
    except pa.ImportError_:
        check("import. absolute path in zip rejected", True)
    old = pa.MAX_IMPORT_BYTES
    pa.MAX_IMPORT_BYTES = 1024
    try:
        pa.import_puppet_file(src, "big.inp", lib)
        check("import. oversized upload rejected (>150 MB limit)", False)
    except pa.ImportError_:
        check("import. oversized upload rejected (>150 MB limit)", True)
    finally:
        pa.MAX_IMPORT_BYTES = old
    nomoc = tmp / "nomoc.zip"
    with zipfile.ZipFile(nomoc, "w") as zf:
        zf.writestr("readme.txt", b"hi")
    try:
        pa.import_puppet_file(nomoc, "nomoc.zip", lib, tmp_root=tmp)
        check("import. zip without .moc3 rejected", False)
    except pa.ImportError_:
        check("import. zip without .moc3 rejected", True)
    leftovers = [p for p in tmp.iterdir() if p.name.startswith("hikari_mint_")]
    check("import. temp work dirs cleaned up", not leftovers, str(leftovers))


def make_live2d_zip(out: Path) -> bool:
    if not LIVE2D_SAMPLE.is_dir() or not pa.DEFAULT_MINT_CLI.is_file():
        return False
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in LIVE2D_SAMPLE.rglob("*"):
            if f.is_file():
                zf.write(f, "HIKKY/" + f.relative_to(LIVE2D_SAMPLE).as_posix())
    return True


# ------------------------------------------------------------------------------------------------ real engine
def test_engine_switch(tmp: Path) -> None:
    if not ENGINE_EXE.is_file() or not SDL_DLL.is_file():
        check("engine. native engine present (skipped)", True)
        return
    from core.engine_bridge import NativeEngineBridge
    hk_copy = tmp / "hk.inp"
    shutil.copyfile(HIKARI, hk_copy)
    before = {p: sha(p) for p in (hk_copy, HIYORI)}
    eng = NativeEngineBridge(ENGINE_EXE, SDL_DLL, hk_copy)
    try:
        eng.start()
        hk_names = list(eng.param_names)
        eng.switch_model(HIYORI)
        check("engine. switch_model -> Hiyori running (67 params, model_path updated)",
              len(eng.param_names) == 67 and eng.model_path == HIYORI and "角度" in eng.param_index, str(len(eng.param_names)))
        info = pa.inspect_puppet(HIYORI, "Hiyori")
        pkt = eng.render_packet(1 / 60, info.recommended_camera, default_config()["shader"], True, info.param_defaults)
        alpha = eng._rgba_view[..., 3]
        cov = float((alpha > 16).mean())
        check("engine. Hiyori renders at the recommended camera (puppet fills the frame)", len(pkt) > 1000 and 0.25 < cov < 0.95, f"{cov:.2f}")
        bad = tmp / "broken.inp"
        bad.write_bytes(pa.INP_MAGIC + (20).to_bytes(4, "big") + b'{"nodes": 1}' + b" " * 8)
        try:
            eng.switch_model(bad)
            check("engine. broken puppet raises", False)
        except Exception:
            check("engine. broken puppet raises", True)
        check("engine. rolled back to the previous puppet (Hiyori, same params)",
              eng.model_path == HIYORI and len(eng.param_names) == 67 and eng._proc is not None and eng._proc.poll() is None)
        try:
            eng.switch_model(tmp / "missing.inp")
            check("engine. missing file -> FileNotFoundError", False)
        except FileNotFoundError:
            check("engine. missing file -> FileNotFoundError", True)
        eng.switch_model(hk_copy)
        check("engine. switch back to Hikari", eng.param_names == hk_names)
        after = {p: sha(p) for p in (hk_copy, HIYORI)}
        check("engine. switch_model never replaces / overwrites files on disk", before == after and bad.exists())
    finally:
        eng.stop()


# ------------------------------------------------------------------------------------------------ v2: physics / arms
HIYORI_PHYS_SPEC = ["髪揺れ\u3000アホ毛", "髪揺れ\u3000前", "髪揺れ\u3000後", "髪飾りの揺れ", "胸リボンの揺れ", "胸\u3000揺れ",
                    "スカートの揺れ"] + [f"[{i}]{c}" for c in ("サイドアップ左", "サイドアップ右", "前髪左", "前髪右") for i in range(7)]
MAO_PHYS_SPEC = ["髪揺れ\u3000前", "髪揺れ\u3000左横", "髪揺れ\u3000右横", "髪揺れ\u3000後", "前髪\u3000ふわ", "横髪\u3000ふわ",
                 "後ろ髪\u3000ふわ", "羽の揺れ", "帽子つばの揺れ", "首飾りの揺れ1", "首飾りの揺れ2", "パーカーひもの揺れ",
                 "ローブの揺れ\u3000左", "ローブの揺れ\u3000右", "ローブのふわ"]
INSPECTOR_KEYS = {"name", "category", "isVec2", "isToggle", "min", "max", "default", "current", "overridden", "mappedTo"}


def _norm_s(pp, x: float) -> float:
    span = (pp["max"][0] - pp["defaults"][0]) if x >= pp["defaults"][0] else (pp["defaults"][0] - pp["min"][0])
    return 0.0 if span == 0 else (x - pp["defaults"][0]) / span


def test_physics() -> None:
    hiyori = pa.inspect_puppet(HIYORI, "Hiyori")
    mao = pa.inspect_puppet(MAO, "Mao")
    mh = hiyori.make_mapper("Hiyori")
    mm = mao.make_mapper("Mao")
    check("physics. Hiyori: every spec physics param detected (35 listed + スカートめくれ = 36, all SimplePhysics outputs)",
          set(HIYORI_PHYS_SPEC) <= set(mh.physics_names) and len(mh.physics_names) == 36
          and set(hiyori.physics_settings) <= set(mh.physics_names), f"{len(mh.physics_names)} {set(HIYORI_PHYS_SPEC) - set(mh.physics_names)}")
    check("physics. Mao: exactly the 15 bound physics params (unbound 帽子リボン / メッシュ skipped)",
          sorted(mm.physics_names) == sorted(MAO_PHYS_SPEC), str(mm.physics_names))
    core = {"角度", "角度 Z", "体の回転\u3000X", "左目\u3000開閉", "口\u3000変形 / 口\u3000開閉", "呼吸", "左腕 A", "肩\u3000すくみ"}
    check("physics. core controls never treated as physics", not (core & set(mh.physics_names)))
    for nm, ok in (("ParamHairFront", True), ("ParamSkirt", True), ("Cape Sway", True), ("Bust Bounce", True),
                   ("ParamEyeLOpen", False), ("ParamAngleX", False), ("Hat 表示", False), ("Clear Gear", False),
                   ("ParamArmLA", False), ("Mouth Spring", False)):
        check(f"physics. keyword rule: {nm!r} -> {ok}", pa.is_physics_param(nm) == ok)
    hk = pa.inspect_puppet(HIKARI, "hikari")
    check("physics. Hikari untouched: passthrough mapper owns no physics (native SimplePhysics)",
          hk.make_mapper("hikari").physics is None and len(hk.physics_settings) > 0)

    # rapid head turn: hair lags, chain cascades, everything stays in range, then settles once physics is off
    meta = {m["name"]: m for m in hiyori.param_meta}
    peak: dict[str, tuple[float, float]] = {}
    finite = in_range = True
    frames = []
    for k in range(420):
        t = k / 60.0
        hx = 0.0 if t < 0.5 else min(1.0, (t - 0.5) / 0.2) * 0.7
        f = dict(hikari_defaults())
        f["Angle"] = (hx * 30.0, 0.0)
        f["Body Angle X"] = (hx * 4.0, 0.0)
        out = mh.step(f, 1.0 / 60.0, physics_enabled=k < 300)
        frames.append(out)
        for n in mh.physics_names:
            x = out[n][0]
            finite &= math.isfinite(x)
            in_range &= meta[n]["min"][0] - 1e-6 <= x <= meta[n]["max"][0] + 1e-6
            s_ = abs(_norm_s(meta[n], x))
            if s_ > peak.get(n, (0.0, 0.0))[0]:
                peak[n] = (s_, t)
    check("physics. finite + clamped to every param's [min, max]", finite and in_range)
    check("physics. rapid head turn swings the hair / ribbon (visible lag)",
          peak["髪揺れ\u3000前"][0] > 0.3 and peak["髪飾りの揺れ"][0] > 0.3 and peak["[0]サイドアップ左"][0] > 0.05,
          str({k_: round(v[0], 2) for k_, v in peak.items() if "揺れ" in k_}))
    check("physics. chain cascade: tail tip [6] peaks AFTER root [0] (wave runs down the twin-tail)",
          peak["[6]サイドアップ左"][1] > peak["[0]サイドアップ左"][1] + 0.05,
          f"{peak['[0]サイドアップ左']} {peak['[6]サイドアップ左']}")
    check("physics. never saturates on a hard turn (peak < 0.9 of range)", max(v[0] for v in peak.values()) < 0.9,
          str(max(peak.items(), key=lambda kv: kv[1][0])))
    last = frames[-1]
    check("physics. physics_enabled False -> springs settle back to the author default",
          all(abs(last[n][0] - meta[n]["defaults"][0]) < 0.02 * (meta[n]["max"][0] - meta[n]["min"][0]) for n in mh.physics_names),
          str({n: last[n] for n in mh.physics_names[:5]}))
    check("physics. map_frame stays stateless (physics params at defaults)",
          all(mh.map_frame(dict(hikari_defaults()))[n] == hiyori.param_defaults[n] for n in mh.physics_names))

    # abuse: huge / NaN inputs never break the solver
    sp = mm.physics
    ok = True
    import random
    rnd = random.Random(7)
    for _ in range(600):
        ch = {k_: rnd.uniform(-5, 5) for k_ in ("head_x", "head_y", "head_z", "body_x", "body_y", "body_z", "breath")}
        if rnd.random() < 0.05:
            ch["head_x"] = float("nan")
        out = sp.step(rnd.uniform(0.0, 0.2), ch, rnd.random() < 0.8)
        ok &= all(math.isfinite(v[0]) and -1.0 - 1e-6 <= v[0] <= 1.0 + 1e-6 for v in out.values())
    check("physics. stable under random huge / NaN / long-dt inputs (Mao ±1 params)", ok)

    # arms: mapping + emote motion through the real rig
    check("arms. Hiyori arm / shoulder / wrist params mapped",
          {mh.channel_of.get(n) for n in ("左腕 A", "右腕 A", "肩\u3000すくみ", "左手B 回転")} == {"arm_l_raise", "arm_r_raise", "shoulder", "arm_l_wrist"})
    check("arms. Mao 腕A shoulder / elbow / wrist + 肩の上下 mapped, wand + hand-shape swaps left alone",
          mm.channel_of.get("右腕A\u3000肩の回転") == "arm_r_raise" and mm.channel_of.get("右腕A\u3000肘の回転") == "arm_r_elbow"
          and mm.channel_of.get("右腕A\u3000手首の回転") == "arm_r_wrist" and mm.channel_of.get("左肩の上下") == "shoulder_l"
          and "杖の回転" not in mm.channel_of and "右手A" not in mm.channel_of and "右腕B\u3000腕のY" not in mm.channel_of)
    for nm, want in (("ParamArmLA", "arm_l_raise"), ("Arm:: Right:: Move", "arm_r_raise"), ("ParamShoulder", "shoulder"),
                     ("ParamLeftShoulderUp", "shoulder_l"), ("ParamHandRAngle", "arm_r_wrist"), ("左腕A\u3000手首の回転", "arm_l_wrist")):
        check(f"arms. rule {nm!r} -> {want}", pa.arm_rule(nm) == want, str(pa.arm_rule(nm)))
    rig = new_rig()
    rig.reload_puppet_meta(mao.param_meta, "Mao", mao.physics_settings)
    cfg = quiet_cfg()
    rest = run(rig, cfg, 20)["右腕A\u3000肩の回転"][0]
    rig.trigger_emote("greet_wave")
    seen = []
    import time as _t
    t_end = _t.perf_counter() + 1.6
    while _t.perf_counter() < t_end:
        p_, _, _ = rig.step(1.0 / 60.0, cfg, None)
        seen.append(p_["右腕A\u3000肩の回転"][0])
        _t.sleep(1 / 240)
    check("arms. Mao greet_wave raises + waves the right arm (肩の回転 up, oscillating, within -10..30)",
          max(seen) > rest + 10 and max(seen) <= 30 and (max(seen[len(seen) // 2:]) - min(seen[len(seen) // 2:])) > 2.0,
          f"rest {rest:.2f} max {max(seen):.2f} min {min(seen):.2f}")
    check("arms. resting arms only sway gently (|rest| < 3 deg)", abs(rest) < 3.0, str(rest))
    check("arms. rig asks the engine to keep MINT drivers off (spring solver owns physics)",
          rig.engine_physics(True) is False and rig.engine_physics(False) is False)
    rig.reload_puppet_meta(None, "hikari")
    check("arms. Hikari: engine physics flag passes through", rig.engine_physics(True) is True and rig.engine_physics(False) is False)


# ------------------------------------------------------------------------------------------------ v2: inspector / overrides
def test_inspector() -> None:
    mao = pa.inspect_puppet(MAO, "Mao")
    mm = mao.make_mapper("Mao")
    params, caps = pa.build_param_inspector(mao.param_meta, mm)
    by = {q["name"]: q for q in params}
    check("inspector. only bound params listed", all(m["bound"] for m in mao.param_meta if m["name"] in by)
          and len(params) == sum(1 for m in mao.param_meta if m["bound"]), str(len(params)))
    check("inspector. exact schema keys", all(set(q) == INSPECTOR_KEYS for q in params))
    magic = ["オーラ\u3000表示", "光\u3000表示", "うさぎ\u3000出現", "目\u3000エフェクト", "爆発光溜め\u3000表示", "光\u3000星"]
    check("inspector. Mao magic (aura / light / bunny / eye fx / explosion / starlight) = extras",
          all(by[n]["category"] == "extras" for n in magic), str({n: by[n]["category"] for n in magic}))
    check("inspector. toggles: オーラ 表示 / 光 表示 isToggle, うさぎ 出現 (0..30) slider, vec2 never toggle",
          by["オーラ\u3000表示"]["isToggle"] and by["光\u3000表示"]["isToggle"] and not by["うさぎ\u3000出現"]["isToggle"]
          and not by["角度"]["isToggle"])
    check("inspector. categories: physics (15) / arms / auto (head, mouth あ)",
          sum(q["category"] == "physics" for q in params) == 15 and by["右腕A\u3000肩の回転"]["category"] == "arms"
          and by["角度"]["category"] == "auto" and by["あ"]["category"] == "auto" and by["羽の揺れ"]["mappedTo"] == "spring"
          and by["角度"]["mappedTo"] == "head_x/head_y" and by["オーラ\u3000表示"]["mappedTo"] is None)
    check("inspector. capabilities (mouth now true for Mao, physicsCount 15, extrasCount > 20)",
          caps["mouth"] and caps["arms"] and caps["physicsCount"] == 15 and caps["extrasCount"] > 20
          and set(caps) == set(pa.CAPABILITY_KEYS) | {"physicsCount", "extrasCount"}, str(caps))

    rig = new_rig()
    cfg = quiet_cfg()
    rig.set_param_override("Angle Z", 12.0, 0.0)
    p = run(rig, cfg, 3)
    check("overrides. Hikari: override applied after everything", p["Angle Z"] == (12.0, 0.0), str(p["Angle Z"]))
    check("overrides. Hikari canonical telemetry frame not mutated", rig.last_canonical is not p)
    rig.reload_puppet_meta(mao.param_meta, "Mao", mao.physics_settings)
    check("overrides. cleared automatically on puppet switch", rig.rig_param_overrides == {})
    for n in ("オーラ\u3000表示", "光\u3000表示", "目\u3000エフェクト"):
        rig.set_param_override(n, 1.0)
    rig.set_param_override("うさぎ\u3000出現", 30.0)
    rig.set_param_override("羽の揺れ", 0.8)
    p = run(rig, cfg, 3)
    check("overrides. Mao extras toggled live (aura / light / eye fx 1.0, bunny 30)",
          p["オーラ\u3000表示"] == (1.0, 0.0) and p["光\u3000表示"] == (1.0, 0.0) and p["目\u3000エフェクト"] == (1.0, 0.0)
          and p["うさぎ\u3000出現"] == (30.0, 0.0))
    check("overrides. beat the auto mapper + physics too (羽の揺れ pinned 0.8)", p["羽の揺れ"] == (0.8, 0.0))
    lst, _ = rig.param_inspector(mao.param_meta, list(mao.physics_settings))
    lb = {q["name"]: q for q in lst}
    check("overrides. inspector shows overridden + live current", lb["オーラ\u3000表示"]["overridden"]
          and lb["オーラ\u3000表示"]["current"] == [1.0, 0.0] and not lb["角度"]["overridden"])
    rig.clear_param_override("羽の揺れ")
    check("overrides. clear one", "羽の揺れ" not in rig.rig_param_overrides and len(rig.rig_param_overrides) == 4)
    rig.clear_param_override(None)
    p = run(rig, cfg, 3)
    check("overrides. clear all -> author defaults back", rig.rig_param_overrides == {} and p["オーラ\u3000表示"] == (0.0, 0.0))
    for bad, exc in ((("NoSuchParam", 1.0), KeyError), (("オーラ\u3000表示", float("nan")), ValueError)):
        try:
            rig.set_param_override(*bad)
            check(f"overrides. reject {bad[0]!r}", False)
        except exc:
            check(f"overrides. reject {bad[0]!r} ({exc.__name__})", True)


# ------------------------------------------------------------------------------------------------ v2: mouth test + synthesis
def _mouth_test_trace(rig: RigController, key: str, axis: int) -> list[float]:
    import time as _t
    cfg = quiet_cfg()
    run(rig, cfg, 10)
    res = rig.trigger_mouth_test(2.2)
    vals = []
    t_end = _t.perf_counter() + 2.6
    while _t.perf_counter() < t_end:
        p_, _, _ = rig.step(1.0 / 60.0, cfg, None)
        vals.append(p_[key][axis])
        _t.sleep(1 / 200)
    vals.append(res["duration"])
    return vals


def test_mouth() -> None:
    rig = new_rig()
    tr = _mouth_test_trace(rig, "VB Mouth", 1)
    dur, tr = tr[-1], tr[:-1]
    n = len(tr)
    check("mouthtest. Hikari: 2.2 s flap through the real lip-sync path (VB Mouth open peaks, closes after)",
          dur == 2.2 and max(tr) > 0.6 and tr[-1] < 0.1, f"max {max(tr):.2f} end {tr[-1]:.2f}")
    opens = sum(1 for i in range(1, n) if tr[i - 1] < 0.25 <= tr[i])
    check("mouthtest. distinct syllable flaps (Ah-Eh-Oh-Ah-Oo: >= 4 openings)", opens >= 4, str(opens))
    hiyori = pa.inspect_puppet(HIYORI, "Hiyori")
    rig.reload_puppet_meta(hiyori.param_meta, "Hiyori", hiyori.physics_settings)
    tr = _mouth_test_trace(rig, "口\u3000変形 / 口\u3000開閉", 1)[:-1]
    check("mouthtest. Hiyori 口 開閉 flaps", max(tr) > 0.6 and tr[-1] < 0.1, f"{max(tr):.2f} {tr[-1]:.2f}")
    mao = pa.inspect_puppet(MAO, "Mao")
    rig.reload_puppet_meta(mao.param_meta, "Mao", mao.physics_settings)
    tr = _mouth_test_trace(rig, "あ", 0)[:-1]
    check("mouthtest. Mao あ flaps (synthesized mouth)", max(tr) > 0.5 and tr[-1] < 0.1, f"{max(tr):.2f} {tr[-1]:.2f}")
    check("mouthtest. mouthMapped true on Mao", rig.mouth_mapped() and rig.trigger_mouth_test()["mouthMapped"])


def test_mouth_synthesis(tmp: Path) -> None:
    check("synth. Hiyori (bound mouth) left alone", pa.synthesize_missing_mouth(HIYORI, write=False)["ok"] is False)
    r = pa.synthesize_missing_mouth(MAO, write=False)
    check("synth. shipped Mao.inp already carries the mouth (idempotent)", r == {"ok": False, "reason": "mouth already bound"}, str(r))
    # rebuild from a stripped copy: same result, texture sections untouched
    cp = tmp / "MaoStripped.inp"
    shutil.copyfile(MAO, cp)
    m = pa.read_puppet_json(cp)
    for q in m["param"]:
        if q["name"] in "あいうえお":
            q["bindings"] = []
    pa._write_puppet_json(cp, m)
    raw_before = cp.read_bytes()
    tail_before = raw_before[12 + int.from_bytes(raw_before[8:12], "big"):]
    r = pa.synthesize_missing_mouth(cp)
    raw = cp.read_bytes()
    check("synth. stripped Mao -> clean layered mouth found + bound to あいうえお",
          r["ok"] and r["params"] == ["あ", "お", "え", "う", "い"] and r["cavity"] == 1071
          and r["layers"]["lowerCover"] == [1075] and r["layers"]["upperCover"] == [1076], str(r))
    check("synth. textures byte-for-byte identical, file still a valid puppet",
          raw[12 + int.from_bytes(raw[8:12], "big"):] == tail_before and pa.read_puppet_json(cp)["param"])
    mm = pa.read_puppet_json(cp)
    a = next(q for q in mm["param"] if q["name"] == "あ")
    zero_ok = all(all(v == [0.0, 0.0] for v in b["values"][0][0]) for b in a["bindings"])
    corners = all(abs(v[1]) < 1e-3 for b in a["bindings"] if b["node"] == 1075
                  for v, x in zip(b["values"][1][0], [0]) )
    check("synth. あ=0 is a zero deform (rest pose unchanged)", zero_ok)
    full = next(b for b in a["bindings"] if b["node"] == 1075)["values"][1][0]
    check("synth. lower lip drops (max dy ~ delta) and corners stay pinned (some verts ~0)",
          max(v[1] for v in full) > 0.9 * r["delta"] and min(abs(v[1]) for v in full) < 0.5 and corners is not None,
          f"{max(v[1] for v in full):.1f}")
    o = next(q for q in mm["param"] if q["name"] == "お")
    fo = next(b for b in o["bindings"] if b["node"] == 1075)["values"][1][0]
    check("synth. お opens 0.8 of あ", abs(max(v[1] for v in fo) - 0.8 * max(v[1] for v in full)) < 0.05)
    # refusal: a baked / collapsed mouth is not hacked
    cp2 = tmp / "MaoBaked.inp"
    shutil.copyfile(cp, cp2)
    m2 = pa.read_puppet_json(cp2)
    for q in m2["param"]:
        if q["name"] in "あいうえお":
            q["bindings"] = []

    def strip(n):
        if n.get("name") == "口":
            n["children"] = n["children"][:2]
        for c in n.get("children") or []:
            strip(c)
    strip(m2["nodes"])
    pa._write_puppet_json(cp2, m2)
    before = cp2.read_bytes()
    r2 = pa.synthesize_missing_mouth(cp2)
    check("synth. baked / too-simple mouth refused honestly, file untouched",
          r2["ok"] is False and "fewer than 3" in r2["reason"] and cp2.read_bytes() == before, str(r2))
    check("synth. ... and capabilities then report mouth false", not pa.inspect_puppet(cp2, "MaoBaked").capabilities["mouth"])


async def http_inspector_hiyori(cl, rt, sel: dict) -> None:
    r = await cl.get("/api/models/params")
    j = await r.json()
    by = {q["name"]: q for q in j.get("params", [])}
    check("http. GET /api/models/params Hiyori: schema + capabilities (physicsCount 36, arms, mouth)",
          r.status == 200 and j["ok"] and j["activeModel"] == "Hiyori" and j["capabilities"]["physicsCount"] == 36
          and j["capabilities"]["arms"] and j["capabilities"]["mouth"] and all(set(q) == INSPECTOR_KEYS for q in j["params"])
          and by["[3]サイドアップ左"]["category"] == "physics" and by["左腕 A"]["category"] == "arms"
          and "左目\u3000笑顔" not in by, str(j.get("capabilities")))
    r = await cl.post("/api/models/params", json={"name": "右腕 A", "x": 7.5, "y": 0.0})
    j = await r.json()
    check("http. POST /api/models/params sets a live override", r.status == 200 and j["ok"] and j["overriddenCount"] == 1
          and j["overrides"] == {"右腕 A": [7.5, 0.0]}, str(j))
    p = run(rt.rig, quiet_cfg(), 2)
    check("http. ... reaches the rendered frame", p["右腕 A"] == (7.5, 0.0))
    r = await cl.post("/api/models/params", json={"name": "右腕 A", "x": 1.0}, headers={"Origin": "https://evil.example"})
    check("http. foreign Origin blocked on params", r.status == 403)
    for body, st in (({"name": "Nope", "x": 1}, 404), ({"name": "右腕 A", "x": "abc"}, 400), ({"x": 1}, 400), ([], 400)):
        r = await cl.post("/api/models/params", json=body)
        check(f"http. params bad body {body!r} -> {st}", r.status == st, str(r.status))
    r = await cl.post("/api/models/test-mouth")
    j = await r.json()
    check("http. POST /api/models/test-mouth", r.status == 200 and j == {"ok": True, "duration": 2.2, "mouthMapped": True}
          and rt.rig.mouth_test_active(), str(j))
    r = await cl.post("/api/models/test-mouth", headers={"Origin": "https://evil.example"})
    check("http. foreign Origin blocked on test-mouth", r.status == 403)


async def http_inspector_mao(cl, rt) -> None:
    check("http. puppet switch cleared the Hiyori override", rt.rig.rig_param_overrides == {})
    for n in ("オーラ\u3000表示", "光\u3000表示", "目\u3000エフェクト"):
        r = await cl.post("/api/models/params", json={"name": n, "x": 1.0, "y": 0.0})
    r = await cl.post("/api/models/params", json={"name": "うさぎ\u3000出現", "x": 30.0})
    j = await r.json()
    check("http. Mao magic extras toggled via POST", j["overriddenCount"] == 4, str(j))
    j = await (await cl.get("/api/models/params")).json()
    by = {q["name"]: q for q in j["params"]}
    check("http. inspector reports Mao extras overridden + toggle flags", by["オーラ\u3000表示"]["overridden"]
          and by["オーラ\u3000表示"]["isToggle"] and by["オーラ\u3000表示"]["category"] == "extras"
          and j["capabilities"]["physicsCount"] == 15 and j["capabilities"]["mouth"], str(j["capabilities"]))
    r = await cl.post("/api/models/params", json={"name": "光\u3000表示", "clear": True})
    j = await r.json()
    check("http. clear one override", j["overriddenCount"] == 3 and "光\u3000表示" not in j["overrides"], str(j))
    r = await cl.post("/api/models/params", json={"clearAll": True})
    j = await r.json()
    check("http. clearAll", j == {"ok": True, "overriddenCount": 0, "overrides": {}}, str(j))


# ------------------------------------------------------------------------------------------------ HTTP
async def test_http(tmp: Path) -> None:
    import server

    lib = tmp / "library"
    lib.mkdir()
    shutil.copyfile(HIYORI, lib / "Hiyori.inp")
    shutil.copyfile(MAO, lib / "Mao.inp")
    hk_file = tmp / "Hikari-v28-Custom.inp"
    shutil.copyfile(HIKARI, hk_file)
    real_lib, real_mint = server.LIBRARY_DIR, server.MINT_CLI
    server.LIBRARY_DIR = lib
    hikari_before = sha(HIKARI)
    lib_before = {p.name: sha(p) for p in lib.iterdir()}

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            self.model_path = path
            self.width, self.height = 960, 1080
            self.switches: list[str] = []
            self.reloads = 0
            self.fail_next = False
            import threading
            self._lock = threading.RLock()
            self._load(path)

        def _load(self, path: Path) -> None:
            self.param_names = [m["name"] for m in pa.inspect_puppet(path).param_meta]

        def switch_model(self, target: Path) -> None:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("engine refused the puppet")
            self.switches.append(Path(target).name)
            self.model_path = Path(target)
            self._load(target)

        def reload_model(self, candidate: Path) -> None:
            self.reloads += 1
            previous = self.model_path
            self.model_path = candidate
            self._load(candidate)
            candidate.replace(previous)
            self.model_path = previous

    class FakeCustomizer:
        builds = 0

        def build_custom_model(self, config, out: Path) -> Path:
            FakeCustomizer.builds += 1
            shutil.copyfile(HIKARI, out)
            return out

    S = server.StudioRuntime

    class FakeRuntime:
        library_model_path = staticmethod(S.library_model_path)
        hikari_default_camera = staticmethod(S.hikari_default_camera)
        model_path_for = S.model_path_for
        _activate_meta = S._activate_meta
        model_entry = S.model_entry
        list_models = S.list_models
        _switch_engine_sync = S._switch_engine_sync
        _camera_now = S._camera_now
        _finish_switch = S._finish_switch
        select_model = S.select_model
        apply_customizer = S.apply_customizer
        _reload_customizer = S._reload_customizer
        sync_rig_flags = S.sync_rig_flags
        param_inspector = S.param_inspector

        def __init__(self) -> None:
            import threading
            self.config = ConfigStore(tmp / "settings.json")
            self.model_path = hk_file
            self.param_defaults, self.param_meta = server._read_param_meta(hk_file)
            self.hikari_param_meta = self.param_meta
            self.rig = RigController(self.param_defaults, FakeLipSync(), HookRegistry())
            self.engine = FakeEngine(hk_file)
            self.customizer = FakeCustomizer()
            self._frame_lock = threading.Lock()
            self._model_lock = asyncio.Lock()
            self._customizer_lock = asyncio.Lock()
            self.active_model_id = "hikari"
            self._shutdown_task = None
            self._fish_ended_on_shutdown = True
            self.broadcasts: list = []
            self.events: list = []
            from modules.decision_engine import DecisionEngine
            self.decisions = DecisionEngine(self.config.data["ai"])

        def start_loop(self, loop) -> None:
            pass

        def stop_loop(self) -> None:
            pass

        def note_stage_event(self, text: str, kind: str = "stage") -> bool:
            self.events.append((kind, text))
            return True

        async def broadcast_json(self, msg, exclude=None) -> None:
            self.broadcasts.append(msg)

    class _Mem:
        async def close(self) -> None:
            pass

    rt = FakeRuntime()
    rt.memory = _Mem()
    app = server.build_app(rt)
    cfg = rt.config.data
    try:
        async with TestClient(TestServer(app, host="127.0.0.1", port=0)) as cl:
            check("http. test server is NOT on port 8800", cl.server.port != 8800, str(cl.server.port))
            r = await cl.get("/api/models")
            j = await r.json()
            ids = [m["id"] for m in j.get("models", [])]
            hk = j["models"][0] if j.get("models") else {}
            check("http. GET /api/models lists hikari + Hiyori + Mao", r.status == 200 and j["ok"] and ids == ["hikari", "Hiyori", "Mao"]
                  and j["activeModel"] == "hikari" and isinstance(j["mintAvailable"], bool), str(j)[:300])
            check("http. hikari entry: builtin, customizable, name, paramCount, mappedChannels, recommendedCamera",
                  hk.get("name") == "Hikari (Custom)" and hk.get("builtin") and hk.get("customizable") and hk.get("paramCount", 0) > 100
                  and "mouth" in hk.get("mappedChannels", []) and set(hk.get("recommendedCamera", {})) == {"scale", "x", "y"}, str(hk))
            hiy = j["models"][1]
            check("http. GET /api/models entries carry capabilities + physicsCount + mouthMapped",
                  all(isinstance(m_.get("capabilities"), dict) and "physicsCount" in m_ and "mouthMapped" in m_ for m_ in j["models"])
                  and hiy["physicsCount"] == 36 and hiy["mouthMapped"] and j["models"][2]["physicsCount"] == 15
                  and j["models"][2]["mouthMapped"] and hk["capabilities"]["mouth"] and hk["capabilities"]["extrasCount"] > 0,
                  str([(m_["id"], m_.get("capabilities")) for m_ in j["models"]]))
            r = await cl.get("/api/models/params")
            jp = await r.json()
            check("http. GET /api/models/params on Hikari: native bound params, auto + arms + physics + extras (VFX)",
                  r.status == 200 and jp["ok"] and jp["activeModel"] == "hikari" and jp["capabilities"]["mouth"]
                  and {q["category"] for q in jp["params"]} >= {"auto", "arms", "extras"}
                  and all(set(q) == INSPECTOR_KEYS for q in jp["params"]), str(jp)[:300])
            check("http. Hiyori entry", hiy["paramCount"] == 67 and not hiy["builtin"] and not hiy["customizable"]
                  and hiy["recommendedCamera"]["scale"] < 1.0, str(hiy))

            cfg["camera"].update({"scale": 1.95, "x": 4.0, "y": 340.0})          # the user's custom Hikari framing
            r = await cl.post("/api/models/select", json={"id": "Hiyori", "autoFrame": True})
            j = await r.json()
            rec = pa.inspect_puppet(lib / "Hiyori.inp").recommended_camera
            check("http. POST /api/models/select Hiyori", r.status == 200 and j["ok"] and j["activeModel"] == "Hiyori"
                  and j["paramCount"] == 67 and "mouth" in j["mappedChannels"] and j["camera"] == rec, str(j))
            check("http. engine + rig switched live", rt.engine.switches[-1] == "Hiyori.inp" and rt.rig.active_model_id == "Hiyori"
                  and rt.rig._mapper is not None and rt.active_model_id == "Hiyori")
            check("http. camera auto-framed + Hikari's camera remembered + activeModel persisted",
                  cfg["camera"]["scale"] == rec["scale"] and cfg["stage"]["savedModelCameras"]["hikari"] == {"scale": 1.95, "x": 4.0, "y": 340.0}
                  and json.loads((tmp / "settings.json").read_text(encoding="utf-8"))["stage"]["activeModel"] == "Hiyori")
            check("http. model_changed + config_sync broadcast", any(b.get("type") == "model_changed" and b["activeModel"] == "Hiyori"
                                                                    for b in rt.broadcasts) and rt.broadcasts[-1]["type"] == "config_sync")
            check("http. paramsMeta now the Hiyori meta", any(m["name"] == "角度" for m in rt.param_meta))
            p = run(rt.rig, quiet_cfg(), 3)
            check("http. rig frames are now Hiyori frames (eyes open)", set(p) == set(rt.engine.param_names) and p["左目\u3000開閉"][0] > 0.95)
            await http_inspector_hiyori(cl, rt, j)

            cfg["camera"].update({"scale": 0.8, "x": 0.0, "y": 1100.0})          # tweak Hiyori's framing
            r = await cl.post("/api/models/select", json={"id": "Mao"})
            j = await r.json()
            check("http. select Mao (autoFrame defaults to true)", r.status == 200 and j["activeModel"] == "Mao" and j["paramCount"] == 130
                  and "mouth" in j["mappedChannels"] and "arms" in j["mappedChannels"], str(j))
            await http_inspector_mao(cl, rt)
            r = await cl.post("/api/models/select", json={"id": "hikari", "autoFrame": True})
            j = await r.json()
            check("http. back to Hikari restores HER saved custom camera", j["activeModel"] == "hikari"
                  and j["camera"] == {"scale": 1.95, "x": 4.0, "y": 340.0} and rt.rig._mapper is None, str(j))
            r = await cl.post("/api/models/select", json={"id": "Hiyori"})
            j = await r.json()
            check("http. back to Hiyori restores the tweaked Hiyori camera", j["camera"] == {"scale": 0.8, "x": 0.0, "y": 1100.0}, str(j))
            r = await cl.post("/api/models/select", json={"id": "Mao", "autoFrame": False})
            j = await r.json()
            check("http. autoFrame false keeps the current camera", j["camera"] == {"scale": 0.8, "x": 0.0, "y": 1100.0}, str(j))

            for body, status, why in (({"id": "Nope"}, 404, "unknown id -> 404"), ({"id": "../settings"}, 400, "path id -> 400"),
                                      ({}, 400, "missing id -> 400")):
                r = await cl.post("/api/models/select", json=body)
                check(f"http. select {why}", r.status == status, str(r.status))
            rt.engine.fail_next = True
            r = await cl.post("/api/models/select", json={"id": "Hiyori"})
            j = await r.json()
            check("http. engine failure -> 500, previous puppet stays active", r.status == 500 and j["activeModel"] == "Mao"
                  and rt.active_model_id == "Mao" and rt.rig.active_model_id == "Mao", str(j))
            r = await cl.post("/api/models/select", json={"id": "Hiyori"}, headers={"Origin": "https://evil.example"})
            check("http. foreign Origin blocked on select", r.status == 403)
            r = await cl.post("/api/config", json={"stage": {"activeModel": "Hiyori", "roomDim": 0.2}, "save": False})
            check("http. POST /api/config cannot flip stage.activeModel behind the engine's back",
                  cfg["stage"]["activeModel"] == "Mao" and abs(cfg["stage"]["roomDim"] - 0.2) < 1e-9)

            # customizer while a library puppet is live: Hikari rebuilt, library file untouched, Hikari back on stage
            mao_sha = sha(lib / "Mao.inp")
            r = await cl.post("/api/customizer/apply", json={"hairColor": "#ff66cc"})
            j = await r.json()
            check("http. customizer while Mao is live -> rebuilt + switched back to hikari",
                  r.status == 200 and j.get("ok") is not False and rt.active_model_id == "hikari" and rt.rig._mapper is None
                  and rt.engine.model_path == hk_file and cfg["stage"]["activeModel"] == "hikari", str(j)[:300])
            check("http. ... library Mao.inp NOT overwritten, no .pending left",
                  sha(lib / "Mao.inp") == mao_sha and not list(tmp.glob("*.pending.inp")))
            r = await cl.post("/api/customizer/apply", json={"hairColor": "#ff77cc"})
            check("http. customizer with Hikari live uses reload_model (unchanged path)", r.status == 200 and rt.engine.reloads == 1)

            # import: .inp upload
            fd = FormData()
            fd.add_field("file", (lib / "Hiyori.inp").read_bytes(), filename="Cool Girl.inp", content_type="application/octet-stream")
            r = await cl.post("/api/models/import", data=fd)
            j = await r.json()
            check("http. POST /api/models/import .inp -> installed + auto-selected",
                  r.status == 200 and j["ok"] and j["model"]["id"] == "Cool Girl" and j["activeModel"] == "Cool Girl"
                  and (lib / "Cool Girl.inp").is_file() and set(j["camera"]) == {"scale", "x", "y"}, str(j)[:300])
            fd = FormData()
            fd.add_field("file", (lib / "Mao.inp").read_bytes(), filename="mao2.inx")
            fd.add_field("autoSelect", "false")
            r = await cl.post("/api/models/import", data=fd)
            j = await r.json()
            check("http. import autoSelect=false keeps the active puppet", r.status == 200 and j["model"]["id"] == "mao2"
                  and j["activeModel"] == "Cool Girl", str(j)[:200])
            fd = FormData()
            fd.add_field("file", b"MZ\x90\x00", filename="virus.exe")
            r = await cl.post("/api/models/import", data=fd)
            check("http. import bad extension -> 415", r.status == 415)
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w") as zf:
                zf.writestr("../../../evil.moc3", b"x")
            fd = FormData()
            fd.add_field("file", zbuf.getvalue(), filename="evil.zip")
            r = await cl.post("/api/models/import", data=fd)
            check("http. import zip-slip -> 415", r.status == 415 and "Unsafe" in (await r.json())["error"])
            fd = FormData()
            fd.add_field("file", (lib / "Hiyori.inp").read_bytes(), filename="x.inp")
            r = await cl.post("/api/models/import", data=fd, headers={"Origin": "https://evil.example"})
            check("http. foreign Origin blocked on import", r.status == 403)

            zpath = tmp / "HIKKY_bundle.zip"
            if make_live2d_zip(zpath):
                fd = FormData()
                fd.add_field("file", zpath.read_bytes(), filename="HIKKY_bundle.zip")
                r = await cl.post("/api/models/import", data=fd)
                j = await r.json()
                check("http. import Live2D .zip -> MINT -> clean .inp in the library, auto-selected",
                      r.status == 200 and j["ok"] and j["activeModel"] == "HIKKY_bundle" and (lib / "HIKKY_bundle.inp").is_file()
                      and j["model"]["paramCount"] > 0, str(j)[:300])
                extra = [p.name for p in lib.iterdir() if p.suffix.lower() not in (".inp", ".inx")]
                check("http. ... ONLY the .inp copied (no psd / cmo3 / temp junk)", not extra, str(extra))
                check("http. ... temp dirs cleaned", not [p for p in (server.TMP_DIR).glob("hikari_mint_*")])
            else:
                check("http. Live2D zip import (MINT / sample missing - skipped)", True)

            # delete
            r = await cl.delete("/api/models/hikari")
            check("http. DELETE hikari -> 400", r.status == 400)
            r = await cl.delete("/api/models/Ghost")
            check("http. DELETE missing -> 404", r.status == 404)
            r = await cl.delete("/api/models/Cool Girl", headers={"Origin": "https://evil.example"})
            check("http. foreign Origin blocked on delete", r.status == 403 and (lib / "Cool Girl.inp").is_file())
            await cl.post("/api/models/select", json={"id": "Cool Girl"})
            active_before = rt.active_model_id
            r = await cl.delete("/api/models/Cool Girl")
            j = await r.json()
            check("http. DELETE active library puppet -> removed + back to hikari",
                  r.status == 200 and j["deleted"] == "Cool Girl" and not (lib / "Cool Girl.inp").exists()
                  and rt.active_model_id == "hikari" and j["activeModel"] == "hikari",
                  f"{active_before} {j}")
            r = await cl.delete("/api/models/mao2")
            check("http. DELETE inactive .inx", r.status == 200 and not (lib / "mao2.inx").exists())
            j = await (await cl.get("/api/models")).json()
            check("http. listing after deletes", "Cool Girl" not in [m["id"] for m in j["models"]] and j["activeModel"] == "hikari")
    finally:
        server.LIBRARY_DIR, server.MINT_CLI = real_lib, real_mint
    check("http. real models/Hikari-v28-Custom.inp untouched by the whole suite", sha(HIKARI) == hikari_before)
    check("http. Hiyori / Mao library copies untouched", all(sha(lib / n) == h for n, h in lib_before.items()))


# ------------------------------------------------------------------------------------------------ startup wiring
def test_startup_fallback(tmp: Path) -> None:
    import server
    real = server.LIBRARY_DIR
    try:
        server.LIBRARY_DIR = tmp / "empty-lib"
        check("startup. missing library puppet -> library_model_path None (server falls back to hikari)",
              server.StudioRuntime.library_model_path("Hiyori") is None)
        server.LIBRARY_DIR = HIYORI.parent
        check("startup. present puppet resolves", server.StudioRuntime.library_model_path("Hiyori") == HIYORI)
        check("startup. traversal ids refused", server.StudioRuntime.library_model_path("..\\Hikari-v28-Custom") is None
              and server.StudioRuntime.library_model_path("hikari") is None)
    finally:
        server.LIBRARY_DIR = real


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hikari-universal-test-") as td:
        tmp = Path(td)
        for name, fn in (("inspect", test_inspect), ("config", test_config), ("importer", test_importer),
                         ("engine", test_engine_switch), ("startup", test_startup_fallback), ("synth", test_mouth_synthesis)):
            sub = tmp / name
            sub.mkdir()
            fn(sub)
        test_mapper()
        test_rig()
        test_physics()
        test_inspector()
        test_mouth()
        sub = tmp / "http"
        sub.mkdir()
        asyncio.run(test_http(sub))
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}", flush=True)
    return 1 if failed else 0


def test_all() -> None:                 # pytest entry point
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
