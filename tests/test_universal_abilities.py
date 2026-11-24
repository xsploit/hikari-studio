"""Verification: universal rig abilities (auto-discovered props / effects of ANY puppet -> RigController -> AI).

  * discover_rig_abilities on Hikari (builtin VFX, never offered to the AI), Hiyori (hand pose; legs off-stage),
    Mao (11 named magic abilities via smart toggle + driver grouping, two-stage explosion, off-stage / hidden props
    filtered from the AI) and a synthetic unknown .inp (English + Japanese auto-labeler, clean fallback names)
  * RigController playback: smooth ramps, finite values, exact return to default, manual sliders still win,
    trigger_emote accepts rig_* ids, catalog swaps on puppet switch
  * DecisionEngine: offers Mao's abilities, picks them on matching moods (heuristic + typed model path), cooldown /
    no-repeat gating, LLM answers validated against the live catalog, instant swap back to Hikari (no stale ids)
  * GET /api/models/abilities, POST /api/models/ability, GET /api/emotes, POST /api/emote with rig ids and the
    local-origin guard on the real server.py handlers (random loopback port, never 8800)

    python tests/test_universal_abilities.py        (or: python -m pytest tests/test_universal_abilities.py)
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import core.puppet_adapter as pa  # noqa: E402
from core.config import ConfigStore, default_config  # noqa: E402
from core.events import HookRegistry  # noqa: E402
from core.rig_controller import RigController, ability_track_frame  # noqa: E402
import modules.decision_engine as de_mod  # noqa: E402
from modules.decision_engine import DecisionEngine, hikari_questions  # noqa: E402
from test_universal_puppets import FakeLipSync, hikari_defaults  # noqa: E402

MODELS = HERE / "models"
HIKARI = MODELS / "Hikari-v28-Custom.inp"
HIYORI = MODELS / "library" / "Hiyori.inp"
MAO = MODELS / "library" / "Mao.inp"

RESULTS: list[tuple[str, bool, str]] = []

MAO_NAMED = {  # id: (label, kind, duration)
    "rig_aura": ("Magic Aura", "moment", 3.2), "rig_starlight": ("Starlight Burst", "moment", 3.0),
    "rig_healing_light": ("Healing Light", "moment", 3.2), "rig_heart_magic": ("Heart Magic", "moment", 3.0),
    "rig_heart_fail": ("Broken Heart", "moment", 2.8), "rig_explosion": ("Magic Explosion", "moment", 3.0),
    "rig_smoke": ("Smoke Puff", "moment", 2.6), "rig_face_ink": ("Paint Splashes", "toggle", 5.5),
    "rig_wand_ink": ("Wand Paint", "toggle", 5.0), "rig_eye_effect": ("Star Pupils", "toggle", 5.0),
    "rig_orb": ("Floating Orb", "moment", 3.0),
}
CORE_PARAMS_MAO = {"角度", "角度 Z", "左目　開閉", "右目　開閉", "目玉", "あ", "呼吸", "照れ", "体の回転　X",
                   "全体の移動　X", "魔法の位置X", "左腕A　肩の回転", "髪揺れ　前", "左眉　左右", "目玉　縮小"}


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail if not ok else ""))
    print(("  PASS " if ok else "  FAIL ") + name + ("" if ok else f"   -> {detail[:300]}"), flush=True)


def quiet_cfg() -> dict:
    cfg = default_config()
    cfg["rig"].update({"autoBlink": False, "autoGaze": False, "mouseLookAt": False, "autoVfx": False})
    return cfg


def by_id(abilities) -> dict:
    return {a.id: a for a in abilities}


# ------------------------------------------------------------------------------------------------ synthetic puppet
def _part(uid: int, name: str, x0: float, y0: float, x1: float, y1: float, opacity: float = 1.0) -> dict:
    return {"uuid": uid, "name": name, "type": "Part", "enabled": True, "opacity": opacity, "zsort": 0.0,
            "transform": {"trans": [0, 0, 0], "rot": [0, 0, 0], "scale": [1, 1]},
            "mesh": {"verts": [x0, y0, x1, y0, x1, y1, x0, y1], "uvs": [0, 0, 1, 0, 1, 1, 0, 1], "indices": [0, 1, 2, 0, 2, 3]},
            "textures": [0], "children": []}


def _param(uid: int, name: str, mn: float, mx: float, d: float, nodes: list[int], kind: str = "deform") -> dict:
    binds = []
    for n in nodes:
        if kind == "opacity":
            binds.append({"node": n, "param_name": "opacity", "values": [[0.0], [1.0]], "isSet": [[True], [True]],
                          "interpolate_mode": "Linear"})
        else:
            z = [[0.0, 0.0]] * 4
            binds.append({"node": n, "param_name": "deform", "values": [[z], [[[0.0, -5.0]] * 4]],
                          "isSet": [[True], [True]], "interpolate_mode": "Linear"})
    return {"uuid": uid, "name": name, "is_vec2": False, "min": [mn, 0.0], "max": [mx, 0.0], "defaults": [d, 0.0],
            "axis_points": [[0.0, 1.0], [0.0]], "merge_mode": "Additive", "bindings": binds}


def make_synthetic(path: Path) -> Path:
    """An arbitrary 'Booth / Inochi Creator' puppet: core face params + custom props in English and Japanese."""
    parts = [
        _part(10, "Body", -300, -600, 300, 900), _part(11, "Face", -200, -900, 200, -500),
        _part(12, "Eye L", 40, -760, 120, -700), _part(13, "Cat Ears", -220, -1050, 220, -900, 0.0),
        _part(14, "Wings", -600, -600, 600, 200, 0.0), _part(15, "Tears", -120, -700, 120, -600, 0.0),
        _part(16, "Gizmo", 150, -300, 260, -200), _part(17, "おばけ", -260, -300, -150, -200),
        _part(18, "Far Prop", 9000, 9000, 9100, 9100, 0.0), _part(19, "Hair", -250, -1000, 250, -400),
    ]
    model = {
        "meta": {"name": "Synthetic Booth Girl", "version": "1.0"},
        "nodes": {"uuid": 1, "name": "Root", "type": "Node", "transform": {"trans": [0, 0, 0], "rot": [0, 0, 0], "scale": [1, 1]},
                  "children": parts},
        "param": [
            _param(100, "ParamAngleX", -30, 30, 0, [11]), _param(101, "ParamEyeLOpen", 0, 1, 1, [12]),
            _param(102, "ParamMouthOpenY", 0, 1, 0, [11]), _param(103, "Hair Sway", -1, 1, 0, [19]),
            _param(104, "Cat Ears Toggle", 0, 1, 0, [13], "opacity"),
            _param(105, "Demon Wings", 0, 1, 0, [14], "opacity"),
            _param(106, "Tears Show", 0, 1, 0, [15], "opacity"), _param(107, "Tears Flow", 0, 30, 0, [15]),
            _param(108, "Mystery Gizmo", 0, 1, 0, [16]), _param(109, "おばけ", 0, 1, 0, [17]),
            _param(110, "Far Away Prop Show", 0, 1, 0, [18], "opacity"),
            _param(111, "Unbound Toggle", 0, 1, 0, []),
        ],
    }
    payload = json.dumps(model, ensure_ascii=False).encode("utf-8")
    path.write_bytes(pa.INP_MAGIC + len(payload).to_bytes(4, "big") + payload)
    return path


# ------------------------------------------------------------------------------------------------ discovery
def test_discovery(tmp: Path) -> None:
    hk = pa.discover_rig_abilities(pa.inspect_puppet(HIKARI, "hikari"))
    check("discover. Hikari: 11 builtin VFX abilities", len(hk) == 11 and all(a.builtin for a in hk), str([a.id for a in hk]))
    check("discover. Hikari builtin abilities are NOT offered to the AI (her 21 emotes cover them)",
          not any(a.ai for a in hk))
    check("discover. Hikari builtin abilities name their emote", by_id(hk)["rig_hearts"].emote == "heart_crush"
          and by_id(hk)["rig_cat_mouth"].params == ["Cat Mouth"])

    hy = by_id(pa.discover_rig_abilities(pa.inspect_puppet(HIYORI, "Hiyori")))
    hp = hy.get("rig_hand_pose")
    check("discover. Hiyori: hand pose ability (左手 + 右手 merged, both hands)",
          hp is not None and set(hp.params) == {"左手", "右手"} and hp.kind == "pose" and hp.ai, str(list(hy)))
    legs = [a for a in hy.values() if "あし" in a.params]
    check("discover. Hiyori: legs (あし) are below the framing -> listed but off-stage, not AI-offered",
          len(legs) == 1 and not legs[0].on_stage and not legs[0].ai)
    check("discover. Hiyori: no core / arm / physics param became an ability",
          not any(p in ("角度", "左腕 A", "左腕 B", "左手B 回転", "髪揺れ　前", "左眉　左右", "胸　揺れ")
                  for a in hy.values() for p in a.params))

    mao_list = pa.discover_rig_abilities(pa.inspect_puppet(MAO, "Mao"))
    mao = by_id(mao_list)
    missing = [k for k in MAO_NAMED if k not in mao]
    check("discover. Mao: all 11 named magic abilities discovered", not missing, str(missing))
    wrong = [(k, mao[k].label, mao[k].kind, mao[k].duration) for k, v in MAO_NAMED.items()
             if k in mao and (mao[k].label, mao[k].kind, mao[k].duration) != v]
    check("discover. Mao: labels / kinds / durations", not wrong, str(wrong))
    tr = {t.param: t for t in mao["rig_aura"].tracks}
    check("discover. Mao aura = オーラ　表示 toggle -> 1 + オーラ sweep 0 -> 24",
          tr["オーラ　表示"].role == "toggle" and tr["オーラ　表示"].target == 1.0
          and tr["オーラ"].role == "sweep" and abs(tr["オーラ"].target - 24.0) < 1e-6, str(mao["rig_aura"].public()))
    tr = {t.param: t for t in mao["rig_starlight"].tracks}
    check("discover. Mao starlight = 光　表示 + 光　星 (0 -> 24)", set(tr) == {"光　表示", "光　星"}
          and abs(tr["光　星"].target - 24.0) < 1e-6)
    check("discover. Mao heart magic = ハート回復 表示 / 緑色 + shared ハート 描画 / 拡縮",
          set(mao["rig_heart_magic"].params) == {"ハート回復　表示", "ハート回復　緑色", "ハート　描画", "ハート　拡縮"})
    check("discover. Mao broken heart = ハート失敗 表示 + 虹色 (+ shared heart drawing)",
          {"ハート失敗　表示", "ハート失敗　虹色"} <= set(mao["rig_heart_fail"].params))
    tr = {t.param: t for t in mao["rig_explosion"].tracks}
    check("discover. Mao explosion: charge (爆発光溜め, stage 1) then burst (爆発, stage 2), show held throughout",
          tr["爆発光溜め"].stage == 1 and tr["爆発"].stage == 2 and tr["爆発光溜め　表示"].stage == 0
          and tr["爆発"].role == "sweep", str(mao["rig_explosion"].public()))
    check("discover. Mao smoke / healing / orb pairs", set(mao["rig_smoke"].params) == {"煙　表示", "煙"}
          and set(mao["rig_healing_light"].params) == {"回復魔法光　表示", "回復魔法光"}
          and set(mao["rig_orb"].params) == {"玉　表示", "玉　移動"})
    ai = [a for a in mao_list if a.ai]
    check("discover. Mao: AI catalog capped at 12", 0 < len(ai) <= pa.AI_ABILITY_CAP, str(len(ai)))
    want_ai = {"rig_aura", "rig_starlight", "rig_healing_light", "rig_heart_magic", "rig_heart_fail", "rig_explosion",
               "rig_smoke", "rig_face_ink", "rig_eye_effect", "rig_orb"}
    check("discover. Mao: the 10 visible named abilities are AI-offered", want_ai <= {a.id for a in ai},
          str(want_ai - {a.id for a in ai}))
    off = {a.id for a in mao_list if not a.on_stage}
    check("discover. Mao: off-stage / hidden props filtered (bunny at her feet, wand-tip ink drip, hidden arm-B wand paint)",
          {"rig_bunny", "rig_paint", "rig_wand_ink"} <= off and not any(mao[i].ai for i in off), str(off))
    used = {p for a in mao_list for p in a.params}
    check("discover. Mao: no core / arm / physics / positioning param in any ability", not (used & CORE_PARAMS_MAO),
          str(used & CORE_PARAMS_MAO))
    check("discover. Mao: unbound params never used", "爆発　表示" not in used and "うさぎの移動　X" not in used)
    check("discover. Mao: every ability has moods from the decision emotions",
          all(a.moods and set(a.moods) <= set(de_mod.EMOTIONS) for a in mao_list),
          str([(a.id, a.moods) for a in mao_list if not set(a.moods) <= set(de_mod.EMOTIONS)]))
    check("discover. cached + independent copies", pa.discover_rig_abilities(pa.inspect_puppet(MAO, "Mao"))[0] is not mao_list[0])

    syn_path = make_synthetic(tmp / "Synthetic.inp")
    syn = by_id(pa.discover_rig_abilities(pa.inspect_puppet(syn_path, "Synthetic")))
    check("discover. synthetic: Cat Ears Toggle -> rig_cat_ears 'Cat Ears' toggle",
          "rig_cat_ears" in syn and syn["rig_cat_ears"].label == "Cat Ears" and syn["rig_cat_ears"].kind == "toggle",
          str({k: (a.label, a.kind) for k, a in syn.items()}))
    check("discover. synthetic: Demon Wings -> rig_wings 'Wings'", "rig_wings" in syn and syn["rig_wings"].label == "Wings")
    t = syn.get("rig_tears")
    check("discover. synthetic: Tears Show + Tears Flow merged -> 'Crying Tears' moment",
          t is not None and set(t.params) == {"Tears Show", "Tears Flow"} and t.label == "Crying Tears" and t.kind == "moment")
    check("discover. synthetic: unknown English prop -> clean fallback 'Mystery Gizmo'",
          "rig_mystery_gizmo" in syn and syn["rig_mystery_gizmo"].label == "Mystery Gizmo")
    jp = [a for a in syn.values() if a.params == ["おばけ"]]
    check("discover. synthetic: unknown Japanese prop -> real ability named by itself",
          len(jp) == 1 and jp[0].label == "おばけ" and jp[0].id.startswith("rig_fx_"))
    far = [a for a in syn.values() if "Far Away Prop Show" in a.params]
    check("discover. synthetic: far off-stage prop listed but not AI-offered", len(far) == 1 and not far[0].on_stage
          and not far[0].ai)
    used = {p for a in syn.values() for p in a.params}
    check("discover. synthetic: core face + physics + unbound params skipped",
          not used & {"ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY", "Hair Sway", "Unbound Toggle"}, str(used))
    meta_only = by_id(pa.discover_abilities_from_meta(pa.inspect_puppet(syn_path, "Synthetic").param_meta, "Synthetic"))
    check("discover. meta-only path (no geometry) still finds every prop", set(meta_only) == set(syn))


# ------------------------------------------------------------------------------------------------ rig playback
def _at(rig: RigController, aid: str, t: float, cfg: dict) -> dict:
    with rig.lock:
        s, d = rig._ability_runs[aid]
        rig._ability_runs[aid] = (time.perf_counter() - t, d)
    p, _, _ = rig.step(1.0 / 60.0, cfg, None)
    return p


def test_rig() -> None:
    cfg = quiet_cfg()
    mao = pa.inspect_puppet(MAO, "Mao")
    rig = RigController(hikari_defaults(), FakeLipSync(), HookRegistry())
    check("rig. Hikari at start: builtin abilities listed", {a["id"] for a in rig.get_rig_abilities()} >= {"rig_hearts", "rig_sparkles"})
    rig.reload_puppet_meta(mao.param_meta, "Mao", mao.physics_settings, abilities=pa.discover_rig_abilities(mao))
    ids = {a["id"] for a in rig.get_rig_abilities()}
    check("rig. Mao loaded: abilities refreshed in reload_puppet_meta", set(MAO_NAMED) <= ids and "rig_hearts" not in ids)
    for _ in range(10):
        rig.step(1 / 60, cfg, None)
    res = rig.trigger_rig_ability("rig_starlight")
    check("rig. trigger_rig_ability -> playing, default duration", res is not None and res["playing"] and res["duration"] == 3.0, str(res))
    check("rig. unknown ability -> None", rig.trigger_rig_ability("rig_nope") is None)
    st = [a for a in rig.get_rig_abilities() if a["id"] == "rig_starlight"][0]
    check("rig. get_rig_abilities reports active", st["active"] is True)
    T = 3.0
    show, star = [], []
    ts = [i / 60.0 for i in range(int(T * 60) + 1)]
    for t in ts:
        p = _at(rig, "rig_starlight", t, cfg)
        show.append(p["光　表示"][0])
        star.append(p["光　星"][0])
    finite = all(math.isfinite(v) for v in show + star)
    check("rig. starlight: all values finite", finite)
    check("rig. starlight: toggle eases 0 -> 1 in 0.2 s", show[0] < 0.05 and show[6] > 0.2 and show[12] > 0.99, str(show[:14]))
    check("rig. starlight: sweep rises smoothly to ~24 (0.8 of 0..30)",
          abs(max(star) - 24.0) < 0.5 and star[int(0.3 * 60)] < star[int(1.2 * 60)] < star[int(2.2 * 60)] + 1e-6, str(max(star)))
    dmax = max(abs(b - a) for a, b in zip(star, star[1:]))
    check("rig. starlight: no pops (max 1-frame change < 1.5 on a 0..30 timeline)", dmax < 1.5, f"{dmax:.3f}")
    check("rig. starlight: both params back to their defaults at the end", abs(show[-1]) < 1e-6 and abs(star[-1]) < 1e-6,
          f"{show[-1]} {star[-1]}")
    rig.trigger_rig_ability("rig_starlight")
    p = _at(rig, "rig_starlight", T + 0.2, cfg)
    check("rig. finished ability is removed + output exactly default",
          p["光　表示"][0] == 0.0 and p["光　星"][0] == 0.0 and not rig.active_rig_abilities())

    # explosion: charge before burst
    ex = by_id(rig.rig_abilities)["rig_explosion"]
    tr = {t.param: t for t in ex.tracks}
    ch = [ability_track_frame(tr["爆発光溜め"], t, ex.duration) for t in (0.6, 1.2, 2.4)]
    bu = [ability_track_frame(tr["爆発"], t, ex.duration) for t in (0.6, 1.2, 2.4)]
    check("rig. explosion: charge plays first, burst later (two-stage)",
          ch[0][1] > 0.5 and bu[0][1] == 0.0 and bu[2][1] > 0.5 and bu[2][0] > 15.0 and ch[2][1] == 0.0, f"{ch} {bu}")

    # orb oscillates around its default
    orb = {t.param: t for t in by_id(rig.rig_abilities)["rig_orb"].tracks}["玉　移動"]
    vals = [ability_track_frame(orb, i / 30.0, 3.0)[0] for i in range(90)]
    check("rig. orb: 玉　移動 oscillates both ways within range", min(vals) < -0.3 and max(vals) > 0.3
          and all(-1.0 <= v <= 1.0 for v in vals))

    # manual slider wins over the ability
    rig.set_param_override("光　表示", 0.0)
    rig.trigger_rig_ability("rig_starlight")
    p = _at(rig, "rig_starlight", 1.5, cfg)
    check("rig. manual inspector override beats the ability", p["光　表示"][0] == 0.0 and p["光　星"][0] > 5.0)
    rig.clear_param_override(None)

    # trigger_emote accepts rig ids
    rig.stop_rig_abilities()
    r = rig.trigger_emote("rig_aura")
    check("rig. trigger_emote('rig_aura') plays the ability", r.get("ok") and r.get("rigAbility") and r["duration"] == 3.2, str(r))
    p = _at(rig, "rig_aura", 1.6, cfg)
    check("rig. aura live in the output frame", p["オーラ　表示"][0] > 0.99 and p["オーラ"][0] > 10.0)
    check("rig. emote_status reports the playing ability", rig.emote_status()["active"] == "rig_aura")
    check("rig. classic emotes still work on Mao", rig.trigger_emote("cheer_bounce")["ok"])
    rig.stop_rig_abilities()
    check("rig. stop_rig_abilities", not rig.active_rig_abilities())

    # Hikari: builtin abilities write her own VFX params
    rig.trigger_rig_ability("rig_heart_magic")
    rig.reload_puppet_meta(None, "hikari")
    check("rig. switch to Hikari: catalog swapped, running Mao abilities dropped",
          not rig.active_rig_abilities() and rig.trigger_rig_ability("rig_heart_magic") is None)
    r = rig.trigger_emote("rig_hearts")
    p = _at(rig, "rig_hearts", 1.5, cfg)
    check("rig. Hikari builtin rig_hearts drives VFX Hearts", r.get("ok") and p["VFX Hearts"][0] > 0.95)
    p = _at(rig, "rig_hearts", 3.2, cfg)
    check("rig. Hikari ability ends cleanly", p["VFX Hearts"][0] < 0.05)
    check("rig. Hikari telemetry frame never mutated by abilities", rig.last_canonical is not rig.last_params or
          rig.last_canonical.get("VFX Hearts", (0, 0))[0] < 0.05)


# ------------------------------------------------------------------------------------------------ decision engine
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _mao_abilities() -> list[dict]:
    mao = pa.inspect_puppet(MAO, "Mao")
    rig = RigController(hikari_defaults(), FakeLipSync(), HookRegistry())
    rig.reload_puppet_meta(mao.param_meta, "Mao", mao.physics_settings, abilities=pa.discover_rig_abilities(mao))
    return rig.get_rig_abilities()


async def test_decisions() -> None:
    import random
    mao_ab = _mao_abilities()
    hk_rig = RigController(hikari_defaults(), FakeLipSync(), HookRegistry())
    clock = Clock()
    de = DecisionEngine({"decisionMode": "heuristic", "emoteCooldownSec": 6.0}, clock=clock, rng=random.Random(7))
    de.set_rig_abilities(hk_rig.get_rig_abilities(), "hikari")
    check("decide. Hikari: builtin abilities never offered", de.rig_abilities == [])
    de.set_rig_abilities(mao_ab, "Mao")
    offered = {a["id"] for a in de.rig_abilities}
    check("decide. Mao loaded: her on-stage abilities offered (<= 12)", {"rig_starlight", "rig_aura", "rig_explosion"} <= offered
          and len(offered) <= 12 and "rig_bunny" not in offered, str(offered))
    check("decide. status exposes the offered catalog", de.status()["rigAbilities"]["model"] == "Mao")

    d = await de.decide("WOW that's so amazing!! incredible!!", "user")
    check("decide. excited / amazed Mao picks one of her own matching abilities",
          d.emote in offered and set(de._rig_abilities[d.emote]["moods"]) & {d.emotion}, f"{d.emotion} {d.emote}")
    first = d.emote
    d2 = await de.decide("WOW amazing again!!", "user")
    check("decide. cooldown gates rig abilities like emotes", d2.emote is None and d2.skipped == "cooldown", str(d2.to_dict()))
    clock.t += 7.0
    d3 = await de.decide("WOW that's amazing!!", "user")
    check("decide. no immediate repeat of the same ability", d3.emote is not None and d3.emote != first, f"{first} {d3.emote}")
    clock.t += 7.0
    d4 = await de.decide("thank you so much, that means a lot", "user")
    check("decide. grateful Mao -> her grateful-mood magic (heart magic / healing light)",
          d4.emotion == "grateful" and d4.emote in ("rig_heart_magic", "rig_healing_light"), f"{d4.emotion} {d4.emote}")

    # typed evaluation-model path: questions carry the abilities; the model's rig pick is played
    seen: dict = {}

    async def fake_typed(cfg, provider, state, questions, **kw):
        seen["q"] = {q.name: q for q in questions}
        return {"emotion": "awe", "emote": seen.get("answer", "rig_starlight"), "wave": "no"}

    real = de_mod.run_typed_decision
    de_mod.run_typed_decision = fake_typed
    try:
        de.resolve_decider = lambda: "gateway"
        clock.t += 7.0
        d5 = await de.decide("look at the night sky", "reply")
        emote_q = seen["q"]["emote"]
        opts = dict(emote_q.options)
        check("decide. evaluation-model question offers rig_* options ('Label - desc')",
              opts.get("rig_starlight") == "Starlight Burst - Sparkling starlight burst" and "rig_" in emote_q.instructions,
              str(list(opts)[-6:]))
        check("decide. evaluation-model pick of rig_starlight is played", d5.emote == "rig_starlight" and d5.backend == "gateway",
              str(d5.to_dict()))
        # swap back to Hikari: zero stale options, a stale model answer is ignored
        de.set_rig_abilities(hk_rig.get_rig_abilities(), "hikari")
        clock.t += 7.0
        seen["answer"] = "rig_aura"
        d6 = await de.decide("look at the night sky", "reply")
        check("decide. switched to Hikari: no rig options in the question",
              not any(k.startswith("rig_") for k in dict(seen["q"]["emote"].options)))
        check("decide. stale rig id from the previous puppet is never played", d6.emote is None or not d6.emote.startswith("rig_"),
              str(d6.emote))
    finally:
        de_mod.run_typed_decision = real
        del de.resolve_decider

    qs = {q.name: q for q in hikari_questions("user", de_mod.DecisionEngine().rig_abilities)}
    check("decide. hikari_questions without abilities unchanged", not any(o.startswith("rig_") for o, _ in qs["emote"].options))

    # LLM path validated against the live catalog
    de.set_rig_abilities(mao_ab, "Mao")
    ans = '{"emotion": "excited", "emote": "rig_explosion", "wave": false}'
    check("decide. LLM answer with an offered rig id accepted", DecisionEngine.parse_llm_answer(ans, de._rig_abilities.keys())
          == ("excited", "rig_explosion", False))
    check("decide. LLM answer with an unoffered rig id fails closed", DecisionEngine.parse_llm_answer(ans) is None)
    req = de.build_llm_request("so cool!!", "user")
    body = json.loads(req["payload"]["messages"][1]["content"])
    check("decide. LLM prompt lists the abilities with labels", "rig_starlight" in body["emotes"]
          and body["rigAbilities"]["rig_aura"].startswith("Magic Aura"))

    # idle moments use the puppet's abilities too
    de2 = DecisionEngine({"decisionMode": "heuristic", "idleEmoteMinSec": 10, "idleEmoteMaxSec": 10, "emoteCooldownSec": 1},
                         clock=clock, rng=random.Random(3))
    de2.set_rig_abilities(mao_ab, "Mao")
    picks = []
    for _ in range(40):
        clock.t += 11.0
        r = de2.idle_tick()
        if r and r.emote:
            picks.append(r.emote)
    check("decide. idle moments include Mao's abilities", any(p.startswith("rig_") for p in picks), str(picks[:10]))
    de2.set_rig_abilities([], "hikari")
    picks = []
    for _ in range(20):
        clock.t += 11.0
        r = de2.idle_tick()
        if r and r.emote:
            picks.append(r.emote)
    check("decide. after the swap idle never picks a rig ability", picks and not any(p.startswith("rig_") for p in picks))


# ------------------------------------------------------------------------------------------------ http
async def test_http(tmp: Path) -> None:
    import threading
    import server
    from modules.decision_engine import DecisionEngine as DE

    lib = tmp / "library"
    lib.mkdir()
    shutil.copyfile(HIYORI, lib / "Hiyori.inp")
    shutil.copyfile(MAO, lib / "Mao.inp")
    hk_file = tmp / "Hikari-v28-Custom.inp"
    shutil.copyfile(HIKARI, hk_file)
    real_lib = server.LIBRARY_DIR
    server.LIBRARY_DIR = lib

    class FakeEngine:
        def __init__(self, path: Path) -> None:
            self.model_path = path
            self._lock = threading.RLock()
            self.param_names = [m["name"] for m in pa.inspect_puppet(path).param_meta]

        def switch_model(self, target: Path) -> None:
            self.model_path = Path(target)
            self.param_names = [m["name"] for m in pa.inspect_puppet(target).param_meta]

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
        sync_rig_flags = S.sync_rig_flags
        param_inspector = S.param_inspector

        def __init__(self) -> None:
            self.config = ConfigStore(tmp / "settings.json")
            self.model_path = hk_file
            self.param_defaults, self.param_meta = server._read_param_meta(hk_file)
            self.hikari_param_meta = self.param_meta
            self.rig = RigController(self.param_defaults, FakeLipSync(), HookRegistry())
            self.engine = FakeEngine(hk_file)
            self._frame_lock = threading.Lock()
            self._model_lock = asyncio.Lock()
            self._customizer_lock = asyncio.Lock()
            self.active_model_id = "hikari"
            self._shutdown_task = None
            self._fish_ended_on_shutdown = True
            self.broadcasts: list = []
            self.events: list = []
            self.decisions = DE(self.config.data["ai"])
            self.decisions.set_rig_abilities(self.rig.get_rig_abilities(), "hikari")

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
    evil = {"Origin": "https://evil.example"}
    try:
        async with TestClient(TestServer(app, host="127.0.0.1", port=0)) as cl:
            check("http. test server is NOT on port 8800", cl.server.port != 8800, str(cl.server.port))
            r = await cl.get("/api/models/abilities")
            j = await r.json()
            check("http. GET /api/models/abilities on Hikari: builtin abilities, contract keys",
                  r.status == 200 and j["ok"] and j["activeModel"] == "hikari" and isinstance(j["aiEnabled"], bool)
                  and len(j["abilities"]) == 11 and all(a["builtin"] and not a["aiEnabled"] for a in j["abilities"]), str(j)[:300])
            r = await cl.post("/api/models/select", json={"id": "Mao"})
            check("http. select Mao", r.status == 200, str(await r.text())[:200])
            check("http. puppet switch feeds the decision engine (Mao catalog)",
                  rt.decisions.rig_model == "Mao" and "rig_starlight" in {a["id"] for a in rt.decisions.rig_abilities})
            r = await cl.get("/api/models/abilities")
            j = await r.json()
            ab = {a["id"]: a for a in j["abilities"]}
            keys = {"id", "label", "desc", "kind", "duration", "moods", "params", "builtin", "onStage", "aiEnabled", "active"}
            check("http. Mao abilities listed with the frontend fields", j["activeModel"] == "Mao" and set(MAO_NAMED) <= set(ab)
                  and keys <= set(ab["rig_starlight"]) and ab["rig_starlight"]["active"] is False, str(list(ab))[:300])
            r = await cl.post("/api/models/ability", json={"id": "rig_starlight"})
            j = await r.json()
            check("http. POST /api/models/ability plays it", r.status == 200 and j == {"ok": True, "id": "rig_starlight",
                  "label": "Starlight Burst", "playing": True, "duration": 3.0}, str(j))
            r = await cl.get("/api/models/abilities")
            j = await r.json()
            check("http. live active flag", {a["id"]: a["active"] for a in j["abilities"]}["rig_starlight"] is True)
            check("http. manual ability starts the decision cooldown", rt.decisions._last_emote_id == "rig_starlight")
            r = await cl.post("/api/models/ability", json={"id": "rig_does_not_exist"})
            check("http. unknown ability -> 404", r.status == 404)
            r = await cl.post("/api/models/ability", json={"nope": 1})
            check("http. missing id -> 400", r.status == 400)
            r = await cl.post("/api/models/ability", data="not json", headers={"Content-Type": "application/json"})
            check("http. bad JSON -> 400", r.status == 400)
            r = await cl.post("/api/models/ability", json={"stopAll": True})
            j = await r.json()
            check("http. stopAll", r.status == 200 and j["ok"] and j["playing"] is False and not rt.rig.active_rig_abilities())
            r = await cl.post("/api/models/ability", json={"id": "rig_aura"}, headers=evil)
            check("http. foreign Origin blocked on POST ability", r.status == 403 and not rt.rig.active_rig_abilities())
            r = await cl.get("/api/models/abilities", headers=evil)
            check("http. foreign Origin blocked on GET abilities", r.status == 403)
            r = await cl.get("/api/models/abilities", headers={"Host": "evil.example:8800"})
            check("http. DNS-rebinding Host blocked", r.status == 403)
            r = await cl.get("/api/emotes")
            j = await r.json()
            check("http. GET /api/emotes keeps emotes + adds rigAbilities", len(j["emotes"]) == 21
                  and "rig_aura" in {a["id"] for a in j["rigAbilities"]})
            r = await cl.post("/api/emote", json={"emote": "rig_aura"})
            j = await r.json()
            check("http. POST /api/emote accepts a rig ability id", r.status == 200 and j["ok"] and j["emote"] == "rig_aura"
                  and "rig_aura" in rt.rig.active_rig_abilities(), str(j))
            r = await cl.post("/api/models/select", json={"id": "Hiyori"})
            check("http. switch to Hiyori -> her catalog", r.status == 200 and rt.decisions.rig_model == "Hiyori"
                  and {a["id"] for a in rt.decisions.rig_abilities} == {"rig_hand_pose"} and not rt.rig.active_rig_abilities())
            r = await cl.post("/api/models/select", json={"id": "hikari"})
            check("http. back to Hikari -> no stale Mao / Hiyori options", r.status == 200 and rt.decisions.rig_model == "hikari"
                  and rt.decisions.rig_abilities == [])
            r = await cl.post("/api/models/ability", json={"id": "rig_starlight"})
            check("http. Mao's ability is unknown on Hikari -> 404", r.status == 404)
            r = await cl.post("/api/models/ability", json={"id": "rig_hearts"})
            check("http. Hikari builtin ability playable", r.status == 200)
    finally:
        server.LIBRARY_DIR = real_lib


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hikari-abilities-test-") as td:
        tmp = Path(td)
        test_discovery(tmp)
        test_rig()
        asyncio.run(test_decisions())
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
