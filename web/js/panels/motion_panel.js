/** MOTION & EMOTIONS — Expression FX & blend shapes, Motion Lab (video-to-VTuber retarget, clip library & blend). */
import { btn, el, group, noteRow, row, slider, toggle } from '../ui.js';

const postMotion = (url, body) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

/* ───────── Expression FX & Blend Shapes ─────────
 * Manual on/off overrides for the "tiddlywinks" VFX + anime blend shapes, via GET/POST /api/vfx.
 * State is fetched once, lazily, the first time the Motion & Emotions tab renders (no polling);
 * every POST response refreshes the cache. Pills repaint in place, so toggling never jumps the scroll. */
const VFX_PARAMS = [
  ['VFX Sparkles', 'Sparkles'],
  ['VFX Hearts', 'Hearts'],
  ['VFX Sweat', 'Sweatdrop'],
  ['VFX Anger', 'Anger Vein'],
  ['VFX Shock', 'Shock Lines'],
  ['VFX Music', 'Music Notes'],
];
const BLEND_PARAMS = [
  ['Eye Star', 'Star Eyes'],
  ['Eye Heart', 'Heart Eyes'],
  ['Cheek Puff', 'Cheek Puff'],
  ['Cat Mouth', 'Cat Mouth :3'],
  ['Pupil Shrink', 'Pupil Shrink'],
];

const vfx = { overrides: {}, loaded: false, loading: null, error: '' };

const fxOn = (param) => Number(vfx.overrides[param]) > 0;
const fxCount = () => Object.keys(vfx.overrides).filter(fxOn).length;
const fxStatusText = () => {
  if (vfx.error) return `FX server unavailable — ${vfx.error}`;
  const n = fxCount();
  return n ? `${n} manual override${n === 1 ? '' : 's'} on` : 'No manual overrides — tap a pill to force it on';
};

function paintFx() {
  document.querySelectorAll('.vfx-chip[data-vfx]').forEach((b) => {
    const on = fxOn(b.dataset.vfx);
    b.classList.toggle('active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  document.querySelectorAll('[data-vfx-status]').forEach((d) => { d.textContent = fxStatusText(); });
}

function applyVfx(res) {
  if (res && res.overrides && typeof res.overrides === 'object') vfx.overrides = { ...res.overrides };
  vfx.error = '';
}

function loadVfx() {
  if (vfx.loaded || vfx.loading) return vfx.loading;
  vfx.loading = (async () => {
    try {
      const r = await fetch('/api/vfx');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      applyVfx(await r.json());
      vfx.loaded = true;
    } catch (err) {
      vfx.error = err.message || 'unavailable';
    } finally {
      vfx.loading = null;
      paintFx();
    }
  })();
  return vfx.loading;
}

async function postVfx(body) {
  try {
    const r = await fetch('/api/vfx', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const res = await r.json().catch(() => ({ ok: r.ok }));
    if (!r.ok || res?.ok === false) throw new Error(res?.error || `HTTP ${r.status}`);
    applyVfx(res);
    vfx.loaded = true;
  } catch (err) {
    vfx.error = err.message || 'request failed';
  }
  paintFx();
}

function toggleFx(param) {
  const next = !fxOn(param);
  // Optimistic: light the pill now; the POST response (when it carries overrides) is authoritative.
  const rest = { ...vfx.overrides };
  delete rest[param];
  vfx.overrides = next ? { ...rest, [param]: 1.0 } : rest;
  paintFx();
  return postVfx({ param, value: next ? 1.0 : null });
}

const fxPills = (items) => el('div', { class: 'chip-grid vfx-grid' }, items.map(([param, label]) => {
  const on = fxOn(param);
  return el('button', {
    type: 'button', class: `chip vfx-chip${on ? ' active' : ''}`, 'data-vfx': param,
    'aria-pressed': String(on), title: `${label} — toggle live`,
    onclick: () => toggleFx(param),
  }, [el('span', { class: 'vfx-dot', 'aria-hidden': 'true' }), label]);
}));

/** Rendered inside the Motion & Emotions tab, right after Face FX (see rig_panel.js). */
export function renderExpressionFx(store) {
  const rig = store.config.rig;
  if (store.activeTab === 'motion') loadVfx();
  return group('Expression FX & Blend Shapes', [
    toggle({
      label: 'Auto emotion VFX',
      desc: 'Trigger sparkles, hearts, sweatdrops, and eye highlights automatically with emotions',
      checked: rig.autoVfx ?? true,
      onChange: (v) => store.patch('rig', { autoVfx: v }),
    }),
    row({ title: 'Expression VFX', desc: 'Sparkles, hearts, sweatdrop, anger vein, shock lines, music notes', stack: true, control: fxPills(VFX_PARAMS) }),
    row({ title: 'Blend shapes', desc: 'Star eyes, heart eyes, cheek puff, cat mouth :3, pupil shrink (shocked)', stack: true, control: fxPills(BLEND_PARAMS) }),
    el('div', { class: 's-row' }, [
      el('div', { class: 's-row-text' }, [
        el('div', { class: 's-title', text: 'Reset FX' }),
        el('div', { class: 's-desc', 'data-vfx-status': '', text: fxStatusText() }),
      ]),
      el('div', { class: 's-ctrl' }, [btn('Reset FX', () => {
        vfx.overrides = {};
        paintFx();
        return postVfx({ clear: true });
      })]),
    ]),
  ], { desc: 'Tap a pill to force it on live · auto mode still fires them with emotions' });
}

export function renderMotionPanel(store) {
  const statusEl = el('div', { class: 's-desc', text: 'Extract head pose, sway, blinks & arms from a short .mp4 / .webm / .mov' });
  const fileIn = el('input', {
    type: 'file', accept: 'video/mp4,video/webm,video/quicktime', hidden: true,
    onchange: async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      statusEl.textContent = `Extracting 60 FPS motion from ${f.name}…`;
      const fd = new FormData();
      fd.append('file', f);
      const r = await (await fetch('/api/motion/extract', { method: 'POST', body: fd })).json();
      if (r.ok) { await store.refreshState(); store.rerenderPanel(); }
      else statusEl.textContent = `Error: ${r.error}`;
    },
  });

  const ms = store.motionStatus || { playing: false, id: '', mix: 0.85, loop: true };
  const motions = store.motions || [];
  const idleId = store.idleClip?.id || '';
  const clipRows = motions.map((c) => {
    const active = ms.playing && ms.id === c.id;
    const isIdle = idleId === c.id;
    return row({
      title: c.name,
      desc: `${c.duration}s · ${c.hasAudio ? 'Audio + motion' : 'Motion'}${active ? ' · playing' : ''}${isIdle ? ' · idle loop' : ''}`,
      cls: active || isIdle ? 'live' : '',
      control: [
        btn(isIdle ? 'Idle ✓' : 'Use as idle', async () => {
          const r = await (await postMotion('/api/motion/idle', { id: isIdle ? '' : c.id })).json();
          if (r.ok) store.idleClip = r.idleClip;
          else statusEl.textContent = `Error: ${r.error}`;
          store.rerenderPanel();
        }, isIdle ? 'primary' : ''),
        btn(active ? 'Stop' : 'Play', async () => {
          const r = await (await postMotion('/api/motion/play',
            active ? { id: '' } : { id: c.id, loop: ms.loop, mix: ms.mix, withAudio: false })).json();
          store.motionStatus = r.status;
          store.rerenderPanel();
        }, active ? 'danger' : ''),
      ],
    });
  });

  const extractRow = el('div', { class: 's-row' }, [
    el('div', { class: 's-row-text' }, [el('div', { class: 's-title', text: 'Extract from video' }), statusEl]),
    el('div', { class: 's-ctrl' }, [
      btn('Extract demo', async () => {
        statusEl.textContent = 'Analyzing demo video via optical flow…';
        const r = await (await fetch('/api/motion/demo', { method: 'POST' })).json();
        if (r.ok) { await store.refreshState(); store.rerenderPanel(); }
        else statusEl.textContent = `Error: ${r.error}`;
      }),
      btn('Upload…', () => fileIn.click(), 'primary'),
      fileIn,
    ]),
  ]);

  return [
    group('Motion Lab', [
      extractRow,
      ...(clipRows.length ? clipRows : [noteRow('No extracted clips yet — upload a video or extract the demo.')]),
      slider({
        label: 'Retarget mix', desc: 'How strongly the clip drives the rig (live mouth flaps stay untouched)',
        min: 0, max: 1, step: 0.05, value: ms.mix ?? 0.85, def: 0.85,
        onInput: async (v) => { ms.mix = v; await postMotion('/api/motion/update', { mix: v }); },
      }),
      toggle({
        label: 'Loop clip', desc: 'Repeat the playing clip continuously', checked: ms.loop ?? true,
        onChange: async (v) => { ms.loop = v; await postMotion('/api/motion/update', { loop: v }); },
      }),
      slider({
        label: 'Idle loop strength',
        desc: idleId ? `Looping “${store.idleClip.name}” under everything: whole-body sway, torso lean, head and drift (her own eyes, blinks, arms and mouth stay hers). Pauses for emotes.`
          : 'Pick “Use as idle” on a clip to loop its whole-body sway as her idle',
        min: 0, max: 2.0, step: 0.05, value: store.config.rig.idleClipMix ?? 1.0, def: 1.0, suffix: '×',
        onInput: (v) => store.patch('rig', { idleClipMix: v }),
      }),
    ], { desc: `${motions.length} clip${motions.length === 1 ? '' : 's'} · video-to-VTuber retargeting` }),
  ];
}
