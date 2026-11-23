/** LIP-SYNC — engine (Hybrid / uLipSync MFCC / IPA formant / amplitude), live phoneme monitor & vowel calibration,
 *  mic input, mouth physics, VTube Studio / VBridger mouth parameter mapping, Viseme Studio. */
import { btn, el, group, more, row, selectRow, slider, toggle } from '../ui.js';

let selectedViseme = 'D';

/* ───────── Engine options ───────── */
const ENGINES = [
  ['hybrid', 'Hybrid (MFCC + IPA Formant — Best)'],
  ['ulipsync_mfcc', 'uLipSync 12-MFCC'],
  ['ipa_formant', 'IPA Formant Vowel Triangle'],
  ['amplitude', 'Classic Amplitude'],
];
const MFCC_SCORING = [['cosine', 'Cosine'], ['l1', 'L1'], ['l2', 'L2']];
// Vowel label → engine viseme code (backend MFCC_PHONEMES: A→D, I→B, U→F, O→E, E→C).
const CALIBRATE_VOWELS = [['AA', 'D'], ['EE', 'B'], ['OO', 'F'], ['OH', 'E'], ['EH', 'C']];

/* ───────── VTube Studio mouth parameters ───────── */
const MOUTH_PARAMS = [
  // id, label, what it drives, bipolar (-1..1)
  ['jawOpen',     'Jaw open',     'Jaw drop amount',        false],
  ['mouthOpen',   'Mouth open',   'Lip opening',            false],
  ['mouthForm',   'Mouth form',   'Frown ↔ smile',          true],
  ['mouthFunnel', 'Mouth funnel', 'O / U pucker',           false],
  ['lipsPress',   'Lips press',   'M / B / P seal',         false],
  ['mouthShrug',  'Mouth shrug',  'F / V lip tuck',         false],
  ['mouthX',      'Mouth X',      'Sideways shift',         true],
  ['tongueOut',   'Tongue out',   'L / TH tongue',          false],
];
const mapDefaults = (bip) => ({ enabled: true, inMin: bip ? -1 : 0, inMax: 1, outMin: bip ? -1 : 0, outMax: 1, smooth: 0.15, curve: 1.0 });
const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
function mapEntry(paramMap, id, bip) {
  const d = mapDefaults(bip);
  const raw = paramMap && typeof paramMap[id] === 'object' && paramMap[id] ? paramMap[id] : {};
  return {
    enabled: raw.enabled ?? d.enabled,
    inMin: num(raw.inMin, d.inMin), inMax: num(raw.inMax, d.inMax),
    outMin: num(raw.outMin, d.outMin), outMax: num(raw.outMax, d.outMax),
    smooth: num(raw.smooth, d.smooth), curve: num(raw.curve, d.curve),
  };
}

/* ───────── Live monitor (polls /api/lipsync/status only while this tab is visible) ───────── */
const VISEME_TO_PH = { A: 'M/B/P', B: 'EE', C: 'EH', D: 'AA', E: 'OH', F: 'OO', G: 'FV', H: 'EH', X: 'Rest' };
function normPhoneme(p) {
  if (p === null || p === undefined) return 'Rest';
  const u = String(p).trim().toUpperCase();
  if (!u || ['REST', 'SIL', 'SILENCE', 'X', 'NONE', '-', 'NEUTRAL'].includes(u)) return 'Rest';
  if (['MBP', 'M/B/P', 'M', 'B', 'P', 'PP', 'MM'].includes(u)) return 'M/B/P';
  if (['FV', 'F', 'V', 'FF', 'F/V'].includes(u)) return 'FV';
  if (['AA', 'A', 'AH'].includes(u)) return 'AA';
  if (['EE', 'I', 'IY', 'IH', 'II'].includes(u)) return 'EE';
  if (['OO', 'U', 'UW', 'UU'].includes(u)) return 'OO';
  if (['OH', 'O', 'OW', 'AO'].includes(u)) return 'OH';
  if (['EH', 'E', 'AE'].includes(u)) return 'EH';
  return u.slice(0, 5);
}

let pollTimer = null;
let statusSupported = null; // null = unknown, true = endpoint live, false = older server (use telemetry)
let lastStatus = null;
let lastProbe = 0;      // re-probe an older server every 10 s so a backend restart is picked up live
let calibMsg = '';
let calibKind = '';

function stopMonitor() {
  clearTimeout(pollTimer);
  pollTimer = null;
}

function startMonitor(store) {
  if (pollTimer) return;
  const tick = async () => {
    const mon = document.getElementById('ls-monitor');
    if (!mon || !mon.isConnected || mon.closest('[hidden]')) { stopMonitor(); return; }

    let phoneme = 'Rest', level = 0, src = '';
    if (statusSupported === false && Date.now() - lastProbe > 10000) statusSupported = null;
    if (statusSupported !== false) {
      lastProbe = Date.now();
      try {
        const r = await fetch('/api/lipsync/status', { cache: 'no-store' });
        if (r.status === 404) statusSupported = false;
        else if (r.ok) {
          const d = await r.json();
          statusSupported = true;
          lastStatus = d;
          // Backend reports the live viseme code (activeViseme: A–H / X); older builds may send a phoneme label.
          const code = d.activeViseme ?? d.viseme;
          phoneme = code !== undefined && code !== null && VISEME_TO_PH[String(code).toUpperCase()]
            ? VISEME_TO_PH[String(code).toUpperCase()]
            : normPhoneme(d.phoneme ?? d.vowel ?? d.activePhoneme ?? d.current ?? code);
          if (d.level !== undefined) {
            level = num(d.level, 0);
            if (level > 1.5) level /= 100;
          } else {
            // Raw RMS (~0.005 quiet … ~0.25 loud) → perceptual 0..1 bar.
            level = Math.min(1, Math.sqrt(Math.max(0, num(d.rms ?? d.volume ?? d.amplitude, 0))) * 2.2);
          }
          src = `${d.engine ? `${String(d.engine).replace(/_/g, ' ')} · ` : ''}${d.source || d.input || 'live'}`;
        }
      } catch { /* transient — try again next tick */ }
    }
    if (statusSupported !== true) {
      const t = store.telemetry || {};
      phoneme = VISEME_TO_PH[t.viseme] || normPhoneme(t.viseme);
      const bands = Array.isArray(t.bands) ? t.bands : [];
      level = bands.length ? Math.min(1, (bands.reduce((a, b) => a + b, 0) / bands.length) * 2.2) : 0;
      src = statusSupported === false ? 'From viseme telemetry (update the studio server for the full monitor)' : 'Connecting…';
    }
    if (!pollTimer) return; // stopped while awaiting

    const badge = document.getElementById('ls-phoneme');
    if (badge && badge.textContent !== phoneme) { badge.textContent = phoneme; badge.dataset.ph = phoneme; }
    const fill = document.getElementById('ls-level-fill');
    if (fill) fill.style.width = `${Math.round(Math.max(0, Math.min(1, level)) * 100)}%`;
    const srcEl = document.getElementById('ls-mon-src');
    if (srcEl && srcEl.textContent !== src) srcEl.textContent = src;

    paintCalibrated(lastStatus);

    pollTimer = setTimeout(tick, statusSupported ? 200 : 140);
  };
  pollTimer = setTimeout(tick, 0);
}

/** Light up calibrated vowel pills from status.calibratedPhonemes ({ D: 3, B: 1, … } viseme code → sample count). */
function paintCalibrated(st) {
  const cal = st?.calibratedPhonemes ?? st?.calibrated ?? st?.calibration;
  const counts = {};
  if (Array.isArray(cal)) cal.forEach((c) => { counts[String(c).toUpperCase()] = 1; });
  else if (cal && typeof cal === 'object') Object.entries(cal).forEach(([c, n]) => { if (n) counts[String(c).toUpperCase()] = Number(n) || 1; });
  document.querySelectorAll('#inspector-body .vowel-pill').forEach((p) => {
    const n = counts[p.dataset.viseme] || 0;
    p.classList.toggle('done', n > 0);
    p.title = `Calibrate “${p.dataset.vowel}” (viseme ${p.dataset.viseme})${n ? ` · ${n} sample${n === 1 ? '' : 's'} saved` : ''}`;
  });
}

async function postCalibrate(body) {
  try {
    const r = await fetch('/api/lipsync/calibrate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (r.status === 404) return { ok: false, error: 'Calibration needs the updated studio server.' };
    const d = await r.json().catch(() => ({ ok: r.ok }));
    return { ok: Boolean(d?.ok ?? r.ok), message: d?.message, error: d?.error, raw: d };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

export function renderLipSyncPanel(store) {
  const ls = store.config.lipsync || {};
  const vis = store.config.visemes || {};
  if (!vis[selectedViseme]) selectedViseme = Object.keys(vis)[0] || 'D';
  const curV = vis[selectedViseme] || {};
  const lp = (label, key, min, max, step, def, suffix = '', desc = '') => slider({
    label, desc, min, max, step, value: ls[key] ?? def, def, suffix, onInput: (v) => store.patch('lipsync', { [key]: v }),
  });

  /* Engine */
  const engine = ENGINES.some(([id]) => id === ls.engine) ? ls.engine : 'hybrid';
  const usesMfcc = engine === 'hybrid' || engine === 'ulipsync_mfcc';

  /* Monitor */
  const vuBars = el('div', { class: 'vu-bars', id: 'vu-bars' },
    Array.from({ length: 12 }, () => el('div', { class: 'vu-bar', style: 'height: 4px' })));
  const monitorRow = el('div', { class: 's-row monitor-row', id: 'ls-monitor' }, [
    el('div', { class: 'ph-badge', id: 'ls-phoneme', 'data-ph': 'Rest', text: 'Rest', 'aria-live': 'polite' }),
    el('div', { class: 's-row-text' }, [
      el('div', { class: 's-title', text: 'Incoming audio' }),
      el('div', { class: 's-desc', id: 'ls-mon-src', text: 'Connecting…' }),
      el('div', { class: 'level-track', 'aria-hidden': 'true' }, [el('div', { class: 'level-fill', id: 'ls-level-fill' })]),
    ]),
  ]);
  const calibStatus = el('div', { class: `apply-status ${calibKind}`.trim(), id: 'ls-calib-msg', role: 'status',
    text: calibMsg || 'Hold a steady vowel, then click its pill to tune MFCC matching to your voice.' });
  const setCalib = (msg, kind) => {
    calibMsg = msg; calibKind = kind;
    calibStatus.textContent = msg;
    calibStatus.className = `apply-status ${kind}`.trim();
  };
  const vowelPills = el('div', { class: 'chip-grid' }, CALIBRATE_VOWELS.map(([v, code]) => el('button', {
    type: 'button', class: 'chip vowel-pill', 'data-vowel': v, 'data-viseme': code, title: `Calibrate “${v}” (viseme ${code})`, text: v,
    onclick: async (e) => {
      const b = e.currentTarget;
      b.disabled = true;
      setCalib(`Listening for “${v}” — hold it steady…`, 'pending');
      const res = await postCalibrate({ phoneme: code });
      b.disabled = false;
      setCalib(res.ok ? (res.message || `Calibrated ${v} ✓`) : `Could not calibrate ${v}: ${res.error || 'unknown error'}`, res.ok ? 'ok' : 'error');
      if (res.ok) {
        if (res.raw?.status) lastStatus = res.raw.status;
        b.classList.add('done');
        paintCalibrated(lastStatus);
      }
    },
  })));

  /* VTube Studio mapping */
  const setParam = (id, field, value) => {
    const pm = (store.config.lipsync || {}).paramMap;
    const full = {};
    for (const [pid, , , bip] of MOUTH_PARAMS) full[pid] = mapEntry(pm, pid, bip);
    full[id][field] = value;
    store.patch('lipsync', { paramMap: full });
  };
  const mapRows = MOUTH_PARAMS.map(([id, label, what, bip]) => {
    const m = mapEntry(ls.paramMap, id, bip);
    const lo = bip ? -1 : 0;
    const ms = (title, field, min, max, step, def, desc = '') => slider({
      label: title, desc, min, max, step, value: m[field], def, onInput: (v) => setParam(id, field, v),
    });
    return more({
      id: `lip.map.${id}`, title: label,
      desc: `${what} · ${m.enabled ? `in ${m.inMin.toFixed(2)}–${m.inMax.toFixed(2)} → out ${m.outMin.toFixed(2)}–${m.outMax.toFixed(2)}` : 'Disabled'}`,
    }, [
      toggle({ label: 'Enabled', desc: `Drive VTube Studio “${id}” from lip-sync`, checked: m.enabled,
        onChange: (v) => { setParam(id, 'enabled', v); store.rerenderPanel(); } }),
      ms('Input min', 'inMin', lo, 1, 0.01, lo, 'Lip-sync value mapped to output min'),
      ms('Input max', 'inMax', lo, 1, 0.01, 1, 'Lip-sync value mapped to output max'),
      ms('Output min', 'outMin', bip ? -1.5 : 0, 1.5, 0.01, lo),
      ms('Output max', 'outMax', bip ? -1.5 : 0, 1.5, 0.01, 1),
      ms('Smooth', 'smooth', 0, 0.95, 0.01, 0.15, 'Higher = steadier, slower'),
      ms('Curve', 'curve', 0.25, 3, 0.05, 1.0, '1 = linear · below 1 snappier · above 1 softer'),
    ]);
  });

  /* Viseme Studio */
  const visemeGrid = el('div', { class: 'viseme-grid' }, Object.entries(vis).map(([code, data]) => {
    const isHeld = ls.holdViseme === code;
    const isSel = selectedViseme === code;
    return el('button', {
      type: 'button',
      class: `viseme-key${isSel ? ' active' : ''}${isHeld ? ' held' : ''}`,
      title: `${data.label} — click to edit`,
      onclick: () => {
        selectedViseme = code;
        if (ls.holdViseme) store.patch('lipsync', { holdViseme: code });
        store.rerenderPanel();
      },
    }, [el('b', { text: code }), el('small', { text: (data.label || '').split(' ')[0] })]);
  }));
  const patchViseme = (axis, val) => {
    vis[selectedViseme][axis] = val;
    store.patch('visemes', { [selectedViseme]: { [axis]: val } });
  };
  const vs = (label, axis, min, max) => slider({
    label, min, max, step: 0.02, value: curV[axis] ?? 0, onInput: (v) => patchViseme(axis, v),
  });

  const out = [
    group('Lip-Sync Engine', [
      selectRow({
        label: 'Lip-sync engine', desc: 'How incoming audio becomes mouth shapes',
        items: ENGINES, value: engine,
        onChange: (v) => { store.patch('lipsync', { engine: v }); store.rerenderPanel(); },
      }),
      selectRow({
        label: 'MFCC scoring method', desc: usesMfcc ? 'How MFCC frames are matched to calibrated vowels' : 'Only used by the Hybrid and uLipSync engines',
        items: MFCC_SCORING, value: MFCC_SCORING.some(([id]) => id === ls.mfccScoring) ? ls.mfccScoring : 'cosine',
        disabled: !usesMfcc, onChange: (v) => store.patch('lipsync', { mfccScoring: v }),
      }),
      lp('Phoneme sharpness (coarticulation)', 'coarticulation', 0.08, 0.60, 0.01, 0.22, '', 'Lower = crisper phonemes · higher = smoother blending'),
    ], { desc: 'VTube Studio / VBridger-style multi-engine lip-sync' }),

    group('Live Monitor', [
      monitorRow,
      row({ title: 'Spectrum', desc: 'Live formant bands', stack: true, control: vuBars }),
      row({
        title: 'Calibrate vowels', desc: 'Tune vowel detection to your voice', stack: true,
        control: el('div', { class: 'calib-wrap' }, [
          vowelPills,
          btn('Reset calibration', async () => {
            setCalib('Resetting calibration…', 'pending');
            const res = await postCalibrate({ reset: true });
            setCalib(res.ok ? (res.message || 'Calibration reset ✓') : `Could not reset: ${res.error || 'unknown error'}`, res.ok ? 'ok' : 'error');
            if (res.ok) {
              lastStatus = res.raw?.status || { ...(lastStatus || {}), calibratedPhonemes: {} };
              paintCalibrated(lastStatus);
            }
          }, 'danger'),
        ]),
      }),
      el('div', { class: 's-row note' }, [calibStatus]),
    ], { desc: 'Incoming audio and the detected phoneme, live' }),

    group('Input', [
      row({
        title: 'Live microphone', desc: ls.micEnabled ? 'Listening — your voice drives her mouth' : 'Drive the mouth from your mic',
        control: [
          btn('Sample take', () => store.playSample()),
          btn(ls.micEnabled ? 'Stop mic' : 'Start mic', () => {
            store.patch('lipsync', { micEnabled: !ls.micEnabled, holdViseme: '' });
            store.rerenderPanel();
          }, ls.micEnabled ? 'danger' : 'primary'),
        ],
      }),
      lp('Input gain', 'gain', 0.2, 3.0, 0.05, 1.45, '×', 'Sensitivity'),
      lp('Noise gate', 'noiseGate', 0.002, 0.12, 0.002, 0.012, '', 'Ignore sounds below this level'),
    ]),

    group('Mouth Physics', [
      lp('Attack speed (ω)', 'attackOmega', 8, 48, 0.5, 30, '', 'How fast the mouth snaps open'),
      lp('Damping (ζ)', 'dampingZeta', 0.40, 1.30, 0.02, 0.78, '', 'Lower = bouncier'),
      lp('Jaw bounce', 'jawBounce', 0.3, 1.5, 0.02, 1.02),
      more({ id: 'lip.phys.more', title: 'Mouth shaping', desc: 'Smile bias, roundness, lip seal, flutter' }, [
        lp('Smile bias', 'smileBias', -0.6, 0.7, 0.02, 0.14),
        lp('O / U roundness', 'roundness', 0.0, 1.5, 0.05, 1.12),
        lp('M / B / P lip seal', 'lipSealSharpness', 0.5, 1.6, 0.05, 1.18),
        lp('Syllable micro-flutter', 'microFlutter', 0.0, 1.0, 0.05, 0.22),
      ]),
    ]),

    group('VTube Studio Mouth Parameter Mapping', [
      ...mapRows,
      row({
        title: 'Reset mapping', desc: 'Restore linear 1:1 mapping for all 8 parameters',
        control: btn('Reset', () => {
          const full = {};
          for (const [pid, , , bip] of MOUTH_PARAMS) full[pid] = mapDefaults(bip);
          store.patch('lipsync', { paramMap: full });
          store.rerenderPanel();
        }, 'danger'),
      }),
    ], { desc: 'Hikari’s 8 real mouth parameters · input → output range, smoothing and curve' }),

    group('Viseme Studio', [
      row({ title: 'Viseme', desc: `Editing [${selectedViseme}] ${curV.label || ''}`, stack: true, control: visemeGrid }),
      toggle({
        label: `Solo-hold [${selectedViseme}]`, desc: 'Freeze this shape on the live mouth while tuning',
        checked: ls.holdViseme === selectedViseme,
        onChange: (on) => { store.patch('lipsync', { holdViseme: on ? selectedViseme : '' }); store.rerenderPanel(); },
      }),
      vs('Mouth open', 'open', 0, 1),
      vs('Form (sad ↔ smile)', 'form', -1, 1),
      vs('Funnel (O/U pucker)', 'funnel', 0, 1),
      more({ id: 'lip.vis.more', title: 'More shape axes', desc: 'Asymmetry, lip tuck, tongue' }, [
        vs('Asymmetry X', 'x', -1, 1),
        vs('Lip tuck (F/V)', 'shrug', 0, 1),
        vs('Tongue (L/TH)', 'tongue', 0, 1),
      ]),
      row({
        title: 'Reset visemes', desc: 'Restore every viseme to the default shape',
        control: btn('Reset all', async () => {
          const r = await (await fetch('/api/visemes/reset', { method: 'POST' })).json();
          if (r.ok) { store.config.visemes = r.visemes; store.rerenderPanel(); }
        }, 'danger'),
      }),
    ]),
  ];

  // Start the light poller after this render is attached to the DOM.
  requestAnimationFrame(() => startMonitor(store));
  return out;
}
