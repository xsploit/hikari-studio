/**
 * Tiny zero-dependency DOM primitives for the Codex-style Settings modal.
 * Anatomy: group(h2 + card) → rows (title + muted subtitle on the left, compact control on the right).
 */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of Array.isArray(children) ? children : [children]) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

/* ---------- Persisted disclosure state ---------- */

const OPEN_KEY = 'hikari.disclosure.v3';
let openState = {};
try { openState = JSON.parse(localStorage.getItem(OPEN_KEY) || '{}') || {}; } catch { openState = {}; }
function saveOpenState() {
  try { localStorage.setItem(OPEN_KEY, JSON.stringify(openState)); } catch { /* storage blocked */ }
}

/* ---------- Layout ---------- */

/** Section heading + rounded card of stacked rows separated by hairlines. */
export function group(title, rows = [], { desc = '', right = null, cls = '' } = {}) {
  return el('section', { class: `s-group ${cls}`.trim() }, [
    el('div', { class: 's-group-head' }, [
      el('div', {}, [el('h2', { text: title }), desc ? el('p', { text: desc }) : null]),
      right,
    ]),
    el('div', { class: 's-card' }, rows),
  ]);
}

/** One settings row. stack=true puts the control under the text (chip sets, text fields, lists). */
export function row({ title, desc = '', control = null, stack = false, cls = '' }) {
  return el('div', { class: `s-row${stack ? ' stack' : ''} ${cls}`.trim() }, [
    el('div', { class: 's-row-text' }, [
      el('div', { class: 's-title', text: title }),
      desc ? el('div', { class: 's-desc', text: desc }) : null,
    ]),
    control ? el('div', { class: 's-ctrl' }, Array.isArray(control) ? control : [control]) : null,
  ]);
}

/** Muted full-width note row. */
export function noteRow(text, attrs = {}) {
  return el('div', { class: 's-row note', ...attrs }, [el('div', { class: 's-desc', text })]);
}
export function note(text, attrs = {}) {
  return el('div', { class: 's-desc', ...attrs, text });
}

/** Disclosure row: clicking reveals extra rows inside the same card (keeps cards short). */
export function more({ id, title, desc = '', open = false }, rows = []) {
  const key = id || title;
  const isOpen = key in openState ? openState[key] : open;
  const body = el('div', { class: 's-more-body' }, rows);
  const head = el('button', { type: 'button', class: 's-row s-more-head', 'aria-expanded': String(isOpen) }, [
    el('div', { class: 's-row-text' }, [
      el('div', { class: 's-title', text: title }),
      desc ? el('div', { class: 's-desc', text: desc }) : null,
    ]),
    el('span', { class: 's-chev', 'aria-hidden': 'true' }),
  ]);
  const wrap = el('div', { class: `s-more${isOpen ? ' open' : ''}` }, [head, body]);
  head.addEventListener('click', () => {
    const next = !wrap.classList.contains('open');
    wrap.classList.toggle('open', next);
    wrap.classList.toggle('just-opened', next);
    head.setAttribute('aria-expanded', String(next));
    openState[key] = next;
    saveOpenState();
  });
  return wrap;
}

/* ---------- Controls (all return full rows) ---------- */

export function slider({ label, desc = '', min, max, step = 0.01, value = 0, suffix = '', def, onInput }) {
  const v0 = Number.isFinite(Number(value)) ? Number(value) : Number(min);
  const valEl = el('span', { class: 'ctrl-val', text: fmt(v0, step, suffix) });
  const inp = el('input', {
    type: 'range', min, max, step, value: v0, 'aria-label': label,
    oninput: (e) => {
      const v = parseFloat(e.target.value);
      valEl.textContent = fmt(v, step, suffix);
      paint();
      onInput?.(v);
    },
  });
  const paint = () => {
    const pct = ((parseFloat(inp.value) - Number(min)) / (Number(max) - Number(min))) * 100;
    inp.style.setProperty('--fill', `${Math.max(0, Math.min(100, pct))}%`);
  };
  paint();
  if (def !== undefined) {
    valEl.title = `Double-click to reset (${fmt(def, step, suffix)})`;
    valEl.classList.add('resettable');
    valEl.addEventListener('dblclick', () => {
      inp.value = def;
      inp.dispatchEvent(new Event('input'));
    });
  }
  return row({ title: label, desc, control: el('div', { class: 's-slider' }, [inp, valEl]), cls: 'has-slider' });
}

export function switchEl({ checked = false, label = '', onChange, disabled = false }) {
  const inp = el('input', { type: 'checkbox', 'aria-label': label, disabled, onchange: (e) => onChange?.(e.target.checked) });
  inp.checked = Boolean(checked);
  return el('label', { class: 'switch' }, [inp, el('span')]);
}

export function toggle({ label, hint = '', desc = '', checked = false, disabled = false, onChange }) {
  return row({ title: label, desc: desc || hint, control: switchEl({ checked, label, onChange, disabled }) });
}

export function colorRow({ label, desc = '', value = '#ffffff', onInput, onChange }) {
  const inp = el('input', {
    type: 'color', value, 'aria-label': label,
    oninput: (e) => onInput?.(e.target.value),
    onchange: (e) => onChange?.(e.target.value),
  });
  const hex = el('span', { class: 'hex', text: String(value).toUpperCase() });
  inp.addEventListener('input', () => { hex.textContent = inp.value.toUpperCase(); });
  return row({ title: label, desc, control: el('div', { class: 'color-pill' }, [hex, inp]) });
}

export function selectEl(items, cur, onChange, attrs = {}) {
  return el('select', { class: 'pill-select', onchange: (e) => onChange(e.target.value), ...attrs },
    items.map(([v, t]) => el('option', { value: v, text: t, selected: cur === v })));
}

export function selectRow({ label, desc = '', items, value, onChange, disabled = false }) {
  return row({ title: label, desc, control: selectEl(items, value, onChange, { disabled, 'aria-label': label }) });
}

/** Segmented pill control for 2–4 options. */
export function segmented(items, { active, onPick, label = '' } = {}) {
  return el('div', { class: 'seg', role: 'group', 'aria-label': label }, items.map(([id, text]) =>
    el('button', {
      type: 'button', class: `seg-btn${active === id ? ' active' : ''}`, 'aria-pressed': String(active === id),
      text, onclick: () => onPick?.(id),
    })));
}
export function segRow({ label, desc = '', items, active, onPick }) {
  return row({ title: label, desc, control: segmented(items, { active, onPick, label }) });
}

/** Chip grid. items: [[id, label, title?]] */
export function chips(items, { active, onPick, accent = false, cls = '' } = {}) {
  return el('div', { class: `chip-grid ${cls}`.trim() }, items.map(([id, label, title]) => {
    const on = typeof active === 'function' ? active(id) : active === id;
    return el('button', {
      type: 'button',
      class: `chip${accent ? ' accent2' : ''}${on ? ' active' : ''}`,
      'aria-pressed': String(Boolean(on)),
      title: title || undefined,
      text: label,
      onclick: () => onPick?.(id),
    });
  }));
}
export function chipsRow({ label, desc = '', items, active, onPick, accent = false }) {
  return row({ title: label, desc, stack: true, control: chips(items, { active, onPick, accent }) });
}

export function btn(text, onclick, variant = '', attrs = {}) {
  return el('button', { type: 'button', class: `btn${variant ? ` ${variant}` : ''}`, text, onclick, ...attrs });
}
export function btnRow(children) {
  return el('div', { class: 'btn-row' }, children);
}
export function actionRow({ label, desc = '', button, onClick, variant = '', disabled = false }) {
  return row({ title: label, desc, control: btn(button, onClick, variant, { disabled }) });
}

export function textIn(attrs = {}) {
  return el('input', { class: 'text-in', type: 'text', ...attrs });
}

export function fmt(v, step, suffix = '') {
  const s = Number(step);
  const dec = s < 0.005 ? 3 : (s < 0.1 ? 2 : (s < 1 ? 1 : 0));
  return `${Number(v).toFixed(dec)}${suffix}`;
}

/** Keep an unknown backend value selectable (newer/older server) instead of silently showing the first option. */
export function withCurrent(items, value) {
  if (value === undefined || value === null || value === '' || items.some(([id]) => id === value)) return items;
  return [[value, String(value)], ...items];
}

/** Codex-style radio list: one row per option (title + subtitle, radio dot on the right). Returns rows. */
export function choiceRows(items, { active, onPick, name = 'choice' } = {}) {
  return items.map(([id, title, desc]) => {
    const on = active === id;
    return el('button', {
      type: 'button', class: `s-row choice-row${on ? ' active' : ''}`, role: 'radio',
      'aria-checked': String(on), 'data-choice': `${name}:${id}`, onclick: () => { if (!on) onPick?.(id); },
    }, [
      el('div', { class: 's-row-text' }, [
        el('div', { class: 's-title', text: title }),
        desc ? el('div', { class: 's-desc', text: desc }) : null,
      ]),
      el('span', { class: 'radio', 'aria-hidden': 'true' }),
    ]);
  });
}

/* ───────────────────────── Settings search ─────────────────────────
 * Index = every group heading, row title, subtitle, disclosure title and control text (buttons, chips,
 * segmented pills, <select> options) of each settings tab, extracted from rendered (detached) panels.
 * So "Full Body", room names, emote names, model names… are all searchable without a hand-kept list. */

export const normSearch = (s) => String(s ?? '').toLowerCase()
  .normalize('NFKD').replace(/[\u0300-\u036f]/g, '')   // strip accents
  .replace(/[^\p{L}\p{N}]+/gu, ' ')                       // every symbol/punctuation/glyph -> word break
  .trim();

// Query term → alternatives (any match counts). Multi-word keys are matched as phrases first.
const SEARCH_SYNONYMS = {
  'full body': ['full body', 'framing'], 'half body': ['waist up', 'framing'], 'upper body': ['waist up', 'framing'],
  'close up': ['bust', 'framing'], closeup: ['bust', 'framing'], portrait: ['bust', 'framing'], headshot: ['bust', 'framing'],
  waist: ['waist up', 'framing'], bust: ['bust', 'framing'], framing: ['framing', 'camera'],
  zoom: ['zoom', 'scale', 'framing'], camera: ['camera', 'framing', 'zoom'], position: ['position', 'lift', 'framing'],
  hand: ['arm', 'gesture'], hands: ['arm', 'gesture'], arm: ['arm'], arms: ['arm'], gesture: ['gesture', 'arm'], wave: ['wave'],
  'lip sync': ['lip sync', 'viseme', 'mouth'], lipsync: ['lip sync', 'viseme', 'mouth'], lips: ['lip', 'mouth', 'viseme'],
  mouth: ['mouth', 'viseme', 'lip'], viseme: ['viseme'], phoneme: ['phoneme', 'viseme', 'vowel'], vowel: ['vowel', 'calibrat'],
  mic: ['mic', 'microphone'], microphone: ['mic', 'microphone'],
  voice: ['voice', 'fish', 'speech', 'tts'], tts: ['tts', 'speech'], speech: ['speech', 'tts', 'voice'], fish: ['fish'],
  call: ['call', 'duplex'], duplex: ['duplex', 'call'],
  gateway: ['gateway', 'vercel'], vercel: ['vercel', 'gateway'], key: ['key', 'token'], token: ['token', 'key'], api: ['api', 'key'],
  hair: ['hair', 'bangs'], bangs: ['bangs', 'hair'], eye: ['eye', 'iris'], eyes: ['eye', 'iris'], iris: ['iris', 'eye'],
  outfit: ['outfit', 'wardrobe', 'fabric', 'halter'], clothes: ['outfit', 'wardrobe', 'fabric'], dress: ['outfit', 'wardrobe'],
  shoes: ['shoes', 'sneakers', 'high tops'], color: ['color', 'tint', 'hue'], colour: ['color', 'tint', 'hue'],
  memory: ['memory', 'memories', 'fact', 'context'], memories: ['memor', 'fact'], remember: ['remember', 'memor', 'note'],
  fact: ['fact', 'memor'], facts: ['fact', 'memor'], note: ['note', 'memor'], notes: ['note', 'memor'], forget: ['forget', 'clear'],
  context: ['context', 'memory', 'turns'], history: ['history', 'turns', 'chat'], chat: ['chat', 'history', 'turns'],
  restart: ['restart', 'persist'], persist: ['persist', 'restart'], continuity: ['continuity', 'context'],
  mood: ['mood', 'expression'], pin: ['pin'],
  background: ['room', 'background'], room: ['room', 'background'], obs: ['obs', 'chroma', 'transparent', 'broadcast'],
  green: ['chroma', 'green'], transparent: ['transparent'], particles: ['particles', 'sakura', 'embers', 'rain', 'dust'],
  light: ['light', 'rim'], lighting: ['light', 'rim'], rim: ['rim'], glow: ['glow', 'emissive', 'bloom'], bloom: ['bloom', 'glow'],
  shader: ['shader', 'look preset'], sharp: ['sharp'], blur: ['blur', 'bokeh'],
  vfx: ['vfx', 'fx'], fx: ['fx', 'vfx'], effects: ['vfx', 'fx'], tiddlywinks: ['vfx'],
  sparkle: ['sparkle'], sparkles: ['sparkle'], sweat: ['sweat'], sweatdrop: ['sweat'],
  anger: ['anger'], angry: ['anger'], shock: ['shock'], music: ['music', 'vibing'],
  star: ['star'], stars: ['star'], starry: ['star'], cat: ['cat', 'smug'], blend: ['blend'], blendshape: ['blend shape'],
  emote: ['emote'], emotes: ['emote'], expression: ['expression', 'mood'], face: ['face', 'expression', 'blush', 'eye'],
  blink: ['blink'], gaze: ['gaze', 'look at', 'mouse'], look: ['look', 'gaze'], physics: ['physics', 'jiggle', 'bounce', 'spring'],
  jiggle: ['jiggle', 'bounce'], bounce: ['bounce', 'jiggle'], chibi: ['chibi', 'proportion'], pat: ['pat', 'poke'], poke: ['poke', 'pat'],
  decider: ['decision', 'decider'], decision: ['decision', 'decider'], brain: ['brain', 'llm', 'decision'], llm: ['llm', 'model'],
  ai: ['ai', 'llm', 'brain', 'gateway'], persona: ['persona', 'personality'], personality: ['personality', 'persona'],
  plugin: ['plugin'], plugins: ['plugin'], param: ['parameter'], params: ['parameter'], parameter: ['parameter'], preset: ['preset'],
  motion: ['motion', 'clip', 'retarget'], video: ['video', 'motion lab'], save: ['save', 'preset'],
};
const SYN_PHRASES = Object.keys(SEARCH_SYNONYMS).filter((k) => k.includes(' ')).sort((a, b) => b.length - a.length);

/** Split a query into term groups: [[alt, alt…], …] — every group must match. */
export function searchTerms(query) {
  let q = ` ${normSearch(query)} `;
  const groups = [];
  for (const phrase of SYN_PHRASES) {
    if (q.includes(` ${phrase} `)) {
      groups.push([...new Set([phrase, ...SEARCH_SYNONYMS[phrase]].map(normSearch))]);
      q = q.replace(` ${phrase} `, ' ');
    }
  }
  for (const w of q.trim().split(' ').filter(Boolean)) {
    groups.push([...new Set([w, ...(SEARCH_SYNONYMS[w] || [])].map(normSearch))]);
  }
  return groups;
}

/** Extract searchable entries from a rendered settings tab. Locator = group + title + nth occurrence. */
export function extractSearchEntries(root, tab, tabTitle) {
  const entries = [];
  const occ = new Map();
  const add = (e) => {
    const k = `${e.kind === 'group' ? 'G' : 'R'}\u0000${e.group}\u0000${e.title}`;
    const n = occ.get(k) || 0;
    occ.set(k, n + 1);
    entries.push({
      ...e, tab, tabTitle, occ: n,
      hTitle: normSearch(e.title), hDesc: normSearch(e.desc), hKw: normSearch(e.kw), hGroup: normSearch(e.group),
    });
  };
  root.querySelectorAll('.s-group').forEach((g) => {
    const group = g.querySelector('.s-group-head h2')?.textContent.trim() || '';
    if (!group) return;
    add({ kind: 'group', group, title: group, desc: g.querySelector('.s-group-head p')?.textContent.trim() || '', kw: '' });
    g.querySelectorAll('.s-row').forEach((r) => {
      const title = r.querySelector(':scope > .s-row-text > .s-title')?.textContent.trim();
      if (!title) return;
      const desc = r.querySelector(':scope > .s-row-text > .s-desc')?.textContent.trim() || '';
      const ctrl = r.querySelector(':scope > .s-ctrl');
      const kw = ctrl ? [...ctrl.querySelectorAll('button, option, optgroup, a')]
        .map((n) => n.label || n.textContent || n.getAttribute('aria-label') || '').join(' ').slice(0, 600) : '';
      const parent = r.parentElement?.closest('.s-more-body')
        ? (r.parentElement.closest('.s-more')?.querySelector(':scope > .s-more-head .s-title')?.textContent.trim() || '') : '';
      add({ kind: r.classList.contains('s-more-head') ? 'section' : 'row', group, title, desc, kw, parent });
    });
  });
  return entries;
}

const atWord = (hay, term) => hay.startsWith(term) || hay.includes(` ${term}`);

/** Score entries against a query; returns the best `limit` matches (every term group must match somewhere). */
export function searchSettings(entries, query, limit = 10) {
  const groups = searchTerms(query);
  if (!groups.length) return [];
  const scored = [];
  for (const e of entries) {
    let score = 0;
    let ok = true;
    for (const alts of groups) {
      let best = 0;
      alts.forEach((a, i) => {
        if (!a) return;
        // The typed word (or the same word spaced differently, e.g. lipsync / lip sync) beats a synonym.
        const w = i === 0 || a.replace(/ /g, '') === alts[0].replace(/ /g, '') ? 1 : 0.8;
        // Match at word starts only ("arm" must not hit "warmth"); descriptions outrank button/option text.
        if (e.hTitle.startsWith(a)) best = Math.max(best, 6 * w);
        else if (atWord(e.hTitle, a)) best = Math.max(best, 4.5 * w);
        else if (atWord(e.hDesc, a)) best = Math.max(best, 2.6 * w);
        else if (atWord(e.hKw, a)) best = Math.max(best, 2.4 * w);
        else if (atWord(e.hGroup, a)) best = Math.max(best, 1.2 * w);
      });
      if (!best) { ok = false; break; }
      score += best;
    }
    if (!ok) continue;
    if (e.kind === 'group') score -= 0.5;           // prefer concrete rows over headings
    if (e.kind === 'section') score -= 0.2;
    score -= Math.min(1, e.title.length / 80);      // shorter titles first on ties
    scored.push([score, e]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  const seen = new Set();
  const out = [];
  for (const [, e] of scored) {
    const k = `${e.tab}|${e.kind}|${e.group}|${e.title}|${e.occ}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(e);
    if (out.length >= limit) break;
  }
  return out;
}

/** Find the live element for an index entry inside the rendered settings body. */
export function findSearchTarget(root, entry) {
  for (const g of root.querySelectorAll('.s-group')) {
    if ((g.querySelector('.s-group-head h2')?.textContent.trim() || '') !== entry.group) continue;
    if (entry.kind === 'group') return g;
    let n = 0;
    for (const r of g.querySelectorAll('.s-row')) {
      if (r.querySelector(':scope > .s-row-text > .s-title')?.textContent.trim() !== entry.title) continue;
      if (n === entry.occ) return r;
      n += 1;
    }
  }
  return null;
}
