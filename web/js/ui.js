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
