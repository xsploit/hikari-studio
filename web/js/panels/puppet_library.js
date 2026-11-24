/** PUPPET LIBRARY — switch between Hikari (Custom), bundled rigs (Hiyori, Mao…) and imported .inp / .inx / Live2D rigs.
 *  Fetches GET /api/models lazily while Look & Props is open (cached 8 s, no background timers).
 *  Below the cards: live rig hookup badges + 1-click mouth-flap test, and an "Extra Rig Controls" inspector
 *  (GET/POST /api/models/params) cached per active puppet and refetched whenever the puppet switches.
 *  Between them: "Rig Abilities & Props" — auto-discovered 1-click ability pills (GET /api/models/abilities,
 *  shared cache in emotes.js, fetched alongside params for the same puppet). */
import { btn, el, group, more, row, segmented, switchEl } from '../ui.js';
import { abilities, abilityFoot, abilityGrid, invalidateAbilities, loadAbilities, syncAbilityUi } from './emotes.js';

const CACHE_MS = 8000;
const HIKARI_ID = 'hikari';
const SEARCH_KW = 'puppet puppets model models avatar avatars rig rigs character switch hikari hiyori mao';
const CAPS_KW = 'capabilities capability hookups hooked auto hooked mapped mouth flap flaps test mouth lip lips head eyes arms physics springs extras';
const PARAMS_KW = 'extra rig controls rig params parameters params extras extra controls toggles props effects fx aura bunny ears hidden physics springs arms hands auto mapped override reset';
const IMPORT_KW = 'import rig inp inx inochi live2d zip mint convert';
const ABIL_KW = 'rig abilities ability props prop spells spell magic starlight explosion aura heart burst moments cast ai trigger autonomous';

const lib = {
  data: null,          // last GET /api/models payload
  fetchedAt: 0,
  loading: null,       // in-flight promise
  error: '',
  busy: false,
  status: '',
  kind: '',            // '' | 'pending' | 'ok' | 'error'
  confirmDelete: null, // model id awaiting a second click
};

const visible = (store) => store.activeTab === 'look' && !document.getElementById('settings')?.hidden;

async function api(url, options) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => null);
  if (!response.ok || !result?.ok) throw new Error(result?.error || result?.detail || `Studio returned HTTP ${response.status}`);
  return result;
}

/** Fetch the model list (force = skip the 8 s cache). Re-renders the panel when it lands. */
export function refreshModels(store, force = false) {
  if (lib.loading) return force ? lib.loading.then(() => refreshModels(store, true)) : lib.loading;
  if (!force && lib.data && Date.now() - lib.fetchedAt < CACHE_MS) return Promise.resolve(lib.data);
  lib.loading = api('/api/models')
    .then((data) => {
      lib.data = data;
      lib.error = '';
      const active = data.activeModel || HIKARI_ID;
      if (store.config?.stage) store.config.stage.activeModel = active;
      store.setPuppetBadge?.(active, modelName(active));
    })
    .catch((error) => { lib.error = error.message; })
    .finally(() => {
      lib.fetchedAt = Date.now();
      lib.loading = null;
      store.rerenderPanel();
    });
  return lib.loading;
}

const models = () => (Array.isArray(lib.data?.models) ? lib.data.models : []);
export function modelName(id) {
  if (!id) return 'Hikari (Custom)';
  return models().find((m) => m.id === id)?.name || (id === HIKARI_ID ? 'Hikari (Custom)' : String(id));
}
export const activeModelId = (store) => store.config?.stage?.activeModel || lib.data?.activeModel || HIKARI_ID;

const summary = (m) => {
  const n = Number(m.paramCount);
  const params = Number.isFinite(n) && n > 0 ? `${n} params` : 'Rig';
  return `${params} · ${m.customizable ? 'Customizable' : 'Auto-mapped'}`;
};
const channelsTitle = (m) => (Array.isArray(m.mappedChannels) && m.mappedChannels.length
  ? `Mapped: ${m.mappedChannels.join(', ')}` : 'No channels reported');

/** Apply a select/import/delete response to the client store so Stage camera sliders stay in sync. */
function applyResult(store, result) {
  const active = result.activeModel || result.model?.id;
  if (active && store.config) {
    store.config.stage = store.config.stage || {};
    store.config.stage.activeModel = active;
    if (lib.data) lib.data.activeModel = active;
  }
  if (result.camera && typeof result.camera === 'object') {
    store.config.camera = { ...(store.config.camera || {}), ...result.camera };
  }
  store.setPuppetBadge?.(active || activeModelId(store), modelName(active));
}

async function run(store, pending, task, success) {
  if (lib.busy) return;
  lib.busy = true;
  lib.status = pending;
  lib.kind = 'pending';
  lib.confirmDelete = null;
  store.rerenderPanel();
  try {
    const result = await task();
    applyResult(store, result);
    lib.status = success(result);
    lib.kind = 'ok';
  } catch (error) {
    lib.status = error.message;
    lib.kind = 'error';
  } finally {
    lib.busy = false;
    rig.key = null;                      // re-read rig params/capabilities for whatever is on stage now
    invalidateAbilities();               // …and its auto-discovered abilities
    await refreshModels(store, true);   // immediate refresh after switch / import / delete
  }
}

const json = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function selectModel(store, m) {
  if (m.id === activeModelId(store)) {
    lib.confirmDelete = null;
    lib.status = `${m.name || m.id} is already on stage.`;
    lib.kind = '';
    store.rerenderPanel();
    return;
  }
  return run(store, `Switching to ${m.name || m.id}…`,
    () => api('/api/models/select', json({ id: m.id, autoFrame: true })),
    () => `Now using ${m.name || m.id} — camera auto-framed ✓`);
}

function deleteModel(store, m) {
  if (lib.confirmDelete !== m.id) {
    lib.confirmDelete = m.id;
    lib.status = `Click Remove again to delete ${m.name || m.id} from the library.`;
    lib.kind = 'pending';
    store.rerenderPanel();
    return;
  }
  const wasActive = m.id === activeModelId(store);
  return run(store, `Removing ${m.name || m.id}…`,
    () => api(`/api/models/${encodeURIComponent(m.id)}`, { method: 'DELETE' }).then(async (result) => {
      // DELETE returns no camera; resync full config if the stage fell back to another puppet.
      if (wasActive && !result.camera) await store.refreshState?.().catch(() => {});
      return result;
    }),
    (result) => `Removed ${m.name || m.id}${wasActive ? ` — back to ${modelName(result.activeModel)}` : ''} ✓`);
}

function importRig(store, file) {
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  return run(store, `Importing & converting rig… (${file.name})`,
    () => api('/api/models/import', { method: 'POST', body: form }),
    (result) => `Imported ${result.model?.name || result.model?.id || file.name} and put it on stage ✓`);
}

function puppetCard(store, m, active) {
  const on = m.id === active;
  const confirming = lib.confirmDelete === m.id;
  const card = el('div', { class: `puppet-card${on ? ' active' : ''}${confirming ? ' confirming' : ''}` }, [
    el('button', {
      type: 'button', class: 'puppet-pick', 'aria-pressed': String(on), title: channelsTitle(m),
      disabled: lib.busy, onclick: () => selectModel(store, m),
    }, [
      el('span', { class: 'puppet-dot', 'aria-hidden': 'true' }),
      el('span', { class: 'puppet-text' }, [
        el('span', { class: 'puppet-name', text: m.name || m.id }),
        el('span', { class: 'puppet-sub', text: summary(m) }),
      ]),
    ]),
    m.builtin ? null : el('button', {
      type: 'button', class: `puppet-del${confirming ? ' confirm' : ''}`, disabled: lib.busy,
      title: confirming ? 'Click to confirm' : `Remove ${m.name || m.id}`,
      'aria-label': confirming ? `Confirm removing ${m.name || m.id}` : `Remove ${m.name || m.id}`,
      text: confirming ? 'Remove' : '×',
      onclick: () => deleteModel(store, m),
    }),
  ]);
  return card;
}

/* ───────────── Rig hookups + Extra Rig Controls (GET/POST /api/models/params) ───────────── */

const PARAM_ROW_CAP = 80;
const SEND_MS = 40;              // slider POST throttle — smooth drags without flooding HTTP
const TEST_FALLBACK_S = 2.2;
const PARAM_TABS = [
  ['extras', 'Extras & FX'],
  ['arms', 'Arms & Hands'],
  ['auto', 'Auto-Mapped'],
  ['physics', 'Physics'],
  ['all', 'All'],
];

const rig = {
  key: null,          // active puppet id the cached params belong to (null = refetch)
  data: null,         // last GET /api/models/params payload
  loading: null,
  error: '',
  tab: null,          // null = pick default for this rig (Extras if any, else All)
  query: '',
  testing: false,
  testMsg: '',
  testKind: '',
  overrides: 0,
};
const sends = new Map(); // param name → { timer, last, inflight, pending }

const params = () => (Array.isArray(rig.data?.params) ? rig.data.params : []);
const num = (v, d = 0) => (Number.isFinite(Number(v)) ? Number(v) : d);
const vec = (v, d = [0, 0]) => (Array.isArray(v) ? [num(v[0], d[0]), num(v[1], d[1])] : [num(v, d[0]), d[1]]);
const countOverrides = () => params().filter((p) => p.overridden).length;

function inCategory(p, tab) {
  if (tab === 'all') return true;
  if (tab === 'auto') return p.category === 'auto' || Boolean(p.mappedTo);
  return p.category === tab;
}
const tabCount = (tab) => params().filter((p) => inCategory(p, tab)).length;
const currentTab = () => rig.tab || (tabCount('extras') > 0 ? 'extras' : 'all');

/** Fetch params for the puppet on stage (once per puppet; switching puppets clears `rig.key`). */
function ensureRig(store, active) {
  if (rig.key === active || rig.loading || lib.busy || !visible(store)) return;
  rig.key = active;
  rig.tab = null;
  rig.query = '';
  rig.loading = api('/api/models/params')
    .then((data) => { rig.data = data; rig.error = ''; rig.overrides = countOverrides(); })
    .catch((error) => { rig.data = null; rig.error = error.message; })
    .finally(() => {
      rig.loading = null;
      // The puppet switched while we were loading: drop this result and fetch again on render.
      if (rig.key !== activeModelId(store)) rig.key = null;
      store.rerenderPanel();
    });
}

/* ---- Capability badges + Test Mouth Flap ---- */

function badge(text, state, title) {
  return el('span', { class: `rig-badge ${state}`, title: title || undefined, text });
}

function capsBadges() {
  if (rig.loading && !rig.data) return [badge('Checking rig…', 'pending')];
  if (rig.error) return [badge('Hookups unknown', 'off', rig.error)];
  const c = rig.data?.capabilities || {};
  const on = (k) => Boolean(c[k]);
  const yes = (label, k, title) => badge(`${label} ${on(k) ? '✓' : '–'}`, on(k) ? 'ok' : 'off',
    title || `${label} ${on(k) ? 'auto-hooked to the live rig' : 'not found on this rig'}`);
  const physics = num(c.physicsCount);
  const extras = num(c.extrasCount);
  const also = ['body', 'gaze', 'brows', 'blush', 'breath'].map((k) => `${k[0].toUpperCase()}${k.slice(1)} ${on(k) ? '✓' : '–'}`).join(' · ');
  return [
    yes('Head', 'head', `Head ${on('head') ? '✓' : '–'} · ${also}`),
    yes('Eyes', 'eyes'),
    on('mouth')
      ? badge('Mouth Flaps ✓', 'ok', 'Speech drives this rig’s mouth')
      : badge('Mouth Flaps ⚠ unbound', 'warn', 'No mouth parameter was auto-mapped — speech will not move the lips'),
    yes('Arms', 'arms'),
    physics > 0 ? badge(`Physics (${physics}) ✓`, 'ok', `${physics} physics springs`) : null,
    extras > 0 ? badge(`Extras (${extras})`, 'accent', `${extras} hidden toggles / effects — see Extra Rig Controls`) : null,
  ].filter(Boolean);
}

const TEST_IDLE = 'Test Mouth Flap';
const TEST_BUSY = 'Testing mouth flaps…';

/** Patch every live copy of the test button/status in place (no full re-render — keeps slider drags alive). */
function syncTestUi() {
  document.querySelectorAll('.rig-test-btn').forEach((b) => {
    b.disabled = rig.testing;
    b.textContent = rig.testing ? TEST_BUSY : TEST_IDLE;
  });
  document.querySelectorAll('.rig-test-status').forEach((s) => {
    s.textContent = rig.testMsg;
    s.className = `rig-test-status apply-status ${rig.testKind}`.trim();
    s.hidden = !rig.testMsg;
  });
}

async function testMouth() {
  if (rig.testing) return;
  rig.testing = true;
  rig.testMsg = 'Watch the lips — playing a 2.2 s speech flap…';
  rig.testKind = 'pending';
  syncTestUi();
  let holdMs = TEST_FALLBACK_S * 1000;
  try {
    const r = await api('/api/models/test-mouth', json({}));
    holdMs = Math.max(400, num(r.duration, TEST_FALLBACK_S) * 1000);
    if (r.mouthMapped === false) {
      rig.testMsg = 'Played, but no mouth parameter is mapped on this rig — lips can’t move.';
      rig.testKind = 'error';
    } else {
      rig.testMsg = 'Mouth flaps playing on stage ✓';
      rig.testKind = 'ok';
    }
  } catch (error) {
    rig.testMsg = `Mouth test failed: ${error.message}`;
    rig.testKind = 'error';
    holdMs = 600;
  }
  syncTestUi();
  setTimeout(() => { rig.testing = false; syncTestUi(); }, holdMs);
}

function renderCapsRow(store, active) {
  const testBtn = btn(rig.testing ? TEST_BUSY : TEST_IDLE, () => testMouth(), 'primary rig-test-btn', {
    disabled: rig.testing || lib.busy, title: 'Play a short speech mouth-flap on the puppet on stage',
  });
  const status = el('div', { class: `rig-test-status apply-status ${rig.testKind}`.trim(), role: 'status', text: rig.testMsg });
  status.hidden = !rig.testMsg;
  const r = row({
    title: 'Rig Hookups', stack: true,
    desc: `What auto-hooked up on ${modelName(active)}`,
    control: el('div', { class: 'rig-caps' }, [
      el('div', { class: 'rig-badges', role: 'list', 'aria-label': 'Rig capabilities' }, capsBadges()),
      el('div', { class: 'rig-test' }, [testBtn, status]),
    ]),
  });
  r.dataset.searchKw = CAPS_KW;
  return r;
}

/* ---- Rig Abilities & Props (shared client in emotes.js) ---- */

/** Fetch abilities once per puppet, alongside the params read (no timers; switching clears the cache). */
function ensureAbilities(store, active) {
  if (abilities.loading || lib.busy || !visible(store)) return;
  if (abilities.forKey === active && abilities.fetchedAt) return;
  loadAbilities({ forKey: active }).then(() => store.rerenderPanel());
}

function renderAbilitiesRow() {
  const list = abilities.list;
  if (!list.length) return null;
  const n = list.length;
  const r = row({
    title: 'Rig Abilities & Props', stack: true,
    desc: abilities.aiEnabled
      ? 'Auto-discovered from this rig · AI decision models can trigger these live'
      : `Auto-discovered from this rig · ${n} ${n === 1 ? 'ability' : 'abilities'} — click to play`,
    control: el('div', { class: 'abil-wrap' }, [abilityGrid(), abilityFoot()]),
    cls: 'abil-row',
  });
  r.dataset.searchKw = ABIL_KW;
  queueMicrotask(syncAbilityUi);
  return r;
}

/* ---- Live overrides ---- */

function postParam(body) {
  return api('/api/models/params', json(body)).then((r) => {
    if (Number.isFinite(Number(r.overriddenCount))) rig.overrides = Number(r.overriddenCount);
    syncResetAll();
    return r;
  });
}

/** Throttled per-param send: at most one request per 40 ms, never two in flight, last value always wins. */
function sendParam(name, body, immediate = false) {
  const s = sends.get(name) || { timer: 0, last: 0, inflight: false, pending: null };
  sends.set(name, s);
  s.pending = { name, ...body };
  const flush = () => {
    s.timer = 0;
    if (s.inflight || !s.pending) return;
    const payload = s.pending;
    s.pending = null;
    s.inflight = true;
    s.last = Date.now();
    postParam(payload).catch((error) => setInspectorMsg(`Couldn’t set ${name}: ${error.message}`, 'error'))
      .finally(() => {
        s.inflight = false;
        if (s.pending && !s.timer) s.timer = setTimeout(flush, Math.max(0, SEND_MS - (Date.now() - s.last)));
      });
  };
  if (s.timer) { if (!immediate) return; clearTimeout(s.timer); s.timer = 0; }
  const wait = immediate ? 0 : SEND_MS - (Date.now() - s.last);
  if (wait <= 0) flush();
  else s.timer = setTimeout(flush, wait);
}

function setInspectorMsg(text, kind = '') {
  document.querySelectorAll('.rig-params-msg').forEach((m) => {
    m.textContent = text;
    m.className = `rig-params-msg apply-status ${kind}`.trim();
    m.hidden = !text;
  });
}

function syncResetAll() {
  document.querySelectorAll('.rig-reset-all').forEach((b) => {
    b.disabled = rig.overrides <= 0;
    b.textContent = rig.overrides > 0 ? `Reset all (${rig.overrides})` : 'Reset all';
  });
}

function markOverridden(p, on, rowEl) {
  const was = p.overridden;
  p.overridden = on;
  if (was !== on) rig.overrides = Math.max(0, rig.overrides + (on ? 1 : -1));
  rowEl?.classList.toggle('overridden', on);
  syncResetAll();
}

function clearParam(p, rowEl) {
  p.current = vec(p.default);
  markOverridden(p, false, rowEl);
  sendParam(p.name, { clear: true }, true);
}

/* ---- Param rows ---- */

function paramMeta(p) {
  const bits = [];
  if (p.category) bits.push(p.category);
  if (p.mappedTo) bits.push(`→ ${p.mappedTo}`);
  return bits.join(' · ');
}

function resetBtn(onClick) {
  return el('button', { type: 'button', class: 'rig-reset', title: 'Reset to rig default', 'aria-label': 'Reset to rig default', text: '↺', onclick: onClick });
}

function axisSlider(p, axis, rowEl, label) {
  const lo = vec(p.min, [0, 0])[axis];
  const hi = vec(p.max, [1, 1])[axis];
  const range = hi - lo || 1;
  const step = Math.max(range / 200, 0.0005);
  const dec = range <= 2 ? 2 : (range <= 20 ? 1 : 0);
  const cur = Math.min(hi, Math.max(lo, vec(p.current, vec(p.default))[axis]));
  const val = el('span', { class: 'ctrl-val', text: cur.toFixed(dec) });
  const inp = el('input', { type: 'range', min: lo, max: hi, step, value: cur, 'aria-label': `${p.name}${label ? ` ${label}` : ''}` });
  const paint = () => inp.style.setProperty('--fill', `${Math.max(0, Math.min(100, ((Number(inp.value) - lo) / range) * 100))}%`);
  paint();
  const push = (immediate) => {
    const v = Number(inp.value);
    const xy = vec(p.current, vec(p.default));
    xy[axis] = v;
    p.current = xy;
    val.textContent = v.toFixed(dec);
    paint();
    markOverridden(p, true, rowEl);
    sendParam(p.name, { x: xy[0], y: xy[1] }, immediate);
  };
  inp.addEventListener('input', () => push(false));
  inp.addEventListener('change', () => push(true));   // release → flush the final value
  return el('div', { class: 's-slider rig-slider' }, [label ? el('span', { class: 'rig-axis', text: label }) : null, inp, val]);
}

function paramRow(p) {
  const rowEl = el('div', { class: `rig-param${p.overridden ? ' overridden' : ''}`, role: 'listitem' });
  const meta = paramMeta(p);
  const text = el('div', { class: 'rig-param-text' }, [
    el('div', { class: 'rig-param-name', title: p.name, text: p.name }),
    meta ? el('div', { class: 'rig-param-meta', text: meta }) : null,
  ]);
  let ctrl;
  if (p.isToggle) {
    const def = vec(p.default)[0] >= 0.5;
    ctrl = [switchEl({
      checked: vec(p.current, vec(p.default))[0] >= 0.5, label: p.name,
      onChange: (v) => {
        p.current = [v ? 1 : 0, 0];
        if (v === def) { clearParam(p, rowEl); return; }
        markOverridden(p, true, rowEl);
        sendParam(p.name, { x: v ? 1 : 0, y: 0 }, true);
      },
    })];
  } else {
    const rebuild = () => { clearParam(p, rowEl); rowEl.replaceWith(paramRow(p)); };
    ctrl = [
      el('div', { class: 'rig-sliders' }, p.isVec2 ? [axisSlider(p, 0, rowEl, 'X'), axisSlider(p, 1, rowEl, 'Y')] : [axisSlider(p, 0, rowEl, '')]),
      resetBtn(rebuild),
    ];
  }
  if (p.isToggle) ctrl.push(resetBtn(() => { clearParam(p, rowEl); rowEl.replaceWith(paramRow(p)); }));
  rowEl.append(text, el('div', { class: 'rig-param-ctrl' }, ctrl));
  return rowEl;
}

function filtered() {
  const tab = currentTab();
  const q = rig.query.trim().toLowerCase();
  return params().filter((p) => inCategory(p, tab) && (!q
    || String(p.name).toLowerCase().includes(q)
    || String(p.category || '').toLowerCase().includes(q)
    || String(p.mappedTo || '').toLowerCase().includes(q)));
}

function fillList(list) {
  const items = filtered();
  const shown = items.slice(0, PARAM_ROW_CAP);
  list.replaceChildren(...(shown.length ? shown.map(paramRow) : [el('div', { class: 'rig-empty', text: rig.query ? 'No parameters match that filter.' : 'Nothing in this category on this rig.' })]));
  if (items.length > shown.length) {
    list.append(el('div', { class: 'rig-empty', text: `Showing ${shown.length} of ${items.length} — type in the filter to narrow down.` }));
  }
}

function renderInspector() {
  const list = el('div', { class: 'rig-param-list', role: 'list' });
  const tabsHost = el('div', { class: 'rig-tabs' });
  const drawTabs = () => tabsHost.replaceChildren(segmented(
    PARAM_TABS.map(([id, t]) => [id, `${t} ${tabCount(id)}`]),
    { active: currentTab(), label: 'Parameter category', onPick: (id) => { rig.tab = id; drawTabs(); fillList(list); } },
  ));
  drawTabs();
  const search = el('input', {
    class: 'text-in rig-search', type: 'search', placeholder: 'Filter parameters…', value: rig.query, 'aria-label': 'Filter parameters',
    oninput: (e) => { rig.query = e.target.value; fillList(list); },
  });
  const resetAll = btn(rig.overrides > 0 ? `Reset all (${rig.overrides})` : 'Reset all', async () => {
    try {
      await postParam({ clearAll: true });
      for (const p of params()) { p.overridden = false; p.current = vec(p.default); }
      rig.overrides = 0;
      syncResetAll();
      fillList(list);
      setInspectorMsg('All rig overrides cleared ✓', 'ok');
    } catch (error) { setInspectorMsg(`Reset failed: ${error.message}`, 'error'); }
  }, 'rig-reset-all', { disabled: rig.overrides <= 0 });
  const msg = el('div', { class: 'rig-params-msg apply-status', role: 'status' });
  msg.hidden = true;
  fillList(list);
  return el('div', { class: 'rig-inspector' }, [
    tabsHost,
    el('div', { class: 'rig-toolbar' }, [search, resetAll]),
    msg,
    list,
  ]);
}

function renderParamsMore(store) {
  const total = params().length;
  const c = rig.data?.capabilities || {};
  let desc;
  let body;
  if (rig.error) {
    desc = 'Rig inspector unavailable';
    body = el('div', { class: 's-row note' }, [
      el('div', { class: 's-desc', text: `Couldn’t read rig parameters: ${rig.error}` }),
      btn('Retry', () => { rig.key = null; rig.error = ''; store.rerenderPanel(); }),
    ]);
  } else if (!rig.data) {
    desc = 'Reading rig parameters…';
    body = el('div', { class: 's-row note' }, [el('div', { class: 's-desc', text: 'Reading rig parameters…' })]);
  } else {
    const extras = num(c.extrasCount, tabCount('extras'));
    desc = `${extras ? `${extras} hidden toggles & FX · ` : ''}${total} parameters — discover, toggle and slide`;
    body = total ? renderInspector() : el('div', { class: 's-row note' }, [el('div', { class: 's-desc', text: 'This rig reports no adjustable parameters.' })]);
  }
  const m = more({ id: 'puppet.params', title: 'Extra Rig Controls', desc }, [body]);
  m.querySelector('.s-more-head').dataset.searchKw = PARAMS_KW;
  return m;
}

/* ───────── 1-click Character Profiles (GET/POST/DELETE /api/profiles) ─────────
 * A profile = puppet + look + Fish voice/model + personality + idle clip + camera. Keys are never stored. */
const PROFILES_KW = 'character profiles character profile profiles profile preset presets save current activate persona voice';
const prof = { busy: '', status: '', kind: '', confirmDelete: null, name: '' };

function profileStatus(store, text, kind = '') {
  prof.status = text;
  prof.kind = kind;
  store.rerenderPanel();
}

async function activateProfile(store, p) {
  if (prof.busy) return;
  prof.busy = p.id;
  profileStatus(store, `Activating ${p.name}…`, 'pending');
  try {
    const r = await api(`/api/profiles/${encodeURIComponent(p.id)}/activate`, { method: 'POST' });
    store.profiles = r.profiles || store.profiles;
    store.activeProfileId = p.id;
    lib.fetchedAt = 0;                         // puppet may have switched: refetch the model list
    invalidateAbilities();
    await store.refreshState();
    prof.busy = '';
    profileStatus(store, `${p.name} is live ✓`, 'ok');
  } catch (error) {
    prof.busy = '';
    profileStatus(store, `Couldn’t activate ${p.name}: ${error.message}`, 'error');
  }
}

async function deleteProfile(store, p) {
  if (prof.busy) return;
  if (prof.confirmDelete !== p.id) {
    prof.confirmDelete = p.id;
    store.rerenderPanel();
    setTimeout(() => { if (prof.confirmDelete === p.id) { prof.confirmDelete = null; store.rerenderPanel(); } }, 3500);
    return;
  }
  prof.confirmDelete = null;
  prof.busy = p.id;
  store.rerenderPanel();
  try {
    const r = await api(`/api/profiles/${encodeURIComponent(p.id)}`, { method: 'DELETE' });
    store.profiles = r.profiles || [];
    if (store.activeProfileId === p.id) store.activeProfileId = '';
    prof.busy = '';
    profileStatus(store, `Deleted ${p.name}`, 'ok');
  } catch (error) {
    prof.busy = '';
    profileStatus(store, `Delete failed: ${error.message}`, 'error');
  }
}

async function saveProfile(store) {
  if (prof.busy) return;
  const name = prof.name.trim();
  if (!name) { profileStatus(store, 'Give the profile a name first', 'error'); return; }
  prof.busy = '__save';
  store.rerenderPanel();
  try {
    const r = await api('/api/profiles', json({ name }));
    store.profiles = r.profiles || store.profiles;
    store.activeProfileId = r.profile?.id || store.activeProfileId;
    prof.name = '';
    prof.busy = '';
    profileStatus(store, `Saved “${r.profile?.name || name}” ✓`, 'ok');
  } catch (error) {
    prof.busy = '';
    profileStatus(store, `Save failed: ${error.message}`, 'error');
  }
}

function profileCard(store, p) {
  const isActive = store.activeProfileId === p.id;
  const confirming = prof.confirmDelete === p.id;
  const bits = [modelName(p.puppetId), p.fishModel, p.fishReferenceId ? 'custom voice' : 'default voice'];
  if (p.idleClip) bits.push('idle clip');
  return el('div', { class: `profile-card${isActive ? ' active' : ''}`, 'data-profile': p.id }, [
    el('div', { class: 'profile-text' }, [
      el('div', { class: 'profile-name', text: p.name }),
      el('div', { class: 'profile-meta', text: bits.filter(Boolean).join(' · ') }),
    ]),
    el('div', { class: 'profile-actions' }, [
      btn(prof.busy === p.id ? 'Activating…' : (isActive ? 'Active ✓' : 'Activate'), () => activateProfile(store, p),
        isActive ? 'primary' : '', { disabled: Boolean(prof.busy), title: `Load ${p.name} (puppet, look, voice, personality, idle)` }),
      btn(confirming ? 'Confirm?' : 'Delete', () => deleteProfile(store, p), confirming ? 'danger' : '',
        { disabled: Boolean(prof.busy), title: confirming ? 'Click again to delete' : `Delete ${p.name}` }),
    ]),
  ]);
}

function renderProfilesRows(store) {
  const list = Array.isArray(store.profiles) ? store.profiles : [];
  const cards = list.length
    ? el('div', { class: 'profile-list', role: 'group', 'aria-label': 'Character profiles' }, list.map((p) => profileCard(store, p)))
    : el('div', { class: 's-desc', text: 'No profiles yet — save the current character below.' });
  const listRow = row({
    title: 'Character Profiles', stack: true,
    desc: '1-click characters: puppet, look, Fish voice, personality, idle clip & camera (API keys never saved)',
    control: cards,
  });
  listRow.dataset.searchKw = PROFILES_KW;
  const nameIn = el('input', {
    class: 'text-in', type: 'text', placeholder: 'Profile name (e.g. Cozy Hikari)', value: prof.name, maxlength: 60,
    'aria-label': 'Profile name', 'data-focus-key': 'profiles.name',
    oninput: (e) => { prof.name = e.target.value; },
    onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); saveProfile(store); } },
  });
  const saveRow = row({
    title: 'Save Current Profile', desc: 'Snapshot what’s on stage now as a reusable character',
    cls: 'wide-ctrl',
    control: [nameIn, btn(prof.busy === '__save' ? 'Saving…' : 'Save Current Profile', () => saveProfile(store), 'primary',
      { disabled: Boolean(prof.busy) })],
  });
  saveRow.dataset.searchKw = PROFILES_KW;
  const statusRow = prof.status
    ? el('div', { class: 's-row note' }, [el('div', { class: `apply-status ${prof.kind}`.trim(), role: 'status', text: prof.status })])
    : null;
  return [listRow, saveRow, statusRow];
}

/** The "Puppet Library" settings group (top of Look & Props). */
export function renderPuppetLibrary(store) {
  const stale = !lib.data || Date.now() - lib.fetchedAt >= CACHE_MS;
  if (stale && !lib.loading && !lib.busy && visible(store)) refreshModels(store);

  const active = activeModelId(store);
  const list = models();
  const mint = Boolean(lib.data?.mintAvailable);
  const activeModel = list.find((m) => m.id === active);

  const fileInput = el('input', {
    type: 'file', accept: mint ? '.inp,.inx,.zip' : '.inp,.inx', hidden: true,
    onchange: (e) => { const f = e.target.files?.[0]; e.target.value = ''; importRig(store, f); },
  });

  let cards;
  if (list.length) {
    cards = el('div', { class: 'puppet-grid', role: 'group', 'aria-label': 'Puppets' }, list.map((m) => puppetCard(store, m, active)));
  } else if (lib.error) {
    cards = el('div', { class: 's-desc', text: `Puppet library unavailable: ${lib.error}` });
  } else {
    cards = el('div', { class: 's-desc', text: 'Loading puppets…' });
  }

  const libRow = row({
    title: 'Puppet', stack: true,
    desc: activeModel ? `On stage: ${activeModel.name || activeModel.id} · ${summary(activeModel)}` : `On stage: ${modelName(active)}`,
    control: cards,
  });
  libRow.dataset.searchKw = SEARCH_KW;

  const importRow = row({
    title: mint ? 'Import Rig (.inp / .inx / Live2D .zip)' : 'Import Rig (.inp / .inx)',
    desc: mint
      ? 'Inochi2D puppets load directly; Live2D .zip models are converted automatically'
      : 'Inochi2D puppets load directly · Live2D .zip conversion needs the mint converter',
    control: [btn(lib.busy && lib.kind === 'pending' && /^Importing/.test(lib.status) ? 'Importing…' : 'Import…',
      () => fileInput.click(), '', { disabled: lib.busy }), fileInput],
  });
  importRow.dataset.searchKw = IMPORT_KW;

  const statusRow = lib.status
    ? el('div', { class: 's-row note' }, [el('div', { class: `apply-status ${lib.kind}`.trim(), role: 'status', text: lib.status })])
    : null;

  ensureRig(store, active);
  ensureAbilities(store, active);
  const capsRow = renderCapsRow(store, active);
  const abilRow = renderAbilitiesRow();
  const paramsMore = renderParamsMore(store);

  const g = group('Puppet Library', [libRow, ...renderProfilesRows(store), capsRow, abilRow, paramsMore, importRow, statusRow],
    { desc: 'Switch avatars, load a saved character or import any rig' });
  g.dataset.searchKw = `${SEARCH_KW} ${IMPORT_KW} ${CAPS_KW} ${ABIL_KW} ${PARAMS_KW} ${PROFILES_KW}`;
  return g;
}

/** Note shown above Wardrobe/Eyes/Hair while a non-Hikari puppet is on stage. */
export function renderPuppetNote(store) {
  const active = activeModelId(store);
  if (active === HIKARI_ID) return null;
  const name = modelName(active);
  return el('div', { class: 'puppet-note', role: 'note' }, [
    el('span', { class: 'puppet-note-dot', 'aria-hidden': 'true' }),
    el('span', { text: `Currently using ${name}. Wardrobe, eye, and hair customizer edits apply to Hikari (Custom) — switch back anytime.` }),
  ]);
}
