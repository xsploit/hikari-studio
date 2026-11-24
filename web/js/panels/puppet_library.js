/** PUPPET LIBRARY — switch between Hikari (Custom), bundled rigs (Hiyori, Mao…) and imported .inp / .inx / Live2D rigs.
 *  Fetches GET /api/models lazily while Look & Props is open (cached 8 s, no background timers). */
import { btn, el, group, row } from '../ui.js';

const CACHE_MS = 8000;
const HIKARI_ID = 'hikari';
const SEARCH_KW = 'puppet puppets model models avatar avatars rig rigs character switch hikari hiyori mao';
const IMPORT_KW = 'import rig inp inx inochi live2d zip mint convert';

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

  const g = group('Puppet Library', [libRow, importRow, statusRow], { desc: 'Switch avatars or import any rig' });
  g.dataset.searchKw = `${SEARCH_KW} ${IMPORT_KW}`;
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
