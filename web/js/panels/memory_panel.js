/** MEMORY — SLEEP memory settings, live status (facts, wake buffer, last sleep, pending dream),
 *  Sleep & Consolidate, Colab 4B LoRA export, saved memories with delete.
 *  Network: fetches only when this tab renders (cached 8 s). No timers — nothing runs while Settings is closed. */
import { btn, el, group, more, noteRow, row, selectRow, slider, toggle, withCurrent } from '../ui.js';

const EMBEDDERS = [['vercel', 'Vercel Gateway (Cheap)'], ['local_hash', 'Local Lightweight Fallback']];
const COLAB_MODELS = [['unsloth/gemma-3-4b-it', 'Gemma 3 4B (Unsloth)'], ['unsloth/Qwen3-4B', 'Qwen 3 4B (Unsloth)']];
const CACHE_MS = 8000;
const MAX_FACTS_SHOWN = 60;

const state = {
  status: null,        // normalized status or null
  statusErr: '',
  facts: null,         // array or null
  factsErr: '',
  fetchedAt: 0,
  loading: false,
  actionMsg: '',
  actionKind: '',
  busy: '',            // 'sleep' | 'export' | ''
  exportUrl: '',
};

const pick = (o, ...keys) => { for (const k of keys) if (o && o[k] !== undefined && o[k] !== null) return o[k]; return undefined; };

function normStatus(d) {
  const s = d?.status && typeof d.status === 'object' ? { ...d, ...d.status } : (d || {});
  const factsRaw = pick(s, 'factsCount', 'factCount', 'facts_saved', 'factsSaved', 'facts');
  const wakeRaw = pick(s, 'wakeBufferTurns', 'wakeTurns', 'bufferTurns', 'wake_buffer_turns', 'wakeBuffer');
  const dreamRaw = pick(s, 'pendingDream', 'latestDream', 'dream', 'pending_dream');
  return {
    facts: Array.isArray(factsRaw) ? factsRaw.length : (Number.isFinite(Number(factsRaw)) ? Number(factsRaw) : null),
    wake: Array.isArray(wakeRaw) ? wakeRaw.length : (Number.isFinite(Number(wakeRaw)) ? Number(wakeRaw) : null),
    lastSleep: pick(s, 'lastSleep', 'lastSleepAt', 'last_sleep', 'lastSleepTime'),
    dream: typeof dreamRaw === 'string' ? dreamRaw : (dreamRaw?.text || dreamRaw?.quote || dreamRaw?.content || ''),
    sleeping: Boolean(pick(s, 'sleeping', 'isSleeping')),
  };
}

function normFacts(d) {
  const list = Array.isArray(d) ? d : (pick(d, 'facts', 'memories', 'items') || []);
  return (Array.isArray(list) ? list : []).map((f, i) => (typeof f === 'string'
    ? { id: String(i), text: f }
    : { id: String(pick(f, 'id', 'uuid', 'key') ?? i), text: String(pick(f, 'text', 'fact', 'content', 'memory') ?? ''), kind: pick(f, 'category', 'kind', 'type') || '' }))
    .filter((f) => f.text);
}

function relTime(v) {
  if (v === undefined || v === null || v === '') return 'Never';
  let t = typeof v === 'number' ? v : Date.parse(v);
  if (!Number.isFinite(t)) return String(v);
  if (t < 1e12) t *= 1000; // epoch seconds
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 0) return 'Just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

async function getJson(url, opts) {
  const r = await fetch(url, { cache: 'no-store', ...opts });
  if (r.status === 404 || r.status === 405) throw Object.assign(new Error('Needs the updated studio server.'), { missing: true });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d?.ok === false) throw new Error(d?.error || `HTTP ${r.status}`);
  return d;
}

async function refresh(store, force = false) {
  if (state.loading || (!force && Date.now() - state.fetchedAt < CACHE_MS)) return;
  state.loading = true;
  const [st, fa] = await Promise.allSettled([getJson('/api/memory/status'), getJson('/api/memory/facts')]);
  state.loading = false;
  state.fetchedAt = Date.now();
  if (st.status === 'fulfilled') { state.status = normStatus(st.value); state.statusErr = ''; } else { state.statusErr = st.reason?.message || 'Unavailable'; }
  if (fa.status === 'fulfilled') { state.facts = normFacts(fa.value); state.factsErr = ''; } else { state.factsErr = fa.reason?.message || 'Unavailable'; }
  if (store.activeTab === 'memory') store.rerenderPanel();
}

export function renderMemoryPanel(store) {
  const mem = store.config.memory || {};
  const set = (vals) => store.patch('memory', vals);
  refresh(store); // cached; re-renders this tab when fresh data arrives

  const st = state.status;
  const stat = (value, label) => el('div', { class: 'stat' }, [el('b', { text: value }), el('span', { text: label })]);
  const statusRows = st
    ? [
        row({ title: 'Status', desc: st.sleeping ? 'Sleeping — consolidating memories…' : 'Awake', stack: true,
          control: el('div', { class: 'stat-grid' }, [
            stat(st.facts ?? '—', 'Facts saved'),
            stat(st.wake ?? '—', 'Wake buffer turns'),
            stat(relTime(st.lastSleep), 'Last sleep'),
          ]) }),
        st.dream ? row({ title: 'Pending dream', desc: 'What Hikari dreamt about last sleep', stack: true,
          control: el('blockquote', { class: 'dream-quote', text: `“${st.dream}”` }) }) : null,
      ]
    : [noteRow(state.loading || !state.fetchedAt ? 'Loading memory status…' : `Memory status: ${state.statusErr || 'unavailable'}`)];

  const msg = el('div', { class: `apply-status ${state.actionKind}`.trim(), role: 'status',
    text: state.actionMsg || 'Sleep consolidates the wake buffer into long-term facts and a dream.' });

  const runAction = async (kind, url, okText) => {
    if (state.busy) return;
    state.busy = kind;
    state.actionMsg = kind === 'sleep' ? 'Hikari is falling asleep and consolidating…' : 'Building the Colab LoRA training batch…';
    state.actionKind = 'pending';
    store.rerenderPanel();
    try {
      const d = await getJson(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(kind === 'export' ? { baseModel: mem.colabBaseModel || COLAB_MODELS[0][0] } : {}),
      });
      state.exportUrl = kind === 'export' ? (pick(d, 'url', 'downloadUrl', 'href') || '') : state.exportUrl;
      const extra = kind === 'export'
        ? [pick(d, 'count', 'examples', 'rows') !== undefined ? `${pick(d, 'count', 'examples', 'rows')} examples` : '', pick(d, 'path', 'file') || ''].filter(Boolean).join(' · ')
        : (pick(d, 'summary', 'message') || '');
      state.actionMsg = `${okText}${extra ? ` — ${extra}` : ''}`;
      state.actionKind = 'ok';
    } catch (err) {
      state.actionMsg = err.message;
      state.actionKind = 'error';
    } finally {
      state.busy = '';
      await refresh(store, true);
      store.rerenderPanel();
    }
  };

  const facts = state.facts || [];
  const factRows = state.facts
    ? (facts.length
        ? facts.slice(0, MAX_FACTS_SHOWN).map((f) => row({
            title: f.text, desc: f.kind || '', cls: 'fact-row',
            control: el('button', {
              type: 'button', class: 'icon-x', title: 'Forget this memory', 'aria-label': `Forget: ${f.text}`, text: '×',
              onclick: async (e) => {
                const b = e.currentTarget;
                b.disabled = true;
                try {
                  await getJson(`/api/memory/facts/${encodeURIComponent(f.id)}`, { method: 'DELETE' });
                  state.facts = state.facts.filter((x) => x.id !== f.id);
                  if (state.status && Number.isFinite(state.status.facts)) state.status.facts = Math.max(0, state.status.facts - 1);
                  store.rerenderPanel();
                } catch (err) {
                  b.disabled = false;
                  b.title = err.message;
                }
              },
            }),
          }))
        : [noteRow('No saved memories yet — they appear after Hikari sleeps.')])
    : [noteRow(state.factsErr ? `Saved memories: ${state.factsErr}` : 'Loading…')];
  if (facts.length > MAX_FACTS_SHOWN) factRows.push(noteRow(`Showing ${MAX_FACTS_SHOWN} of ${facts.length}.`));

  return [
    group('SLEEP Memory', [
      toggle({ label: 'SLEEP memory enabled', desc: 'Remember facts about you across sessions',
        checked: mem.enabled ?? true, onChange: (v) => set({ enabled: v }) }),
      selectRow({ label: 'Embedding provider', desc: 'How memories are indexed for recall',
        items: withCurrent(EMBEDDERS, mem.embeddingProvider), value: mem.embeddingProvider || 'vercel',
        onChange: (v) => set({ embeddingProvider: v }) }),
      slider({ label: 'Auto-sleep after idle', desc: '0 = manual only', min: 0, max: 120, step: 5,
        value: mem.autoSleepIdleMin ?? 0, def: 0, suffix: ' min', onInput: (v) => set({ autoSleepIdleMin: v }) }),
    ], { desc: 'Wake buffer → sleep → long-term facts & dreams' }),

    group('Memory Status', [
      ...statusRows,
      row({
        title: 'Sleep & consolidate now', desc: 'Turn the wake buffer into facts and a dream',
        control: btn(state.busy === 'sleep' ? 'Sleeping…' : 'Sleep now', () => runAction('sleep', '/api/memory/sleep', 'Sleep complete ✓'),
          'primary', { disabled: Boolean(state.busy) }),
      }),
      selectRow({ label: 'Colab 4B base model', desc: 'Base model the LoRA batch targets',
        items: withCurrent(COLAB_MODELS, mem.colabBaseModel), value: mem.colabBaseModel || COLAB_MODELS[0][0],
        onChange: (v) => set({ colabBaseModel: v }) }),
      row({
        title: 'Export Colab LoRA batch', desc: 'Training data for the local 4B adapter',
        control: [
          state.exportUrl ? el('a', { class: 'btn', href: state.exportUrl, download: '', text: 'Download' }) : null,
          btn(state.busy === 'export' ? 'Exporting…' : 'Export', () => runAction('export', '/api/memory/export-colab', 'Colab batch exported ✓'),
            '', { disabled: Boolean(state.busy) }),
        ],
      }),
      el('div', { class: 's-row note' }, [msg]),
      more({ id: 'memory.facts', title: 'Saved memories', desc: state.facts ? `${facts.length} fact${facts.length === 1 ? '' : 's'}` : 'Long-term facts' }, factRows),
      row({ title: 'Refresh', desc: state.fetchedAt ? `Updated ${relTime(state.fetchedAt)}` : 'Not loaded yet',
        control: btn('Refresh', () => refresh(store, true), '', { disabled: state.loading }) }),
    ], { desc: 'Live from the studio' }),
  ];
}
