/** MEMORY — SLEEP memory settings, live status (facts, pinned, wake log, last sleep, mood, recap, pending dreams),
 *  Sleep & Consolidate, Colab 4B LoRA export, saved memories with pin/unpin + delete.
 *  Network: fetches only when this tab renders (cached 8 s). No timers — nothing runs while Settings is closed. */
import { btn, el, group, more, noteRow, row, selectRow, slider, toggle, withCurrent } from '../ui.js';

const EMBEDDERS = [['vercel', 'Vercel Gateway (Cheap)'], ['local_hash', 'Local Lightweight Fallback']];
const COLAB_MODELS = [['unsloth/gemma-3-4b-it', 'Gemma 3 4B (Unsloth)'], ['unsloth/Qwen3-4B', 'Qwen 3 4B (Unsloth)']];
const CACHE_MS = 8000;
const MAX_FACTS_SHOWN = 60;
const MAX_DREAMS_SHOWN = 5;

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

const numOrNull = (v) => (v === undefined || v === null || v === '' || !Number.isFinite(Number(v)) ? null : Number(v));

/** lastSleep may be an object ({ ts, cards, dreams, turns, … }), a timestamp, an ISO string, or null. */
function sleepTs(v) {
  if (v === undefined || v === null || v === '') return null;
  if (typeof v === 'object') return sleepTs(pick(v, 'ts', 'time', 'at', 'timestamp') ?? null);
  return v;
}

function normDream(d) {
  if (typeof d === 'string') return d.trim() ? { text: d.trim(), offered: false } : null;
  if (!d || typeof d !== 'object') return null;
  const text = String(pick(d, 'text', 'quote', 'content', 'dream') ?? '').trim();
  return text ? { text, offered: Boolean(pick(d, 'offered', 'used', 'shared')) } : null;
}

function normStatus(d) {
  const s = d?.status && typeof d.status === 'object' ? { ...d, ...d.status } : (d || {});
  const factsRaw = pick(s, 'factCount', 'factsCount', 'facts_saved', 'factsSaved', 'facts');
  const lastSleepRaw = pick(s, 'lastSleep', 'last_sleep', 'lastSleepAt', 'lastSleepTime');
  const lastObj = lastSleepRaw && typeof lastSleepRaw === 'object' ? lastSleepRaw : null;
  const dreamsRaw = pick(s, 'pendingDreams', 'pending_dreams');
  const legacyDream = pick(s, 'pendingDream', 'latestDream', 'dream');
  const dreams = (Array.isArray(dreamsRaw) ? dreamsRaw : (legacyDream ? [legacyDream] : [])).map(normDream).filter(Boolean);
  const moodRaw = s.mood && typeof s.mood === 'object' ? s.mood : null;
  const recapRaw = pick(s, 'recap', 'lastRecap', 'sessionRecap');
  return {
    facts: Array.isArray(factsRaw) ? factsRaw.length : numOrNull(factsRaw),
    pinned: numOrNull(pick(s, 'pinnedCount', 'pinned_count')),
    wake: numOrNull(pick(s, 'wakeLogCount', 'wake_log_count', 'wakeBufferTurns')) ?? 0,
    lastSleepTs: sleepTs(lastSleepRaw),
    lastSleepInfo: lastObj ? {
      cards: numOrNull(lastObj.cards), dreams: numOrNull(lastObj.dreams), turns: numOrNull(lastObj.turns),
    } : null,
    dreams,
    recap: typeof recapRaw === 'string' ? recapRaw.trim() : String(recapRaw?.text || recapRaw?.summary || '').trim(),
    mood: moodRaw ? {
      label: String(moodRaw.label || '').trim(),
      valence: numOrNull(moodRaw.valence),
      arousal: numOrNull(moodRaw.arousal),
    } : null,
    sleeping: Boolean(pick(s, 'sleeping', 'isSleeping')),
  };
}

function normFacts(d) {
  const list = Array.isArray(d) ? d : (pick(d, 'facts', 'memories', 'items') || []);
  const facts = (Array.isArray(list) ? list : []).map((f, i) => (typeof f === 'string'
    ? { id: String(i), text: f, kind: '', pinned: false, order: i }
    : {
        id: String(pick(f, 'id', 'uuid', 'key') ?? i),
        text: String(pick(f, 'text', 'fact', 'content', 'memory') ?? ''),
        kind: pick(f, 'category', 'kind', 'type') || '',
        pinned: Boolean(pick(f, 'pinned', 'pin', 'isPinned')),
        order: i,
      }))
    .filter((f) => f.text);
  return sortFacts(facts);
}

/** Pinned first; otherwise keep the server's order (most recently updated first). */
function sortFacts(facts) {
  return facts.sort((a, b) => (Number(b.pinned) - Number(a.pinned)) || (a.order - b.order));
}

function moodText(m) {
  if (!m) return '';
  if (m.label) return m.label[0].toUpperCase() + m.label.slice(1);
  if (m.valence === null) return '';
  return m.valence > 0.25 ? 'Upbeat' : (m.valence < -0.25 ? 'Low' : 'Calm');
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

function absTime(v) {
  if (v === undefined || v === null || v === '') return '';
  let t = typeof v === 'number' ? v : Date.parse(v);
  if (!Number.isFinite(t)) return '';
  if (t < 1e12) t *= 1000;
  return new Date(t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
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
  const stat = (value, label, title = '') => el('div', { class: 'stat', title: title || undefined }, [el('b', { text: String(value) }), el('span', { text: label })]);
  let statusRows;
  if (st) {
    const ls = st.lastSleepInfo;
    const sleepDetail = ls ? [ls.cards !== null ? `${ls.cards} card${ls.cards === 1 ? '' : 's'}` : '', ls.dreams !== null ? `${ls.dreams} dream${ls.dreams === 1 ? '' : 's'}` : '', ls.turns !== null ? `${ls.turns} turns` : ''].filter(Boolean).join(' · ') : '';
    const mood = moodText(st.mood);
    const tiles = [
      stat(st.facts ?? '—', st.pinned ? `Facts saved · ${st.pinned} pinned` : 'Facts saved'),
      stat(st.wake, 'Wake log turns', 'Turns waiting to be consolidated at the next sleep'),
      stat(st.lastSleepTs === null ? 'Never' : relTime(st.lastSleepTs), 'Last sleep', [absTime(st.lastSleepTs), sleepDetail].filter(Boolean).join(' — ')),
      mood ? stat(mood, 'Mood', st.mood.valence !== null ? `valence ${st.mood.valence.toFixed(2)}${st.mood.arousal !== null ? ` · arousal ${st.mood.arousal.toFixed(2)}` : ''}` : '') : null,
    ].filter(Boolean);
    const dreams = st.dreams.slice(0, MAX_DREAMS_SHOWN);
    statusRows = [
      row({
        title: 'Status',
        desc: st.sleeping ? 'Sleeping — consolidating memories…' : (sleepDetail ? `Awake · last sleep made ${sleepDetail}` : 'Awake'),
        stack: true, control: el('div', { class: 'stat-grid' }, tiles),
      }),
      st.recap ? row({ title: 'Last session recap', desc: 'What she remembers about your last session', stack: true,
        control: el('p', { class: 'recap-text', text: st.recap }) }) : null,
      dreams.length ? row({
        title: dreams.length === 1 ? 'Pending dream' : `Pending dreams · ${st.dreams.length}`,
        desc: 'Reflections from sleep she may bring up next time you talk', stack: true,
        control: el('div', { class: 'dream-list' }, [
          ...dreams.map((dr) => el('blockquote', { class: `dream-quote${dr.offered ? ' offered' : ''}` }, [
            `“${dr.text}”`,
            dr.offered ? el('span', { class: 'badge', text: 'Offered' }) : null,
          ])),
          st.dreams.length > dreams.length ? el('div', { class: 's-desc', text: `+${st.dreams.length - dreams.length} more` }) : null,
        ]),
      }) : null,
    ];
  } else {
    statusRows = [noteRow(state.loading || !state.fetchedAt ? 'Loading memory status…' : `Memory status: ${state.statusErr || 'unavailable'}`)];
  }

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
  const pinIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l-1 6 4 4H6l4-4-1-6Z"/><path d="M12 13v8"/></svg>';
  const factRows = state.facts
    ? (facts.length
        ? facts.slice(0, MAX_FACTS_SHOWN).map((f) => row({
            title: f.text, desc: [f.pinned ? 'Pinned — always remembered' : '', f.kind].filter(Boolean).join(' · '),
            cls: `fact-row${f.pinned ? ' pinned' : ''}`,
            control: [
              f.pinned ? el('span', { class: 'badge pin', text: 'Pinned' }) : null,
              el('button', {
                type: 'button', class: `icon-pin${f.pinned ? ' on' : ''}`, html: pinIcon,
                title: f.pinned ? 'Unpin' : 'Pin — keep this memory first and never let it fade',
                'aria-label': `${f.pinned ? 'Unpin' : 'Pin'}: ${f.text}`, 'aria-pressed': String(f.pinned),
                onclick: async (e) => {
                  const b = e.currentTarget;
                  const next = !f.pinned;
                  b.disabled = true;
                  try {
                    await getJson(`/api/memory/facts/${encodeURIComponent(f.id)}/pin`, {
                      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pinned: next }),
                    });
                    f.pinned = next;
                    sortFacts(state.facts);
                    if (state.status && state.status.pinned !== null) state.status.pinned = Math.max(0, state.status.pinned + (next ? 1 : -1));
                    store.rerenderPanel();
                  } catch (err) {
                    b.disabled = false;
                    b.title = `Could not ${next ? 'pin' : 'unpin'}: ${err.message}`;
                    state.actionMsg = b.title;
                    state.actionKind = 'error';
                  }
                },
              }),
              el('button', {
                type: 'button', class: 'icon-x', title: 'Forget this memory', 'aria-label': `Forget: ${f.text}`, text: '×',
                onclick: async (e) => {
                  const b = e.currentTarget;
                  b.disabled = true;
                  try {
                    await getJson(`/api/memory/facts/${encodeURIComponent(f.id)}`, { method: 'DELETE' });
                    state.facts = state.facts.filter((x) => x.id !== f.id);
                    if (state.status && Number.isFinite(state.status.facts)) state.status.facts = Math.max(0, state.status.facts - 1);
                    if (f.pinned && state.status && state.status.pinned !== null) state.status.pinned = Math.max(0, state.status.pinned - 1);
                    store.rerenderPanel();
                  } catch (err) {
                    b.disabled = false;
                    b.title = err.message;
                  }
                },
              }),
            ],
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
      more({
        id: 'memory.facts', title: 'Saved memories',
        desc: state.facts ? `${facts.length} fact${facts.length === 1 ? '' : 's'}${facts.some((f) => f.pinned) ? ` · ${facts.filter((f) => f.pinned).length} pinned` : ''}` : 'Long-term facts',
      }, factRows),
      row({ title: 'Refresh', desc: state.fetchedAt ? `Updated ${relTime(state.fetchedAt)}` : 'Not loaded yet',
        control: btn('Refresh', () => refresh(store, true), '', { disabled: state.loading }) }),
    ], { desc: 'Live from the studio' }),
  ];
}
