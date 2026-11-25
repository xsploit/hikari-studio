/** MEMORY & CONTEXT — persistent conversation continuity across restarts, immediate long-term memories
 *  (add / pin / forget), recent saved turns and clear-chat-history.
 *  Network: fetches only when this tab renders (cached 8 s). No timers — nothing runs while Settings is closed. */
import { btn, el, group, more, noteRow, row, selectRow, slider, switchEl, textIn, toggle, withCurrent } from '../ui.js';

const EMBEDDERS = [['vercel', 'Vercel AI Gateway'], ['local', 'Local lightweight (offline)']];
const CATEGORY_LABELS = {
  fact: 'Fact', preference: 'Preference', project: 'Project', inside_joke: 'Inside joke', viewer: 'Viewer', note: 'Note',
};
const SELF_KINDS = [['opinion', 'Opinion'], ['inside_joke', 'Inside joke'], ['trait', 'Trait'], ['promise', 'Promise']];
const SELF_KIND_LABELS = Object.fromEntries(SELF_KINDS);
const CACHE_MS = 8000;
const MAX_FACTS_SHOWN = 60;
const HISTORY_LIMIT = 12;

const state = {
  status: null,        // normalized status or null
  statusErr: '',
  facts: null,         // array or null
  factsErr: '',
  history: null,       // array or null
  historyErr: '',
  fetchedAt: 0,
  loading: false,
  msg: '',             // feedback line under "Add a memory note"
  msgKind: '',         // 'pending' | 'ok' | 'error' | ''
  draft: '',           // unsaved note text (survives re-renders)
  draftPinned: false,
  saving: false,
  clearArmed: false,   // two-step confirm for Clear chat history
  clearing: false,
  historyMsg: '',
  historyMsgKind: '',
  selfNotes: null,     // Hikari's own opinions / inside jokes / promises
  selfDraft: '',
  selfKind: 'opinion',
  selfSaving: false,
};

const pick = (o, ...keys) => { for (const k of keys) if (o && o[k] !== undefined && o[k] !== null) return o[k]; return undefined; };
const numOrNull = (v) => (v === undefined || v === null || v === '' || !Number.isFinite(Number(v)) ? null : Number(v));
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

function normStatus(d) {
  const s = d || {};
  const moodRaw = s.mood && typeof s.mood === 'object' ? s.mood : null;
  const recRaw = s.reconsider && typeof s.reconsider === 'object' ? s.reconsider : null;
  return {
    enabled: s.enabled !== false,
    facts: numOrNull(s.factCount),
    pinned: numOrNull(s.pinnedCount) ?? 0,
    turns: numOrNull(s.conversationTurns),
    lastTs: numOrNull(s.lastInteractionTs),
    mood: moodRaw ? {
      label: String(moodRaw.label || '').trim(),
      valence: numOrNull(moodRaw.valence),
      arousal: numOrNull(moodRaw.arousal),
    } : null,
    reconsider: recRaw ? {
      llmReady: Boolean(recRaw.llmReady),
      reconsidered: numOrNull(recRaw.reconsidered) ?? 0,
      heldBack: numOrNull(recRaw.heldBack) ?? 0,
      implicit: numOrNull(recRaw.implicit) ?? 0,
      askFirst: numOrNull(recRaw.askFirst) ?? 0,
      last: recRaw.last && typeof recRaw.last === 'object' ? recRaw.last : {},
    } : null,
    companion: s.companion && typeof s.companion === 'object' ? {
      daysKnown: numOrNull(s.companion.daysKnown) ?? 0,
      sessionCount: numOrNull(s.companion.sessionCount) ?? 0,
      carryoverActive: Boolean(s.companion.carryoverActive),
    } : null,
  };
}

function normFact(f, i = 0) {
  return {
    id: String(pick(f, 'id') ?? i),
    text: String(pick(f, 'text') ?? '').trim(),
    category: String(pick(f, 'category') || 'note'),
    evidence: String(pick(f, 'evidence') || '').trim(),
    eventDate: String(pick(f, 'eventDate') || '').trim(),
    temporalKind: String(pick(f, 'temporalKind') || 'timeless'),
    temporalLabel: String(pick(f, 'temporalLabel') || '').trim(),
    pinned: Boolean(f?.pinned),
    updatedAt: numOrNull(pick(f, 'updatedAt', 'createdAt')) ?? 0,
  };
}

/** Pinned first, then most recently updated. */
function sortFacts(facts) {
  return facts.sort((a, b) => (Number(b.pinned) - Number(a.pinned)) || (b.updatedAt - a.updatedAt));
}

function normFacts(d) {
  const list = Array.isArray(d?.facts) ? d.facts : [];
  return sortFacts(list.map(normFact).filter((f) => f.text));
}

function normHistory(d) {
  const list = Array.isArray(d?.turns) ? d.turns : [];
  return list
    .map((t) => ({
      role: t?.role === 'assistant' ? 'assistant' : 'user',
      text: String(t?.text || '').trim(),
      channel: t?.channel === 'voice' ? 'voice' : 'chat',
      ts: numOrNull(t?.ts),
    }))
    .filter((t) => t.text);
}

function moodText(m) {
  if (!m) return '';
  if (m.label) return m.label[0].toUpperCase() + m.label.slice(1);
  if (m.valence === null) return '';
  return m.valence > 0.25 ? 'Upbeat' : (m.valence < -0.25 ? 'Low' : 'Calm');
}

function toMs(v) {
  if (v === undefined || v === null || v === '') return null;
  let t = typeof v === 'number' ? v : Date.parse(v);
  if (!Number.isFinite(t)) return null;
  if (t < 1e12) t *= 1000; // epoch seconds
  return t;
}

function relTime(v) {
  const t = toMs(v);
  if (t === null) return 'Never';
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 5) return 'Just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function absTime(v) {
  const t = toMs(v);
  if (t === null) return '';
  return new Date(t).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short', hour12: true });
}

async function getJson(url, opts) {
  const r = await fetch(url, { cache: 'no-store', ...opts });
  if (r.status === 404 || r.status === 405) throw Object.assign(new Error('Needs the updated studio server.'), { missing: true });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d?.ok === false) throw new Error(d?.error || `HTTP ${r.status}`);
  return d;
}

const postJson = (url, body) => getJson(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}),
});

async function refresh(store, force = false) {
  if (state.loading || (!force && Date.now() - state.fetchedAt < CACHE_MS)) return;
  state.loading = true;
  const [st, fa, hi, sf] = await Promise.allSettled([
    getJson('/api/memory/status'),
    getJson('/api/memory/facts'),
    getJson(`/api/chat/history?limit=${HISTORY_LIMIT}`),
    getJson('/api/memory/self'),
  ]);
  state.loading = false;
  state.fetchedAt = Date.now();
  if (st.status === 'fulfilled') { state.status = normStatus(st.value); state.statusErr = ''; } else { state.statusErr = st.reason?.message || 'Unavailable'; }
  if (fa.status === 'fulfilled') { state.facts = normFacts(fa.value); state.factsErr = ''; } else { state.factsErr = fa.reason?.message || 'Unavailable'; }
  if (hi.status === 'fulfilled') { state.history = normHistory(hi.value); state.historyErr = ''; } else { state.historyErr = hi.reason?.message || 'Unavailable'; }
  if (sf.status === 'fulfilled') { state.selfNotes = Array.isArray(sf.value?.notes) ? sf.value.notes : []; }
  if (store.activeTab === 'memory') store.rerenderPanel();
}

function bumpCounts({ facts = 0, pinned = 0 }) {
  const s = state.status;
  if (!s) return;
  if (s.facts !== null) s.facts = Math.max(0, s.facts + facts);
  s.pinned = Math.max(0, s.pinned + pinned);
}

const PIN_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l-1 6 4 4H6l4-4-1-6Z"/><path d="M12 13v8"/></svg>';

export function renderMemoryPanel(store) {
  const mem = store.config.memory || {};
  const set = (vals) => store.patch('memory', vals);
  // Re-renders replace the DOM; keep the note field focused if the user was typing in it.
  const noteWasFocused = Boolean(document.activeElement?.classList?.contains('mem-note-in'));
  refresh(store); // cached; re-renders this tab when fresh data arrives

  /* ── Live status tiles ── */
  const st = state.status;
  const stat = (value, label, title = '') => el('div', { class: 'stat', title: title || undefined }, [
    el('b', { text: String(value) }), el('span', { text: label }),
  ]);
  let statusRows;
  if (st) {
    const mood = moodText(st.mood);
    const moodTip = st.mood && st.mood.valence !== null
      ? `valence ${st.mood.valence.toFixed(2)}${st.mood.arousal !== null ? ` · arousal ${st.mood.arousal.toFixed(2)}` : ''}` : '';
    statusRows = [
      row({
        title: 'Status',
        desc: st.enabled ? 'Memory is on · context reloads automatically after a restart' : 'Memory is off — Hikari won’t remember new things',
        stack: true,
        control: el('div', { class: 'stat-grid' }, [
          stat(st.facts ?? '—', st.pinned > 0 ? `Saved memories · ${st.pinned} pinned` : 'Saved memories'),
          stat(st.turns ?? '—', 'Saved chat turns', 'Reloaded automatically after restart'),
          stat(relTime(st.lastTs), 'Last talked', absTime(st.lastTs)),
          mood ? stat(mood, 'Mood', moodTip) : null,
        ]),
      }),
      st.reconsider ? (() => {
        const rec = st.reconsider;
        const last = rec.last || {};
        const held = Array.isArray(last.heldBack) ? last.heldBack : [];
        const soft = Array.isArray(last.softened) ? last.softened : [];
        const notes = [
          ...held.map((h) => `held back (${h.reason || h.check || 'filtered'})`),
          ...soft.map((s) => `${s.mode === 'ask_first' ? 'ask first' : 'implicit'} (${s.reason || s.check || 'softened'})`),
        ].slice(0, 2);
        const modeTxt = rec.llmReady ? '5 checks · 4 modes (gateway + offline)' : '5 checks · 4 modes (offline heuristic)';
        const detail = notes.length
          ? `Last turn (${last.tone || 'neutral'}): ${notes.join(' · ')}`
          : (rec.reconsidered > 0 ? `${modeTxt} · last turn: ${last.tone || 'neutral'}` : `${modeTxt} · ready for next turn`);
        return row({
          title: 'Reconsider gate',
          desc: detail,
          stack: true,
          control: el('div', { class: 'stat-grid' }, [
            stat(rec.reconsidered, 'Checked'),
            stat(rec.implicit, 'Implicit only', 'Shaped tone silently without stating aloud'),
            stat(rec.askFirst, 'Ask first', 'Sensitive — checks in before bringing up'),
            stat(rec.heldBack, 'Held back', 'Omitted for tone, boundary, or staleness'),
          ]),
        });
      })() : null,
    ].filter(Boolean);
  } else {
    statusRows = [noteRow(state.loading || !state.fetchedAt ? 'Loading memory status…' : `Memory status: ${state.statusErr || 'unavailable'}`)];
  }

  /* ── Add a memory note ── */
  const saveNote = async () => {
    const text = state.draft.trim();
    if (!text || state.saving) return;
    state.saving = true;
    state.msg = 'Saving…';
    state.msgKind = 'pending';
    store.rerenderPanel();
    try {
      const d = await postJson('/api/memory/facts', { text, category: 'note', pinned: state.draftPinned });
      const fact = normFact(d?.fact || { text, category: 'note', pinned: state.draftPinned, updatedAt: Date.now() / 1000 });
      if (fact.text && state.facts) state.facts = sortFacts([fact, ...state.facts.filter((f) => f.id !== fact.id)]);
      bumpCounts({ facts: 1, pinned: fact.pinned ? 1 : 0 });
      state.msg = fact.pinned ? 'Saved & pinned ✓ — Hikari will always keep this in mind' : 'Saved ✓ — Hikari will remember this';
      state.msgKind = 'ok';
      state.draft = '';
      state.draftPinned = false;
    } catch (err) {
      state.msg = `Could not save: ${err.message}`;
      state.msgKind = 'error';
    } finally {
      state.saving = false;
      store.rerenderPanel();
    }
  };
  const noteInput = textIn({
    class: 'text-in mem-note-in', value: state.draft, placeholder: 'e.g. My cat is called Mochi', 'aria-label': 'Memory note', maxlength: 500,
    oninput: (e) => { state.draft = e.target.value; },
    onkeydown: (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); saveNote(); } },
  });
  if (noteWasFocused) {
    requestAnimationFrame(() => {
      if (!noteInput.isConnected) return;
      noteInput.focus({ preventScroll: true });
      noteInput.setSelectionRange(noteInput.value.length, noteInput.value.length);
    });
  }
  const addRow = el('div', { class: 's-row stack mem-add' }, [
    el('div', { class: 's-row-text' }, [
      el('div', { class: 's-title', text: 'Add a memory note' }),
      el('div', { class: 's-desc', text: 'Saved immediately and recalled whenever it’s relevant' }),
    ]),
    el('div', { class: 's-ctrl mem-add-ctrl' }, [
      noteInput,
      el('span', { class: 'mem-pin-toggle', title: 'Pinned memories are always in Hikari’s context' }, [
        switchEl({ checked: state.draftPinned, label: 'Pin this note', onChange: (v) => { state.draftPinned = v; } }),
        el('span', { text: 'Pin' }),
      ]),
      btn(state.saving ? 'Saving…' : 'Save', saveNote, 'primary', { disabled: state.saving }),
    ]),
  ]);
  const addMsg = state.msg
    ? el('div', { class: 's-row note' }, [el('div', { class: `apply-status ${state.msgKind}`.trim(), role: 'status', text: state.msg })])
    : null;

  /* ── Saved memories ── */
  const facts = state.facts || [];
  const pinnedFacts = facts.filter((f) => f.pinned).length;
  const factRows = state.facts
    ? (facts.length
        ? facts.slice(0, MAX_FACTS_SHOWN).map((f) => row({
            title: f.text,
            desc: [f.pinned ? 'Pinned — always in context' : '', f.evidence ? `“${f.evidence}”` : ''].filter(Boolean).join(' · '),
            cls: `fact-row${f.pinned ? ' pinned' : ''}`,
            control: [
              f.temporalLabel ? el('span', {
                class: `badge temporal-pill t-${String(f.temporalKind || 'dated').replace(/[^a-z_-]/gi, '')}`,
                title: f.eventDate ? `Dated memory · ${f.eventDate}` : 'Dated memory', text: f.temporalLabel,
              }) : null,
              el('span', { class: `badge cat cat-${f.category}`, text: CATEGORY_LABELS[f.category] || f.category }),
              el('button', {
                type: 'button', class: `icon-pin${f.pinned ? ' on' : ''}`, html: PIN_ICON,
                title: f.pinned ? 'Unpin' : 'Pin — always keep this in Hikari’s context',
                'aria-label': `${f.pinned ? 'Unpin' : 'Pin'}: ${f.text}`, 'aria-pressed': String(f.pinned),
                onclick: async (e) => {
                  const b = e.currentTarget;
                  const next = !f.pinned;
                  b.disabled = true;
                  try {
                    const d = await postJson(`/api/memory/facts/${encodeURIComponent(f.id)}/pin`, { pinned: next });
                    f.pinned = typeof d?.pinned === 'boolean' ? d.pinned : next;
                    sortFacts(state.facts);
                    bumpCounts({ pinned: f.pinned ? 1 : -1 });
                  } catch (err) {
                    state.msg = `Could not ${next ? 'pin' : 'unpin'}: ${err.message}`;
                    state.msgKind = 'error';
                  }
                  store.rerenderPanel();
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
                    bumpCounts({ facts: -1, pinned: f.pinned ? -1 : 0 });
                    state.msg = 'Forgotten ✓';
                    state.msgKind = 'ok';
                  } catch (err) {
                    state.msg = `Could not forget: ${err.message}`;
                    state.msgKind = 'error';
                  }
                  store.rerenderPanel();
                },
              }),
            ],
          }))
        : [noteRow('No saved memories yet — add a note above, or just chat and Hikari will remember the important bits.')])
    : [noteRow(state.factsErr ? `Saved memories: ${state.factsErr}` : 'Loading…')];
  if (facts.length > MAX_FACTS_SHOWN) factRows.push(noteRow(`Showing ${MAX_FACTS_SHOWN} of ${facts.length}.`));

  /* ── Recent saved turns + clear chat history ── */
  const history = state.history || [];
  const historyRows = state.history
    ? (history.length
        ? [el('div', { class: 's-row stack' }, [
            el('div', { class: 'turn-list' }, history.map((t) => el('div', { class: `turn ${t.role}` }, [
              el('div', { class: 'turn-meta' }, [
                el('b', { text: t.role === 'assistant' ? 'Hikari' : 'You' }),
                el('span', { class: `badge ch-${t.channel}`, text: t.channel }),
                el('span', { class: 'turn-time', text: relTime(t.ts), title: absTime(t.ts) || undefined }),
              ]),
              el('div', { class: 'turn-text', text: t.text }),
            ]))),
          ])]
        : [noteRow('No saved turns yet — chat or call and they’ll show up here.')])
    : [noteRow(state.historyErr ? `Recent turns: ${state.historyErr}` : 'Loading…')];

  const clearChat = async () => {
    if (state.clearing) return;
    if (!state.clearArmed) { state.clearArmed = true; store.rerenderPanel(); return; }
    state.clearArmed = false;
    state.clearing = true;
    store.rerenderPanel();
    try {
      const d = await postJson('/api/chat/clear');
      const n = numOrNull(d?.cleared);
      state.history = [];
      if (state.status) state.status.turns = 0;
      state.historyMsg = `Chat history cleared${n !== null ? ` (${plural(n, 'turn')})` : ''} ✓ — saved memories kept`;
      state.historyMsgKind = 'ok';
    } catch (err) {
      state.historyMsg = `Could not clear: ${err.message}`;
      state.historyMsgKind = 'error';
    } finally {
      state.clearing = false;
      store.rerenderPanel();
    }
  };
  historyRows.push(row({
    title: 'Clear chat history',
    desc: state.clearArmed ? 'Click again to confirm — saved long-term memories stay intact' : 'Forgets recent turns only — saved long-term memories stay intact',
    control: [
      state.clearArmed ? btn('Cancel', () => { state.clearArmed = false; store.rerenderPanel(); }) : null,
      btn(state.clearing ? 'Clearing…' : (state.clearArmed ? 'Confirm clear' : 'Clear history'), clearChat, 'danger', { disabled: state.clearing }),
    ],
  }));
  if (state.historyMsg) {
    historyRows.push(el('div', { class: 's-row note' }, [
      el('div', { class: `apply-status ${state.historyMsgKind}`.trim(), role: 'status', text: state.historyMsg }),
    ]));
  }

  const selfNotes = state.selfNotes || [];
  const saveSelfNote = async () => {
    const text = state.selfDraft.trim();
    if (!text || state.selfSaving) return;
    state.selfSaving = true;
    store.rerenderPanel();
    try {
      const d = await postJson('/api/memory/self', { text, kind: state.selfKind });
      if (d?.note) state.selfNotes = [d.note, ...selfNotes.filter((n) => n.id !== d.note.id)];
      state.selfDraft = '';
    } catch { /* ignore */ } finally {
      state.selfSaving = false;
      store.rerenderPanel();
    }
  };
  const selfInput = textIn({
    class: 'text-in', value: state.selfDraft, placeholder: 'e.g. Still refuses to admit pineapple belongs on pizza',
    'aria-label': 'Hikari continuity note', maxlength: 220,
    oninput: (e) => { state.selfDraft = e.target.value; },
    onkeydown: (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); saveSelfNote(); } },
  });
  const selfKindSel = el('select', {
    class: 'sel', 'aria-label': 'Note kind',
    onchange: (e) => { state.selfKind = e.target.value; },
  }, SELF_KINDS.map(([v, lbl]) => el('option', { value: v, text: lbl, selected: v === state.selfKind })));
  const compMeta = st?.companion;
  const selfDesc = [
    compMeta?.daysKnown ? `Day ${compMeta.daysKnown} together` : '',
    compMeta?.carryoverActive ? 'Emotional carryover active (implicit)' : '',
    `${selfNotes.length} ${selfNotes.length === 1 ? 'note' : 'notes'}`,
  ].filter(Boolean).join(' · ');
  const selfRows = [
    el('div', { class: 's-row stack mem-add' }, [
      el('div', { class: 's-row-text' }, [
        el('div', { class: 's-title', text: 'Add her opinion, joke or promise' }),
        el('div', { class: 's-desc', text: 'Hikari’s own persistent personality continuity across days' }),
      ]),
      el('div', { class: 's-ctrl mem-add-ctrl' }, [
        selfInput, selfKindSel,
        btn(state.selfSaving ? 'Saving…' : 'Add', saveSelfNote, 'primary', { disabled: state.selfSaving }),
      ]),
    ]),
    ...(selfNotes.length ? selfNotes.map((n) => row({
      title: n.text,
      control: [
        el('span', { class: 'badge cat', text: SELF_KIND_LABELS[n.kind] || n.kind || 'Opinion' }),
        el('button', {
          type: 'button', class: 'icon-x', title: 'Remove', 'aria-label': `Remove: ${n.text}`, text: '×',
          onclick: async () => {
            try {
              await getJson(`/api/memory/self/${encodeURIComponent(n.id)}`, { method: 'DELETE' });
              state.selfNotes = (state.selfNotes || []).filter((x) => x.id !== n.id);
              store.rerenderPanel();
            } catch { /* ignore */ }
          },
        }),
      ],
    })) : [noteRow('No character continuity notes yet — add an opinion, inside joke or promise above.')]),
  ];

  return [
    group('Memory & Context', [
      toggle({ label: 'Memory enabled', desc: 'Remember you and your conversations across sessions',
        checked: mem.enabled ?? true, onChange: (v) => set({ enabled: v }) }),
      toggle({ label: 'Persist conversation across restarts', desc: 'Save chat & voice turns to disk so Hikari picks up right where you left off',
        checked: mem.persistHistory ?? true, onChange: (v) => set({ persistHistory: v }) }),
      toggle({ label: 'Auto-remember facts', desc: 'Extract preferences, projects, and details immediately — no sleep cycle needed',
        checked: mem.autoExtract ?? true, onChange: (v) => set({ autoExtract: v }) }),
      slider({ label: 'Context window', desc: 'Recent turns reloaded when Hikari starts', min: 4, max: 60, step: 2,
        value: mem.historyTurns ?? 24, def: 24, suffix: ' turns', onInput: (v) => set({ historyTurns: v }) }),
      more({ id: 'memory.advanced', title: 'Advanced', desc: 'Embedding provider and recall depth' }, [
        selectRow({ label: 'Embedding provider', desc: 'How memories are indexed for recall',
          items: withCurrent(EMBEDDERS, mem.embeddingProvider), value: mem.embeddingProvider || 'vercel',
          onChange: (v) => set({ embeddingProvider: v }) }),
        slider({ label: 'Recall top-K facts', desc: 'Most relevant memories added to each reply', min: 2, max: 15, step: 1,
          value: mem.recallTopK ?? 6, def: 6, onInput: (v) => set({ recallTopK: v }) }),
      ]),
    ], { desc: 'Persistent chat continuity & long-term memory across restarts' }),

    group('Live Context & Status', [
      ...statusRows,
      addRow,
      addMsg,
      more({
        id: 'memory.facts', title: 'Saved memories', open: facts.length > 0,
        desc: state.facts ? `${facts.length} ${facts.length === 1 ? 'memory' : 'memories'}${pinnedFacts ? ` · ${pinnedFacts} pinned` : ''}` : 'Long-term notes & facts',
      }, factRows),
      more({
        id: 'memory.self', title: 'Hikari’s continuity (opinions & inside jokes)', open: selfNotes.length > 0,
        desc: selfDesc || 'Her own evolving tastes, running gags & emotional carryover',
      }, selfRows),
      more({
        id: 'memory.history', title: 'Recent saved turns',
        desc: state.history ? `Last ${plural(history.length, 'turn')} · chat & voice` : 'What gets reloaded after a restart',
      }, historyRows),
      row({ title: 'Refresh', desc: state.fetchedAt ? `Updated ${relTime(state.fetchedAt)}` : 'Not loaded yet',
        control: btn(state.loading ? 'Refreshing…' : 'Refresh', () => refresh(store, true), '', { disabled: state.loading }) }),
    ], { desc: 'Saved automatically on every turn' }),
  ];
}
