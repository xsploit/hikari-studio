/** VOICE & BRAIN — brain/voice mode (Fish duplex · Vercel AI Gateway + Fish Speech), Fish TTS transport,
 *  live gateway model + provider pickers, live Fish voice picker + voice designer, live call, speech & ASR, custom LLM, decision engine, body language, personality. */
import { btn, choiceRows, el, group, more, row, selectEl, segmented, selectRow, slider, textIn, toggle, withCurrent } from '../ui.js';
import { jsonFetch, renderVoiceDesigner, softRerender, tagged } from './voice_design.js';

const BRAIN_MODES = [
  ['fish_duplex', 'Fish Real-Time Duplex', 'Fish LLM + live full-duplex voice — lowest latency'],
  ['vercel_fish', 'Vercel AI Gateway + Fish Speech', 'Any gateway model with provider pinning, voiced by Fish'],
];

const FISH_TRANSPORTS = [
  ['sse', 'SSE Streaming (Fastest)'],
  ['ws', 'WebSocket Stream'],
  ['http', 'HTTP Standard'],
];

const PROVIDERS = [
  ['offline', 'Offline persona'],
  ['openai', 'OpenAI'],
  ['lmstudio', 'LM Studio (local)'],
  ['ollama', 'Ollama (local)'],
  ['openrouter', 'OpenRouter'],
  ['custom', 'Custom URL'],
];

const DEFAULT_VISION_MODEL = 'openai/gpt-5-mini';
// Keep the picker and custom field aligned with modules.vercel_gateway.is_vision_model_id.
function isVisionModelId(id) {
  if (typeof id !== 'string' || id.length > 160 || !/^[A-Za-z0-9][A-Za-z0-9._+-]*(?:\/[A-Za-z0-9._:@+-]+)+$/.test(id)) return false;
  const parts = id.toLowerCase().split('/');
  const leaf = parts.at(-1).replaceAll('_', '.');
  return parts[0] !== 'openai' || !(/^o1(?:$|[.-]|pro)/.test(leaf) || /(^|[.-])pro([.-]|$)/.test(leaf));
}

const VISION_MODES = [
  ['describe', 'Vision Eye → Chat Model (Recommended for DeepSeek / text models)'],
  ['direct', 'Direct Multimodal Reply (Vision model answers directly)'],
];

const FISH_MODELS = [
  ['drama-3', 'Drama 3 · Hikari voice'],
  ['s2.1-pro-free', 'S2.1-Pro Free'],
  ['s2.1-pro', 'S2.1-Pro'],
];

// Decider modes, grouped. Vercel AI Gateway (one AI_GATEWAY_API_KEY) is the primary path.
const DECISION_MODE_GROUPS = [
  ['Recommended', [
    ['auto', 'Auto (Vercel AI Gateway)'],
    ['gateway', 'Vercel AI Gateway (Evaluation Model)'],
  ]],
  ['Offline & fallback', [
    ['heuristic', 'Fast Local Heuristic (Offline)'],
    ['llm', 'Chat LLM Fallback'],
  ]],
  ['Direct providers (optional)', [
    ['clef', 'Cloudflare Clef (Direct)'],
    ['pplx', 'Perplexity (Direct / Gateway)'],
    ['luna', 'OpenAI Luna (Direct / Gateway)'],
  ]],
];
const DECISION_MODE_DESC = {
  auto: 'Vercel AI Gateway first, then direct providers, then local heuristics',
  gateway: 'Always decide with the Gateway evaluation model below',
  heuristic: 'Instant offline keyword rules — no network, no keys',
  llm: 'Reuse the chat model to pick emotions and gestures',
  clef: 'Cloudflare Workers AI directly — needs the Cloudflare overrides below',
  pplx: 'Perplexity Decider — direct key if set, otherwise via the Gateway',
  luna: 'OpenAI Decisions — direct key if set, otherwise via the Gateway',
  jev: 'Typesafe Jev via the Vercel AI Gateway',
};
const GATEWAY_MODES = new Set(['auto', 'gateway', 'pplx', 'luna', 'jev']);
// Keys whose change can flip the gateway status / active decider.
const GATEWAY_KEY_FIELDS = new Set(['vercelApiKey', 'cloudflareAccountId', 'cloudflareApiToken', 'perplexityApiKey']);

const DEFAULT_EVAL_MODEL = 'perplexity/pplx-decider-flash-v1.3';
const FALLBACK_EVAL_MODELS = [
  { id: 'perplexity/pplx-decider-flash-v1.3', provider: 'perplexity', label: 'Perplexity Decider Flash v1.3' },
  { id: 'openai/gpt-6-luna-decisions', provider: 'openai', label: 'OpenAI GPT-6 Luna Decisions' },
  { id: 'typesafe-ai/jev', provider: 'typesafe-ai', label: 'Typesafe Jev' },
  { id: 'google/gemini-decision-1', provider: 'google', label: 'Google Gemini Decision 1' },
  { id: 'microsoft/decision-1.1', provider: 'microsoft', label: 'Microsoft Decision 1.1' },
  { id: 'liquid/d1.2', provider: 'liquid', label: 'Liquid D1.2' },
];

// GET /api/deciders/models — fetched only while the Voice & Brain tab renders (cached 30 s, no timers).
const deciders = { models: null, gatewayConfigured: null, activeModel: '', activeMode: '', fetchedAt: 0, loading: false, error: '', queued: false };
async function loadDeciders(store, force = false) {
  if (deciders.loading) { if (force) deciders.queued = true; return; } // re-run once the in-flight fetch finishes
  if (!force && Date.now() - deciders.fetchedAt < 30000) return;
  deciders.loading = true;
  try {
    const r = await fetch('/api/deciders/models', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status === 404 ? 'needs the updated studio server' : `HTTP ${r.status}`);
    const d = await r.json();
    const list = Array.isArray(d?.models) ? d.models.filter((m) => m && m.id) : [];
    deciders.models = list.length ? list : null;
    deciders.gatewayConfigured = typeof d?.gatewayConfigured === 'boolean' ? d.gatewayConfigured : null;
    deciders.activeModel = String(d?.activeModel || '');
    deciders.activeMode = String(d?.activeMode || '');
    deciders.error = '';
  } catch (err) {
    deciders.error = err.message || 'unavailable';
  } finally {
    deciders.loading = false;
    deciders.fetchedAt = Date.now();
    if (deciders.queued) { deciders.queued = false; loadDeciders(store, true); return; }
    softRerender(store);
  }
}

/** Re-check the gateway right after a key changes (one-shot, after the patch has reached the server). */
function refreshDecidersSoon(store) {
  setTimeout(() => loadDeciders(store, true), 400);
}

/* ───────── Live Vercel AI Gateway models: GET /api/vercel/models (cached 60 s, no timers) ───────── */
const gw = { data: null, fetchedAt: 0, loading: false, error: '', queued: false, filter: '', vfilter: '' };
async function loadVercelModels(store, force = false) {
  if (gw.loading) { if (force) gw.queued = true; return; }
  if (!force && Date.now() - gw.fetchedAt < 60000) return;
  gw.loading = true;
  try {
    const d = await jsonFetch(`/api/vercel/models?force=${force ? 1 : 0}`);
    gw.data = {
      configured: typeof d.configured === 'boolean' ? d.configured : null,
      activeModel: String(d.activeModel || ''),
      activeProvider: String(d.activeProvider || ''),
      models: (Array.isArray(d.models) ? d.models : []).filter((m) => m && m.id),
      providers: (Array.isArray(d.providers) ? d.providers : []).filter(Boolean).map(String),
    };
    gw.error = '';
  } catch (err) {
    gw.error = err.message || 'unavailable';
  } finally {
    gw.loading = false;
    gw.fetchedAt = Date.now();
    if (gw.queued) { gw.queued = false; loadVercelModels(store, true); return; }
    softRerender(store);
  }
}

/* ───────── Live Fish voices: GET /api/fish/voices (cached 30 s per scope/query, no timers) ───────── */
const fv = { scope: 'self', q: '', cache: new Map(), loading: '', queued: false, activeVoiceId: '' };
const fvKey = () => (fv.scope === 'self' ? 'self' : `community|${fv.q.trim().toLowerCase()}`);
async function loadFishVoices(store, force = false) {
  const key = fvKey();
  const hit = fv.cache.get(key);
  if (!force && hit && Date.now() - hit.at < 30000) return;
  if (fv.loading) { if (force || fv.loading !== key) fv.queued = true; return; }
  fv.loading = key;
  const scope = fv.scope;
  const qs = new URLSearchParams({ scope, q: scope === 'self' ? '' : fv.q.trim(), page: '1', limit: '30', force: force ? '1' : '0' });
  try {
    const d = await jsonFetch(`/api/fish/voices?${qs}`);
    fv.cache.set(key, {
      at: Date.now(), error: '',
      configured: typeof d.configured === 'boolean' ? d.configured : null,
      voices: (Array.isArray(d.voices) ? d.voices : []).filter((v) => v && v.id),
    });
    if (d.activeVoiceId) fv.activeVoiceId = String(d.activeVoiceId);
  } catch (err) {
    fv.cache.set(key, { at: Date.now(), error: err.message || 'unavailable', configured: null, voices: hit?.voices || [] });
  } finally {
    fv.loading = '';
    // A scope/search switch (or forced refresh) arrived mid-flight: fetch whatever is current now.
    if (fv.queued) { fv.queued = false; loadFishVoices(store, fvKey() === key); return; }
    softRerender(store);
  }
}
/** Find a voice by id in any cached list (labels a preserved/custom id when possible). */
function knownVoice(id) {
  for (const e of fv.cache.values()) { const v = e.voices.find((x) => x.id === id); if (v) return v; }
  return null;
}
const voiceLabel = (v) => {
  const t = String(v.title || v.id);
  const vis = String(v.visibility || '').toLowerCase();
  return (v.mine || fv.scope === 'self') && vis && vis !== 'public' ? `${t} (${vis})` : t;
};

// Inline sample player for the selected voice (one at a time).
const preview = { audio: null, id: '' };
function togglePreview(store, voice) {
  const url = voice?.samples?.[0];
  if (!url) return;
  if (preview.audio && preview.id === voice.id && !preview.audio.paused) {
    preview.audio.pause();
  } else {
    preview.audio?.pause();
    preview.audio = new Audio(url);
    preview.id = voice.id;
    preview.audio.addEventListener('ended', () => softRerender(store));
    preview.audio.addEventListener('pause', () => softRerender(store));
    preview.audio.play().catch(() => softRerender(store));
  }
  softRerender(store);
}

const orderList = (v) => (Array.isArray(v) ? v : String(v || '').split(',')).map((x) => String(x).trim()).filter(Boolean);

const TTS = [
  ['fish', 'Fish Drama 3'],
  ['openai', 'OpenAI-compatible'],
  ['windows', 'Windows voice'],
];

export function renderAiPanel(store) {
  const ai = store.config.ai;
  const ls = store.config.lipsync || {};
  const rig = store.config.rig || {};
  const fs = store.fishStatus || {};
  const fc = store.fishCall;
  const mode = ai.fishAgentMode || 'drama-3';
  const modeLabel = mode === 'drama-3' ? 'Drama 3' : 'Live 3';
  const set = (vals) => store.patch('ai', vals);

  /* Saved credential overrides: the server masks them ("********") and ignores "" / the echoed mask,
     so inputs never show the value; typing replaces it; Clear sends null (explicit clear). */
  // A new/cleared key changes what the live lists can see: re-fetch once the patch has reached the server.
  const keyChanged = (field) => {
    if (field === 'vercelApiKey') setTimeout(() => loadVercelModels(store, true), 400);
    if (field === 'fishApiKey') { fv.cache.clear(); setTimeout(() => loadFishVoices(store, true), 400); }
  };
  const secretControl = (field, label, { type = 'password', blankHint = 'Leave blank to use .env.local' } = {}) => {
    const isSet = Boolean(ai[field]);
    const input = textIn({
      type, autocomplete: 'off', 'aria-label': label,
      placeholder: isSet ? 'Saved — type to replace' : blankHint,
      onchange: (e) => {
        const v = e.target.value.trim();
        if (!v) return;
        set({ [field]: v });
        e.target.value = '';
        if (GATEWAY_KEY_FIELDS.has(field)) refreshDecidersSoon(store);
        keyChanged(field);
        store.rerenderPanel();
      },
    });
    const clear = () => {
      set({ [field]: null });
      if (GATEWAY_KEY_FIELDS.has(field)) refreshDecidersSoon(store);
      keyChanged(field);
      store.rerenderPanel();
    };
    return isSet ? [input, btn('Clear', clear)] : input;
  };

  const callState = fc?.connecting ? 'connecting' : (fc?.connected ? 'live' : 'idle');
  const callRow = el('div', { class: `s-row call-row ${callState}` }, [
    el('span', { class: 'call-dot', 'aria-hidden': 'true' }),
    el('div', { class: 's-row-text' }, [
      el('div', { class: 's-title', text: fc?.connected ? `${fc.mode === 'drama-3' ? 'Drama 3' : 'Live 3'} call live` : (fc?.connecting ? 'Connecting…' : 'Live voice call') }),
      el('div', { class: 's-desc', text: fc?.connected ? `Agent ${fc.agentMode} · mic ${fc.micMuted ? 'muted' : 'on'}` : 'Full-duplex WebRTC voice, instant STT and 60 FPS lip-sync' }),
    ]),
    el('div', { class: 's-ctrl' }, [
      fc?.connected ? btn(fc.micMuted ? 'Unmute' : 'Mute', () => fc.toggleMic()) : null,
      fc?.connected ? btn('Interrupt', () => fc.interrupt()) : null,
      btn(fc?.connecting ? 'Connecting…' : (fc?.connected ? 'End call' : `Connect ${modeLabel}`),
        () => store.toggleFishCall(), fc?.connected ? 'danger' : 'primary', { disabled: Boolean(fc?.connecting) }),
    ]),
  ]);

  /* Fish voice dropdown — My Voices / Community, inline ▶ sample preview, current id always preserved */
  const fishVoiceRow = () => {
    loadFishVoices(store);
    const entry = fv.cache.get(fvKey());
    const loading = fv.loading === fvKey();
    const voices = entry?.voices || [];
    const cur = String(ai.fishReferenceId || '');
    const curVoice = voices.find((v) => v.id === cur) || knownVoice(cur);
    const opts = [];
    if (cur && !voices.some((v) => v.id === cur)) {
      opts.push(el('option', { value: cur, selected: true, text: `Current · ${curVoice ? voiceLabel(curVoice) : cur}` }));
    } else if (!cur) {
      opts.push(el('option', { value: '', selected: true, text: 'Hikari default voice' }));
    }
    voices.forEach((v) => opts.push(el('option', { value: v.id, text: voiceLabel(v), selected: v.id === cur, title: v.description || v.id })));
    if (!voices.length) opts.push(el('option', { value: '__none', disabled: true, text: loading ? 'Loading voices…' : 'No voices found' }));
    const select = el('select', {
      class: 'pill-select wide', 'aria-label': 'Fish voice',
      onchange: (e) => { if (e.target.value !== '__none') { set({ fishReferenceId: e.target.value }); softRerender(store); } },
    }, opts);
    const sampleUrl = curVoice?.samples?.[0];
    const playing = Boolean(preview.audio && preview.id === curVoice?.id && !preview.audio.paused);
    const status = loading ? (fv.scope === 'self' ? 'Loading your Fish voices…' : 'Searching community voices…')
      : entry?.configured === false ? 'Fish API key not configured — add one below'
        : entry?.error ? `Voice list unavailable (${entry.error}) — paste a voice ID below`
          : entry ? `${voices.length} ${fv.scope === 'self' ? 'voices on your account · design a new one in Voice Designer' : 'community voices'}`
            : 'Checking Fish…';
    const pickScope = (id) => {
      if (id === fv.scope) return;
      fv.scope = id;
      loadFishVoices(store);
      softRerender(store);
    };
    const runSearch = () => { loadFishVoices(store); softRerender(store); };
    const control = el('div', { class: 'fv-picker' }, [
      el('div', { class: 'fv-bar' }, [
        segmented([['self', 'My Voices'], ['community', 'Community']], { active: fv.scope, onPick: pickScope, label: 'Voice source' }),
        btn(loading ? 'Loading…' : 'Refresh', () => loadFishVoices(store, true), '', { disabled: Boolean(loading), title: 'Reload voices from Fish' }),
      ]),
      fv.scope === 'community' ? el('div', { class: 'fv-bar' }, [
        textIn({
          type: 'search', placeholder: 'Search community voices…', value: fv.q, 'aria-label': 'Search community voices',
          'data-focus-key': 'fish.voiceSearch',
          oninput: (e) => { fv.q = e.target.value; },
          onkeydown: (e) => { if (e.key === 'Enter') runSearch(); },
        }),
        btn('Search', runSearch),
      ]) : null,
      el('div', { class: 'fv-bar' }, [
        select,
        sampleUrl ? btn(playing ? '■' : '▶', () => togglePreview(store, curVoice), 'icon-btn',
          { title: playing ? 'Stop sample' : 'Play voice sample', 'aria-label': playing ? 'Stop sample' : 'Play voice sample' }) : null,
      ]),
      el('div', { class: `fv-status${entry?.error || entry?.configured === false ? ' warn' : ''}`, text: status }),
    ]);
    return tagged(row({ title: 'Fish voice', desc: 'Pick one of your voices or browse the community library', stack: true, control }),
      'fish voices voice picker my voices community reference voice');
  };

  /* Subtle Live Call voice sync hint: chat TTS uses the voice at once, Fish calls on the next connect. */
  const voiceSyncHint = () => {
    const cur = String(ai.fishReferenceId || '');
    if (!cur) return null;
    let text = 'Applies immediately to chat TTS and on the next Live Call';
    if (fc?.connected) {
      text = fs.voiceStatus?.sessionVoiceId === cur
        ? 'Chat TTS and the current Live Call use this voice'
        : 'Chat TTS uses this voice now. The next Live Call will request it.';
    }
    return el('div', { class: 's-row note voice-sync-hint', 'data-voice-sync': '' }, [
      el('div', { class: 's-desc' }, [el('span', { class: 'sync-dot', 'aria-hidden': 'true' }), text]),
    ]);
  };

  /* A designed voice was saved: adopt the server's ai config, select it, refresh the voice list. */
  const onVoiceSaved = (res) => {
    const v = res?.voice || {};
    if (res?.ai && typeof res.ai === 'object') Object.assign(store.config.ai, res.ai);
    if (res?.voiceSync && typeof res.voiceSync === 'object') store.fishStatus = { ...(store.fishStatus || {}), voiceSync: res.voiceSync };
    if (v.id) {
      if (res.applied) store.config.ai.fishReferenceId = res.ai?.fishReferenceId || v.id;
      else store.patch('ai', { fishReferenceId: v.id });
      const self = fv.cache.get('self');
      if (self && !self.voices.some((x) => x.id === v.id)) self.voices.unshift({ ...v, mine: true });
    }
    fv.scope = 'self';
    loadFishVoices(store, true);
  };

  const fishRows = ai.ttsProvider === 'fish' ? [
    row({
      title: 'Fish API key',
      desc: fs.keyReady ? 'Loaded server-side · Drama 3 TTS + transcribe-1 ASR ready' : 'Not found in env or .env.local — paste an override',
      control: el('span', { class: `badge ${fs.keyReady ? 'ok' : 'warn'}`, text: fs.keyReady ? 'Ready' : 'Missing' }),
    }),
    selectRow({ label: 'Fish model', desc: 'Voice model for chat replies', items: FISH_MODELS, value: ai.fishModel, onChange: (v) => set({ fishModel: v }) }),
    tagged(toggle({
      label: 'Fish vocal expression tags',
      desc: 'Keep Drama-3 performance directions and S2 tags like [giggles], [whisper] in Fish speech',
      checked: ai.fishVocalTags ?? true,
      onChange: (v) => set({ fishVocalTags: v }),
    }), 'vocal tags drama 3 s2 giggles whisper'),
    fishVoiceRow(),
    voiceSyncHint(),
    tagged(row({ title: 'Custom voice ID', desc: 'Paste any Fish voice / reference ID — the dropdown above follows it', cls: 'wide-ctrl',
      control: textIn({ placeholder: 'Paste a Fish voice ID', value: ai.fishReferenceId || '', 'aria-label': 'Custom voice ID',
        'data-focus-key': 'fish.customVoice', spellcheck: 'false',
        onchange: (e) => { set({ fishReferenceId: e.target.value.trim() }); softRerender(store); } }) }),
    'reference voice id fish voices'),
    slider({ label: 'Speech speed', min: 0.6, max: 1.6, step: 0.05, value: ai.fishSpeed ?? 1.0, def: 1.0, suffix: '×', onInput: (v) => set({ fishSpeed: v }) }),
    row({ title: 'API key override', desc: 'Optional — overrides the server key', control: secretControl('fishApiKey', 'Fish API key override'), cls: 'wide-ctrl' }),
  ] : [];

  /* Decision engine — Vercel AI Gateway first */
  loadDeciders(store);
  const decMode = String(ai.decisionMode || 'auto').toLowerCase();
  const knownMode = DECISION_MODE_GROUPS.some(([, items]) => items.some(([id]) => id === decMode));
  const modeSelect = el('select', {
    class: 'pill-select wide', 'aria-label': 'Decision engine',
    onchange: (e) => { set({ decisionMode: e.target.value }); store.rerenderPanel(); },
  }, [
    knownMode ? null : el('option', { value: decMode, text: decMode, selected: true }),
    ...DECISION_MODE_GROUPS.map(([label, items]) => el('optgroup', { label },
      items.map(([id, text]) => el('option', { value: id, text, selected: id === decMode })))),
  ]);
  const evalModels = deciders.models || FALLBACK_EVAL_MODELS;
  const evalModel = ai.gatewayEvalModel || DEFAULT_EVAL_MODEL;
  const evalItems = withCurrent(evalModels.map((m) => [m.id, m.label || m.id]), evalModel);
  const evalInfo = evalModels.find((m) => m.id === evalModel);
  // A leftover "Custom decision model" silently beats the Evaluation model dropdown in gateway modes — surface it.
  const overrideActive = Boolean((ai.decisionModel || '').trim()) && (decMode === 'auto' || decMode === 'gateway')
    && ai.decisionModel.trim() !== evalModel;
  // Prefer the server's answer; otherwise a saved override implies a key (env key presence is unknown client-side).
  const gwReady = deciders.gatewayConfigured ?? (ai.vercelApiKey ? true : null);
  const gatewayStatusRow = row({
    title: 'Vercel AI Gateway',
    desc: gwReady === true
      ? 'Vercel AI Gateway ready — one AI_GATEWAY_API_KEY powers every evaluation model; no Perplexity, OpenAI or Cloudflare keys needed'
      : (gwReady === false
          ? 'Set AI_GATEWAY_API_KEY in Gateway settings (above) or in .env.local'
          : (deciders.loading || !deciders.fetchedAt
              ? 'Checking the gateway…'
              : `Status unknown (${deciders.error || 'no response'}) — AI_GATEWAY_API_KEY in .env.local is used if present`)),
    control: el('span', {
      class: `badge ${gwReady === true ? 'ok' : (gwReady === false ? 'warn' : '')}`.trim(),
      text: gwReady === true ? 'Ready' : (gwReady === false ? 'Key missing' : 'Checking'),
    }),
  });

  /* Vercel AI Gateway — live model dropdown (grouped by owner) + provider pin limited to that model's providers */
  loadVercelModels(store);
  const gwModels = gw.data?.models || [];
  const chatModels = gwModels.filter((m) => !m.type || m.type === 'language');
  const embedModels = gwModels.filter((m) => m.type === 'embedding');
  const curModel = String(ai.vercelModel || '');
  const curInfo = gwModels.find((m) => m.id === curModel);
  const order = orderList(ai.vercelProviderOrder);

  const fillModelOptions = (select) => {
    const f = gw.filter.trim().toLowerCase();
    const match = (m) => !f || `${m.id} ${m.name || ''} ${m.owner || ''}`.toLowerCase().includes(f);
    const byOwner = new Map();
    chatModels.filter(match).forEach((m) => {
      const owner = String(m.owner || m.id.split('/')[0] || 'other');
      if (!byOwner.has(owner)) byOwner.set(owner, []);
      byOwner.get(owner).push(m);
    });
    const known = chatModels.some((m) => m.id === curModel);
    const nodes = [
      el('option', { value: '', text: `Server default${gw.data?.activeModel && !curModel ? ` (${gw.data.activeModel})` : ' (AI_GATEWAY_MODEL)'}`, selected: !curModel }),
      curModel && !known ? el('option', { value: curModel, text: `Custom · ${curModel}`, selected: true }) : null,
      ...[...byOwner.keys()].sort().map((owner) => el('optgroup', { label: owner },
        byOwner.get(owner).sort((a, b) => String(a.name || a.id).localeCompare(String(b.name || b.id)))
          .map((m) => el('option', { value: m.id, text: m.name && m.name !== m.id ? `${m.name} · ${m.id}` : m.id, selected: m.id === curModel })))),
      f && !byOwner.size ? el('option', { value: '__none', disabled: true, text: `No models match “${gw.filter.trim()}”` }) : null,
    ].filter(Boolean);
    select.replaceChildren(...nodes);
  };

  const vercelModelRow = () => {
    const select = el('select', {
      class: 'pill-select wide', 'aria-label': 'Vercel model',
      onchange: (e) => { if (e.target.value === '__none') return; set({ vercelModel: e.target.value }); softRerender(store); },
    });
    fillModelOptions(select);
    const owners = new Set(chatModels.map((m) => m.owner || m.id.split('/')[0])).size;
    const desc = gw.loading && !gw.data ? 'Loading live models from the gateway…'
      : gw.error ? `Live list unavailable (${gw.error}) — paste a model ID below`
        : gw.data?.configured === false ? 'Add an AI Gateway key (below) to load live models — or paste a model ID'
          : gw.data ? `${chatModels.length} live models from ${owners} owners · grouped by owner`
            : 'Checking the gateway…';
    const control = el('div', { class: 'gw-picker' }, [
      el('div', { class: 'fv-bar' }, [
        textIn({
          type: 'search', placeholder: 'Filter models…', value: gw.filter, 'aria-label': 'Filter models', 'data-focus-key': 'gw.filter',
          oninput: (e) => { gw.filter = e.target.value; fillModelOptions(select); },
        }),
        btn(gw.loading ? 'Loading…' : 'Refresh', () => loadVercelModels(store, true), '', { disabled: gw.loading, title: 'Reload live gateway models' }),
      ]),
      select,
      textIn({
        placeholder: 'Or paste any model ID, e.g. google/gemini-2.5-flash', value: curModel && !chatModels.some((m) => m.id === curModel) ? curModel : '',
        'aria-label': 'Custom model ID', 'data-focus-key': 'gw.customModel', spellcheck: 'false',
        onchange: (e) => { set({ vercelModel: e.target.value.trim() }); softRerender(store); },
      }),
    ]);
    return tagged(row({ title: 'Vercel model', desc, stack: true, control }), 'vercel models gateway model llm chat model');
  };

  const providerRow = () => {
    const fromModel = Array.isArray(curInfo?.providers) && curInfo.providers.length > 0;
    const provs = (fromModel ? curInfo.providers : (gw.data?.providers || [])).map(String);
    const pin = order[0] || '';
    const unlisted = Boolean(pin && provs.length && !provs.includes(pin));
    const items = [['', 'Auto (Gateway default)'], ...provs.map((p) => [p, p])];
    if (pin && !provs.includes(pin)) items.push([pin, `${pin} · not listed for this model`]);
    const desc = unlisted
      ? `“${pin}” isn't listed for ${curModel || 'this model'} — pick Auto or a listed provider`
      : fromModel ? `${provs.length} upstream provider${provs.length === 1 ? '' : 's'} serve ${curModel}`
        : (curModel ? 'Model not in the live list — showing every gateway provider' : 'Showing every gateway provider');
    return tagged(row({
      title: 'Provider pinning', desc, cls: unlisted ? 'warn-row' : '',
      control: selectEl(items, pin, (p) => {
        set({ vercelProviderOrder: p ? [p, ...order.filter((x) => x !== p)].join(',') : '' });
        softRerender(store);
      }, { 'aria-label': 'Provider pinning', class: 'pill-select wide' }),
    }), 'provider pinning pin provider gateway upstream');
  };

  /* Screen Vision — vision model (only gateway models tagged "vision", from the list already fetched above),
     its own provider pin and the relay mode. Writes the `vision` section; zero extra network calls. */
  const vcfg = store.config.vision || {};
  const visionModels = chatModels.filter((m) => Array.isArray(m.tags) && m.tags.includes('vision') && isVisionModelId(m.id));
  const curVModel = String(vcfg.model || '');
  const curVInfo = gwModels.find((m) => m.id === (curVModel || DEFAULT_VISION_MODEL));
  const setVision = (vals) => store.patch('vision', vals);

  const fillVisionOptions = (select) => {
    const f = (gw.vfilter || '').trim().toLowerCase();
    const match = (m) => !f || `${m.id} ${m.name || ''} ${m.owner || ''}`.toLowerCase().includes(f);
    const byOwner = new Map();
    visionModels.filter(match).forEach((m) => {
      const owner = String(m.owner || m.id.split('/')[0] || 'other');
      if (!byOwner.has(owner)) byOwner.set(owner, []);
      byOwner.get(owner).push(m);
    });
    const selectedModel = visionModels.find((m) => m.id === curVModel);
    select.replaceChildren(...[
      el('option', { value: '', text: `Auto (${DEFAULT_VISION_MODEL})`, selected: !curVModel }),
      curVModel && (!selectedModel || !match(selectedModel))
        ? el('option', { value: curVModel, text: `${selectedModel ? 'Current' : 'Custom'} · ${curVModel}`, selected: true }) : null,
      ...[...byOwner.keys()].sort().map((owner) => el('optgroup', { label: owner },
        byOwner.get(owner).sort((a, b) => String(a.name || a.id).localeCompare(String(b.name || b.id)))
          .map((m) => el('option', { value: m.id, text: m.name && m.name !== m.id ? `${m.name} · ${m.id}` : m.id, selected: m.id === curVModel })))),
      f && !byOwner.size ? el('option', { value: '__none', disabled: true, text: `No vision models match “${gw.vfilter.trim()}”` }) : null,
    ].filter(Boolean));
  };

  const visionModelRow = () => {
    const select = el('select', {
      class: 'pill-select wide', 'aria-label': 'Vision model',
      onchange: (e) => { if (e.target.value === '__none') return; setVision({ model: e.target.value }); softRerender(store); },
    });
    fillVisionOptions(select);
    const desc = gw.loading && !gw.data ? 'Loading vision-capable models from the gateway…'
      : gw.error ? `Live list unavailable (${gw.error}) — paste a vision model ID below`
        : gw.data ? `${visionModels.length} vision-capable gateway models · sees the screen for text-only chat models`
          : 'Checking the gateway…';
    const error = el('span', { class: 's-desc', role: 'status', hidden: true });
    const validateModel = (input) => {
      const id = input.value.trim();
      const valid = !id || isVisionModelId(id);
      error.textContent = valid ? '' : 'Use a routed provider/model ID up to 160 characters. OpenAI o1 and Pro models are unavailable.';
      error.hidden = valid;
      input.setAttribute('aria-invalid', String(!valid));
      input.setCustomValidity(error.textContent);
      return valid;
    };
    const control = el('div', { class: 'gw-picker' }, [
      el('div', { class: 'fv-bar' }, [
        textIn({
          type: 'search', placeholder: 'Filter vision models…', value: gw.vfilter || '', 'aria-label': 'Filter vision models',
          'data-focus-key': 'gw.vfilter', oninput: (e) => { gw.vfilter = e.target.value; fillVisionOptions(select); },
        }),
        btn(gw.loading ? 'Loading…' : 'Refresh', () => loadVercelModels(store, true), '', { disabled: gw.loading, title: 'Reload live gateway models' }),
      ]),
      select,
      textIn({
        placeholder: 'Or paste any vision model ID, e.g. google/gemini-2.5-flash',
        value: curVModel && !visionModels.some((m) => m.id === curVModel) ? curVModel : '',
        // Saved on change. Let softRerender defer while typing so a catalog refresh cannot commit a partial ID.
        'aria-label': 'Custom vision model ID', spellcheck: 'false',
        oninput: (e) => validateModel(e.target),
        onchange: (e) => {
          if (!validateModel(e.target)) return;
          setVision({ model: e.target.value.trim() });
          requestAnimationFrame(() => softRerender(store));
        },
      }),
      error,
    ]);
    return tagged(row({ title: 'Vision model', desc, stack: true, control }), 'vision model screen vision multimodal image model');
  };

  const visionProviderRow = () => {
    const vorder = orderList(vcfg.providerOrder);
    const fromModel = Array.isArray(curVInfo?.providers) && (curVInfo.providersKnown || curVInfo.providers.length > 0);
    const provs = (fromModel ? curVInfo.providers : (gw.data?.providers || [])).map(String);
    const pin = vorder[0] || '';
    const unlisted = Boolean(pin && (fromModel || provs.length) && !provs.includes(pin));
    const items = [['', 'Auto (Gateway default)'], ...provs.map((p) => [p, p])];
    if (pin && !provs.includes(pin)) items.push([pin, `${pin} · not listed for this model`]);
    const vName = curVModel || DEFAULT_VISION_MODEL;
    const desc = unlisted
      ? `“${pin}” isn't listed for ${vName} — pick Auto or a listed provider`
      : fromModel ? `${provs.length} upstream provider${provs.length === 1 ? '' : 's'} serve ${vName}`
        : 'Showing every gateway provider';
    return tagged(row({
      title: 'Vision provider pinning', desc, cls: unlisted ? 'warn-row' : '',
      control: selectEl(items, pin, (p) => {
        setVision({ providerOrder: p ? [p, ...vorder.filter((x) => x !== p)].join(',') : '' });
        softRerender(store);
      }, { 'aria-label': 'Vision provider pinning', 'data-focus-key': 'gw.visionProvider', class: 'pill-select wide' }),
    }), 'vision provider pinning screen vision upstream');
  };

  const visionModeRow = () => tagged(selectRow({
    label: 'Vision mode',
    desc: (vcfg.mode || 'describe') === 'direct'
      ? 'The vision model answers screen turns itself (its own style, image attached)'
      : 'The Vision Eye describes your screen; your chat model replies in character with its own voice tags',
    items: withCurrent(VISION_MODES, vcfg.mode || 'describe'), value: vcfg.mode || 'describe',
    onChange: (v) => { setVision({ mode: v }); softRerender(store); },
  }), 'vision mode screen vision multimodal describe direct vision eye');

  /* Brain & voice mode */
  // A retired mode left in an old settings.json falls back to Fish duplex in the picker.
  const brainMode = BRAIN_MODES.some(([id]) => id === ai.brainMode) ? ai.brainMode : 'fish_duplex';
  const vercelRows = [
    vercelModelRow(),
    providerRow(),
    tagged(row({ title: 'Provider fallback order', desc: 'Comma-separated, tried in order — the pin above goes first', cls: 'wide-ctrl',
      control: textIn({ placeholder: 'vertex,groq,fireworks', value: order.join(','), 'aria-label': 'Provider fallback order',
        'data-focus-key': 'gw.order', spellcheck: 'false',
        onchange: (e) => { set({ vercelProviderOrder: orderList(e.target.value).join(',') }); softRerender(store); } }) }),
    'provider pinning order fallback gateway'),
    toggle({ label: 'Strict provider pin (only)', desc: 'Never fall back to providers outside the list',
      // Backend may store a provider list ([] = off) or a boolean; an empty list must never read as "on".
      checked: Array.isArray(ai.vercelProviderOnly) ? ai.vercelProviderOnly.length > 0 : Boolean(ai.vercelProviderOnly),
      onChange: (v) => set({ vercelProviderOnly: v === true }) }),
    row({ title: 'Vercel embedding model', desc: 'Indexes saved memories when Memory & Context uses the Vercel embedder', cls: 'wide-ctrl',
      control: [
        textIn({ placeholder: 'openai/text-embedding-3-small', value: ai.vercelEmbeddingModel || '', 'aria-label': 'Vercel embedding model',
          list: embedModels.length ? 'gw-embed-models' : undefined, onchange: (e) => set({ vercelEmbeddingModel: e.target.value.trim() }) }),
        embedModels.length ? el('datalist', { id: 'gw-embed-models' }, embedModels.map((m) => el('option', { value: m.id, label: m.name || m.id }))) : null,
      ] }),
    row({ title: 'AI Gateway key', desc: 'AI_GATEWAY_API_KEY — auto-loaded server-side from .env.local if left blank', cls: 'wide-ctrl',
      control: secretControl('vercelApiKey', 'AI Gateway key') }),
  ];

  return [
    group('Brain & Voice Mode', [
      ...choiceRows(BRAIN_MODES, {
        name: 'brainMode', active: brainMode,
        onPick: (v) => { set({ brainMode: v }); store.rerenderPanel(); },
      }),
      selectRow({
        label: 'Fish Speech transport', desc: 'How synthesized speech streams back from Fish',
        items: withCurrent(FISH_TRANSPORTS, ai.fishTtsTransport), value: ai.fishTtsTransport || 'sse',
        onChange: (v) => set({ fishTtsTransport: v }),
      }),
    ], { desc: 'Who thinks and who speaks' }),

    brainMode === 'vercel_fish'
      ? group('Vercel AI Gateway', vercelRows, { desc: 'Model, provider pinning and embeddings' })
      : group('Vercel AI Gateway', [
          more({ id: 'voice.vercel.more', title: 'Gateway settings', desc: 'Model, provider pinning, embeddings, key' }, vercelRows),
        ]),

    group('Live Call', [
      callRow,
      row({
        title: 'Voice agent', desc: 'Drama 3 uses Hikari’s custom voice; Live 3 is Clara full-duplex',
        control: selectEl([['drama-3', 'Drama 3'], ['live-3', 'Live 3']], mode,
          (v) => { set({ fishAgentMode: v }); store.rerenderPanel(); },
          { disabled: fc?.connected || fc?.connecting, 'aria-label': 'Voice agent' }),
      }),
    ]),

    group('Chat & Speech', [
      toggle({ label: 'Speak replies aloud', desc: 'Voice text-chat replies with TTS', checked: ai.ttsEnabled, onChange: (v) => set({ ttsEnabled: v }) }),
      selectRow({ label: 'TTS engine', desc: 'Push-to-talk always uses Fish ASR', items: TTS, value: ai.ttsProvider, onChange: (v) => { set({ ttsProvider: v }); store.rerenderPanel(); } }),
      ...fishRows,
      tagged(toggle({
        label: 'Screen Vision',
        desc: 'Allow on-demand screen look via the stage eye button or saying "look at my screen"',
        checked: store.config.vision?.enabled ?? true,
        onChange: (v) => store.patch('vision', { enabled: v }),
      }), 'screen vision look at my screen screenshot'),
      visionModelRow(),
      visionProviderRow(),
      visionModeRow(),
      row({ title: 'Sample voice take', desc: 'Play a test line through the lip-sync pipeline', control: btn('Play', () => store.playSample()) }),
    ]),

    group('Voice Designer (Fish Audio)', [renderVoiceDesigner(store, { onSaved: onVoiceSaved })],
      { desc: 'Design a brand-new voice from a description and save it to your Fish account' }),

    group('Custom LLM Brain', [
      selectRow({
        label: 'LLM provider', desc: 'OpenAI-compatible provider for text chat', items: withCurrent(PROVIDERS, ai.provider), value: ai.provider,
        onChange: (p) => {
          const defaults = { lmstudio: 'http://127.0.0.1:1234/v1', ollama: 'http://127.0.0.1:11434/v1', openrouter: 'https://openrouter.ai/api/v1', openai: 'https://api.openai.com/v1' };
          const patch = { provider: p };
          if (defaults[p]) patch.baseUrl = defaults[p];
          set(patch);
          store.rerenderPanel();
        },
      }),
      row({ title: 'Model', desc: 'Model ID sent to the provider', control: textIn({ value: ai.model, onchange: (e) => set({ model: e.target.value }) }), cls: 'wide-ctrl' }),
      row({ title: 'Base URL', desc: 'OpenAI-compatible endpoint', control: textIn({ value: ai.baseUrl, onchange: (e) => set({ baseUrl: e.target.value }) }), cls: 'wide-ctrl' }),
      row({ title: 'API key', desc: 'Optional if set in the environment', control: secretControl('apiKey', 'LLM API key'), cls: 'wide-ctrl' }),
      slider({ label: 'Creativity', desc: 'Sampling temperature', min: 0, max: 1.4, step: 0.05, value: ai.temperature, onInput: (v) => set({ temperature: v }) }),
    ]),

    group('Decision Engine', [
      row({
        title: 'Decision engine',
        desc: `${DECISION_MODE_DESC[decMode] || 'Custom decider mode'}${brainMode === 'fish_duplex' && decMode === 'auto' ? ' · live Fish calls only use deciders fast enough for real time' : ''}`,
        control: modeSelect,
      }),
      deciders.activeModel ? row({
        title: 'Active decider',
        desc: `What is actually deciding right now${deciders.activeMode ? ` · ${deciders.activeMode} mode` : ''}`,
        control: el('code', { class: 'model-chip', text: deciders.activeModel, title: deciders.activeModel }),
      }) : null,
      overrideActive ? row({
        title: 'Custom model override is active',
        desc: `“${ai.decisionModel}” overrides the dropdown — clear it to use ${evalModel}`,
        cls: 'warn-row',
        control: btn('Clear override', () => { set({ decisionModel: '' }); refreshDecidersSoon(store); store.rerenderPanel(); }),
      }) : null,
      GATEWAY_MODES.has(decMode) ? gatewayStatusRow : null,
      (decMode === 'auto' || decMode === 'gateway') ? row({
        title: 'Evaluation model',
        desc: `${evalInfo?.provider ? `${evalInfo.provider} · ` : ''}${evalModel}${deciders.models ? '' : ' · built-in list'}`,
        control: selectEl(evalItems, evalModel, (v) => { set({ gatewayEvalModel: v }); store.rerenderPanel(); },
          { 'aria-label': 'Evaluation model', class: 'pill-select wide' }),
      }) : null,
      more({
        id: 'voice.decider.direct', title: 'Optional direct-provider overrides (bypass Vercel Gateway)',
        desc: 'Cloudflare Clef, Perplexity key, custom decider model ID', open: decMode === 'clef',
      }, [
        row({
          title: 'Custom decision model', desc: 'Model ID for direct modes (blank = mode default)', cls: 'wide-ctrl',
          control: textIn({
            placeholder: decMode === 'clef' ? '@cf/cloudflare/clef-flash' : (decMode === 'pplx' ? 'pplx-decider-flash' : 'Mode default'),
            value: ai.decisionModel || '', 'aria-label': 'Custom decision model', onchange: (e) => set({ decisionModel: e.target.value.trim() }),
          }),
        }),
        row({ title: 'Cloudflare Account ID', desc: 'Auto-loaded from env/.env.local if blank', cls: 'wide-ctrl',
          control: secretControl('cloudflareAccountId', 'Cloudflare Account ID', { type: 'text', blankHint: 'Account ID (blank = .env.local)' }) }),
        row({ title: 'Cloudflare API Token', desc: 'Workers AI token — auto-loaded server-side if blank', cls: 'wide-ctrl',
          control: secretControl('cloudflareApiToken', 'Cloudflare API Token') }),
        row({ title: 'Perplexity API key', desc: 'Direct Perplexity — otherwise Perplexity routes via the Gateway', cls: 'wide-ctrl',
          control: secretControl('perplexityApiKey', 'Perplexity API key') }),
      ]),
      toggle({
        label: 'Autonomous gestures & emotes', desc: 'Let Hikari pick emotes on her own from the conversation',
        checked: ai.autonomousGestures ?? true,
        onChange: (v) => { set({ autonomousGestures: v }); store.rerenderPanel(); },
      }),
      toggle({
        label: 'Auto-wave on greetings', desc: 'Let the AI wave on hi / bye (off: waves only come from stream events like a Twitch follow)',
        checked: ai.autoWaveOnGreeting ?? false, disabled: ai.autonomousGestures === false,
        onChange: (v) => set({ autoWaveOnGreeting: v }),
      }),
      toggle({
        label: 'Gentle idle emotes', desc: 'Small idle moments while nobody is talking',
        checked: ai.autoIdleEmotes ?? true, disabled: ai.autonomousGestures === false,
        onChange: (v) => set({ autoIdleEmotes: v }),
      }),
    ], { desc: 'How Hikari decides what to feel and do' }),

    group('Speech Body Language', [
      toggle({
        label: 'Talking gestures', desc: 'Relaxed, bent arms move naturally while she speaks (also in Motion → Arms)',
        checked: rig.autoGestureOnSpeech ?? true, onChange: (v) => store.patch('rig', { autoGestureOnSpeech: v }),
      }),
      slider({ label: 'Speech → head nod', min: 0, max: 1.5, step: 0.05, value: ls.speechToHeadNod ?? 0.8, def: 0.80, onInput: (v) => store.patch('lipsync', { speechToHeadNod: v }) }),
      slider({ label: 'Speech → eyebrow lift', min: 0, max: 1.5, step: 0.05, value: ls.speechToBrows ?? 0.72, def: 0.72, onInput: (v) => store.patch('lipsync', { speechToBrows: v }) }),
      slider({ label: 'Talking gesture intensity', desc: 'How much speech moves her arms', min: 0, max: 1.5, step: 0.05, value: ls.speechToArmGesture ?? 0.85, def: 0.85, onInput: (v) => store.patch('lipsync', { speechToArmGesture: v }) }),
    ]),

    group('Personality', [
      row({
        title: 'Persona', desc: 'Who Hikari is — saved when you click away', stack: true,
        control: el('textarea', { rows: 6, text: ai.persona || '', 'aria-label': 'Persona', onchange: (e) => set({ persona: e.target.value }) }),
      }),
      row({
        title: 'Memory & Context', desc: 'Saved memories, conversation continuity and chat history',
        control: btn('Open', () => document.querySelector('.nav-item[data-tab="memory"]')?.click()),
      }),
    ]),
  ];
}
