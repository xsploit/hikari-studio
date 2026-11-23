/** VOICE & BRAIN — brain/voice mode (Fish duplex · Vercel AI Gateway · local 4B LoRA), Fish TTS transport,
 *  gateway provider pinning, live call, speech & ASR, custom LLM, decision engine, body language, personality. */
import { btn, choiceRows, el, group, more, row, selectEl, selectRow, slider, textIn, toggle, withCurrent } from '../ui.js';

const BRAIN_MODES = [
  ['fish_duplex', 'Fish Real-Time Duplex', 'Fish LLM + live full-duplex voice — lowest latency'],
  ['vercel_fish', 'Vercel AI Gateway + Fish Speech', 'Any gateway model with provider pinning, voiced by Fish'],
  ['local_4b_fish', 'Experimental: Local 4B SLEEP LoRA', 'Google SLEEP continual-learning test — keep Vercel/Fish for live streaming'],
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
const deciders = { models: null, gatewayConfigured: null, activeModel: '', fetchedAt: 0, loading: false, error: '' };
async function loadDeciders(store, force = false) {
  if (deciders.loading || (!force && Date.now() - deciders.fetchedAt < 30000)) return;
  deciders.loading = true;
  try {
    const r = await fetch('/api/deciders/models', { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status === 404 ? 'needs the updated studio server' : `HTTP ${r.status}`);
    const d = await r.json();
    const list = Array.isArray(d?.models) ? d.models.filter((m) => m && m.id) : [];
    deciders.models = list.length ? list : null;
    deciders.gatewayConfigured = typeof d?.gatewayConfigured === 'boolean' ? d.gatewayConfigured : null;
    deciders.activeModel = String(d?.activeModel || '');
    deciders.error = '';
  } catch (err) {
    deciders.error = err.message || 'unavailable';
  } finally {
    deciders.loading = false;
    deciders.fetchedAt = Date.now();
    if (store.activeTab === 'voice') store.rerenderPanel();
  }
}

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
  const secretControl = (field, label, { type = 'password', blankHint = 'Leave blank to use .env.local' } = {}) => {
    const isSet = Boolean(ai[field]);
    const input = textIn({
      type, autocomplete: 'off', 'aria-label': label,
      placeholder: isSet ? 'Saved — type to replace' : blankHint,
      onchange: (e) => { const v = e.target.value.trim(); if (v) { set({ [field]: v }); e.target.value = ''; store.rerenderPanel(); } },
    });
    return isSet ? [input, btn('Clear', () => { set({ [field]: null }); store.rerenderPanel(); })] : input;
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

  const fishRows = ai.ttsProvider === 'fish' ? [
    row({
      title: 'Fish API key',
      desc: fs.keyReady ? 'Loaded server-side · Drama 3 TTS + transcribe-1 ASR ready' : 'Not found in env or .env.local — paste an override',
      control: el('span', { class: `badge ${fs.keyReady ? 'ok' : 'warn'}`, text: fs.keyReady ? 'Ready' : 'Missing' }),
    }),
    selectRow({ label: 'Fish model', desc: 'Voice model for chat replies', items: FISH_MODELS, value: ai.fishModel, onChange: (v) => set({ fishModel: v }) }),
    row({ title: 'Reference voice ID', desc: 'Fish voice to clone', control: textIn({ placeholder: 'Voice ID', value: ai.fishReferenceId, onchange: (e) => set({ fishReferenceId: e.target.value }) }), cls: 'wide-ctrl' }),
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

  /* Brain & voice mode */
  const brainMode = ai.brainMode || 'fish_duplex';
  const vercelRows = [
    row({ title: 'Vercel model', desc: 'Gateway model ID, e.g. google/gemini-2.5-flash', cls: 'wide-ctrl',
      control: textIn({ placeholder: 'google/gemini-2.5-flash', value: ai.vercelModel || '', 'aria-label': 'Vercel model',
        onchange: (e) => set({ vercelModel: e.target.value.trim() }) }) }),
    row({ title: 'Provider pinning order', desc: 'Comma-separated, tried in order', cls: 'wide-ctrl',
      control: textIn({ placeholder: 'vertex,groq,fireworks', value: ai.vercelProviderOrder || '', 'aria-label': 'Provider pinning order',
        onchange: (e) => set({ vercelProviderOrder: e.target.value.split(',').map((x) => x.trim()).filter(Boolean).join(',') }) }) }),
    toggle({ label: 'Strict provider pin (only)', desc: 'Never fall back to providers outside the list',
      // Backend may store a provider list ([] = off) or a boolean; an empty list must never read as "on".
      checked: Array.isArray(ai.vercelProviderOnly) ? ai.vercelProviderOnly.length > 0 : Boolean(ai.vercelProviderOnly),
      onChange: (v) => set({ vercelProviderOnly: v === true }) }),
    row({ title: 'Vercel embedding model', desc: 'Used by SLEEP memory when the Vercel embedder is on', cls: 'wide-ctrl',
      control: textIn({ placeholder: 'openai/text-embedding-3-small', value: ai.vercelEmbeddingModel || '', 'aria-label': 'Vercel embedding model',
        onchange: (e) => set({ vercelEmbeddingModel: e.target.value.trim() }) }) }),
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

    brainMode === 'local_4b_fish' ? group('Local 4B LoRA', [
      row({ title: 'SLEEP LoRA adapter path', desc: 'Folder or .safetensors from your Colab export', cls: 'wide-ctrl',
        control: textIn({ placeholder: 'models/hikari-sleep-lora', value: ai.localLoraPath || '', 'aria-label': 'SLEEP LoRA adapter path',
          onchange: (e) => set({ localLoraPath: e.target.value.trim() }) }) }),
    ]) : null,

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
      row({ title: 'Sample voice take', desc: 'Play a test line through the lip-sync pipeline', control: btn('Play', () => store.playSample()) }),
    ]),

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
      row({ title: 'Decision engine', desc: DECISION_MODE_DESC[decMode] || 'Custom decider mode', control: modeSelect }),
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
        label: 'Auto-wave on greetings', desc: 'Wave on hi / bye, and when you come back after a while',
        checked: ai.autoWaveOnGreeting ?? true, disabled: ai.autonomousGestures === false,
        onChange: (v) => set({ autoWaveOnGreeting: v }),
      }),
      toggle({
        label: 'Gentle idle emotes', desc: 'Small idle moments while nobody is talking',
        checked: ai.autoIdleEmotes ?? true, disabled: ai.autonomousGestures === false,
        onChange: (v) => set({ autoIdleEmotes: v }),
      }),
    ], { desc: 'How Hikari decides what to feel and do' }),

    group('Speech Body Language', [
      toggle({ label: 'Gesture while speaking', desc: 'Raise and move arms automatically during speech', checked: rig.autoGestureOnSpeech, onChange: (v) => store.patch('rig', { autoGestureOnSpeech: v }) }),
      slider({ label: 'Speech → head nod', min: 0, max: 1.5, step: 0.05, value: ls.speechToHeadNod ?? 0.8, def: 0.80, onInput: (v) => store.patch('lipsync', { speechToHeadNod: v }) }),
      slider({ label: 'Speech → eyebrow lift', min: 0, max: 1.5, step: 0.05, value: ls.speechToBrows ?? 0.72, def: 0.72, onInput: (v) => store.patch('lipsync', { speechToBrows: v }) }),
      slider({ label: 'Speech → arm gesture', min: 0, max: 1.5, step: 0.05, value: ls.speechToArmGesture ?? 0.85, def: 0.85, onInput: (v) => store.patch('lipsync', { speechToArmGesture: v }) }),
    ]),

    group('Personality & Memory', [
      row({
        title: 'Persona', desc: 'Who Hikari is — saved when you click away', stack: true,
        control: el('textarea', { rows: 6, text: ai.persona || '', 'aria-label': 'Persona', onchange: (e) => set({ persona: e.target.value }) }),
      }),
      row({
        title: 'Conversation memory', desc: 'Forget the chat history',
        control: btn('Clear memory', async (e) => {
          const b = e.currentTarget;
          await fetch('/api/memory/clear', { method: 'POST' });
          b.textContent = 'Cleared ✓';
        }, 'danger'),
      }),
    ]),
  ];
}
