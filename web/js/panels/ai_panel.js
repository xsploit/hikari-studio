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

const DECISION_MODES = [
  ['auto', 'Auto (Clef → pplx → Jev → Luna → Local)'],
  ['clef', 'Cloudflare Clef-Flash (<40ms)'],
  ['pplx', 'Perplexity Decider (pplx-decider-flash)'],
  ['jev', 'Jev (typesafe-ai/jev · Vercel Eval)'],
  ['luna', 'OpenAI Decisions (gpt-6-luna)'],
  ['llm', 'Chat LLM Fallback'],
  ['heuristic', 'Local Heuristic Only'],
];

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
    row({ title: 'API key override', desc: 'Optional — overrides the server key', control: textIn({ type: 'password', placeholder: 'sk-…', value: ai.fishApiKey, onchange: (e) => set({ fishApiKey: e.target.value }) }), cls: 'wide-ctrl' }),
  ] : [];

  /* Decision engine (Cloudflare Clef first) */
  const decMode = String(ai.decisionMode || 'auto').toLowerCase();
  const usesClef = decMode === 'clef' || decMode === 'auto';
  const cfTokenSet = Boolean(ai.cloudflareApiToken);
  const pplxKeySet = Boolean(ai.perplexityApiKey);
  const pplxKeyIn = textIn({
    type: 'password', autocomplete: 'off', 'aria-label': 'Perplexity API key',
    placeholder: pplxKeySet ? 'Saved — type to replace' : 'Leave blank to use .env.local',
    onchange: (e) => { const v = e.target.value.trim(); if (v) { set({ perplexityApiKey: v }); e.target.value = ''; store.rerenderPanel(); } },
  });
  const cfTokenIn = textIn({
    type: 'password', autocomplete: 'off', 'aria-label': 'Cloudflare API Token',
    placeholder: cfTokenSet ? 'Saved — type to replace' : 'Leave blank to use .env.local',
    onchange: (e) => { const v = e.target.value.trim(); if (v) { set({ cloudflareApiToken: v }); e.target.value = ''; store.rerenderPanel(); } },
  });

  /* Brain & voice mode */
  const brainMode = ai.brainMode || 'fish_duplex';
  const vercelKeySet = Boolean(ai.vercelApiKey);
  const vercelKeyIn = textIn({
    type: 'password', autocomplete: 'off', 'aria-label': 'Vercel API key',
    placeholder: vercelKeySet ? 'Saved — type to replace' : 'Leave blank to use .env.local',
    onchange: (e) => { const v = e.target.value.trim(); if (v) { set({ vercelApiKey: v }); e.target.value = ''; store.rerenderPanel(); } },
  });
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
    row({ title: 'Vercel API key', desc: 'Auto-loaded server-side from .env.local if left blank', cls: 'wide-ctrl',
      control: vercelKeySet
        ? [vercelKeyIn, btn('Clear', () => { set({ vercelApiKey: '' }); store.rerenderPanel(); })]
        : vercelKeyIn }),
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
      row({ title: 'API key', desc: 'Optional if set in the environment', control: textIn({ type: 'password', value: ai.apiKey, onchange: (e) => set({ apiKey: e.target.value }) }), cls: 'wide-ctrl' }),
      slider({ label: 'Creativity', desc: 'Sampling temperature', min: 0, max: 1.4, step: 0.05, value: ai.temperature, onInput: (v) => set({ temperature: v }) }),
    ]),

    group('Decision Engine', [
      selectRow({
        label: 'Decision engine mode', desc: 'Auto tries Clef, Perplexity, Jev, Luna, then local heuristics',
        items: withCurrent(DECISION_MODES, decMode), value: decMode,
        onChange: (v) => { set({ decisionMode: v }); store.rerenderPanel(); },
      }),
      row({
        title: 'Decision model', desc: 'Optional fast model override for emotion & gesture picks',
        control: textIn({
          placeholder: usesClef ? '@cf/cloudflare/clef-flash (or clef / clef-omni / typesafe-ai/jev)'
            : (decMode === 'pplx' ? 'pplx-decider-flash' : 'typesafe-ai/jev (or gpt-6-luna)'),
          value: ai.decisionModel || '', 'aria-label': 'Decision model', onchange: (e) => set({ decisionModel: e.target.value.trim() }),
        }),
        cls: 'wide-ctrl',
      }),
      ...(usesClef ? [
        row({
          title: 'Cloudflare Account ID', desc: 'Auto-loaded from env/.env.local if blank', cls: 'wide-ctrl',
          control: textIn({ placeholder: 'Account ID', value: ai.cloudflareAccountId || '', autocomplete: 'off', 'aria-label': 'Cloudflare Account ID',
            onchange: (e) => set({ cloudflareAccountId: e.target.value.trim() }) }),
        }),
        row({
          title: 'Cloudflare API Token', desc: 'Workers AI token — auto-loaded server-side if blank', cls: 'wide-ctrl',
          control: cfTokenSet ? [cfTokenIn, btn('Clear', () => { set({ cloudflareApiToken: '' }); store.rerenderPanel(); })] : cfTokenIn,
        }),
      ] : []),
      ...(decMode === 'pplx' ? [
        row({
          title: 'Perplexity API key', desc: 'Auto-loaded from env/.env.local if blank', cls: 'wide-ctrl',
          control: pplxKeySet ? [pplxKeyIn, btn('Clear', () => { set({ perplexityApiKey: '' }); store.rerenderPanel(); })] : pplxKeyIn,
        }),
      ] : []),
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
