/** VOICE DESIGNER — describe a voice, generate Fish Audio candidates, preview them inline,
 *  then save one to the user's Fish account and select it (POST /api/fish/voice-design, /api/fish/voices/save). */
import { btn, el, more, row, selectEl, textIn } from '../ui.js';

const LANGS = [['en', 'English'], ['ja', 'Japanese'], ['zh', 'Chinese'], ['ko', 'Korean'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German']];
const COUNTS = [['2', '2 takes'], ['3', '3 takes']];
const VISIBILITY = [['private', 'Private'], ['unlisted', 'Unlisted'], ['public', 'Public']];
export const DEFAULT_PREVIEW_LINE = 'Hey! Welcome back to the stream, I missed you!';

// Module state survives re-renders (the settings page is rebuilt on every change).
const vd = {
  instruction: '', referenceText: DEFAULT_PREVIEW_LINE, language: 'en', n: '2',
  designing: false, saving: false, error: '', savedMsg: '',
  candidates: [], selected: '', title: '', visibility: 'private',
};

/** Tag a row (or a disclosure head) with explicit Settings-search keywords. */
export function tagged(node, kw) {
  const target = node?.classList?.contains('s-more') ? node.querySelector(':scope > .s-more-head') : node;
  if (target) target.dataset.searchKw = kw;
  return node;
}

/** Shared JSON helper: throws a readable error for HTTP failures and {ok:false} payloads. */
export async function jsonFetch(url, opts = {}) {
  const r = await fetch(url, { cache: 'no-store', ...opts });
  let d = null;
  try { d = await r.json(); } catch { /* non-JSON body */ }
  if (!r.ok || d?.ok === false) {
    const msg = d?.error || d?.detail || d?.message;
    throw new Error(msg ? String(msg) : (r.status === 404 ? 'needs the updated studio server' : `HTTP ${r.status}`));
  }
  return d || {};
}

/**
 * Re-render the Voice & Brain page without stealing the caret: inputs tagged data-focus-key get focus and
 * selection restored; while the user types in any other field the re-render waits until it loses focus.
 */
let blurWait = false;
export function softRerender(store) {
  if (store.activeTab !== 'voice') return;
  const body = document.getElementById('inspector-body');
  const a = document.activeElement;
  const typing = a && body?.contains(a)
    && (a.tagName === 'TEXTAREA' || (a.tagName === 'INPUT' && /^(text|search|password|url|)$/.test(a.type || '')));
  if (typing && !a.dataset.focusKey) {
    if (!blurWait) {
      blurWait = true;
      a.addEventListener('blur', () => { blurWait = false; softRerender(store); }, { once: true });
    }
    return;
  }
  const key = typing ? a.dataset.focusKey : '';
  let s0 = null; let s1 = null;
  if (key) { try { s0 = a.selectionStart; s1 = a.selectionEnd; } catch { /* no selection API */ } }
  store.rerenderPanel();
  if (!key || !body) return;
  const n = body.querySelector(`[data-focus-key="${key}"]`);
  if (!n) return;
  n.focus();
  if (s0 !== null) { try { n.setSelectionRange(s0, s1); } catch { /* ignore */ } }
}

/* <audio> players are kept per candidate and moved into the freshly rendered slot after each re-render,
   so a preview keeps playing when the page rebuilds (detached search-index renders never touch them). */
function audioFor(c) {
  if (!c._el) {
    c._el = el('audio', { controls: true, preload: 'metadata', class: 'vd-audio', src: `data:${c.mime || 'audio/wav'};base64,${c.audioBase64 || ''}` });
  }
  return c._el;
}
function mountAudio() {
  queueMicrotask(() => {
    const body = document.getElementById('inspector-body');
    if (!body) return;
    for (const c of vd.candidates) {
      const slot = [...body.querySelectorAll('.vd-audio-slot')].find((s) => s.dataset.cand === c.id);
      if (!slot) continue;
      const a = audioFor(c);
      if (a.parentNode !== slot) slot.replaceChildren(a);
      if (c._wasPlaying && a.paused) a.play().catch(() => {});
    }
  });
}

async function design(store) {
  if (vd.designing) return;
  const instruction = vd.instruction.trim();
  if (!instruction) { vd.error = 'Describe the voice first'; softRerender(store); return; }
  vd.designing = true; vd.error = ''; vd.savedMsg = '';
  softRerender(store);
  try {
    const d = await jsonFetch('/api/fish/voice-design', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        instruction, referenceText: vd.referenceText.trim() || DEFAULT_PREVIEW_LINE,
        language: vd.language, n: Number(vd.n) || 2,
      }),
    });
    const list = (Array.isArray(d.candidates) ? d.candidates : []).filter((c) => c && c.id && c.audioBase64);
    vd.candidates.forEach((c) => c._el?.pause());
    vd.candidates = list;
    vd.selected = list[0]?.id || '';
    if (!list.length) vd.error = 'Fish returned no candidates — try a more detailed description';
  } catch (err) {
    vd.error = `Voice design failed: ${err.message || 'unavailable'}`;
  } finally {
    vd.designing = false;
    softRerender(store);
  }
}

async function save(store, onSaved) {
  if (vd.saving) return;
  const title = vd.title.trim();
  if (!vd.selected) { vd.error = 'Pick a candidate first'; softRerender(store); return; }
  if (!title) { vd.error = 'Name your voice first'; softRerender(store); return; }
  vd.saving = true; vd.error = ''; vd.savedMsg = '';
  softRerender(store);
  try {
    const d = await jsonFetch('/api/fish/voices/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidateId: vd.selected, title, description: vd.instruction.trim().slice(0, 500),
        visibility: vd.visibility, applyNow: true,
      }),
    });
    onSaved?.(d);
    vd.savedMsg = `Saved '${d.voice?.title || title}' to your Fish account & selected ✓`;
  } catch (err) {
    vd.error = `Save failed: ${err.message || 'unavailable'}`;
  } finally {
    vd.saving = false;
    softRerender(store);
  }
}

function candidateCard(store, c, i, onSaved) {
  const on = c.id === vd.selected;
  c._wasPlaying = Boolean(c._el && !c._el.paused);
  const pick = () => { if (!on) { vd.selected = c.id; vd.error = ''; softRerender(store); } };
  const dur = Number(c.durationSec);
  return el('div', { class: `vd-cand${on ? ' selected' : ''}`, 'data-cand': c.id }, [
    el('label', { class: 'vd-cand-head' }, [
      el('input', { type: 'radio', name: 'vd-candidate', value: c.id, checked: on, 'aria-label': `Candidate ${i + 1}`, onchange: pick }),
      el('span', { class: 'vd-cand-title', text: `Candidate ${i + 1}` }),
      Number.isFinite(dur) && dur > 0 ? el('span', { class: 'vd-chip', text: `${dur.toFixed(1)} s` }) : null,
    ]),
    el('span', { class: 'vd-audio-slot', 'data-cand': c.id }),
    on ? el('div', { class: 'vd-save' }, [
      el('div', { class: 'vd-save-label', text: 'Save to Fish Account' }),
      el('div', { class: 'vd-save-row' }, [
        textIn({
          placeholder: 'Voice name', value: vd.title, 'aria-label': 'Voice name', 'data-focus-key': 'vd.title', maxlength: 80,
          oninput: (e) => { vd.title = e.target.value; },
          onkeydown: (e) => { if (e.key === 'Enter') save(store, onSaved); },
        }),
        selectEl(VISIBILITY, vd.visibility, (v) => { vd.visibility = v; }, { 'aria-label': 'Visibility' }),
        btn(vd.saving ? 'Saving…' : 'Save & Use Voice', () => save(store, onSaved), 'primary', { disabled: vd.saving }),
      ]),
    ]) : null,
  ]);
}

/** The "Create & Save a Voice" disclosure. onSaved(result) receives the /api/fish/voices/save payload. */
export function renderVoiceDesigner(store, { onSaved } = {}) {
  const rows = [
    row({
      title: 'Voice description', desc: 'Describe how the new voice should sound', stack: true,
      control: el('textarea', {
        rows: 3, class: 'vd-prompt', text: vd.instruction, 'aria-label': 'Voice description', 'data-focus-key': 'vd.instruction',
        placeholder: 'Bright, energetic anime VTuber girl with a warm playful voice…',
        oninput: (e) => { vd.instruction = e.target.value; },
      }),
    }),
    row({
      title: 'Preview line', desc: 'What each candidate says', cls: 'wide-ctrl',
      control: textIn({
        value: vd.referenceText, placeholder: DEFAULT_PREVIEW_LINE, 'aria-label': 'Preview line', 'data-focus-key': 'vd.ref',
        oninput: (e) => { vd.referenceText = e.target.value; },
      }),
    }),
    row({
      title: 'Generate candidates', desc: 'Language and number of takes',
      control: [
        selectEl(LANGS, vd.language, (v) => { vd.language = v; }, { 'aria-label': 'Voice language' }),
        selectEl(COUNTS, vd.n, (v) => { vd.n = v; }, { 'aria-label': 'Candidates' }),
        btn(vd.designing ? 'Designing…' : 'Design Voice', () => design(store), 'primary', { disabled: vd.designing }),
      ],
    }),
  ];
  if (vd.candidates.length) {
    rows.push(row({
      title: 'Candidates', desc: 'Listen, pick one, name it and save it to your Fish account', stack: true,
      control: el('div', { class: 'vd-cands' }, vd.candidates.map((c, i) => candidateCard(store, c, i, onSaved))),
    }));
    mountAudio();
  }
  if (vd.error) rows.push(el('div', { class: 's-row note vd-msg err', role: 'alert' }, [el('div', { class: 's-desc', text: vd.error })]));
  if (vd.savedMsg) rows.push(el('div', { class: 's-row note vd-msg ok', role: 'status' }, [el('div', { class: 's-desc', text: vd.savedMsg })]));
  return tagged(more({
    id: 'voice.designer', title: 'Create & Save a Voice',
    desc: 'Describe a voice, preview candidates and save it to your Fish account',
  }, rows), 'voice design designer create voice new voice clone fish voices');
}

/** Test hook (smoke tests only). */
export const _voiceDesignState = vd;
