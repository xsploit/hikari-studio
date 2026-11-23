/** Tweened emotes — loaded from GET /api/emotes (falls back to the built-in 8 on older servers).
 *  Fired with POST /api/emote. Shared by the Motion & Emotions tab and the floating popover. */
import { el } from '../ui.js';

const GLYPHS = {
  joy: '✦', greeting: '〜', shy: '◠', sad: '☁', curious: '?', annoyed: '◔', love: '♥',
  smug: '◡', awe: '✧', sleepy: 'z', caring: '❀', surprised: '!',
};

const BUILTIN = [
  { id: 'excited_hop',   label: 'Excited Hop',   category: 'joy' },
  { id: 'shy_blush',     label: 'Shy Tilt',      category: 'shy' },
  { id: 'wink_wave',     label: 'Wink & Wave',   category: 'greeting' },
  { id: 'gloom_shake',   label: 'Shock / Gloom', category: 'sad' },
  { id: 'curious_lean',  label: 'Curious Lean',  category: 'curious' },
  { id: 'chibi_pop',     label: 'Chibi Pop',     category: 'joy' },
  { id: 'pout_puff',     label: 'Pout Puff',     category: 'annoyed' },
  { id: 'heart_sparkle', label: 'Starry Eyes',   category: 'love' },
];

const decorate = (e) => ({
  id: String(e.id),
  label: e.label || String(e.id).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  category: e.category || '',
  description: e.description || e.desc || '',
  duration: Number(e.duration) || 0,
  isNew: Boolean(e.new),
  glyph: GLYPHS[e.category] || '•',
});

/** Live binding — importers always see the latest list. */
export let EMOTES = BUILTIN.map(decorate);
export let emotesSource = 'builtin';

let loading = null;
export function loadEmotes() {
  if (loading) return loading;
  loading = (async () => {
    try {
      const r = await fetch('/api/emotes');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const list = Array.isArray(data?.emotes) ? data.emotes.filter((e) => e && e.id) : [];
      if (list.length) {
        EMOTES = list.map(decorate);
        emotesSource = 'server';
      }
    } catch { /* older server — keep the built-in 8 */ }
    return EMOTES;
  })();
  return loading;
}

export async function triggerEmote(id) {
  try {
    const r = await fetch('/api/emote', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emote: id }),
    });
    const res = await r.json().catch(() => ({ ok: r.ok }));
    return { ok: Boolean(res?.ok), duration: Number(res?.duration) || 2.6, error: res?.error };
  } catch (err) {
    return { ok: false, duration: 0, error: err.message };
  }
}

/** Pill grid of emotes; each pill shows a progress sweep while its emote plays. */
export function emoteGrid({ compact = false, showKeys = false } = {}) {
  return el('div', { class: `emote-grid${compact ? ' compact' : ''}` }, EMOTES.map((e, i) => {
    const pill = el('button', {
      type: 'button', class: 'emote-pill', 'data-emote': e.id,
      title: e.description ? `${e.label} — ${e.description}` : e.label,
    }, [
      // In the popover the first 9 glyph circles show their hotkey number instead of the category glyph.
      showKeys && i < 9
        ? el('span', { class: 'emote-glyph key', text: String(i + 1), title: `Key ${i + 1}` })
        : el('span', { class: 'emote-glyph', text: e.glyph, 'aria-hidden': 'true' }),
      el('span', { class: 'emote-label', text: e.label }),
      e.isNew && !compact ? el('span', { class: 'emote-new', text: 'New' }) : null,
    ]);
    pill.addEventListener('click', () => firePill(pill, e.id));
    return pill;
  }));
}

export async function firePill(pill, id) {
  pill.classList.remove('firing', 'failed');
  const res = await triggerEmote(id);
  if (!pill.isConnected) return res;
  void pill.offsetWidth; // restart the CSS sweep animation
  if (res.ok) {
    pill.style.setProperty('--dur', `${res.duration}s`);
    pill.classList.add('firing');
    clearTimeout(pill._t);
    pill._t = setTimeout(() => pill.classList.remove('firing'), res.duration * 1000);
  } else {
    pill.classList.add('failed');
    pill.title = res.error || 'Emote failed';
    setTimeout(() => pill.classList.remove('failed'), 1200);
  }
  return res;
}
