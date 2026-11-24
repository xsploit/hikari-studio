/** Tweened emotes — loaded from GET /api/emotes (falls back to the built-in catalog on older servers;
 *  the 7 Expression-FX emotes are always included).
 *  Fired with POST /api/emote. Shared by the Motion & Emotions tab and the floating popover. */
import { el } from '../ui.js';

const GLYPHS = {
  joy: '✦', greeting: '〜', shy: '◠', sad: '☁', curious: '?', annoyed: '◔', love: '♥',
  smug: '◡', awe: '✧', sleepy: 'z', caring: '❀', surprised: '!',
  angry: '#', nervous: '﹅', music: '♪', vibe: '♪', starry: '★', cat: 'ω',
};
// Per-emote glyphs win over the category glyph (the new Expression-FX emotes read better with their own mark).
const ID_GLYPHS = {
  starry_eyes: '★', heart_crush: '♥', pout_puff: '◔', cat_smug: 'ω',
  anime_sweat: '﹅', anger_vein: '#', vibe_music: '♪',
};

/** Expression-FX emotes (blend shapes + VFX). Always listed — appended if an older server's catalog lacks them. */
const FX_EMOTES = [
  { id: 'starry_eyes', label: 'Starry Eyes', category: 'awe',     description: 'Star-shaped eye highlights with a sparkle burst', new: true },
  { id: 'heart_crush', label: 'Heart Crush', category: 'love',    description: 'Heart eyes and floating hearts', new: true },
  { id: 'pout_puff',   label: 'Pout Puff',   category: 'annoyed', description: 'Turns away with puffed cheeks', new: true },
  { id: 'cat_smug',    label: 'Smug :3',     category: 'smug',    description: 'Cat-mouth :3 smirk', new: true },
  { id: 'anime_sweat', label: 'Sweatdrop',   category: 'nervous', description: 'Awkward anime sweatdrop', new: true },
  { id: 'anger_vein',  label: 'Anger Vein',  category: 'angry',   description: 'Throbbing anger vein, tsundere huff', new: true },
  { id: 'vibe_music',  label: 'Vibing',      category: 'music',   description: 'Head-bob vibing with floating music notes', new: true },
];

const BUILTIN = [
  { id: 'excited_hop',   label: 'Excited Hop',   category: 'joy' },
  { id: 'shy_blush',     label: 'Shy Blush',     category: 'shy' },
  { id: 'wink_wave',     label: 'Wink Wave',     category: 'greeting' },
  { id: 'gloom_shake',   label: 'Gloom Shake',   category: 'sad' },
  { id: 'curious_lean',  label: 'Curious Lean',  category: 'curious' },
  { id: 'chibi_pop',     label: 'Chibi Pop',     category: 'joy' },
  { id: 'heart_sparkle', label: 'Heart Sparkle', category: 'love' },
  { id: 'greet_wave',    label: 'Greet Wave',    category: 'greeting' },
  { id: 'cheer_bounce',  label: 'Cheer Bounce',  category: 'joy' },
  { id: 'smug_tease',    label: 'Smug Tease',    category: 'smug' },
  { id: 'starry_awe',    label: 'Starry Awe',    category: 'awe' },
  { id: 'flustered_shy', label: 'Flustered Shy', category: 'shy' },
  { id: 'sleepy_yawn',   label: 'Sleepy Yawn',   category: 'sleepy' },
  { id: 'gentle_nod',    label: 'Gentle Nod',    category: 'caring' },
  ...FX_EMOTES,
];

/** Server list first (its order, labels, durations), then any FX emote it does not know yet. */
const withFx = (list) => {
  const have = new Set(list.map((e) => String(e.id)));
  return [...list, ...FX_EMOTES.filter((e) => !have.has(e.id))];
};

const decorate = (e) => ({
  id: String(e.id),
  label: e.label || String(e.id).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  category: e.category || '',
  description: e.description || e.desc || '',
  duration: Number(e.duration) || 0,
  isNew: Boolean(e.new),
  glyph: ID_GLYPHS[e.id] || GLYPHS[e.category] || '•',
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
        EMOTES = withFx(list).map(decorate);
        emotesSource = 'server';
      }
    } catch { /* older server — keep the built-in catalog */ }
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
