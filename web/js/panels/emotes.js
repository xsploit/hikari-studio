/** Tweened emotes — loaded from GET /api/emotes (falls back to the built-in catalog on older servers;
 *  the 7 Expression-FX emotes are always included).
 *  Fired with POST /api/emote. Shared by the Motion & Emotions tab and the floating popover.
 *  Also home of the shared Rig Abilities & Props client (GET /api/models/abilities · POST /api/models/ability),
 *  used by Look & Props → Puppet Library and the "Rig Abilities" strip inside the ✦ Quick Emote popover. */
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
      // Newer servers also list the active puppet's auto-discovered rig abilities — seed the cache with them.
      if (Array.isArray(data?.rigAbilities) && !abilities.fetchedAt) applyAbilities(data);
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

/* ───────────── Rig Abilities & Props (GET /api/models/abilities · POST /api/models/ability) ─────────────
 * One shared cache + playing state for every view. No polling: data is fetched when Look & Props renders
 * for a new puppet, after a puppet switch/import (`invalidateAbilities`), and whenever the ✦ popover opens.
 * Pills are patched in place (`syncAbilityUi`) so a playing sweep never forces a full panel re-render. */

const HIKARI_ID = 'hikari';
const POP_MAX_AGE = 3000;            // re-opening the popover within 3 s reuses the fresh copy
const KIND_GLYPH = { moment: '✦', prop: '◆', toggle: '◐', loop: '↻', aura: '✺' };

export const abilities = {
  list: [],
  activeModel: '',   // as reported by the server
  aiEnabled: false,
  forKey: null,      // client-side puppet id the cache was fetched for (null = refetch)
  fetchedAt: 0,
  loading: null,
  error: '',
  status: '',
  kind: '',          // '' | 'pending' | 'ok' | 'error'
};
const playing = new Map();           // id → { start, dur, timer }
const listeners = new Set();

const toNum = (v, d = 0) => (Number.isFinite(Number(v)) && Number(v) > 0 ? Number(v) : d);
const strList = (v) => (Array.isArray(v) ? v.map(String).filter(Boolean) : []);

const decorateAbility = (a) => ({
  id: String(a.id),
  label: a.label || String(a.id).replace(/^rig_/, '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  desc: a.desc || a.description || '',
  kind: String(a.kind || 'moment'),
  duration: toNum(a.duration, 3),
  params: strList(a.params),
  moods: strList(a.moods),
  active: Boolean(a.active),
});

function applyAbilities(data) {
  const raw = Array.isArray(data?.abilities) ? data.abilities : (Array.isArray(data?.rigAbilities) ? data.rigAbilities : []);
  abilities.list = raw.filter((a) => a && a.id).map(decorateAbility);
  abilities.activeModel = String(data?.activeModel || '');
  abilities.aiEnabled = Boolean(data?.aiEnabled);
  abilities.error = '';
  abilities.fetchedAt = Date.now();
  // Something already playing on stage (e.g. the AI fired it) lights up when you open the view.
  for (const a of abilities.list) if (a.active && !playing.has(a.id)) markPlaying(a.id, a.duration, false);
}

export const onAbilitiesChange = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const notify = () => listeners.forEach((fn) => { try { fn(abilities); } catch { /* view gone */ } });

/** The puppet on stage is a non-Hikari rig with discovered abilities (what the popover strip needs). */
export const hasRigAbilities = () => abilities.list.length > 0 && abilities.activeModel.toLowerCase() !== HIKARI_ID;

/**
 * Fetch abilities. `forKey` = client puppet id: cached until the puppet changes.
 * `maxAge` = accept a copy younger than this many ms (popover). Older servers (404) just give an empty list.
 */
export function loadAbilities({ forKey, maxAge } = {}) {
  if (abilities.loading) return abilities.loading;
  const fresh = maxAge !== undefined
    ? abilities.fetchedAt && Date.now() - abilities.fetchedAt < maxAge
    : abilities.fetchedAt && abilities.forKey !== null && (forKey === undefined || abilities.forKey === forKey);
  if (fresh) return Promise.resolve(abilities);
  if (forKey !== undefined) abilities.forKey = forKey;
  abilities.loading = fetch('/api/models/abilities')
    .then(async (r) => {
      const data = await r.json().catch(() => null);
      if (!r.ok || !data?.ok) throw new Error(data?.error || data?.detail || `HTTP ${r.status}`);
      applyAbilities(data);
    })
    .catch((error) => {
      abilities.list = [];
      abilities.error = error.message;
      abilities.fetchedAt = Date.now();
    })
    .finally(() => {
      abilities.loading = null;
      notify();
      syncAbilityUi();
    });
  return abilities.loading;
}

/** Puppet switched / imported / removed: forget the cache and anything that was lit up. */
export function invalidateAbilities() {
  abilities.forKey = null;
  abilities.fetchedAt = 0;
  abilities.status = '';
  abilities.kind = '';
  for (const p of playing.values()) clearTimeout(p.timer);
  playing.clear();
  syncAbilityUi();
}

const labelOf = (id) => abilities.list.find((a) => a.id === id)?.label || id;

function markPlaying(id, dur, sync = true) {
  const prev = playing.get(id);
  if (prev) clearTimeout(prev.timer);
  const entry = { start: Date.now(), dur };
  entry.timer = setTimeout(() => {           // one-shot "finished" timer, not a poll
    if (playing.get(id) !== entry) return;
    playing.delete(id);
    if (abilities.kind === 'ok' && abilities.status.startsWith(labelOf(id))) abilities.status = '';
    syncAbilityUi();
  }, dur * 1000);
  playing.set(id, entry);
  if (sync) syncAbilityUi();
}

export function setAbilityStatus(text, kind = '') {
  abilities.status = text;
  abilities.kind = kind;
  syncAbilityUi();
}

const postAbility = (body) => fetch('/api/models/ability', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
}).then(async (r) => {
  const res = await r.json().catch(() => null);
  if (!r.ok || !res?.ok) throw new Error(res?.error || res?.detail || `HTTP ${r.status}`);
  return res;
});

/** Trigger one ability. Lights its pill(s) pink for the returned duration. */
export async function fireAbility(id) {
  const label = labelOf(id);
  const a = abilities.list.find((x) => x.id === id);
  setAbilityStatus(`Casting ${label}…`, 'pending');
  try {
    const res = await postAbility({ id });
    if (res.playing === false) {
      setAbilityStatus(`${label} didn’t start on this rig.`, 'error');
      return { ok: false, duration: 0 };
    }
    const dur = toNum(res.duration, a?.duration || 3);
    markPlaying(id, dur, false);
    setAbilityStatus(`${label} playing on stage ✓ · ${dur.toFixed(1)} s`, 'ok');
    return { ok: true, duration: dur };
  } catch (error) {
    document.querySelectorAll('.ability-pill[data-ability]').forEach((p) => {
      if (p.dataset.ability !== id) return;
      p.classList.add('failed');
      setTimeout(() => p.classList.remove('failed'), 1200);
    });
    setAbilityStatus(`${label} failed: ${error.message}`, 'error');
    return { ok: false, duration: 0, error: error.message };
  }
}

export async function stopAbilities() {
  try {
    await postAbility({ stopAll: true });
    for (const p of playing.values()) clearTimeout(p.timer);
    playing.clear();
    setAbilityStatus('Stopped all rig abilities ✓', 'ok');
  } catch (error) {
    setAbilityStatus(`Stop failed: ${error.message}`, 'error');
  }
}

/** Patch every live pill / Stop button / status line in place. */
export function syncAbilityUi() {
  if (typeof document === 'undefined') return;
  document.querySelectorAll('.ability-pill[data-ability]').forEach((pill) => {
    const p = playing.get(pill.dataset.ability);
    if (!p) {
      pill.classList.remove('playing');
      pill.setAttribute('aria-pressed', 'false');
      delete pill.dataset.start;
      return;
    }
    if (pill.dataset.start !== String(p.start)) {   // (re)start the sweep from where this cast is now
      pill.classList.remove('playing');
      void pill.offsetWidth;
      const left = Math.max(0.2, p.dur - (Date.now() - p.start) / 1000);
      pill.style.setProperty('--dur', `${left}s`);
      pill.dataset.start = String(p.start);
    }
    pill.classList.add('playing');
    pill.setAttribute('aria-pressed', 'true');
  });
  const any = playing.size > 0;
  document.querySelectorAll('.ability-stop').forEach((b) => { b.hidden = !any; });
  document.querySelectorAll('.ability-status').forEach((s) => {
    s.textContent = abilities.status;
    s.className = `ability-status apply-status ${abilities.kind}`.trim();
    s.hidden = !abilities.status;
  });
  document.querySelectorAll('.abil-foot').forEach((f) => { f.hidden = !any && !abilities.status; });
}

function abilityTitle(a) {
  const lines = [a.desc ? `${a.label} — ${a.desc}` : a.label];
  if (a.moods.length) lines.push(`Moods: ${a.moods.join(', ')}`);
  if (abilities.aiEnabled) lines.push('The AI decision model can trigger this on its own');
  return lines.join('\n');
}

/** Pill row of the active rig's abilities. */
export function abilityGrid({ compact = false } = {}) {
  const grid = el('div', { class: `abil-grid${compact ? ' compact' : ''}`, role: 'group', 'aria-label': 'Rig abilities' },
    abilities.list.map((a) => el('button', {
      type: 'button', class: 'ability-pill', 'data-ability': a.id, 'aria-pressed': 'false', title: abilityTitle(a),
      onclick: () => fireAbility(a.id),
    }, [
      el('span', { class: 'ability-glyph', 'aria-hidden': 'true', text: KIND_GLYPH[a.kind] || '✧' }),
      el('span', { class: 'ability-label', text: a.label }),
    ])));
  queueMicrotask(syncAbilityUi);           // light up anything still playing once mounted
  return grid;
}

/** Status line + subtle Stop button (both hidden until there is something to say / stop). */
export function abilityFoot({ popover = false } = {}) {
  const status = el('div', { class: 'ability-status apply-status', role: 'status' });
  status.hidden = true;
  const stop = el('button', {
    type: 'button', class: popover ? 'pop-link ability-stop' : 'btn ability-stop', text: 'Stop',
    title: 'Stop every playing rig ability', onclick: () => stopAbilities(),
  });
  stop.hidden = true;
  const foot = el('div', { class: 'abil-foot' }, [status, stop]);
  foot.hidden = true;
  return foot;
}

/* ---- ✦ Quick Emote popover strip (self-mounting, right under #emote-popover-grid) ---- */

function paintPopover(host) {
  if (!hasRigAbilities()) { host.hidden = true; host.replaceChildren(); return; }
  host.hidden = false;
  host.replaceChildren(
    el('div', { class: 'pop-sub-head' }, [
      el('span', { text: 'Rig Abilities' }),
      el('span', { class: 'pop-sub-count', text: String(abilities.list.length) }),
      el('span', { class: 'grow' }),
      abilities.aiEnabled ? el('span', { class: 'ai-chip', text: 'AI ✓', title: 'The AI decision model can fire these on its own' }) : null,
    ]),
    abilityGrid({ compact: true }),
    abilityFoot({ popover: true }),
  );
  syncAbilityUi();
}

function mountPopoverAbilities() {
  const pop = document.getElementById('emote-popover');
  const grid = document.getElementById('emote-popover-grid');
  if (!pop || !grid || document.getElementById('emote-popover-abilities')) return;
  const host = el('div', { id: 'emote-popover-abilities', class: 'pop-abil', role: 'group', 'aria-label': 'Rig abilities' });
  host.hidden = true;
  grid.after(host);
  const paint = () => paintPopover(host);
  onAbilitiesChange(paint);
  // Refresh each time the picker opens (an attribute flip, not a timer) — covers puppet switches from anywhere.
  new MutationObserver(() => {
    if (pop.hidden) return;
    paint();
    loadAbilities({ maxAge: POP_MAX_AGE });
  }).observe(pop, { attributes: true, attributeFilter: ['hidden'] });
  if (!pop.hidden) loadAbilities({ maxAge: POP_MAX_AGE });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountPopoverAbilities, { once: true });
  else mountPopoverAbilities();
}
