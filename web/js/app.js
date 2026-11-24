/** Hikari Studio — client bootstrap: stage renderers, Codex-style Settings modal, quick emotes,
 *  head-pat/poke interaction, built-in Fish Drama 3 / Live 3 call, Fish ASR push-to-talk & 60 FPS WebSocket bridge. */
import { AvatarGLRenderer, ProceduralRoomRenderer } from './stage.js';
import { FishCallManager } from './fish_call.js';
import { renderStagePanel, renderPresetsSection } from './panels/stage_panel.js';
import { renderShaderPanel } from './panels/shader_panel.js';
import { renderLipSyncPanel } from './panels/lipsync_panel.js';
import { renderRigPanel, renderParamInspector } from './panels/rig_panel.js';
import { renderCustomizerPanel } from './panels/customizer_panel.js';
import { renderMotionPanel } from './panels/motion_panel.js';
import { renderAiPanel } from './panels/ai_panel.js';
import { renderPluginsPanel } from './panels/plugins_panel.js';
import { renderMemoryPanel } from './panels/memory_panel.js';
import { EMOTES, emoteGrid, firePill, loadEmotes } from './panels/emotes.js';
import { extractSearchEntries, findSearchTarget, searchSettings } from './ui.js';

const TABS = {
  general:  { title: 'General & Stage',    sub: 'Room, OBS output, atmosphere, framing and presets.',
              render: (s) => [...renderStagePanel(s), renderPresetsSection(s)] },
  look:     { title: 'Look & Props',       sub: 'Puppet library, eyes, hair, outfit, shoes, glow and proportions.',
              render: (s) => renderCustomizerPanel(s) },
  motion:   { title: 'Motion & Emotions',  sub: 'Emotes, expressions, physics, head-pats, arms, gaze and Motion Lab.',
              render: (s) => [...renderRigPanel(s), ...renderMotionPanel(s)] },
  voice:    { title: 'Voice & Brain',      sub: 'Brain & voice mode, gateway pinning, live call, speech and personality.',
              render: (s) => renderAiPanel(s) },
  memory:   { title: 'Memory & Context',   sub: 'Conversation continuity across restarts, saved memories, notes and chat history.',
              render: (s) => renderMemoryPanel(s) },
  lipsync:  { title: 'Lip-Sync',           sub: 'Microphone, mouth physics and the Viseme Studio.',
              render: (s) => renderLipSyncPanel(s) },
  advanced: { title: 'Advanced',           sub: 'Shader & lighting, plugins and the raw parameter inspector.',
              render: (s) => [...renderShaderPanel(s), ...renderPluginsPanel(s), renderParamInspector(s)] },
};

const ROOM_TITLES = {
  twilight_atelier: 'Twilight Atelier',
  midnight_loft: 'Midnight Loft',
  sakura_pavilion: 'Sakura Pavilion',
  obsidian_studio: 'Obsidian Studio',
  chroma_green: 'OBS Chroma Green',
  transparent: 'OBS Transparent',
  custom: 'Custom Room',
};

const TAB_KEY = 'hikari.settings.tab';
const loadTab = () => { try { const t = localStorage.getItem(TAB_KEY); return TABS[t] ? t : 'general'; } catch { return 'general'; } };
const saveTab = (t) => { try { localStorage.setItem(TAB_KEY, t); } catch { /* storage blocked */ } };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

const settingsRoot = document.getElementById('settings');

const store = {
  config: null,
  shaderPresets: {},
  expressionPresets: [],
  roomAmbient: {},
  paramsMeta: [],
  plugins: [],
  hooks: {},
  motions: [],
  motionStatus: {},
  idleClip: {},
  fishStatus: {},
  fishCall: null,
  telemetry: { viseme: 'X', bands: [] },
  activeTab: loadTab(),
  ws: null,

  sendWs(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
  },

  patch(section, values) {
    if (!this.config[section]) this.config[section] = {};
    Object.assign(this.config[section], values);
    this.sendWs({ type: 'patch', section, values });
    // Sections the WS bridge may not know yet (e.g. "memory") also go through POST /api/config, debounced.
    if (section === 'memory') {
      this._restPending = { ...(this._restPending || {}), ...values };
      clearTimeout(this._restTimer);
      this._restTimer = setTimeout(() => {
        const vals = this._restPending;
        this._restPending = null;
        fetch('/api/config', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ memory: vals, save: false }),
        }).catch(() => {});
      }, 350);
    }
    if (section === 'stage' && values.room) this.setRoomTitle(values.room);
  },

  setRoomTitle(room) {
    document.getElementById('hud-room-name').textContent = ROOM_TITLES[room] || room || 'Studio';
  },

  /** Subtle top-bar badge naming the puppet on stage — hidden while Hikari herself is active. */
  setPuppetBadge(id, name) {
    const badge = document.getElementById('hud-puppet');
    if (!badge) return;
    const other = Boolean(id) && id !== 'hikari';
    badge.hidden = !other;
    badge.textContent = other ? (name || id) : '';
    badge.title = other ? `Puppet on stage: ${name || id}` : '';
  },

  async refreshState() {
    const st = await (await fetch('/api/state')).json();
    this.config = st.config;
    this.shaderPresets = st.shaderPresets || {};
    this.expressionPresets = st.expressionPresets || [];
    this.eyeStylePresets = st.eyeStylePresets || {};
    this.roomAmbient = st.roomAmbient || {};
    this.paramsMeta = st.paramsMeta || [];
    this.plugins = st.plugins || [];
    this.hooks = st.hooks || {};
    this.motions = st.motions || [];
    this.motionStatus = st.motionStatus || {};
    this.idleClip = st.idleClip || {};
    this.fishStatus = st.fishStatus || {};
    this.setRoomTitle(this.config.stage.room);
    this.setPuppetBadge(this.config.stage.activeModel || 'hikari', this.config.stage.activeModelName);
  },

  async playSample() {
    await fetch('/api/audio/sample', { method: 'POST' });
  },

  async toggleFishCall() {
    if (!this.fishCall) return;
    if (this.fishCall.connected || this.fishCall.connecting) {
      await this.fishCall.disconnect();
    } else {
      const mode = this.config?.ai?.fishAgentMode || 'drama-3';
      await this.fishCall.connect(mode, true);
    }
  },

  /** Re-render the active settings page, keeping scroll position. Skipped while the modal is closed. */
  rerenderPanel(animate = false) {
    if (settingsRoot.hidden) return;
    const p = TABS[this.activeTab] || TABS.general;
    document.getElementById('inspector-title').textContent = p.title;
    document.getElementById('inspector-sub').textContent = p.sub;
    document.querySelectorAll('.nav-item[data-tab]').forEach((b) => {
      const on = b.dataset.tab === this.activeTab;
      b.classList.toggle('active', on);
      b.setAttribute('aria-current', on ? 'page' : 'false');
    });
    const body = document.getElementById('inspector-body');
    const scroll = animate ? 0 : body.scrollTop;
    body.classList.remove('page-enter');
    body.replaceChildren(...p.render(this).filter(Boolean));
    body.scrollTop = scroll;
    if (animate) {
      void body.offsetWidth;
      body.classList.add('page-enter');
    }
  },
};

async function boot() {
  await store.refreshState();

  const roomCanvas = document.getElementById('room-canvas');
  const customImg = document.getElementById('custom-room-img');
  const avatarCanvas = document.getElementById('avatar-canvas');
  const roomRenderer = new ProceduralRoomRenderer(roomCanvas, customImg);
  const avatarRenderer = new AvatarGLRenderer(avatarCanvas);

  const app = document.getElementById('app');
  const audioEl = document.getElementById('voice-audio');
  const subBar = document.getElementById('subtitle-bar');
  const subTag = document.getElementById('subtitle-tag');
  const subTxt = document.getElementById('subtitle-text');
  const btnInterrupt = document.getElementById('btn-interrupt');
  const btnFishCall = document.getElementById('btn-fish-call');
  const btnMic = document.getElementById('btn-mic');
  const btnEmotes = document.getElementById('btn-emotes');
  const btnSettings = document.getElementById('btn-settings');
  const emotePop = document.getElementById('emote-popover');
  const connPill = document.getElementById('hud-conn');
  const connLabel = document.getElementById('hud-conn-label');
  const fxLayer = document.getElementById('stage-fx');

  let ptrX = 0, ptrY = 0;
  let draggingAvatar = false, dragLastX = 0, dragLastY = 0;
  let subTimeout = null;
  let wsOnline = false;
  let fishUserPost = Promise.resolve(); // last viewer-transcript POST (agent turns wait on it)

  function showSubtitle(tag, text, durationSec = 5.5) {
    if (!text) return;
    subTag.textContent = tag || 'hikari';
    subTxt.textContent = text;
    subBar.hidden = false;
    clearTimeout(subTimeout);
    subTimeout = setTimeout(() => { subBar.hidden = true; }, Math.max(3800, durationSec * 1000));
  }

  /* ───────── Status pill (top-right) ───────── */
  function updateStatus() {
    const fc = store.fishCall;
    let state = 'offline', label = 'Reconnecting…';
    if (fc?.connecting) { state = 'connecting'; label = 'Connecting call…'; }
    else if (fc?.connected) { state = 'call'; label = `${fc.mode === 'drama-3' ? 'Drama 3' : 'Live 3'} · ${fc.agentMode}`; }
    else if (wsOnline) { state = 'online'; label = 'Live'; }
    connPill.dataset.state = state;
    connLabel.textContent = label;
  }

  /* ───────── Built-in full-duplex Fish Drama 3 / Live 3 ───────── */
  store.fishCall = new FishCallManager({
    onState: (st) => {
      btnFishCall.classList.toggle('connecting', Boolean(st.connecting));
      btnFishCall.classList.toggle('live', Boolean(st.connected && !st.connecting));
      if (st.connecting) {
        btnFishCall.title = 'Connecting… (click to cancel)';
      } else if (st.connected) {
        const lbl = st.mode === 'drama-3' ? 'Drama 3' : 'Live 3';
        btnFishCall.title = `End ${lbl} call (${st.agentMode})`;
        btnInterrupt.hidden = false;
        btnMic.classList.toggle('active', !st.micMuted);
      } else {
        const mode = store.config?.ai?.fishAgentMode || 'drama-3';
        btnFishCall.title = mode === 'drama-3' ? 'Connect Drama 3 voice call' : 'Connect Live 3 voice call';
        btnInterrupt.hidden = audioEl.paused;
        btnMic.classList.remove('active');
      }
      updateStatus();
      if (store.activeTab === 'voice') store.rerenderPanel();
    },
    onMessage: async ({ role, text }) => {
      if (!text) return;
      if (role === 'agent') {
        // Let the viewer's line land first so the backend pairs both sides into one saved turn (bounded wait).
        await Promise.race([fishUserPost, new Promise((r) => setTimeout(r, 1500))]);
        const r = await fetch('/api/fish/emote', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: 'agent', text }),
        }).then((x) => x.json()).catch(() => ({ tag: 'drama-3', clean: text }));
        showSubtitle(r.tag || 'drama-3', r.clean || text, 7.0);
      } else {
        showSubtitle('you', text, 3.5);
        // Fire-and-forget: viewer emotion/wave decisions + pending viewer turn for conversation memory.
        fishUserPost = fetch('/api/fish/emote', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ role: 'user', text }),
        }).then(() => {}, () => {});
      }
    },
    onError: (err) => showSubtitle('fish error', err, 6.0),
  });

  /* ───────── Settings modal ───────── */
  let lastFocus = null;
  let closeTimer = null;
  function openSettings(tab) {
    closeEmotes();
    if (tab && TABS[tab]) { store.activeTab = tab; saveTab(tab); }
    searchIndex = null;
    clearTimeout(closeTimer);
    lastFocus = document.activeElement;
    settingsRoot.hidden = false;
    store.rerenderPanel(true);
    requestAnimationFrame(() => {
      settingsRoot.classList.add('open');
      app.classList.add('settings-open'); // slides the stage so Hikari stays centred in the open area
    });
    btnSettings.classList.add('active');
    settingsRoot.querySelector('.modal')?.focus({ preventScroll: true });
  }
  function closeSettings() {
    if (settingsRoot.hidden) return;
    settingsRoot.classList.remove('open');
    app.classList.remove('settings-open');
    closeSearchList();
    btnSettings.classList.remove('active');
    closeTimer = setTimeout(() => { settingsRoot.hidden = true; }, 360);
    lastFocus?.focus?.({ preventScroll: true });
  }
  btnSettings.addEventListener('click', () => (settingsRoot.hidden ? openSettings() : closeSettings()));
  settingsRoot.querySelectorAll('[data-close]').forEach((n) => n.addEventListener('click', closeSettings));
  settingsRoot.querySelectorAll('.nav-item[data-tab]').forEach((b) => b.addEventListener('click', () => {
    if (store.activeTab === b.dataset.tab) return;
    store.activeTab = b.dataset.tab;
    saveTab(store.activeTab);
    store.rerenderPanel(true);
  }));
  document.getElementById('btn-save-settings').addEventListener('click', (e) => {
    store.sendWs({ type: 'save' });
    const b = e.currentTarget;
    b.textContent = 'Saved ✓';
    b.classList.add('saved');
    setTimeout(() => { b.textContent = 'Save settings'; b.classList.remove('saved'); }, 1400);
  });

  /* ───────── Settings search (all tabs, incl. collapsed sections) ───────── */
  const searchIn = document.getElementById('settings-search');
  const searchClear = document.getElementById('settings-search-clear');
  const searchList = document.getElementById('settings-search-results');
  const searchBox = searchIn.closest('.side-search');
  let searchIndex = null;
  let searchIndexAt = 0;
  let searchResults = [];
  let searchActive = -1;

  // Render every tab off-screen and index its headings, rows, hints and control labels.
  // Built lazily on first keystroke; rebuilt at most every 3 s so it tracks live settings.
  function buildSearchIndex() {
    if (searchIndex && Date.now() - searchIndexAt < 3000) return searchIndex;
    const all = [];
    for (const [key, tab] of Object.entries(TABS)) {
      try {
        const box = document.createElement('div');
        box.append(...tab.render(store).filter(Boolean));
        all.push(...extractSearchEntries(box, key, tab.title));
      } catch (err) {
        console.warn('Settings search could not index', key, err);
      }
    }
    searchIndex = all;
    searchIndexAt = Date.now();
    return all;
  }

  // Wrap typed words found in the title with <mark> (built with text nodes — no HTML injection).
  function highlightTitle(title, query) {
    const words = query.toLowerCase().split(/\s+/).filter((w) => w.length > 1);
    const lower = title.toLowerCase();
    const spans = [];
    for (const w of words) {
      let i = lower.indexOf(w);
      while (i >= 0) { spans.push([i, i + w.length]); i = lower.indexOf(w, i + w.length); }
    }
    spans.sort((a, b) => a[0] - b[0]);
    const out = [];
    let pos = 0;
    for (const [a, b] of spans) {
      if (a < pos) continue;
      if (a > pos) out.push(document.createTextNode(title.slice(pos, a)));
      const m = document.createElement('mark');
      m.textContent = title.slice(a, b);
      out.push(m);
      pos = b;
    }
    if (pos < title.length) out.push(document.createTextNode(title.slice(pos)));
    return out;
  }

  function setSearchActive(i) {
    searchActive = i;
    [...searchList.querySelectorAll('.ssr')].forEach((b, j) => {
      b.classList.toggle('active', j === i);
      b.setAttribute('aria-selected', String(j === i));
      if (j === i) {
        b.scrollIntoView({ block: 'nearest' });
        searchIn.setAttribute('aria-activedescendant', b.id);
      }
    });
  }

  function closeSearchList() {
    searchList.hidden = true;
    searchIn.setAttribute('aria-expanded', 'false');
    searchIn.removeAttribute('aria-activedescendant');
  }

  function renderSearchResults() {
    const q = searchIn.value.trim();
    searchClear.hidden = !q;
    searchBox.classList.toggle('has-value', Boolean(q));
    if (!q) { closeSearchList(); return; }
    searchResults = searchSettings(buildSearchIndex(), q, 10);
    const items = searchResults.map((r, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'ssr';
      b.id = `ssr-${i}`;
      b.setAttribute('role', 'option');
      const t = document.createElement('span');
      t.className = 'ssr-title';
      t.append(...highlightTitle(r.title, q));
      const meta = document.createElement('span');
      meta.className = 'ssr-meta';
      const where = [r.tabTitle];
      if (r.kind !== 'group' && r.group) where.push(r.group);
      if (r.parent) where.push(r.parent);
      meta.textContent = where.join(' · ');
      b.append(t, meta);
      b.addEventListener('mousedown', (e) => e.preventDefault()); // keep focus in the input
      b.addEventListener('mousemove', () => { if (searchActive !== i) setSearchActive(i); });
      b.addEventListener('click', () => jumpToSetting(r));
      return b;
    });
    if (!items.length) {
      const none = document.createElement('div');
      none.className = 'ssr-empty';
      none.textContent = `No settings match “${q}”`;
      items.push(none);
    }
    searchList.replaceChildren(...items);
    searchList.hidden = false;
    searchIn.setAttribute('aria-expanded', 'true');
    setSearchActive(searchResults.length ? 0 : -1);
  }

  function jumpToSetting(r) {
    closeSearchList();
    if (store.activeTab !== r.tab) {
      store.activeTab = r.tab;
      saveTab(r.tab);
      store.rerenderPanel(true);
    }
    const body = document.getElementById('inspector-body');
    const target = findSearchTarget(body, r);
    if (!target) return;
    // Open every collapsed disclosure around the target (and the target's own, for section hits).
    const toOpen = [];
    for (let m = target.parentElement?.closest('.s-more'); m; m = m.parentElement?.closest('.s-more')) toOpen.push(m);
    if (target.classList.contains('s-more-head')) toOpen.push(target.closest('.s-more'));
    toOpen.reverse().forEach((m) => {
      if (m && !m.classList.contains('open')) m.querySelector(':scope > .s-more-head')?.click();
    });
    const flash = target.matches('.s-group') ? (target.querySelector('.s-card') || target) : target;
    requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: 'smooth', block: target.matches('.s-group') ? 'start' : 'center' });
      flash.classList.remove('search-flash');
      void flash.offsetWidth; // restart the animation
      flash.classList.add('search-flash');
      setTimeout(() => flash.classList.remove('search-flash'), 2000);
    });
  }

  function focusSearch() {
    if (settingsRoot.hidden) openSettings();
    requestAnimationFrame(() => { searchIn.focus(); searchIn.select(); });
  }

  searchIn.addEventListener('input', renderSearchResults);
  searchIn.addEventListener('focus', () => { if (searchIn.value.trim()) renderSearchResults(); });
  searchIn.addEventListener('blur', () => closeSearchList());
  searchIn.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (searchList.hidden) { renderSearchResults(); return; }
      e.preventDefault();
      if (!searchResults.length) return;
      const n = searchResults.length;
      setSearchActive((searchActive + (e.key === 'ArrowDown' ? 1 : -1) + n) % n);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const r = searchResults[searchActive];
      if (r && !searchList.hidden) jumpToSetting(r);
    } else if (e.key === 'Escape') {
      if (searchIn.value) {           // first Esc clears; an empty field lets Esc close Settings
        e.preventDefault();
        e.stopPropagation();
        searchIn.value = '';
        renderSearchResults();
      }
    }
  });
  searchClear.addEventListener('click', () => {
    searchIn.value = '';
    renderSearchResults();
    searchIn.focus();
  });

  /* ───────── Quick-emote popover ───────── */
  const buildEmotePopover = () =>
    document.getElementById('emote-popover-grid').replaceChildren(emoteGrid({ compact: true, showKeys: true }));
  buildEmotePopover();
  loadEmotes().then(() => {
    buildEmotePopover();
    if (store.activeTab === 'motion') store.rerenderPanel();
  });
  function openEmotes() {
    emotePop.hidden = false;
    btnEmotes.classList.add('active');
    btnEmotes.setAttribute('aria-expanded', 'true');
  }
  function closeEmotes() {
    emotePop.hidden = true;
    btnEmotes.classList.remove('active');
    btnEmotes.setAttribute('aria-expanded', 'false');
  }
  btnEmotes.addEventListener('click', (e) => { e.stopPropagation(); emotePop.hidden ? openEmotes() : closeEmotes(); });
  document.addEventListener('pointerdown', (e) => {
    if (!emotePop.hidden && !emotePop.contains(e.target) && !btnEmotes.contains(e.target)) closeEmotes();
  });
  document.getElementById('btn-quick-wave').addEventListener('click', () => {
    fetch('/api/hook/trigger', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'wave' }) });
  });
  document.getElementById('btn-pop-motion').addEventListener('click', () => openSettings('motion'));

  /* ───────── Global keys ───────── */
  window.addEventListener('keydown', (e) => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
    if (e.key === 'Escape') {
      if (!emotePop.hidden) { closeEmotes(); return; }
      if (!settingsRoot.hidden) { closeSettings(); return; }
    }
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      focusSearch();
      return;
    }
    if (e.key === '/' && !typing && !settingsRoot.hidden && !e.ctrlKey && !e.metaKey && !e.altKey) {
      e.preventDefault();
      focusSearch();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key === ',') {
      e.preventDefault();
      settingsRoot.hidden ? openSettings() : closeSettings();
      return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return; // emote hotkeys work with Settings open too
    if (e.key === 'e' || e.key === 'E') { emotePop.hidden ? openEmotes() : closeEmotes(); return; }
    if (!emotePop.hidden && /^[1-9]$/.test(e.key) && EMOTES[Number(e.key) - 1]) {
      const id = EMOTES[Number(e.key) - 1].id;
      const pill = emotePop.querySelector(`[data-emote="${id}"]`);
      if (pill) firePill(pill, id);
    }
  });

  /* ───────── Room animation loop (60 FPS procedural environment & particles) ───────── */
  function animRoom(now) {
    roomRenderer.render(now * 0.001, store.config.stage, ptrX, ptrY);
    requestAnimationFrame(animRoom);
  }
  requestAnimationFrame(animRoom);

  /* ───────── Stage interaction: look-at, Shift/right-drag move, wheel scale, head-pat & poke ───────── */
  // On-screen rect of the avatar quad — mirrors AvatarGLRenderer.drawPacket's NDC placement.
  function avatarRect() {
    const cfg = store.config.stage || {};
    const W = window.innerWidth, H = window.innerHeight;
    const aspect = avatarRenderer._aspect || 0.75;
    const sy = Number(cfg.avatarScale ?? 1.0);
    const sx = (sy * aspect) / (W / H);
    const ox = (Number(cfg.avatarX ?? 0.5) * 2 - 1) - ptrX * 0.012;
    const oy = -1.0 + sy + Number(cfg.avatarY ?? 0) * 2.0;
    const left = ((ox - sx + 1) / 2) * W, right = ((ox + sx + 1) / 2) * W;
    const top = ((1 - (oy + sy)) / 2) * H, bottom = ((1 - (oy - sy)) / 2) * H;
    return { left, top, w: right - left, h: bottom - top, cx: (left + right) / 2 };
  }
  // Client → canvas-local coordinates. The stage slides right while Settings is open, so always measure
  // against the canvas's live on-screen rect (also correct mid-transition).
  function toLocal(clientX, clientY) {
    const b = avatarCanvas.getBoundingClientRect();
    return { x: clientX - b.left, y: clientY - b.top, w: b.width || window.innerWidth, h: b.height || window.innerHeight };
  }
  // nx: -1 (left) … 0 (centre) … 1 (right) across the avatar; ny: 0 (top) … 1 (bottom) of the avatar.
  function hitTest(clientX, clientY) {
    const r = avatarRect();
    const p = toLocal(clientX, clientY);
    const nx = (p.x - r.cx) / (r.w / 2);
    const ny = (p.y - r.top) / r.h;
    const headFrac = clamp(0.2 * Number(store.config.camera?.scale ?? 1.82), 0.14, 0.5);
    return {
      nx, ny,
      onBody: Math.abs(nx) < 0.62 && ny >= -0.02 && ny <= 1,
      onHead: Math.abs(nx) < 0.5 && ny >= -0.02 && ny <= headFrac,
    };
  }
  const patEnabled = () => store.config.rig?.headPatEnabled ?? true;

  function spawnFx(x, y, glyph, cls = '') {
    const n = document.createElement('span');
    n.className = `fx-pop ${cls}`.trim();
    n.textContent = glyph;
    const b = fxLayer.getBoundingClientRect(); // fx layer moves with the stage
    n.style.left = `${x - b.left}px`;
    n.style.top = `${y - b.top}px`;
    n.style.setProperty('--dx', `${(Math.random() - 0.5) * 40}px`);
    fxLayer.appendChild(n);
    setTimeout(() => n.remove(), 950);
  }

  const PAT_INTERVAL_MS = 50; // ~20 Hz
  let down = null;            // { x, y, t, moved }
  let patLast = { t: 0, x: 0, y: 0 };
  let fxLast = 0;

  avatarCanvas.addEventListener('pointermove', (e) => {
    const lp = toLocal(e.clientX, e.clientY);
    ptrX = (lp.x / lp.w) * 2 - 1;
    ptrY = (lp.y / lp.h) * 2 - 1;
    store.sendWs({ type: 'pointer', x: ptrX, y: ptrY });

    if (draggingAvatar) {
      const dx = (e.clientX - dragLastX) / window.innerWidth;
      const dy = (e.clientY - dragLastY) / window.innerHeight;
      dragLastX = e.clientX;
      dragLastY = e.clientY;
      store.patch('stage', {
        avatarX: clamp(store.config.stage.avatarX + dx, 0.1, 0.9),
        avatarY: clamp(store.config.stage.avatarY - dy, -0.25, 0.45),
      });
      return;
    }

    const hit = hitTest(e.clientX, e.clientY);
    const enabled = patEnabled();
    avatarCanvas.classList.toggle('pat-zone', enabled && hit.onHead);
    avatarCanvas.classList.toggle('poke-zone', enabled && hit.onBody && !hit.onHead);

    if (!down || !(e.buttons & 1)) return;
    if (!down.moved && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) down.moved = true;
    if (!down.moved || !enabled || !hit.onHead) return;

    avatarCanvas.classList.add('patting');
    const now = performance.now();
    if (now - patLast.t < PAT_INTERVAL_MS) return;
    const dt = Math.max(1, now - patLast.t) / 1000;
    const r = avatarRect();
    const dist = patLast.t ? Math.hypot(e.clientX - patLast.x, e.clientY - patLast.y) : 0;
    const speed = clamp((dist / dt) / Math.max(120, r.w * 1.2), 0.35, 2.0);
    store.sendWs({ type: 'interact', kind: 'pat', x: +hit.nx.toFixed(3), y: +hit.ny.toFixed(3), speed: +speed.toFixed(2) });
    patLast = { t: now, x: e.clientX, y: e.clientY };
    if (now - fxLast > 220) { fxLast = now; spawnFx(e.clientX, e.clientY - 10, '♥'); }
  });

  avatarCanvas.addEventListener('pointerdown', (e) => {
    closeEmotes();
    if (e.shiftKey || e.button === 2) {
      draggingAvatar = true;
      avatarCanvas.classList.add('moving');
      dragLastX = e.clientX;
      dragLastY = e.clientY;
    } else if (e.button === 0) {
      down = { x: e.clientX, y: e.clientY, t: performance.now(), moved: false };
      patLast = { t: 0, x: e.clientX, y: e.clientY };
    }
  });

  window.addEventListener('pointerup', (e) => {
    if (down && !draggingAvatar && !down.moved && performance.now() - down.t < 300 && patEnabled()) {
      const hit = hitTest(e.clientX, e.clientY);
      if (hit.onBody) {
        store.sendWs({ type: 'interact', kind: 'poke', x: +hit.nx.toFixed(3), y: +hit.ny.toFixed(3) });
        spawnFx(e.clientX, e.clientY, '!', 'poke');
      }
    }
    down = null;
    draggingAvatar = false;
    avatarCanvas.classList.remove('moving', 'patting');
  });
  avatarCanvas.addEventListener('pointerleave', () => avatarCanvas.classList.remove('pat-zone', 'poke-zone'));
  avatarCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
  avatarCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const next = clamp(store.config.stage.avatarScale - e.deltaY * 0.0007, 0.5, 1.65);
    store.patch('stage', { avatarScale: next });
  }, { passive: false });

  /* ───────── Synchronized voice audio playback + interrupt ───────── */
  audioEl.addEventListener('timeupdate', () => {
    if (!audioEl.paused) store.sendWs({ type: 'audio_sync', time: audioEl.currentTime });
  });
  audioEl.addEventListener('ended', () => {
    if (!store.fishCall.connected) btnInterrupt.hidden = true;
  });

  function showSpeech(msg) {
    if (msg.text) showSubtitle(msg.tag || 'hikari', msg.text, (msg.duration || 4) + 2);
    if (msg.audioUrl) {
      audioEl.src = msg.audioUrl;
      audioEl.currentTime = 0;
      audioEl.play().catch(() => {});
      btnInterrupt.hidden = false;
    }
  }

  /* ───────── WebSocket bridge with auto-reconnect ───────── */
  function connectWs() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    store.ws = ws;
    ws.onopen = () => { wsOnline = true; updateStatus(); };
    ws.onmessage = async (ev) => {
      if (typeof ev.data !== 'string') {
        await avatarRenderer.drawPacket(ev.data, store.config.stage, ptrX, ptrY);
        return;
      }
      const msg = JSON.parse(ev.data);
      if (msg.type === 'telemetry') {
        document.getElementById('hud-fps').textContent = `${msg.fps} FPS`;
        document.getElementById('hud-viseme').textContent = `Viseme: ${msg.viseme}`;
        store.telemetry = { viseme: msg.viseme, bands: msg.bands || [] };
        store.hooks = msg.hooks || store.hooks;
        store.motionStatus = msg.motion || store.motionStatus;
        const vu = document.getElementById('vu-bars');
        if (vu && msg.bands) {
          const bars = vu.children;
          for (let i = 0; i < Math.min(bars.length, msg.bands.length); i++) {
            bars[i].style.height = `${Math.max(3, Math.round(msg.bands[i] * 28))}px`;
          }
        }
      } else if (msg.type === 'speech') {
        showSpeech(msg);
      } else if (msg.type === 'interrupt') {
        audioEl.pause();
        if (!store.fishCall.connected) btnInterrupt.hidden = true;
      } else if (msg.type === 'config_sync') {
        // Keep locally-set sections an older server doesn't know about yet (e.g. memory).
        if (store.config?.memory && msg.config && !msg.config.memory) msg.config.memory = store.config.memory;
        store.config = msg.config;
      }
    };
    ws.onclose = () => { wsOnline = false; updateStatus(); setTimeout(connectWs, 1000); };
  }
  connectWs();

  /* ───────── Bottom bar: call, mic (push-to-talk / call mute), chat, interrupt ───────── */
  btnFishCall.addEventListener('click', () => store.toggleFishCall());

  let pttRecorder = null;
  let pttChunks = [];
  let pttStream = null;

  async function sendChatText(message) {
    if (!message) return;
    if (store.fishCall.connected) {
      store.fishCall.sendMessage(message);
      showSubtitle('you', message, 3.5);
      return;
    }
    showSubtitle('thinking', '…', 10.0);
    await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
  }

  // In a live call → toggles WebRTC mic mute. Outside a call → Fish ASR push-to-talk that auto-sends chat.
  btnMic.addEventListener('click', async () => {
    if (store.fishCall.connected) {
      await store.fishCall.toggleMic();
      return;
    }
    if (pttRecorder && pttRecorder.state === 'recording') {
      pttRecorder.stop();
      btnMic.classList.remove('active');
      showSubtitle('fish asr', 'Transcribing…', 6.0);
      return;
    }
    try {
      pttStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      pttChunks = [];
      pttRecorder = new MediaRecorder(pttStream);
      pttRecorder.ondataavailable = (e) => { if (e.data.size > 0) pttChunks.push(e.data); };
      pttRecorder.onstop = async () => {
        pttStream?.getTracks().forEach((t) => t.stop());
        pttStream = null;
        const blob = new Blob(pttChunks, { type: pttRecorder.mimeType || 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'mic.webm');
        const r = await (await fetch('/api/stt', { method: 'POST', body: fd })).json();
        if (r.ok && r.text) {
          showSubtitle('you', r.text, 3.5);
          await sendChatText(r.text);
        } else {
          showSubtitle('stt', r.error || 'No speech detected.', 3.5);
        }
      };
      pttRecorder.start();
      btnMic.classList.add('active');
      showSubtitle('listening', 'Speak now — click the mic again to send via Fish ASR…', 15.0);
    } catch (err) {
      showSubtitle('mic error', err.message || 'Microphone access denied.', 4.0);
    }
  });

  btnInterrupt.addEventListener('click', async () => {
    if (store.fishCall.connected) store.fishCall.interrupt();
    await fetch('/api/interrupt', { method: 'POST' });
    audioEl.pause();
    if (!store.fishCall.connected) btnInterrupt.hidden = true;
  });

  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;
    chatInput.value = '';
    await sendChatText(message);
  });

  updateStatus();

  // Deep link: /#settings or /#settings/<tab> opens the Settings modal (handy for Stream Deck / bookmarks).
  const openFromHash = () => {
    const m = location.hash.match(/^#settings(?:\/(\w+))?/);
    if (m) openSettings(m[1]);
    else if (location.hash === '#emotes') openEmotes();
  };
  window.addEventListener('hashchange', openFromHash);
  openFromHash();
}

boot();
