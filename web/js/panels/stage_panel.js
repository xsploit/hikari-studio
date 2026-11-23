/** GENERAL & STAGE — room/background, OBS output, atmosphere, framing & camera, saved studio presets. */
import { btn, chips, chipsRow, el, group, more, noteRow, row, segRow, slider, toggle } from '../ui.js';

const ROOMS = [
  ['twilight_atelier', 'Twilight Atelier'],
  ['midnight_loft', 'Midnight Loft'],
  ['sakura_pavilion', 'Sakura Pavilion'],
  ['obsidian_studio', 'Obsidian Studio'],
];
const OBS_ROOMS = [['chroma_green', 'Chroma green'], ['transparent', 'Transparent']];

const PARTICLES = [
  ['auto', 'Auto'], ['embers', 'Embers'], ['rain', 'Neon Rain'], ['sakura', 'Sakura'], ['dust', 'Dust'], ['none', 'Off'],
];

const FRAMING = [
  ['bust',  'Bust',      { scale: 2.25, y: 420 }, { avatarScale: 1.0, avatarY: 0.0 }],
  ['waist', 'Waist-up',  { scale: 1.82, y: 355 }, { avatarScale: 1.0, avatarY: 0.0 }],
  ['full',  'Full body', { scale: 0.84, y: 0 },   { avatarScale: 0.96, avatarY: 0.02 }],
];

export function renderStagePanel(store) {
  const st = store.config.stage;
  const cam = store.config.camera;
  const ss = (label, key, min, max, step, def, suffix = '', desc = '') => slider({
    label, desc, min, max, step, value: st[key] ?? def, def, suffix, onInput: (v) => store.patch('stage', { [key]: v }),
  });

  const pickRoom = (id) => {
    if (id === 'custom' && !st.customRoomUrl) return;
    store.patch('stage', { room: id });
    if (st.autoMatchLighting && store.roomAmbient[id]) store.patch('shader', store.roomAmbient[id]);
    store.rerenderPanel();
  };

  const fileInput = el('input', {
    type: 'file', accept: 'image/png,image/jpeg,image/webp', hidden: true,
    onchange: async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append('file', f);
      const res = await (await fetch('/api/upload/room', { method: 'POST', body: fd })).json();
      if (res.ok) {
        store.config.stage.room = 'custom';
        store.config.stage.customRoomUrl = res.url;
        store.setRoomTitle?.('custom');
        store.rerenderPanel();
      }
    },
  });

  const curFrame = FRAMING.find(([, , c]) => Math.abs((cam.scale ?? 0) - c.scale) < 0.01 && Math.abs((cam.y ?? 0) - c.y) < 1)?.[0];
  const roomItems = st.customRoomUrl ? [...ROOMS, ['custom', 'Custom image']] : ROOMS;

  return [
    group('Room', [
      chipsRow({ label: 'Background', desc: 'Procedural animated rooms, drawn live', items: roomItems, active: st.room, onPick: pickRoom }),
      row({ title: 'Custom room', desc: 'Use your own PNG / JPG / WebP backdrop', control: [btn('Upload image…', () => fileInput.click()), fileInput] }),
      toggle({ label: 'Auto-match lighting', desc: 'Re-light Hikari’s rim & ambient to the room', checked: st.autoMatchLighting, onChange: (v) => store.patch('stage', { autoMatchLighting: v }) }),
    ]),

    group('OBS Output', [
      segRow({
        label: 'Broadcast background', desc: 'Keyable backgrounds for streaming',
        items: [['', 'Room'], ...OBS_ROOMS],
        active: OBS_ROOMS.some(([id]) => id === st.room) ? st.room : '',
        onPick: (id) => {
          if (id) pickRoom(id);
          else if (OBS_ROOMS.some(([r]) => r === st.room)) pickRoom('twilight_atelier');
        },
      }),
      row({
        title: 'OBS browser source', desc: 'Transparent avatar-only page — add /obs as a Browser Source',
        control: el('a', { class: 'btn', href: '/obs', target: '_blank', rel: 'noopener', text: 'Open /obs ↗' }),
      }),
    ]),

    group('Atmosphere', [
      row({ title: 'Particles', desc: 'Ambient effect layer', stack: true,
        control: chips(PARTICLES, { accent: true, active: st.particles, onPick: (id) => { store.patch('stage', { particles: id }); store.rerenderPanel(); } }) }),
      ss('Bokeh blur', 'roomBlur', 0, 14, 0.5, 1.5, 'px', 'Depth-of-field on the room'),
      ss('Room dim', 'roomDim', 0, 0.75, 0.02, 0.14, '', 'Darken the room so she pops'),
      ss('Parallax', 'parallax', 0, 1.5, 0.05, 0.55, '', 'Room shifts with your cursor'),
    ]),

    group('Framing & Camera', [
      segRow({
        label: 'Framing', desc: 'Camera preset',
        items: FRAMING.map(([id, lbl]) => [id, lbl]), active: curFrame,
        onPick: (id) => {
          const f = FRAMING.find((x) => x[0] === id);
          store.patch('camera', f[2]); store.patch('stage', f[3]); store.rerenderPanel();
        },
      }),
      ss('Avatar scale', 'avatarScale', 0.5, 1.65, 0.02, 1.0, '×', 'Also: scroll on stage'),
      more({ id: 'stage.cam.more', title: 'Position & rig camera', desc: 'X / Y placement, zoom, vertical' }, [
        ss('Position X', 'avatarX', 0.1, 0.9, 0.01, 0.5, '', 'Also: Shift-drag on stage'),
        ss('Vertical lift', 'avatarY', -0.25, 0.45, 0.01, 0, '', '0 = flush with the bottom'),
        slider({ label: 'Camera zoom', min: 0.65, max: 2.6, step: 0.02, value: cam.scale, def: 1.82, suffix: '×', onInput: (v) => store.patch('camera', { scale: v }) }),
        slider({ label: 'Camera vertical', min: -60, max: 520, step: 2, value: cam.y, def: 355, onInput: (v) => store.patch('camera', { y: v }) }),
      ]),
    ]),
  ];
}

/** Save / apply full-studio presets. */
export function renderPresetsSection(store) {
  const presetNameIn = el('input', { class: 'text-in', placeholder: 'Preset name', 'aria-label': 'Preset name' });
  const savedKeys = Object.keys(store.config.userPresets || {});
  const save = async () => {
    const name = presetNameIn.value.trim();
    if (!name) return;
    const r = await (await fetch('/api/presets/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
    })).json();
    if (r.ok) { store.config.userPresets = r.userPresets; presetNameIn.value = ''; store.rerenderPanel(); }
  };
  presetNameIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });

  return group('Studio Presets', [
    row({ title: 'Save current studio', desc: 'Room, look, rig and voice settings together', control: [presetNameIn, btn('Save', save, 'primary')], cls: 'wide-ctrl' }),
    savedKeys.length
      ? row({
          title: 'Saved presets', desc: 'Click to apply', stack: true,
          control: chips(savedKeys.map((k) => [k, k]), {
            onPick: async (k) => {
              const r = await (await fetch('/api/presets/apply', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: k }),
              })).json();
              if (r.ok) { store.config = r.config; store.setRoomTitle?.(r.config?.stage?.room); store.rerenderPanel(); }
            },
          }),
        })
      : noteRow('No saved presets yet.'),
  ]);
}
