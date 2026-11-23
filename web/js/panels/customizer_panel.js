/** LOOK & PROPS — wardrobe, eyes, hair & bangs, outfit tint, glow & proportions, imports, props slot. */
import { actionRow, btn, chipsRow, colorRow, el, group, more, row, segRow, slider } from '../ui.js';

const HAIR_PRESETS = [
  ['Hikari Sakura',   { hairColor: '#ff6a9e', hairColorStrength: 0.0, hairHue: 0, hairSat: 1.00, hairVal: 1.00 }],
  ['Neon Rose',       { hairColor: '#ff2e88', hairColorStrength: 1.0, hairHue: 0, hairSat: 1.15, hairVal: 1.02 }],
  ['Lavender Dream',  { hairColor: '#c084ff', hairColorStrength: 1.0, hairHue: 0, hairSat: 0.95, hairVal: 1.05 }],
  ['Platinum Silver', { hairColor: '#e4e8f5', hairColorStrength: 1.0, hairHue: 0, hairSat: 0.80, hairVal: 1.12 }],
  ['Cyber Cyan',      { hairColor: '#38dcff', hairColorStrength: 1.0, hairHue: 0, hairSat: 1.05, hairVal: 1.04 }],
  ['Honey Blonde',    { hairColor: '#ffd275', hairColorStrength: 1.0, hairHue: 0, hairSat: 1.00, hairVal: 1.08 }],
  ['Midnight Violet', { hairColor: '#5630a8', hairColorStrength: 1.0, hairHue: 0, hairSat: 1.05, hairVal: 0.92 }],
];

const OUTFIT_PRESETS = [
  ['Original Black', { outfitTint: '#ffffff', outfitTintStrength: 0.0,  outfitVal: 1.0 }],
  ['Crimson Tech',   { outfitTint: '#d6284c', outfitTintStrength: 0.78, outfitVal: 1.15 }],
  ['Royal Navy',     { outfitTint: '#3658c8', outfitTintStrength: 0.78, outfitVal: 1.15 }],
  ['Cyber Violet',   { outfitTint: '#8a48e8', outfitTintStrength: 0.76, outfitVal: 1.15 }],
  ['Snow White',     { outfitTint: '#f4f2fa', outfitTintStrength: 0.82, outfitVal: 1.65 }],
];

let applyStatus = '';
let applyKind = '';
let applying = false;

const swatchChips = (presets, colorOf, onPick) => el('div', { class: 'chip-grid' }, presets.map(([lbl, vals]) =>
  el('button', { type: 'button', class: 'chip', onclick: () => onPick(vals) }, [
    el('span', { class: 'swatch', style: `background:${colorOf(vals)}`, 'aria-hidden': 'true' }), lbl,
  ])));

export function renderCustomizerPanel(store) {
  const saved = store.config.customizer || {};
  const c = { ...saved };
  const rig = store.config.rig || {};
  const eyePresets = store.eyeStylePresets || {};
  let dirty = false;

  const statusLine = el('div', {
    class: `apply-status ${applyKind}`.trim(), role: 'status',
    text: applyStatus || 'Presets, styles and color pickers apply instantly. Fine-tune sliders apply with Apply.',
  });
  const applyBtn = btn(applying ? 'Applying…' : 'Apply', () => applyNow(), 'primary');
  const markDirty = () => {
    if (dirty) return;
    dirty = true;
    applyBtn.classList.add('pulse');
    statusLine.textContent = 'Slider changes pending — press Apply.';
    statusLine.className = 'apply-status pending';
  };

  const submit = async (url, options, pending, success) => {
    if (applying) return;
    applying = true;
    applyStatus = pending;
    applyKind = 'pending';
    store.rerenderPanel();
    try {
      const response = await fetch(url, options);
      const result = await response.json().catch(() => null);
      if (!response.ok || !result?.ok) throw new Error(result?.error || `Studio returned HTTP ${response.status}`);
      if (result.customizer) store.config.customizer = result.customizer;
      applyStatus = success(result);
      applyKind = 'ok';
    } catch (error) {
      applyStatus = `Could not apply: ${error.message}`;
      applyKind = 'error';
    } finally {
      applying = false;
      store.rerenderPanel();
    }
  };

  const applyNow = (patch = {}) => {
    const edits = Object.fromEntries(Object.entries(c).filter(([key, value]) => value !== saved[key]));
    return submit('/api/customizer/apply', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...edits, ...patch }),
    }, 'Applying and updating the avatar…', (result) => `Applied in ${result.elapsedMs} ms ✓`);
  };

  const upload = (file, route, pending, success) => {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    return submit(route, { method: 'POST', body: form }, pending, () => success);
  };
  const fileInput = (accept, route, pending, success) => el('input', {
    type: 'file', accept, hidden: true,
    onchange: (e) => upload(e.target.files?.[0], route, pending, success),
  });
  const vrmInput = fileInput('.vrm,.glb', '/api/customizer/upload-vrm', 'Extracting VRM eyes…', 'Applied VRM eyes ✓');
  const pngInput = fileInput('image/png,image/jpeg,image/webp', '/api/customizer/upload-eye', 'Mapping the custom iris…', 'Applied custom eye texture ✓');
  const frontHairInput = fileInput('image/png,image/webp', '/api/customizer/upload-front-hair', 'Applying custom front hair PNG…', 'Applied custom front hair ✓');

  // Customizer slider (pending until Apply).
  const cs = (label, key, min, max, step, def, suffix = '', desc = '') => slider({
    label, desc, min, max, step, value: c[key] ?? def, def, suffix,
    onInput: (v) => { c[key] = v; markDirty(); },
  });

  const eyeItems = Object.entries(eyePresets).map(([id, p]) => [id, p.label || id]);

  const out = [
    el('div', { class: 'apply-bar' }, [statusLine, applyBtn]),

    group('Wardrobe', [
      segRow({
        label: 'Outfit', desc: 'Top / halter variant',
        items: [['original', 'Original'], ['hikari-stripe', 'Hikari Stripe'], ['refined-vrm', 'Refined VRM']],
        active: c.outfitStyle || 'original', onPick: (id) => applyNow({ outfitStyle: id }),
      }),
      segRow({
        label: 'Shoes', desc: 'Footwear variant',
        items: [['original', 'Sneakers'], ['d3-high-top', 'High-Tops']],
        active: c.shoeStyle || 'original', onPick: (id) => applyNow({ shoeStyle: id }),
      }),
    ]),

    group('Eyes', [
      eyeItems.length
        ? chipsRow({
            label: 'Eye style', desc: 'Extracted from the refined VRM, incl. solid-pupil original',
            items: eyeItems, active: c.eyeStyle,
            onPick: (id) => {
              const p = eyePresets[id] || {};
              applyNow({ eyeStyle: id, eyeHue: p.eyeHue ?? 0, eyeSat: p.eyeSat ?? 1.05, eyeVal: p.eyeVal ?? 1.08, eyeHighlight: p.eyeHighlight ?? 1.0 });
            },
          })
        : row({ title: 'Eye style', desc: 'No eye style presets reported by the studio.' }),
      cs('Hue shift', 'eyeHue', -180, 180, 2, 0, '°', 'Rotate the iris color'),
      cs('Highlights', 'eyeHighlight', 0.0, 1.5, 0.05, 1.0, '', 'Chromatic catch-light strength'),
      more({ id: 'look.eyes.more', title: 'Fine-tune iris', desc: 'Saturation, brightness, scale, offsets' }, [
        cs('Saturation', 'eyeSat', 0.2, 1.8, 0.05, 1.05),
        cs('Brightness', 'eyeVal', 0.6, 1.5, 0.04, 1.08),
        cs('Iris scale', 'eyeScale', 0.80, 1.25, 0.02, 1.04),
        cs('Gaze convergence X', 'eyeOffsetX', -0.15, 0.15, 0.01, 0),
        cs('Vertical offset Y', 'eyeOffsetY', -0.15, 0.15, 0.01, 0),
      ]),
    ]),

    group('Hair & Bangs', [
      row({ title: 'Color presets', desc: 'Applies to all hair, twintails and brows', stack: true,
        control: swatchChips(HAIR_PRESETS, (v) => v.hairColor, (vals) => applyNow(vals)) }),
      colorRow({
        label: 'Exact hair color', desc: 'Pick any color — applies on release',
        value: c.hairColor || '#ff6a9e',
        onChange: (v) => applyNow({ hairColor: v, hairColorStrength: Math.max(0.95, c.hairColorStrength || 0) }),
      }),
      more({ id: 'look.hair.more', title: 'Fine-tune hair', desc: 'Blend, hue, saturation, brightness' }, [
        cs('Color blend', 'hairColorStrength', 0.0, 1.0, 0.04, 0.0),
        cs('Hue', 'hairHue', -180, 180, 2, 0, '°'),
        cs('Saturation', 'hairSat', 0.0, 1.6, 0.04, 1.0),
        cs('Brightness', 'hairVal', 0.6, 1.35, 0.04, 1.0),
      ]),
      segRow({
        label: 'Bangs', desc: 'Front hair rendering',
        items: [['solid', 'Solid'], ['seethrough', 'See-through'], ...(c.customFrontHairPath ? [['custom', 'Custom PNG']] : [])],
        active: c.bangsMode || 'solid', onPick: (id) => applyNow({ bangsMode: id }),
      }),
      row({
        title: 'Custom bangs', desc: 'Download the template, paint it, import it back',
        control: [
          el('a', { class: 'btn', href: '/api/customizer/front-hair-template.png', download: 'front_hair_editable.png', text: 'Template' }),
          btn('Import PNG…', () => frontHairInput.click()),
          frontHairInput,
        ],
      }),
    ]),

    group('Outfit Color', [
      row({ title: 'Fabric presets', desc: 'Skin-safe tint — only the fabric changes', stack: true,
        control: swatchChips(OUTFIT_PRESETS, (v) => (v.outfitTintStrength ? v.outfitTint : '#151518'), (vals) => applyNow(vals)) }),
      colorRow({
        label: 'Fabric color', desc: 'Pick any color — applies on release',
        value: c.outfitTint || '#d6284c',
        onChange: (v) => applyNow({
          outfitTint: v,
          outfitTintStrength: (c.outfitTintStrength || 0) < 0.15 ? 0.80 : c.outfitTintStrength,
          outfitVal: (c.outfitVal || 1.0) <= 1.02 ? 1.18 : c.outfitVal,
        }),
      }),
      cs('Tint strength', 'outfitTintStrength', 0.0, 1.0, 0.04, 0.0),
      cs('Brightness', 'outfitVal', 0.6, 1.8, 0.04, 1.0),
    ]),

    group('Glow & Proportions', [
      slider({
        label: 'Emissive glow', desc: 'Galaxy / solid-pupil eyes and halter top LEDs',
        min: 0, max: 2, step: 0.05, value: rig.emissiveGlow ?? 1.0, def: 1.0, suffix: '×',
        onInput: (v) => store.patch('rig', { emissiveGlow: v }),
      }),
      slider({
        label: 'Proportion (Chibi)', desc: 'Full-body chibi transform',
        min: 0, max: 1, step: 0.05, value: rig.chibiMode ?? 0.0, def: 0.0,
        onInput: (v) => store.patch('rig', { chibiMode: v }),
      }),
    ], { desc: 'Live — no Apply needed' }),

    group('Imports & Props', [
      row({
        title: 'Import eyes', desc: 'Extract eyes from a .vrm, or map a custom iris image',
        control: [btn('.VRM…', () => vrmInput.click()), btn('.PNG…', () => pngInput.click()), vrmInput, pngInput],
      }),
      row({ title: 'Props & accessories', desc: 'Hats, glasses and held items — coming soon', control: el('span', { class: 'badge', text: 'Soon' }) }),
      actionRow({
        label: 'Reset character', desc: 'Restore default eyes, hair, outfit and shoes', button: 'Reset', variant: 'danger',
        onClick: () => applyNow({
          eyeStyle: 'vrm_refined', eyeHue: 0, eyeSat: 1.05, eyeVal: 1.08, eyeHighlight: 1.0,
          eyeScale: 1.04, eyeOffsetX: 0, eyeOffsetY: 0,
          hairColor: '#ff6a9e', hairColorStrength: 0.0, hairHue: 0, hairSat: 1.0, hairVal: 1.0,
          outfitStyle: 'original', shoeStyle: 'original',
          outfitTint: '#d6284c', outfitTintStrength: 0.0, outfitVal: 1.0,
        }),
      }),
    ]),
  ];

  if (applying) {
    for (const node of out) {
      for (const control of node.querySelectorAll('button:not(.s-more-head), input, a.btn')) {
        if (control.tagName === 'A') control.classList.add('disabled');
        else control.disabled = true;
      }
    }
  }
  return out;
}
