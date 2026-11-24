/** MOTION & EMOTIONS — emotes, expression, face FX, physics & head-pat, arms, gaze. ADVANCED — param inspector. */
import { btn, chipsRow, el, group, more, noteRow, row, segRow, slider, toggle } from '../ui.js';
import { EMOTES, emoteGrid } from './emotes.js';
import { renderExpressionFx } from './motion_panel.js';

let paramFilter = '';

const ARM_PRESETS = [
  ['rest',  'Rest',   { armRPose: 0, armLPose: 0 }],
  ['wave',  'Wave',   { armRPose: 1, armLPose: 0 }],
  ['left',  'Left',   { armRPose: 0, armLPose: 1 }],
  ['cheer', 'Cheer',  { armRPose: 1, armLPose: 1 }],
];

export function renderRigPanel(store) {
  const rig = store.config.rig;
  const set = (vals) => store.patch('rig', vals);
  const rs = (label, key, min, max, step, def, suffix = '', desc = '') => slider({
    label, desc, min, max, step, value: rig[key] ?? def, def, suffix, onInput: (v) => set({ [key]: v }),
  });

  const curArm = ARM_PRESETS.find(([, , v]) =>
    Math.round(rig.armRPose ?? 0) === v.armRPose && Math.round(rig.armLPose ?? 0) === v.armLPose)?.[0];

  return [
    group('Quick Emotes', [
      row({ title: 'Tweened emotes', desc: 'One click — also in the ✦ popover on stage (keys 1–9)', stack: true, control: emoteGrid() }),
    ], { desc: `${EMOTES.length} spring-tweened emotes · hover for details` }),

    group('Expression', [
      chipsRow({
        label: 'Base expression', desc: 'Resting mood between emotes',
        items: store.expressionPresets.map((n) => [n, n[0].toUpperCase() + n.slice(1)]),
        active: rig.expression,
        onPick: (name) => { set({ expression: name }); store.rerenderPanel(); },
      }),
    ]),

    group('Physics & Interaction', [
      toggle({ label: 'Secondary physics', desc: '60 FPS spring pendulums for hair, chest and accessories', checked: rig.physicsEnabled, onChange: (v) => set({ physicsEnabled: v }) }),
      rs('Bounce & jiggle', 'jiggleScale', 0, 2.2, 0.05, 1.25, '×', 'Spring pendulum intensity'),
      toggle({
        label: 'Head-pat & poke', desc: 'Stroke her head to pat, quick-click her to poke',
        checked: rig.headPatEnabled ?? true, onChange: (v) => set({ headPatEnabled: v }),
      }),
    ]),

    group('Face FX', [
      rs('Sparkle eyes', 'sparkleEyes', 0, 1, 0.02, 0),
      rs('Blush lines', 'blushLines', 0, 1, 0.02, 0.15, '', 'Anime /// cheek lines'),
      rs('Anime gloom', 'gloom', 0, 1, 0.05, 0, '', 'Shadowed forehead & eyes'),
      more({ id: 'motion.fx.more', title: 'More face FX', desc: 'Cheek puff, squint, tongue' }, [
        rs('Cheek puff', 'cheekPuff', 0, 1, 0.02, 0),
        rs('Eye squint', 'eyeSquint', 0, 1, 0.02, 0, '', 'Duchenne smile squint'),
        rs('Tongue out', 'tongueOut', 0, 1, 0.02, 0),
      ]),
    ]),

    renderExpressionFx(store),

    group('Arms', [
      segRow({
        label: 'Arm pose', desc: '3-bone rigid toggle arms',
        items: ARM_PRESETS.map(([id, lbl]) => [id, lbl]), active: curArm,
        onPick: (id) => { set(ARM_PRESETS.find((p) => p[0] === id)[2]); store.rerenderPanel(); },
      }),
      toggle({
        label: 'Talking gestures', desc: 'Relaxed, bent arms move naturally while she speaks',
        checked: rig.autoGestureOnSpeech ?? true, onChange: (v) => set({ autoGestureOnSpeech: v }),
      }),
      slider({
        label: 'Talking gesture intensity', min: 0, max: 1.5, step: 0.05,
        value: (store.config.lipsync || {}).speechToArmGesture ?? 0.85, def: 0.85,
        onInput: (v) => store.patch('lipsync', { speechToArmGesture: v }),
      }),
      more({ id: 'motion.arms.more', title: 'Fine-tune arms', desc: 'Gesture and shoulder lift per side' }, [
        rs('Right gesture', 'armRGesture', -0.48, 0.92, 0.02, 0),
        rs('Right shoulder lift', 'armRLift', -0.7, 0.8, 0.02, 0),
        rs('Left gesture', 'armLGesture', -0.48, 0.92, 0.02, 0),
        rs('Left shoulder lift', 'armLLift', -0.7, 0.8, 0.02, 0),
      ]),
    ]),

    group('Gaze & Idle Life', [
      toggle({ label: 'Natural gaze', desc: 'Autonomous looking around & saccades', checked: rig.autoGaze ?? true, onChange: (v) => set({ autoGaze: v }) }),
      rs('Gaze liveliness', 'autoGazeLiveliness', 0, 1.8, 0.05, 1.0, '×'),
      toggle({ label: 'Follow mouse', desc: 'Looks at your cursor, then drifts back', checked: rig.mouseLookAt, onChange: (v) => set({ mouseLookAt: v }) }),
      rs('Look-at strength', 'mouseLookStrength', 0, 1.5, 0.05, 0.72),
      toggle({ label: 'Auto-blink', desc: 'Natural single & double blinks', checked: rig.autoBlink, onChange: (v) => set({ autoBlink: v }) }),
      rs('Blink rate', 'blinkRate', 0.3, 2.2, 0.05, 1.0, '×'),
      rs('Idle sway & breathing', 'idleSway', 0, 2.0, 0.05, 1.0, '×'),
    ]),
  ];
}

/** ADVANCED — searchable direct rig parameter overrides. */
export function renderParamInspector(store) {
  const rig = store.config.rig;
  const meta = store.paramsMeta || [];
  const ov = rig.overrides || {};
  const q = paramFilter.toLowerCase();
  const matching = meta.filter((m) => !q || m.name.toLowerCase().includes(q));
  const shown = matching.slice(0, 24);

  const rows = shown.map((m) => {
    const cur = ov[m.name] ? ov[m.name][0] : m.defaults[0];
    return slider({
      label: m.name + (ov[m.name] ? ' ●' : ''),
      min: m.min[0], max: m.max[0],
      step: (m.max[0] - m.min[0]) > 10 ? 0.5 : 0.02,
      value: cur, def: m.defaults[0],
      onInput: (v) => {
        const next = { ...(rig.overrides || {}), [m.name]: [v, ov[m.name]?.[1] ?? m.defaults[1]] };
        store.patch('rig', { overrides: next });
      },
    });
  });

  const searchIn = el('input', {
    class: 'text-in search', type: 'search', placeholder: `Filter ${meta.length} rig params…`, value: paramFilter,
    'aria-label': 'Filter rig parameters',
    oninput: (e) => {
      paramFilter = e.target.value;
      const pos = e.target.selectionStart;
      store.rerenderPanel();
      const again = document.querySelector('#inspector-body input.search');
      if (again) { again.focus(); again.setSelectionRange(pos, pos); }
    },
  });

  const overrideCount = Object.keys(ov).length;
  return group('Parameter Inspector', [
    row({ title: 'Search', desc: 'Override any bound rig parameter directly (● = overridden)', stack: true, control: searchIn }),
    ...rows,
    matching.length > shown.length ? noteRow(`Showing ${shown.length} of ${matching.length} — refine the filter.`) : null,
    row({
      title: 'Clear overrides', desc: `${overrideCount} parameter${overrideCount === 1 ? '' : 's'} overridden`,
      control: btn('Clear all', () => {
        store.config.rig.overrides = {};
        store.sendWs({ type: 'override_clear' });
        store.rerenderPanel();
      }, 'danger'),
    }),
  ], { desc: `${meta.length} bound parameters` });
}

