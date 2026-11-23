/** ADVANCED — shader look presets, sharpness, rim lighting, bloom, color grade. */
import { chipsRow, colorRow, group, more, slider, toggle } from '../ui.js';

export function renderShaderPanel(store) {
  const sh = store.config.shader;
  const set = (vals) => store.patch('shader', vals);
  const s = (label, key, min, max, step, def, suffix = '', desc = '') => slider({
    label, desc, min, max, step, value: sh[key] ?? def, def, suffix, onInput: (v) => set({ [key]: v }),
  });
  const presets = Object.entries(store.shaderPresets || {});

  return [
    group('Shader & Look', [
      toggle({ label: 'Real-time GLSL shader', desc: 'GPU sharpening, rim light, bloom and grading', checked: sh.enabled, onChange: (v) => set({ enabled: v }) }),
      chipsRow({
        label: 'Look preset', desc: 'Overwrites the lighting values below',
        items: presets.map(([key, p]) => [key, p.label || key]),
        onPick: (key) => { set({ ...store.shaderPresets[key] }); store.rerenderPanel(); },
      }),
      s('Lineart sharpness', 'sharpness', 0, 1.5, 0.02, 0.35, '', 'Contrast-adaptive sharpening'),
      s('Chromatic aberration', 'chromaAb', 0, 0.8, 0.02, 0),
    ]),

    group('Lighting', [
      s('Rim strength', 'rimStrength', 0, 2.2, 0.02, 0.72),
      s('Key light angle', 'keyAngle', 0, 360, 1, 138, '°'),
      colorRow({ label: 'Key rim color', value: sh.keyColor, onInput: (v) => set({ keyColor: v }) }),
      colorRow({ label: 'Fill rim color', value: sh.fillColor, onInput: (v) => set({ fillColor: v }) }),
      more({ id: 'adv.light.more', title: 'Rim width & room wrap', desc: 'Width, ambient wrap amount and color' }, [
        s('Rim width', 'rimWidth', 0.4, 2.4, 0.05, 1.0),
        s('Ambient wrap', 'ambientWrap', 0, 1.2, 0.02, 0.24, '', 'Room light bleeding onto her edges'),
        colorRow({ label: 'Ambient color', value: sh.ambientColor, onInput: (v) => set({ ambientColor: v }) }),
      ]),
    ]),

    group('Bloom & Color Grade', [
      s('Bloom strength', 'bloomStrength', 0, 1.8, 0.02, 0.20, '', 'Selective eye & hair specular bloom'),
      s('Bloom threshold', 'bloomThresh', 0.70, 0.98, 0.01, 0.94, '', 'Higher = skin-safe'),
      s('Vibrance', 'vibrance', 0.5, 1.6, 0.02, 1.08),
      more({ id: 'adv.grade.more', title: 'More grading', desc: 'Contrast, brightness, warmth, tint' }, [
        s('Contrast', 'contrast', 0.7, 1.4, 0.02, 1.04),
        s('Brightness', 'brightness', 0.65, 1.35, 0.02, 1.01),
        s('Warmth', 'warmth', -0.8, 0.8, 0.02, 0.08),
        colorRow({ label: 'Global tint', value: sh.tint, onInput: (v) => set({ tint: v }) }),
      ]),
    ]),
  ];
}
