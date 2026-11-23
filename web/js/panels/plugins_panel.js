/** ADVANCED — drop-in plugins, automation webhook test, live hook telemetry. */
import { btn, el, group, noteRow, row, slider, switchEl } from '../ui.js';

const postPlugins = async (body) => (await fetch('/api/plugins', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})).json();

export function renderPluginsPanel(store) {
  const plugins = store.plugins || [];

  const pluginRows = plugins.flatMap((p) => {
    const head = row({
      title: `${p.name}`, desc: p.description,
      cls: 'plugin-row',
      control: [
        el('span', { class: 'badge', text: `v${p.version}` }),
        switchEl({
          checked: p.enabled, label: p.name,
          onChange: async (on) => {
            const r = await postPlugins({ id: p.id, enabled: on });
            if (r.ok) { store.plugins = r.plugins; store.hooks = r.hooks; store.rerenderPanel(); }
          },
        }),
      ],
    });
    if (!p.enabled) return [head];
    const settings = Object.entries(p.settings || {}).map(([k, v]) => {
      if (typeof v === 'boolean') {
        return row({
          title: k, cls: 'nested',
          control: switchEl({
            checked: v, label: k,
            onChange: async (nv) => {
              const r = await postPlugins({ id: p.id, settings: { [k]: nv } });
              if (r.ok) { store.plugins = r.plugins; store.hooks = r.hooks; }
            },
          }),
        });
      }
      if (typeof v === 'number') {
        const s = slider({
          label: k, min: 0, max: Math.max(2, v * 3), step: 0.02, value: v,
          onInput: (nv) => { postPlugins({ id: p.id, settings: { [k]: nv } }); },
        });
        s.classList.add('nested');
        return s;
      }
      return null;
    }).filter(Boolean);
    return [head, ...settings];
  });

  const hookRows = Object.entries(store.hooks || {}).map(([name, entries]) =>
    row({ title: name, desc: entries.map((e) => `${e.plugin} · ${e.lastMs}ms`).join(', '), cls: 'mono-title' }));

  const triggerTest = (payload) => fetch('/api/hook/trigger', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });

  const enabledCount = plugins.filter((p) => p.enabled).length;
  return [
    group('Plugins', pluginRows.length ? pluginRows : [noteRow('No plugins found in plugins/*.py.')],
      { desc: `${enabledCount} of ${plugins.length} enabled · drop-in plugins/*.py` }),
    group('Automation', [
      row({
        title: 'Webhook test', desc: 'POST /api/hook/trigger — for OBS, Stream Deck, bots and agents',
        control: [
          btn('Wave', () => triggerTest({ action: 'wave' })),
          btn('Cheer', () => triggerTest({ action: 'cheer' })),
          btn('Neon look', () => triggerTest({ action: 'shader_preset', value: 'midnight_neon' })),
        ],
      }),
      ...(hookRows.length ? hookRows : [noteRow('No hooks active.')]),
    ], { desc: 'Live hook registry telemetry' }),
  ];
}
