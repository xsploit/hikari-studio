/** MOTION & EMOTIONS — Motion Lab: video-to-VTuber retarget extraction, clip library & playback blend. */
import { btn, el, group, noteRow, row, slider, toggle } from '../ui.js';

const postMotion = (url, body) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

export function renderMotionPanel(store) {
  const statusEl = el('div', { class: 's-desc', text: 'Extract head pose, sway, blinks & arms from a short .mp4 / .webm / .mov' });
  const fileIn = el('input', {
    type: 'file', accept: 'video/mp4,video/webm,video/quicktime', hidden: true,
    onchange: async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      statusEl.textContent = `Extracting 60 FPS motion from ${f.name}…`;
      const fd = new FormData();
      fd.append('file', f);
      const r = await (await fetch('/api/motion/extract', { method: 'POST', body: fd })).json();
      if (r.ok) { await store.refreshState(); store.rerenderPanel(); }
      else statusEl.textContent = `Error: ${r.error}`;
    },
  });

  const ms = store.motionStatus || { playing: false, id: '', mix: 0.85, loop: true };
  const motions = store.motions || [];
  const clipRows = motions.map((c) => {
    const active = ms.playing && ms.id === c.id;
    return row({
      title: c.name,
      desc: `${c.duration}s · ${c.hasAudio ? 'Audio + motion' : 'Motion'}${active ? ' · playing' : ''}`,
      cls: active ? 'live' : '',
      control: btn(active ? 'Stop' : 'Play', async () => {
        const r = await (await postMotion('/api/motion/play',
          active ? { id: '' } : { id: c.id, loop: ms.loop, mix: ms.mix, withAudio: false })).json();
        store.motionStatus = r.status;
        store.rerenderPanel();
      }, active ? 'danger' : ''),
    });
  });

  const extractRow = el('div', { class: 's-row' }, [
    el('div', { class: 's-row-text' }, [el('div', { class: 's-title', text: 'Extract from video' }), statusEl]),
    el('div', { class: 's-ctrl' }, [
      btn('Extract demo', async () => {
        statusEl.textContent = 'Analyzing demo video via optical flow…';
        const r = await (await fetch('/api/motion/demo', { method: 'POST' })).json();
        if (r.ok) { await store.refreshState(); store.rerenderPanel(); }
        else statusEl.textContent = `Error: ${r.error}`;
      }),
      btn('Upload…', () => fileIn.click(), 'primary'),
      fileIn,
    ]),
  ]);

  return [
    group('Motion Lab', [
      extractRow,
      ...(clipRows.length ? clipRows : [noteRow('No extracted clips yet — upload a video or extract the demo.')]),
      slider({
        label: 'Retarget mix', desc: 'How strongly the clip drives the rig (live mouth flaps stay untouched)',
        min: 0, max: 1, step: 0.05, value: ms.mix ?? 0.85, def: 0.85,
        onInput: async (v) => { ms.mix = v; await postMotion('/api/motion/update', { mix: v }); },
      }),
      toggle({
        label: 'Loop clip', desc: 'Repeat the playing clip continuously', checked: ms.loop ?? true,
        onChange: async (v) => { ms.loop = v; await postMotion('/api/motion/update', { loop: v }); },
      }),
    ], { desc: `${motions.length} clip${motions.length === 1 ? '' : 's'} · video-to-VTuber retargeting` }),
  ];
}
