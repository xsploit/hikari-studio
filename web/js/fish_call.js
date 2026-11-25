/**
 * Native Fish Audio Full-Duplex Client (Drama 3 default / Live 3) running inside Hikari Studio.
 * Uses @fishaudio/agent-client + RemotePcm AudioWorklet -> streams 48kHz Int16 PCM directly
 * over ws://127.0.0.1:8800/ws/fish-pcm into Hikari v28's 60 FPS VBridger lip-sync!
 */
import { AgentSession } from './vendor/fish-agent-client.js';

const SAMPLE_RATE = 48000;

export class FishCallManager {
  constructor({ onState, onMessage, onError } = {}) {
    this.onState = onState || (() => {});
    this.onMessage = onMessage || (() => {});
    this.onError = onError || (() => {});

    this.session = null;
    this.output = null;
    this.mode = 'drama-3';
    this.connected = false;
    this.connecting = false;
    this.micMuted = true;
    this.agentMode = 'idle';
    this.outputDeviceId = '';
    this._hasServerSession = false;
    this.sessionId = '';      // Fish session_id of OUR call — scopes /end so an old tab can't kill a newer tab's call
    this._connectSeq = 0;     // bumped by every connect()/disconnect(); a stale handshake aborts itself

    window.addEventListener('pagehide', () => {
      if (!this._hasServerSession) return;
      const body = new Blob([JSON.stringify({ sessionId: this.sessionId })], { type: 'application/json' });
      navigator.sendBeacon('/api/fish/session/end', body);
    });
  }

  _endServerSession(sessionId) {
    return fetch('/api/fish/session/end', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId: sessionId || '' }),
    }).catch(() => {});
  }

  _emitState() {
    this.onState({
      connected: this.connected,
      connecting: this.connecting,
      mode: this.mode,
      micMuted: this.micMuted,
      agentMode: this.agentMode,
    });
  }

  async _setupPcmPipeline(room) {
    const context = new AudioContext({ sampleRate: SAMPLE_RATE });
    let node;
    try {
      await context.resume();
      await context.audioWorklet.addModule('/static/js/pcm-worklet.js');
      node = new AudioWorkletNode(context, 'remote-pcm', {
        channelCount: 1,
        channelCountMode: 'explicit',
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      node.connect(context.destination);
    } catch (err) {
      await context.close().catch(() => {});
      throw err;
    }

    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${wsProto}//${location.host}/ws/fish-pcm`);
    const tracks = new Map();

    const attach = (track) => {
      if (track?.kind !== 'audio' || tracks.has(track)) return;
      const src = context.createMediaStreamSource(new MediaStream([track.mediaStreamTrack]));
      src.connect(node);
      tracks.set(track, src);
    };
    const detach = (track) => {
      tracks.get(track)?.disconnect();
      tracks.delete(track);
    };

    node.port.onmessage = ({ data }) => {
      if (socket.readyState !== WebSocket.OPEN || socket.bufferedAmount >= 64000) return;
      const samples = new Float32Array(data);
      const pcm = new DataView(new ArrayBuffer(samples.length * 2));
      for (let i = 0; i < samples.length; i++) {
        const v = Math.max(-1, Math.min(1, samples[i]));
        pcm.setInt16(i * 2, Math.round(v * (v < 0 ? 32768 : 32767)), true);
      }
      socket.send(pcm.buffer);
    };

    room.on('trackSubscribed', attach);
    room.on('trackUnsubscribed', detach);
    for (const p of room.remoteParticipants.values()) {
      for (const pub of p.audioTrackPublications.values()) attach(pub.track);
    }

    return {
      context,
      node,
      disconnect: () => {
        room.off('trackSubscribed', attach);
        room.off('trackUnsubscribed', detach);
        for (const t of tracks.keys()) detach(t);
        node.port.onmessage = null;
        try { socket.close(); } catch {}
        try { node.disconnect(); } catch {}
        context.close().catch(() => {});
      },
    };
  }

  async connect(mode = 'drama-3', startMicOn = true) {
    if (this.connecting || this.connected) return;
    const seq = ++this._connectSeq;
    const stale = () => seq !== this._connectSeq; // true once disconnect()/cancel ran mid-handshake
    this.mode = mode === 'live-3' ? 'live-3' : 'drama-3';
    this.connecting = true;
    this.agentMode = 'connecting';
    this._emitState();

    let session = null;
    let output = null;
    try {
      const resp = await fetch('/api/fish/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: this.mode }),
      });
      const sessionToken = await resp.json();
      if (!resp.ok || sessionToken.error) {
        throw new Error(sessionToken.error || 'Could not start Fish Audio session.');
      }
      const sid = String(sessionToken.session_id || sessionToken.sessionId || '');
      if (stale()) { // canceled while the server was creating the session — release it, touch nothing else
        await this._endServerSession(sid);
        return;
      }
      this._hasServerSession = true;
      this.sessionId = sid;

      // Microphone starts OFF; it is enabled only after the handshake is confirmed current (below).
      session = await AgentSession.start({
        sessionToken,
        microphone: false,
        wakeLock: false,
        audio: this.outputDeviceId ? { outputDeviceId: this.outputDeviceId } : undefined,
        callbacks: {
          onMessage: (msg) => { if (!stale()) this.onMessage(msg); },
          onModeChange: (m) => {
            if (stale()) return;
            this.agentMode = m || 'listening';
            this._emitState();
          },
          onError: (err) => { if (!stale()) this.onError(err?.code || err?.message || 'Fish voice error'); },
          onDisconnect: () => {
            if (!stale() && this.connected) this.disconnect().catch(() => {});
          },
        },
      });
      if (stale()) { // disconnect() already ended the server session; just drop this local one
        await session.end().catch(() => {});
        return;
      }
      this.session = session;

      const room = session.getRoom();
      if (!room) throw new Error('Fish session did not provide a LiveKit room.');
      output = await this._setupPcmPipeline(room);
      if (stale()) {
        output.disconnect();
        return;
      }
      this.output = output;

      if (startMicOn) {
        try {
          await session.setMicMuted(false);
        } catch (err) {
          this.onError(err?.message || 'Microphone unavailable — call continues muted.');
        }
        if (stale()) return; // disconnect() during mic start already ended the session (and the mic)
      }

      this.connected = true;
      this.connecting = false;
      this.micMuted = Boolean(session.micMuted);
      this.agentMode = 'ready';
      this._emitState();
    } catch (err) {
      if (stale()) {
        if (output && output !== this.output) output.disconnect();
        if (session && session !== this.session) await session.end().catch(() => {});
        return;
      }
      await this.disconnect().catch(() => {});
      this.onError(err?.message || String(err));
    }
  }

  async disconnect() {
    this._connectSeq++; // cancels any in-flight connect()
    const cur = this.session;
    const sid = this.sessionId;
    this.session = null;
    this.connected = false;
    this.connecting = false;
    this.micMuted = true;
    this.agentMode = 'idle';
    if (this.output) {
      this.output.disconnect();
      this.output = null;
    }
    try {
      if (cur) await cur.end();
    } catch {}
    if (this._hasServerSession) {
      this._hasServerSession = false;
      this.sessionId = '';
      await this._endServerSession(sid);
    }
    this._emitState();
  }

  async toggleMic() {
    if (!this.session) return false;
    await this.session.setMicMuted(!this.session.micMuted);
    this.micMuted = Boolean(this.session.micMuted);
    this._emitState();
    return !this.micMuted;
  }

  interrupt() {
    this.session?.interrupt();
  }

  sendMessage(text) {
    if (!this.session || !text.trim()) return false;
    this.session.sendUserMessage(text.trim(), { audio: true });
    return true;
  }

  sendSilentContext(text) {
    if (!this.session || !this.connected || !text || !text.trim()) return false;
    // Never interrupt or talk over Hikari while she is speaking.
    if (this.agentMode === 'speaking' || this.session.isSpeaking) return false;
    this.session.sendUserMessage(
      `[Silent screen context — do NOT reply out loud or acknowledge this; just keep it in mind when I talk]: ${text.trim()}`,
      { audio: false },
    );
    return true;
  }

  async setOutputDevice(id) {
    this.outputDeviceId = id || '';
    if (this.session) await this.session.setOutputDevice(this.outputDeviceId);
  }
}
