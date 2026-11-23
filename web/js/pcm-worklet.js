class RemotePcm extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Float32Array(960);
    this.used = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const value of channel) {
      this.frame[this.used++] = value;
      if (this.used === this.frame.length) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Float32Array(960);
        this.used = 0;
      }
    }
    return true;
  }
}

registerProcessor("remote-pcm", RemotePcm);
