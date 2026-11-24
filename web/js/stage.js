/**
 * Stage Renderer:
 * 1) AvatarGLRenderer — WebGL2 premultiplied-alpha compositor decoding [rgb_jpg][alpha_jpg] at 60 FPS with ZERO edge halos.
 * 2) ProceduralRoomRenderer — 100% original procedural architectural/anime studio rooms + animated atmosphere particles
 *    (Twilight Atelier, Midnight Loft, Sakura Pavilion, Obsidian Broadcast Studio, Chroma Green, Transparent, or Custom Upload).
 */

const VS_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUV;
uniform vec2 uOffset;
uniform vec2 uScale;
void main() {
  vec2 p = aPos * uScale + uOffset;
  gl_Position = vec4(p, 0.0, 1.0);
  vUV = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
}`;

const FS_SRC = `#version 300 es
precision mediump float;
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uRgb;
uniform sampler2D uAlpha;
void main() {
  vec3 rgb = texture(uRgb, vUV).rgb;
  float a = texture(uAlpha, vUV).r;
  // Engine already outputs premultiplied RGB; clamp cleanly so JPEG quantization never leaks bright fringes
  vec3 clean = min(rgb, vec3(a));
  fragColor = vec4(clean, a);
}`;

export class AvatarGLRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) throw new Error('WebGL2 required');
    this.gl = gl;
    this._busy = false;
    this._aspect = 1440 / 1080;
    this._texW = 0;
    this._texH = 0;

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      return sh;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS_SRC));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS_SRC));
    gl.linkProgram(prog);
    this.prog = prog;

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    const makeTex = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.texRgb = makeTex();
    this.texAlpha = makeTex();
    this.uOffset = gl.getUniformLocation(prog, 'uOffset');
    this.uScale = gl.getUniformLocation(prog, 'uScale');
    this.uRgb = gl.getUniformLocation(prog, 'uRgb');
    this.uAlpha = gl.getUniformLocation(prog, 'uAlpha');
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, Math.floor(window.innerWidth * dpr));
    this.canvas.height = Math.max(1, Math.floor(window.innerHeight * dpr));
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  async drawPacket(arrayBuf, stageCfg, px = 0, py = 0) {
    if (this._busy) return;
    this._busy = true;
    let bmRgb = null, bmA = null;
    try {
      const view = new DataView(arrayBuf);
      const rgbLen = view.getUint32(0, false);
      // Zero-copy Uint8Array subviews (no arrayBuf.slice() heap copy 60x/sec!)
      const rgbBlob = new Blob([new Uint8Array(arrayBuf, 4, rgbLen)], { type: 'image/jpeg' });
      const aBytes = new Uint8Array(arrayBuf, 4 + rgbLen);
      const aType = (aBytes.length > 3 && aBytes[0] === 0x89 && aBytes[1] === 0x50) ? 'image/png' : 'image/jpeg';
      const alphaBlob = new Blob([aBytes], { type: aType });
      // allSettled (not all): if one decode rejects, the sibling that succeeded is still closed in `finally`.
      const [rRgb, rA] = await Promise.allSettled([createImageBitmap(rgbBlob), createImageBitmap(alphaBlob)]);
      if (rRgb.status === 'fulfilled') bmRgb = rRgb.value;
      if (rA.status === 'fulfilled') bmA = rA.value;
      if (!bmRgb || !bmA) return; // corrupt/partial frame — skip it (finally releases whatever decoded)
      const bw = bmRgb.width, bh = bmRgb.height;
      this._aspect = bw / bh;

      const gl = this.gl;
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.prog);

      // Allocate GPU VRAM storage ONCE per resolution, then stream via texSubImage2D (zero VRAM churn!)
      const realloc = (bw !== this._texW || bh !== this._texH);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texRgb);
      if (realloc) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmRgb);
      else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, bmRgb);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.uniform1i(this.uRgb, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.texAlpha);
      if (realloc) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmA);
      else gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, bmA);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.uniform1i(this.uAlpha, 1);
      if (realloc) { this._texW = bw; this._texH = bh; }

      const cw = this.canvas.width, ch = this.canvas.height;
      const winAspect = cw / ch;
      const sc = Number(stageCfg.avatarScale ?? 1.0);
      const sy = sc;
      const sx = (sy * this._aspect) / winAspect;
      const ox = (Number(stageCfg.avatarX ?? 0.5) * 2 - 1) - px * 0.012;
      // Anchor bottom of avatar quad FLUSH to bottom of viewport (-1.0) so skirt/body NEVER floats cut-off in mid-air!
      const userShiftY = Number(stageCfg.avatarY ?? 0.0);
      const oy = -1.0 + sy + userShiftY * 2.0 + Math.min(0.0, py * 0.008);
      gl.uniform2f(this.uScale, sx, sy);
      gl.uniform2f(this.uOffset, ox, oy);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    } finally {
      if (bmRgb) bmRgb.close();
      if (bmA) bmA.close();
      this._busy = false;
    }
  }
}

export class ProceduralRoomRenderer {
  constructor(canvas, customImg) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.customImg = customImg;
    this.particles = Array.from({ length: 68 }, () => ({
      x: Math.random(), y: Math.random(),
      s: 0.4 + Math.random() * 1.4,
      vx: (Math.random() - 0.5) * 0.02,
      vy: 0.015 + Math.random() * 0.045,
      ph: Math.random() * Math.PI * 2,
    }));
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  render(t, stageCfg, px, py) {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const room = stageCfg.room || 'twilight_atelier';
    const par = Number(stageCfg.parallax ?? 0.55);
    const ox = px * 26 * par;
    const oy = py * 18 * par;

    // Handle custom uploaded background image
    if (room === 'custom' && stageCfg.customRoomUrl) {
      if (this.customImg.getAttribute('src') !== stageCfg.customRoomUrl) {
        this.customImg.src = stageCfg.customRoomUrl;
      }
      this.customImg.hidden = false;
      this.customImg.style.filter = `blur(${Number(stageCfg.roomBlur || 0)}px) brightness(${1 - Number(stageCfg.roomDim || 0)})`;
      this.customImg.style.transform = `scale(1.06) translate(${(-ox * 0.6).toFixed(1)}px, ${(-oy * 0.6).toFixed(1)}px)`;
      ctx.clearRect(0, 0, W, H);
      this._drawParticles(ctx, W, H, t, stageCfg.particles === 'auto' ? 'dust' : stageCfg.particles);
      return;
    }
    this.customImg.hidden = true;

    if (room === 'transparent') {
      if (this._lastRoom !== 'transparent') {
        ctx.clearRect(0, 0, W, H);
        this._lastRoom = 'transparent';
      }
      return;
    }
    this._lastRoom = room;
    if (room === 'chroma_green') {
      ctx.fillStyle = '#00b140';
      ctx.fillRect(0, 0, W, H);
      return;
    }

    ctx.save();
    if (room === 'twilight_atelier') this._drawTwilightAtelier(ctx, W, H, t, ox, oy);
    else if (room === 'midnight_loft') this._drawMidnightLoft(ctx, W, H, t, ox, oy);
    else if (room === 'sakura_pavilion') this._drawSakuraPavilion(ctx, W, H, t, ox, oy);
    else this._drawObsidianStudio(ctx, W, H, t, ox, oy);

    // Optional dim overlay
    const dim = Number(stageCfg.roomDim || 0);
    if (dim > 0.005) {
      ctx.fillStyle = `rgba(8, 6, 14, ${Math.min(0.8, dim)})`;
      ctx.fillRect(0, 0, W, H);
    }

    const pMode = stageCfg.particles === 'auto'
      ? (room === 'midnight_loft' ? 'rain' : room === 'sakura_pavilion' ? 'sakura' : room === 'twilight_atelier' ? 'embers' : 'dust')
      : stageCfg.particles;
    this._drawParticles(ctx, W, H, t, pMode);
    ctx.restore();
    this.canvas.style.filter = Number(stageCfg.roomBlur || 0) > 0.1 ? `blur(${Number(stageCfg.roomBlur)}px)` : 'none';
  }

  _drawTwilightAtelier(ctx, W, H, t, ox, oy) {
    // Sunset gradient through panoramic studio window
    const sky = ctx.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0.00, '#181836');
    sky.addColorStop(0.36, '#432d5e');
    sky.addColorStop(0.64, '#a65371');
    sky.addColorStop(0.84, '#eb9466');
    sky.addColorStop(1.00, '#2a1a22');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, W, H);

    // Soft distant horizon glow & layered mountain ridges with parallax
    const sunG = ctx.createRadialGradient(W * 0.52 - ox * 0.4, H * 0.72 - oy * 0.4, 20, W * 0.52, H * 0.72, W * 0.48);
    sunG.addColorStop(0, 'rgba(255, 210, 148, 0.55)');
    sunG.addColorStop(1, 'rgba(255, 130, 120, 0)');
    ctx.fillStyle = sunG;
    ctx.fillRect(0, 0, W, H);

    // Distant ridges
    ctx.fillStyle = 'rgba(32, 20, 44, 0.55)';
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let x = 0; x <= W; x += 60) {
      const y = H * 0.68 + Math.sin(x * 0.004 + 1.2) * 32 + Math.cos(x * 0.011) * 14 - oy * 0.3;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fill();

    // Minimalist architectural studio window mullions & warm left/right walls
    const wallW = W * 0.15;
    ctx.fillStyle = '#18131d';
    ctx.fillRect(-40 - ox, 0, wallW + 40, H);
    ctx.fillRect(W - wallW - ox, 0, wallW + 80, H);
    ctx.fillRect(0, H * 0.80 - oy * 0.8, W, H * 0.25);

    // Window mullions
    ctx.strokeStyle = 'rgba(26, 20, 32, 0.92)';
    ctx.lineWidth = 10;
    for (const f of [0.33, 0.67]) {
      ctx.beginPath();
      ctx.moveTo(W * f - ox * 0.85, 0);
      ctx.lineTo(W * f - ox * 0.85, H * 0.80 - oy * 0.8);
      ctx.stroke();
    }
    // Warm amber desk lamp bounce on left + cool twilight fill on right
    const lamp = ctx.createRadialGradient(W * 0.16, H * 0.66, 10, W * 0.16, H * 0.66, W * 0.36);
    lamp.addColorStop(0, `rgba(255, 182, 102, ${0.26 + 0.03 * Math.sin(t * 2.4)})`);
    lamp.addColorStop(1, 'rgba(255, 160, 80, 0)');
    ctx.fillStyle = lamp;
    ctx.fillRect(0, 0, W, H);
  }

  _drawMidnightLoft(ctx, W, H, t, ox, oy) {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#090b1a');
    g.addColorStop(0.55, '#131632');
    g.addColorStop(1, '#090912');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // Soft neon skyline bokeh pillars
    const colors = ['#ff4db8', '#35e4ff', '#9c6bff', '#ff79d1', '#42c6ff'];
    for (let i = 0; i < 18; i++) {
      const bx = ((i * 0.064 + 0.02) * W) - ox * (0.3 + (i % 3) * 0.18);
      const bh = (0.24 + ((i * 37) % 10) * 0.045) * H;
      const bw = 26 + (i % 4) * 16;
      ctx.fillStyle = 'rgba(18, 22, 46, 0.85)';
      ctx.fillRect(bx, H * 0.78 - bh - oy * 0.4, bw, bh + H * 0.25);
      // Neon accent strip
      ctx.fillStyle = colors[i % colors.length];
      ctx.globalAlpha = 0.28 + 0.12 * Math.sin(t * 1.4 + i);
      ctx.fillRect(bx + 3, H * 0.78 - bh + 12 - oy * 0.4, 3, bh * 0.55);
      ctx.globalAlpha = 1.0;
    }

    // Wet balcony horizon reflection glow
    const pinkGlow = ctx.createRadialGradient(W * 0.24, H * 0.55, 10, W * 0.24, H * 0.55, W * 0.4);
    pinkGlow.addColorStop(0, 'rgba(255, 84, 198, 0.20)');
    pinkGlow.addColorStop(1, 'rgba(255, 84, 198, 0)');
    ctx.fillStyle = pinkGlow;
    ctx.fillRect(0, 0, W, H);

    const cyanGlow = ctx.createRadialGradient(W * 0.78, H * 0.48, 10, W * 0.78, H * 0.48, W * 0.4);
    cyanGlow.addColorStop(0, 'rgba(58, 232, 255, 0.18)');
    cyanGlow.addColorStop(1, 'rgba(58, 232, 255, 0)');
    ctx.fillStyle = cyanGlow;
    ctx.fillRect(0, 0, W, H);

    // Balcony railing
    ctx.fillStyle = 'rgba(10, 10, 18, 0.92)';
    ctx.fillRect(0, H * 0.79 - oy * 0.8, W, 8);
    ctx.fillRect(0, H * 0.86 - oy * 0.8, W, H * 0.2);
  }

  _drawSakuraPavilion(ctx, W, H, t, ox, oy) {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#131836');
    g.addColorStop(0.5, '#2b2550');
    g.addColorStop(0.85, '#593760');
    g.addColorStop(1, '#1c1528');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // Luminous full moon
    const mx = W * 0.70 - ox * 0.25, my = H * 0.22 - oy * 0.25;
    const mg = ctx.createRadialGradient(mx, my, 12, mx, my, 180);
    mg.addColorStop(0, 'rgba(255, 246, 232, 0.92)');
    mg.addColorStop(0.24, 'rgba(255, 224, 236, 0.42)');
    mg.addColorStop(1, 'rgba(255, 180, 220, 0)');
    ctx.fillStyle = mg;
    ctx.fillRect(0, 0, W, H);

    // Soft blossom canopy glow top-left
    const sg = ctx.createRadialGradient(W * 0.16 - ox * 0.6, H * 0.14 - oy * 0.6, 20, W * 0.16, H * 0.14, W * 0.42);
    sg.addColorStop(0, 'rgba(255, 182, 216, 0.34)');
    sg.addColorStop(1, 'rgba(255, 160, 205, 0)');
    ctx.fillStyle = sg;
    ctx.fillRect(0, 0, W, H);

    // Reflective water horizon
    ctx.fillStyle = 'rgba(22, 18, 38, 0.72)';
    ctx.fillRect(0, H * 0.74 - oy * 0.5, W, H * 0.3);
  }

  _drawObsidianStudio(ctx, W, H, t, ox, oy) {
    const g = ctx.createRadialGradient(W * 0.5 - ox * 0.4, H * 0.38 - oy * 0.4, 40, W * 0.5, H * 0.45, Math.max(W, H) * 0.78);
    g.addColorStop(0, '#272438');
    g.addColorStop(0.55, '#13111d');
    g.addColorStop(1, '#08070d');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    const g1 = ctx.createRadialGradient(W * 0.18, H * 0.26, 10, W * 0.18, H * 0.26, W * 0.45);
    g1.addColorStop(0, 'rgba(244, 152, 198, 0.14)');
    g1.addColorStop(1, 'transparent');
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, W, H);

    const g2 = ctx.createRadialGradient(W * 0.84, H * 0.45, 10, W * 0.84, H * 0.45, W * 0.45);
    g2.addColorStop(0, 'rgba(123, 228, 255, 0.11)');
    g2.addColorStop(1, 'transparent');
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);
  }

  _drawParticles(ctx, W, H, t, mode) {
    if (!mode || mode === 'none') return;
    for (const p of this.particles) {
      if (mode === 'rain') {
        const y = ((p.y + t * (0.45 + p.s * 0.25)) % 1.1) * H;
        const x = ((p.x - t * 0.08 + 10) % 1.0) * W;
        ctx.strokeStyle = 'rgba(165, 225, 255, 0.26)';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x - 6, y + 18 * p.s);
        ctx.stroke();
      } else if (mode === 'sakura') {
        const y = ((p.y + t * (0.04 + p.s * 0.02)) % 1.08) * H;
        const x = ((p.x + Math.sin(t * 0.9 + p.ph) * 0.04 + t * 0.02) % 1.05) * W;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(t * 0.9 + p.ph);
        ctx.fillStyle = 'rgba(255, 195, 224, 0.55)';
        ctx.beginPath();
        ctx.ellipse(0, 0, 5.5 * p.s, 2.8 * p.s, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      } else {
        // embers or dust motes gently rising
        const y = (1.05 - ((p.y + t * (0.018 + p.s * 0.012)) % 1.1)) * H;
        const x = ((p.x + Math.sin(t * 0.6 + p.ph) * 0.025 + 1) % 1.0) * W;
        const alpha = 0.22 + 0.28 * (0.5 + 0.5 * Math.sin(t * 2.1 + p.ph));
        ctx.fillStyle = mode === 'embers' ? `rgba(255, 204, 138, ${alpha})` : `rgba(244, 210, 232, ${alpha * 0.7})`;
        ctx.beginPath();
        ctx.arc(x, y, 1.8 * p.s, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
