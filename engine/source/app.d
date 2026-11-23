module app;

import bindbc.opengl;
import core.stdc.string : memcpy;
import core.sys.windows.windows : FreeLibrary, GetProcAddress, LoadLibraryW;
import inochi2d;
import std.conv : to;
import std.exception : enforce;
import std.math : cos, sin, PI;
import core.memory : GC;
import std.process : environment;
import std.stdio : stderr, stdin, stdout;
import std.string : fromStringz, join, toStringz;

private double clockTime = 0.0;
private double clock() { return clockTime; }

private immutable string vertSrc = q{
#version 330 core
out vec2 vUV;
void main() {
    vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
    vUV = 0.5 * (pos[gl_VertexID] + 1.0);
}
};

// Real-Time 60 FPS Fully Tweakable Anime VTuber Post-Process Shader:
// - Directional dual-key/fill silhouette rim lighting (live colors, angle, width, strength)
// - Selective eye-sparkle & vivid hair specular bloom (live strength & threshold, skin-safe)
// - Room-matched ambient edge light wrap (integrates character seamlessly into any room!)
// - Color grading (vibrance, contrast, brightness, warmth/temp, tint RGB, subtle chromatic aberration)
private immutable string fragSrc = q{
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTex;
uniform sampler2D uEmissive;
uniform vec2 uTexel;
uniform vec2 uSrcTexel;

uniform float uRimStrength;
uniform float uRimWidth;
uniform vec3 uKeyColor;
uniform vec3 uFillColor;
uniform vec2 uKeyDir;
uniform vec2 uFillDir;

uniform float uBloomStrength;
uniform float uBloomThresh;

uniform float uVibrance;
uniform float uContrast;
uniform float uBrightness;
uniform float uWarmth;
uniform vec3 uTint;

uniform vec3 uAmbientColor;
uniform float uAmbientWrap;
uniform float uChromaAb;
uniform float uSharpness;

// The shader may sample Inochi2D's own framebuffer textures, which keep the
// default REPEAT wrap. Clamping to the edge texel centres gives exactly
// CLAMP_TO_EDGE results without changing Inochi2D's texture state.
vec4 tex0(vec2 uv) { return texture(uTex, clamp(uv, uSrcTexel * 0.5, vec2(1.0) - uSrcTexel * 0.5)); }
vec4 tex1(vec2 uv) { return texture(uEmissive, clamp(uv, uSrcTexel * 0.5, vec2(1.0) - uSrcTexel * 0.5)); }

vec4 sampleSS(vec2 uv) {
    vec2 o = uSrcTexel * 0.5;
    vec4 s0 = tex0(clamp(uv + vec2(-o.x, -o.y), vec2(0.0), vec2(1.0)));
    vec4 s1 = tex0(clamp(uv + vec2( o.x, -o.y), vec2(0.0), vec2(1.0)));
    vec4 s2 = tex0(clamp(uv + vec2(-o.x,  o.y), vec2(0.0), vec2(1.0)));
    vec4 s3 = tex0(clamp(uv + vec2( o.x,  o.y), vec2(0.0), vec2(1.0)));
    return 0.25 * (s0 + s1 + s2 + s3);
}

vec3 sampleStraightRGB(vec2 uv) {
    vec4 s = sampleSS(uv);
    return (s.a > 0.01) ? (s.rgb / s.a) : vec3(0.0);
}

void main() {
    vec4 raw = sampleSS(vUV);
    float a = raw.a;
    vec3 col = (a > 0.001) ? (raw.rgb / a) : vec3(0.0);

    // 0. Contrast-Adaptive Anime Lineart Sharpening (FidelityFX CAS-inspired, halo-free!)
    if (uSharpness > 0.001 && a > 0.05) {
        vec4 sN = sampleSS(vUV + vec2( 0.0,  1.0) * uTexel);
        vec4 sS = sampleSS(vUV + vec2( 0.0, -1.0) * uTexel);
        vec4 sE = sampleSS(vUV + vec2( 1.0,  0.0) * uTexel);
        vec4 sW = sampleSS(vUV + vec2(-1.0,  0.0) * uTexel);
        vec3 n = (sN.a > 0.01) ? (sN.rgb / sN.a) : col;
        vec3 s = (sS.a > 0.01) ? (sS.rgb / sS.a) : col;
        vec3 e = (sE.a > 0.01) ? (sE.rgb / sE.a) : col;
        vec3 w = (sW.a > 0.01) ? (sW.rgb / sW.a) : col;
        // Only sharpen solidly opaque interior (preserves anti-aliased outer silhouette!)
        float inner = smoothstep(0.85, 0.995, min(min(sN.a, sS.a), min(sE.a, sW.a)));
        vec3 mn = min(col, min(min(n, s), min(e, w)));
        vec3 mx = max(col, max(max(n, s), max(e, w)));
        // Adaptive weight suppresses ringing on high-contrast edges while crisping soft lineart
        vec3 amp = clamp(min(mn, 1.0 - mx) / max(mx, vec3(0.001)), vec3(0.0), vec3(1.0));
        vec3 wSharp = -sqrt(amp) * (0.13 * clamp(uSharpness, 0.0, 1.4));
        vec3 sharpened = (col + (n + s + e + w) * wSharp) / (1.0 + 4.0 * wSharp);
        col = mix(col, clamp(sharpened, vec3(0.0), vec3(1.0)), inner);
    }

    // Optional subtle anime lens chromatic aberration on outer edges
    if (uChromaAb > 0.001 && a > 0.01) {
        vec2 dir = (vUV - vec2(0.5)) * uChromaAb * 6.0 * uTexel;
        vec4 sR = tex0(clamp(vUV + dir, vec2(0.0), vec2(1.0)));
        vec4 sB = tex0(clamp(vUV - dir, vec2(0.0), vec2(1.0)));
        float rC = (sR.a > 0.01) ? (sR.r / sR.a) : col.r;
        float bC = (sB.a > 0.01) ? (sB.b / sB.a) : col.b;
        col.r = mix(col.r, rC, 0.75);
        col.b = mix(col.b, bC, 0.75);
    }

    // 1. Selective Specular Bloom (eye sparkles & vivid hair gloss, skin-safe!)
    vec3 bloom = vec3(0.0);
    if (uBloomStrength > 0.001) {
        float wSum = 0.0;
        const float dirs[8] = float[8](0.0, 0.785398, 1.570796, 2.356194, 3.141592, 3.926990, 4.712389, 5.497787);
        float threshWhite = clamp(uBloomThresh, 0.65, 0.98);
        float threshVivid = clamp(uBloomThresh - 0.14, 0.45, 0.92);
        for (int r = 1; r <= 3; ++r) {
            float rad = float(r) * 2.8;
            float w = (r == 1) ? 0.52 : ((r == 2) ? 0.30 : 0.18);
            for (int i = 0; i < 8; ++i) {
                vec2 off = vec2(cos(dirs[i]), sin(dirs[i])) * rad * uTexel;
                vec4 s = tex0(clamp(vUV + off, vec2(0.0), vec2(1.0)));
                vec3 sc = (s.a > 0.01) ? (s.rgb / s.a) : vec3(0.0);
                float mx = max(sc.r, max(sc.g, sc.b));
                float mn = min(sc.r, min(sc.g, sc.b));
                float sat = (mx > 0.01) ? ((mx - mn) / mx) : 0.0;
                float whiteHi = smoothstep(threshWhite, min(0.995, threshWhite + 0.08), mn);
                float vividHi = smoothstep(threshVivid, min(0.99, threshVivid + 0.18), mx) * smoothstep(0.30, 0.56, sat);
                float hi = max(whiteHi, vividHi * 0.68);
                bloom += s.rgb * hi * w;
                wSum += w;
            }
        }
        bloom /= max(wSum, 0.001);
    }

    // 2. Directional Dual Rim-Lighting + Room Ambient Edge Wrap
    vec3 rim = vec3(0.0);
    if (a > 0.05) {
        float rw = max(0.3, uRimWidth);
        if (uRimStrength > 0.001) {
            float rad1 = 2.0 * rw;
            float rad2 = 4.4 * rw;
            float aKey1 = tex0(vUV + uKeyDir * rad1 * uTexel).a;
            float aKey2 = tex0(vUV + uKeyDir * rad2 * uTexel).a;
            float keyEdge = clamp(1.0 - min(aKey1, aKey2), 0.0, 1.0);
            float aInKey = tex0(vUV - uKeyDir * 1.5 * uTexel).a;
            keyEdge *= smoothstep(0.25, 0.90, aInKey);

            float aFill1 = tex0(vUV + uFillDir * rad1 * uTexel).a;
            float aFill2 = tex0(vUV + uFillDir * rad2 * uTexel).a;
            float fillEdge = clamp(1.0 - min(aFill1, aFill2), 0.0, 1.0);
            float aInFill = tex0(vUV - uFillDir * 1.5 * uTexel).a;
            fillEdge *= smoothstep(0.25, 0.90, aInFill);

            float darkBoost = mix(1.0, 0.44, smoothstep(0.65, 0.92, dot(col, vec3(0.299, 0.587, 0.114))));
            rim += (uKeyColor * keyEdge * 0.36 + uFillColor * fillEdge * 0.25) * uRimStrength * darkBoost;
        }

        // Room Ambient Light Wrap (soft colored spill around outer silhouette + subtle global tint)
        if (uAmbientWrap > 0.001) {
            float aRing = 0.0;
            aRing += tex0(vUV + vec2( 7.0,  0.0) * uTexel).a;
            aRing += tex0(vUV + vec2(-7.0,  0.0) * uTexel).a;
            aRing += tex0(vUV + vec2( 0.0,  7.0) * uTexel).a;
            aRing += tex0(vUV + vec2( 0.0, -7.0) * uTexel).a;
            aRing *= 0.25;
            float wrapEdge = clamp(1.0 - aRing, 0.0, 1.0);
            col = mix(col, col * mix(vec3(1.0), uAmbientColor * 1.25, 0.28), uAmbientWrap * 0.55);
            rim += uAmbientColor * wrapEdge * uAmbientWrap * 0.32;
        }

        // Subtle anime studio shading gradient
        float vertGrad = mix(0.95, 1.02, smoothstep(0.12, 0.82, vUV.y));
        col *= vertGrad;
    }

    // 3. Color Grading: Warmth, Brightness, Vibrance, Contrast, Tint
    col *= uBrightness;
    col.r *= (1.0 + uWarmth * 0.14);
    col.b *= (1.0 - uWarmth * 0.14);
    col *= uTint;

    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(luma), col, uVibrance);
    col = mix(vec3(0.5), col, uContrast);
    col = clamp(col + rim, vec3(0.0), vec3(1.0));

    // 3.5 Native Inochi2D Per-Part GPU Emissive Glow (Galaxy/Solid-Pupil Eyes + Cyber Halter LEDs!)
    vec3 emCore = tex1(vUV).rgb;
    vec3 emHalo = vec3(0.0);
    {
        const float edirs[8] = float[8](0.0, 0.785398, 1.570796, 2.356194, 3.141592, 3.926990, 4.712389, 5.497787);
        for (int r = 1; r <= 2; ++r) {
            float rad = float(r) * 3.2;
            float w = (r == 1) ? 0.62 : 0.38;
            for (int i = 0; i < 8; ++i) {
                vec2 off = vec2(cos(edirs[i]), sin(edirs[i])) * rad * uTexel;
                emHalo += tex1(clamp(vUV + off, vec2(0.0), vec2(1.0))).rgb * w;
            }
        }
        emHalo /= 8.0;
    }
    vec3 emTotal = emCore * 0.62 + emHalo * 0.42;

    // Re-premultiply and add dreamy specular bloom + true per-part GPU emissive glow!
    vec3 premul = col * a + bloom * (uBloomStrength * 0.44) + emTotal;
    float emLuma = dot(emTotal, vec3(0.299, 0.587, 0.114));
    float outA = clamp(a + dot(bloom * (uBloomStrength * 0.44), vec3(0.33)) * 0.35 + emLuma * 0.45, 0.0, 1.0);
    fragColor = vec4(clamp(premul, vec3(0.0), vec3(1.0)), outA);
}
};

private GLuint compileShader(GLenum type, string src) {
    auto sh = glCreateShader(type);
    const(char)* ptr = src.ptr;
    GLint len = cast(GLint)src.length;
    glShaderSource(sh, 1, &ptr, &len);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char[1024] log;
        GLsizei n = 0;
        glGetShaderInfoLog(sh, 1024, &n, log.ptr);
        enforce(false, "Shader compile failed: " ~ cast(string)log[0 .. n].idup);
    }
    return sh;
}

// ---------------------------------------------------------------------------
// Zero-allocation command parsing (runs once per frame at 60 FPS).

private bool isSpace(char c) @nogc nothrow pure @safe {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/// Whitespace tokenizer over a slice of the reused line buffer. No copies.
private struct Tokens {
    const(char)[] s;
    size_t i;

    const(char)[] next() @nogc nothrow pure @safe {
        while (i < s.length && isSpace(s[i])) ++i;
        immutable b = i;
        while (i < s.length && !isSpace(s[i])) ++i;
        return s[b .. i];
    }
}

private bool sameIgnoreCase(const(char)[] a, string b) @nogc nothrow pure @safe {
    if (a.length != b.length) return false;
    foreach (k; 0 .. a.length) {
        char c = a[k];
        if (c >= 'A' && c <= 'Z') c = cast(char)(c + 32);
        if (c != b[k]) return false;
    }
    return true;
}

/// Parses a decimal number like "-12.3450", "1e-05", "nan" or "inf".
/// Returns false (leaving v untouched) on anything else.
private bool parseNum(const(char)[] t, ref double v) @nogc nothrow pure @safe {
    static immutable double[23] p10 = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
        1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22];
    size_t i = 0;
    bool neg = false;
    if (t.length && (t[0] == '-' || t[0] == '+')) { neg = t[0] == '-'; i = 1; }
    auto rest = t[i .. $];
    if (sameIgnoreCase(rest, "nan")) { v = double.nan; return true; }
    if (sameIgnoreCase(rest, "inf") || sameIgnoreCase(rest, "infinity")) {
        v = neg ? -double.infinity : double.infinity;
        return true;
    }

    ulong mant = 0;
    int exp10 = 0;
    bool any = false;
    for (; i < t.length && t[i] >= '0' && t[i] <= '9'; ++i) {
        any = true;
        if (mant < 1_000_000_000_000_000_000UL) mant = mant * 10 + (t[i] - '0');
        else ++exp10;
    }
    if (i < t.length && t[i] == '.') {
        for (++i; i < t.length && t[i] >= '0' && t[i] <= '9'; ++i) {
            any = true;
            if (mant < 1_000_000_000_000_000_000UL) { mant = mant * 10 + (t[i] - '0'); --exp10; }
        }
    }
    if (!any) return false;
    if (i < t.length && (t[i] == 'e' || t[i] == 'E')) {
        ++i;
        bool eneg = false;
        if (i < t.length && (t[i] == '-' || t[i] == '+')) { eneg = t[i] == '-'; ++i; }
        int e = 0;
        bool ed = false;
        for (; i < t.length && t[i] >= '0' && t[i] <= '9'; ++i) {
            ed = true;
            if (e < 10_000) e = e * 10 + (t[i] - '0');
        }
        if (!ed) return false;
        exp10 += eneg ? -e : e;
    }
    if (i != t.length) return false;

    double r = cast(double)mant;
    if (mant != 0) {
        int e = exp10 < 0 ? -exp10 : exp10;
        while (e > 22) { r = exp10 < 0 ? r / 1e22 : r * 1e22; e -= 22; }
        r = exp10 < 0 ? r / p10[e] : r * p10[e];
    }
    v = neg ? -r : r;
    return true;
}

unittest {
    double v = 7;
    assert(parseNum("0.0166667", v) && v == 0.0166667);
    assert(parseNum("-12.3450", v) && v == -12.345);
    assert(parseNum("365.0000", v) && v == 365.0);
    assert(parseNum("1e-05", v) && v == 1e-05);
    assert(parseNum("+2.5E3", v) && v == 2500.0);
    assert(parseNum("135.00", v) && v == 135.0);
    assert(parseNum("0", v) && v == 0.0);
    assert(parseNum("-0.00000", v) && v == 0.0);
    assert(parseNum("nan", v) && v != v);
    assert(parseNum("-inf", v) && v == -double.infinity);
    v = 7;
    assert(!parseNum("", v) && v == 7);
    assert(!parseNum("-", v) && v == 7);
    assert(!parseNum("1.2.3", v) && v == 7);
    assert(!parseNum("abc", v) && v == 7);
    assert(!parseNum("1e", v) && v == 7);
    auto tk = Tokens("  F 0.5\t2 | 3 \n");
    assert(tk.next() == "F" && tk.next() == "0.5" && tk.next() == "2" && tk.next() == "|");
    assert(tk.next() == "3" && tk.next() == "" && tk.next() == "");
}

// ---------------------------------------------------------------------------

/// Texture name attached to `attachment` of `fbo`, or 0 when it is a
/// renderbuffer, missing, or not mip level 0 of a plain 2D texture.
private GLuint attachedTexture(GLuint fbo, GLenum attachment) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    scope (exit) glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    GLint type = GL_NONE, name = 0, level = -1, face = 0;
    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
    if (type != GL_TEXTURE) return 0;
    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &level);
    glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, attachment,
        GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE, &face);
    return (name > 0 && level == 0 && face == 0) ? cast(GLuint)name : 0;
}

/// Fallback copy target (only created if Inochi2D ever attaches a renderbuffer).
private GLuint makeCopyTexture(int w, int h) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, null);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

void main(string[] args) {
    enforce(args.length >= 3,
        "Usage: hikari-live-engine <SDL2 DLL> <model.inp> [--width W] [--height H]");
    int width = 720;
    int height = 810;
    for (size_t i = 3; i < args.length; ++i) {
        if (args[i] == "--width" && i + 1 < args.length) width = args[++i].to!int;
        else if (args[i] == "--height" && i + 1 < args.length) height = args[++i].to!int;
        else enforce(false, "Unknown argument: " ~ args[i]);
    }

    auto sdl = LoadLibraryW((args[1] ~ "\0").to!wstring.ptr);
    enforce(sdl !is null, "Could not load SDL2");
    scope (exit) FreeLibrary(sdl);
    auto symbol(string name, T)() {
        auto entry = cast(T)GetProcAddress(sdl, name.toStringz);
        enforce(entry !is null, "Missing SDL2 function: " ~ name);
        return entry;
    }
    alias Init = extern(C) int function(uint);
    alias Quit = extern(C) void function();
    alias Attribute = extern(C) int function(int, int);
    alias Window = extern(C) void* function(const(char)*, int, int, int, int, uint);
    alias WindowCleanup = extern(C) void function(void*);
    alias Context = extern(C) void* function(void*);
    alias ContextCleanup = extern(C) void function(void*);
    alias ErrorText = extern(C) const(char)* function();

    auto errorText = symbol!("SDL_GetError", ErrorText)();
    auto init = symbol!("SDL_Init", Init)();
    enforce(init(0x20) == 0, "SDL video init failed: " ~ errorText().fromStringz.idup);
    scope (exit) symbol!("SDL_Quit", Quit)()();
    auto attribute = symbol!("SDL_GL_SetAttribute", Attribute)();
    enforce(attribute(17, 3) == 0 && attribute(18, 2) == 0 && attribute(21, 1) == 0 &&
            attribute(7, 8) == 0 && attribute(5, 0) == 0, "SDL OpenGL setup failed");
    auto window = symbol!("SDL_CreateWindow", Window)()("Hikari Live Engine", 0, 0, width, height, 0xA);
    enforce(window !is null, "SDL window failed: " ~ errorText().fromStringz.idup);
    scope (exit) symbol!("SDL_DestroyWindow", WindowCleanup)()(window);
    auto context = symbol!("SDL_GL_CreateContext", Context)()(window);
    enforce(context !is null, "SDL context failed: " ~ errorText().fromStringz.idup);
    scope (exit) symbol!("SDL_GL_DeleteContext", ContextCleanup)()(context);
    auto support = loadOpenGL();
    enforce(support >= GLSupport.gl32, "OpenGL 3.2 required, found " ~ support.to!string);

    enum int SSAA = 2;
    int ssW = width * SSAA;
    int ssH = height * SSAA;

    inInit(&clock);
    inSetViewport(ssW, ssH);
    inSetClearColor(0, 0, 0, 0);
    auto camera = inGetCamera();
    camera.scale = vec2(1.55f * SSAA);
    camera.position = vec2(0.0f, 365.0f);

    // Compile shader + post FBO
    auto vs = compileShader(GL_VERTEX_SHADER, vertSrc);
    auto fs = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    auto prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint locTex = glGetUniformLocation(prog, "uTex");
    GLint locEmissive = glGetUniformLocation(prog, "uEmissive");
    GLint locTexel = glGetUniformLocation(prog, "uTexel");
    GLint locSrcTexel = glGetUniformLocation(prog, "uSrcTexel");
    GLint locRimStr = glGetUniformLocation(prog, "uRimStrength");
    GLint locRimWid = glGetUniformLocation(prog, "uRimWidth");
    GLint locKeyCol = glGetUniformLocation(prog, "uKeyColor");
    GLint locFillCol = glGetUniformLocation(prog, "uFillColor");
    GLint locKeyDir = glGetUniformLocation(prog, "uKeyDir");
    GLint locFillDir = glGetUniformLocation(prog, "uFillDir");
    GLint locBloomStr = glGetUniformLocation(prog, "uBloomStrength");
    GLint locBloomThr = glGetUniformLocation(prog, "uBloomThresh");
    GLint locVib = glGetUniformLocation(prog, "uVibrance");
    GLint locCon = glGetUniformLocation(prog, "uContrast");
    GLint locBri = glGetUniformLocation(prog, "uBrightness");
    GLint locWarm = glGetUniformLocation(prog, "uWarmth");
    GLint locTint = glGetUniformLocation(prog, "uTint");
    GLint locAmbCol = glGetUniformLocation(prog, "uAmbientColor");
    GLint locAmbWrap = glGetUniformLocation(prog, "uAmbientWrap");
    GLint locChroma = glGetUniformLocation(prog, "uChromaAb");
    GLint locSharp = glGetUniformLocation(prog, "uSharpness");

    // srcTex/emTex are only created as a fallback; normally the shader reads
    // Inochi2D's framebuffer textures directly (no 31.6 MB/frame copy).
    GLuint vao = 0, srcTex = 0, emTex = 0, postTex = 0, postFbo = 0;
    glGenVertexArrays(1, &vao);

    glGenTextures(1, &postTex);
    glBindTexture(GL_TEXTURE_2D, postTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenFramebuffers(1, &postFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, postFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postTex, 0);
    enforce(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Post FBO incomplete");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto puppet = inLoadPuppet(args[2]);
    enforce(puppet !is null, "Could not load the Inochi2D puppet");
    puppet.enableDrivers = true;

    string[] pNames;
    foreach (p; puppet.parameters) pNames ~= p.name;
    stderr.writeln("READY ", pNames.length, " ", pNames.join("|"));
    stderr.flush();

    size_t rowSize = cast(size_t)width * 4;
    auto pixels = new ubyte[rowSize * height];
    auto upright = new ubyte[pixels.length];

    // Everything the loop needs is allocated once, here.
    size_t nParams = puppet.parameters.length;
    auto paramVals = new double[nParams * 2];
    // readln is handed the full-length storage every frame, so it writes in
    // place and never grows (passing it last frame's shorter slice would make
    // Phobos reallocate whenever a line gets longer again). Sized for ~31
    // header numbers + 2 per parameter, each well under 24 characters.
    auto lineStore = new char[4096 + nParams * 2 * 24];
    char[] lineBuf;

    // Header fields after "F". Bad numbers keep last frame's value instead of crashing.
    enum size_t HEADER = 31;  // "F" + 30 values; the last 3 are optional
    double[HEADER] hv = 0.0;

    // Inochi2D's color attachments, resolved once per framebuffer object.
    GLuint fbSeen = 0, albedoTex = 0, emissiveTex = 0;

    // HIKARI_ENGINE_GCSTATS=1 prints GC bytes allocated by this thread after
    // 120 warm-up frames, on exit. Never printed in normal runs (stderr is
    // only read up to READY).
    immutable bool gcStats = environment.get("HIKARI_ENGINE_GCSTATS", "") == "1";
    ulong frames = 0, gcAtWarm = 0;
    scope (exit) if (gcStats) {
        stderr.writeln("GCSTATS frames=", frames, " bytes_after_warmup=",
            GC.allocatedInCurrentThread() - gcAtWarm,
            " path=", albedoTex ? "direct" : "copy");
        stderr.flush();
    }

    // Interactive real-time loop: reads one command line per frame, emits raw RGBA bytes
    while (true) {
        lineBuf = lineStore;
        if (stdin.readln(lineBuf) == 0) break;
        if (lineBuf.length > lineStore.length / 2) lineStore = new char[lineBuf.length * 2];  // rare: oversize line
        const(char)[] line = lineBuf;
        size_t lo = 0, hi = line.length;
        while (lo < hi && isSpace(line[lo])) ++lo;
        while (hi > lo && isSpace(line[hi - 1])) --hi;
        line = line[lo .. hi];
        if (line.length == 0) continue;
        if (line == "QUIT") break;

        size_t bar = 0;
        while (bar < line.length && line[bar] != '|') ++bar;
        if (bar == line.length) continue;

        // Header: "F dt scale x y phys shader rim... sharpness"
        auto th = Tokens(line[0 .. bar]);
        if (th.next() != "F") continue;
        hv[28] = 0.0; hv[29] = 0.0; hv[30] = 0.65;  // optional field defaults
        size_t nh = 1;
        for (auto t = th.next(); t.length; t = th.next()) {
            if (nh < HEADER) parseNum(t, hv[nh]);
            ++nh;
        }
        if (nh < 28) continue;

        double dt = hv[1];
        float camScale = hv[2], camX = hv[3], camY = hv[4];
        bool physOn = hv[5] != 0.0;
        bool shaderOn = hv[6] != 0.0;

        float rimStr = hv[7];
        float rimWid = hv[8];
        float keyR = hv[9], keyG = hv[10], keyB = hv[11];
        float fillR = hv[12], fillG = hv[13], fillB = hv[14];
        float keyDeg = hv[15];
        float bloomStr = hv[16];
        float bloomThr = hv[17];
        float vib = hv[18];
        float con = hv[19];
        float bri = hv[20];
        float warm = hv[21];
        float tintR = hv[22], tintG = hv[23], tintB = hv[24];
        float ambR = hv[25], ambG = hv[26], ambB = hv[27];
        float ambWrap = hv[28];
        float chromaAb = hv[29];
        float sharpness = hv[30];

        // Parameter pairs: only applied when the full set parsed cleanly.
        auto tv = Tokens(line[bar + 1 .. $]);
        size_t nv = 0;
        bool valsOk = true;
        for (auto t = tv.next(); t.length && nv < paramVals.length; t = tv.next())
            valsOk &= parseNum(t, paramVals[nv++]);
        if (valsOk && nv == paramVals.length) {
            foreach (i; 0 .. nParams)
                puppet.parameters[i].value = vec2(cast(float)paramVals[i * 2], cast(float)paramVals[i * 2 + 1]);
        }

        camera.scale = vec2(camScale * cast(float)SSAA);
        camera.position = vec2(camX, camY);
        puppet.enableDrivers = physOn;

        clockTime += (dt > 0.0005 && dt < 0.1) ? dt : (1.0 / 60.0);
        inUpdate();
        puppet.update();
        inBeginScene();
        puppet.draw();
        inEndScene();

        GLuint fb = inGetFramebuffer();
        if (fb != fbSeen) {
            fbSeen = fb;
            albedoTex = attachedTexture(fb, GL_COLOR_ATTACHMENT0);
            emissiveTex = attachedTexture(fb, GL_COLOR_ATTACHMENT1);
            if (!albedoTex && !srcTex) srcTex = makeCopyTexture(ssW, ssH);
            if (!emissiveTex && !emTex) emTex = makeCopyTexture(ssW, ssH);
        }

        if (shaderOn) {
            // Fallback only: copy attachments that are not sampleable textures.
            if (!albedoTex || !emissiveTex) {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
                if (!albedoTex) {
                    glReadBuffer(GL_COLOR_ATTACHMENT0);
                    glBindTexture(GL_TEXTURE_2D, srcTex);
                    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, ssW, ssH);
                }
                if (!emissiveTex) {
                    glReadBuffer(GL_COLOR_ATTACHMENT1);
                    glBindTexture(GL_TEXTURE_2D, emTex);
                    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, ssW, ssH);
                }
                glReadBuffer(GL_COLOR_ATTACHMENT0);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, postFbo);
            glViewport(0, 0, width, height);
            glDisable(GL_BLEND);
            glDisable(GL_STENCIL_TEST);
            glUseProgram(prog);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, albedoTex ? albedoTex : srcTex);
            glUniform1i(locTex, 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, emissiveTex ? emissiveTex : emTex);
            glUniform1i(locEmissive, 1);
            glUniform2f(locTexel, 1.0f / width, 1.0f / height);
            glUniform2f(locSrcTexel, 1.0f / ssW, 1.0f / ssH);
            glUniform1f(locRimStr, rimStr);
            glUniform1f(locRimWid, rimWid);
            glUniform3f(locKeyCol, keyR, keyG, keyB);
            glUniform3f(locFillCol, fillR, fillG, fillB);

            float radKey = keyDeg * cast(float)(PI / 180.0);
            float radFill = radKey + cast(float)(PI * 0.72);
            glUniform2f(locKeyDir, cos(radKey), sin(radKey));
            glUniform2f(locFillDir, cos(radFill), sin(radFill));

            glUniform1f(locBloomStr, bloomStr);
            glUniform1f(locBloomThr, bloomThr);
            glUniform1f(locVib, vib);
            glUniform1f(locCon, con);
            glUniform1f(locBri, bri);
            glUniform1f(locWarm, warm);
            glUniform3f(locTint, tintR, tintG, tintB);
            glUniform3f(locAmbCol, ambR, ambG, ambB);
            glUniform1f(locAmbWrap, ambWrap);
            glUniform1f(locChroma, chromaAb);
            glUniform1f(locSharp, sharpness);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // Unbind Inochi2D's textures so its next frame never renders into
            // a texture that is still bound for sampling.
            glBindTexture(GL_TEXTURE_2D, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, postFbo);
        } else {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
            // Always blit the albedo, never a leftover emissive read buffer.
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, postFbo);
            glBlitFramebuffer(0, 0, ssW, ssH, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, postFbo);
        }

        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.ptr);
        foreach (y; 0 .. height)
            memcpy(upright.ptr + y * rowSize, pixels.ptr + (height - 1 - y) * rowSize, rowSize);
        stdout.rawWrite(upright);
        stdout.flush();

        if (gcStats && ++frames == 120) gcAtWarm = GC.allocatedInCurrentThread();
    }
}
