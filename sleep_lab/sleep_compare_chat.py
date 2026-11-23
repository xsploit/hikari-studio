"""Hikari SLEEP Lab — 2-Box Side-by-Side Zero-Context Test Chat (Base vs Slept LoRA).

Supports BOTH backends seamlessly:
  1. PEFT / Transformers Mode (works right now with `Qwen/Qwen3-0.6B` + `adapters/hikari_sleep`!)
  2. GGUF / CUDA 12 `llama-server.exe` Mode (for Colab's `Qwen3-4B` GGUF + `hikari-sleep-lora.gguf`!)

Zero memories or RAG are ever injected into the prompt — pure Base vs Slept LoRA comparison!
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlparse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAB_DIR = Path(__file__).resolve().parent
QUIZ_PATH = LAB_DIR / "datasets" / "eval_recall_quiz.jsonl"
CPU_TEST_ADAPTER = LAB_DIR / "outputs" / "_cpu_test" / "outputs" / "adapters" / "hikari_sleep"

DEFAULT_SYSTEM_PROMPT = (
    "You are Hikari-chan (Hikky C), an AI streamer girl talking with SUBSECT. "
    "Answer naturally and concisely from your own memory. If you don't know, say so honestly."
)


def find_llama_server() -> str:
    env_p = os.environ.get("LLAMA_SERVER_EXE", "").strip()
    if env_p and Path(env_p).is_file():
        return env_p
    home = Path.home()
    candidates = sorted(
        home.glob(".lmstudio/extensions/backends/**/llama-server.exe"),
        key=lambda p: ("cuda12" in str(p).lower(), "cuda" in str(p).lower(), str(p)),
        reverse=True,
    )
    return str(candidates[0]) if candidates else ""


def find_gguf(kind: str) -> str:
    home = Path.home()
    if kind == "lora":
        roots = [LAB_DIR / "outputs", LAB_DIR / "adapters", LAB_DIR, home / "Downloads"]
        hits: list[Path] = []
        for r in roots:
            if r.exists():
                for p in r.rglob("*lora*.gguf"):
                    if p.is_file() and p.stat().st_size > 100_000:
                        hits.append(p)
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(hits[0])
        if CPU_TEST_ADAPTER.is_dir():
            return str(CPU_TEST_ADAPTER)
        return ""
    roots = [
        LAB_DIR / "models",
        home / ".lmstudio" / "models",
        home / ".cache" / "lm-studio" / "models",
    ]
    hits = []
    for r in roots:
        if r.exists():
            for p in r.rglob("*.gguf"):
                name = p.name.lower()
                if any(k in name for k in ("lora", "embed", "nomic", "mmproj")):
                    continue
                if p.is_file() and p.stat().st_size > 400_000_000:
                    hits.append(p)
    if hits:
        hits.sort(key=lambda p: ("qwen3" in p.name.lower() and "4b" in p.name.lower(), p.stat().st_mtime), reverse=True)
        return str(hits[0])
    return "Qwen/Qwen3-0.6B"


def load_quiz() -> list[dict[str, Any]]:
    if not QUIZ_PATH.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in QUIZ_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


class UnifiedRunner:
    def __init__(self, port: int = 8810) -> None:
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.lock = threading.Lock()
        self.mode: str = ""  # "gguf" | "peft"
        self.proc: subprocess.Popen | None = None
        self.peft_model: Any = None
        self.peft_tok: Any = None
        self.peft_device: str = "cpu"
        self.llama_exe: str = find_llama_server()
        self.model_path: str = ""
        self.lora_path: str = ""
        self.status_msg: str = "Ready — click Load Model (Qwen3-0.6B PEFT or Qwen3-4B GGUF)."
        self.ready: bool = False
        atexit.register(self.stop)

    def stop(self) -> None:
        with self.lock:
            p = self.proc
            self.proc = None
            self.peft_model = None
            self.peft_tok = None
            self.ready = False
            self.mode = ""
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=4)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def start(self, model_path: str, lora_path: str, llama_exe: str = "") -> dict[str, Any]:
        model_path = (model_path or "").strip()
        lora_path = (lora_path or "").strip()
        if not model_path:
            return {"ok": False, "error": "Enter a Hugging Face model ID (e.g. Qwen/Qwen3-0.6B) or local .gguf path."}
        if model_path.lower().endswith(".gguf"):
            return self._start_gguf(model_path, lora_path, llama_exe)
        return self._start_peft(model_path, lora_path)

    def _start_peft(self, model_id: str, adapter_dir: str) -> dict[str, Any]:
        self.stop()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except Exception as e:
            return {"ok": False, "error": f"PyTorch/PEFT not available in this Python ({e}). Launch via launch_compare_chat.cmd!"}

        if adapter_dir and not Path(adapter_dir).is_dir():
            return {"ok": False, "error": f"PEFT adapter folder not found: {adapter_dir}"}

        with self.lock:
            self.status_msg = f"Loading {model_id} + PEFT adapter..."
            self.model_path = model_id
            self.lora_path = adapter_dir
        try:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if dev == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(dev)
            if adapter_dir:
                mdl = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            else:
                mdl = base
            mdl.eval()
            with self.lock:
                self.peft_tok = tok
                self.peft_model = mdl
                self.peft_device = dev
                self.mode = "peft"
                self.ready = True
                self.status_msg = f"Loaded {model_id} on {dev.upper()} (PEFT mode — zero-context A/B ready!)"
            return {"ok": True, "status": self.status_msg}
        except Exception as e:
            self.status_msg = f"PEFT load failed: {e}"
            return {"ok": False, "error": self.status_msg}

    def _start_gguf(self, model_path: str, lora_path: str, llama_exe: str = "") -> dict[str, Any]:
        exe = (llama_exe or self.llama_exe or find_llama_server()).strip()
        if not exe or not Path(exe).is_file():
            return {"ok": False, "error": "llama-server.exe not found."}
        if not Path(model_path).is_file():
            return {"ok": False, "error": f"Base model GGUF not found: {model_path}"}
        if lora_path and not Path(lora_path).is_file():
            return {"ok": False, "error": f"LoRA GGUF not found: {lora_path}"}

        self.stop()
        cmd = [
            exe, "-m", model_path,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", "99", "-c", "4096", "--no-webui",
        ]
        if lora_path:
            cmd.extend(["--lora", lora_path, "--lora-init-without-apply"])
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with self.lock:
            self.llama_exe = exe
            self.model_path = model_path
            self.lora_path = lora_path
            self.status_msg = "Loading GGUF onto RTX GPU..."
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags)

        deadline = time.perf_counter() + 45.0
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                err = (self.proc.stderr.read() or b"").decode("utf-8", "replace")[-400:]
                self.status_msg = f"llama-server exited: {err.strip()}"
                return {"ok": False, "error": self.status_msg}
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=1.5) as r:
                    if r.status == 200 and json.loads(r.read().decode("utf-8", "replace")).get("status") == "ok":
                        self.mode = "gguf"
                        self.ready = True
                        self.status_msg = "Loaded GGUF on RTX GPU — zero-context A/B ready!"
                        return {"ok": True, "status": self.status_msg}
            except Exception:
                pass
            time.sleep(0.35)
        self.stop()
        return {"ok": False, "error": "Timed out waiting for llama-server."}

    def _peft_generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> tuple[str, float]:
        import torch
        tok = self.peft_tok
        mdl = self.peft_model
        msgs = []
        if system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": prompt})
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(self.peft_device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = mdl.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.05),
                temperature=max(0.05, temperature) if temperature > 0.05 else 1.0,
                pad_token_id=tok.pad_token_id,
            )
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        out_ids = gen[0][inputs["input_ids"].shape[1]:]
        ans = tok.decode(out_ids, skip_special_tokens=True).strip()
        return ans, ms

    def _set_lora_scale(self, scale: float) -> None:
        if not self.lora_path:
            return
        req = urllib.request.Request(
            f"{self.base_url}/lora-adapters",
            data=json.dumps([{"id": 0, "scale": float(scale)}]).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            r.read()

    def _gguf_chat(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> tuple[str, float]:
        msgs = []
        if system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        msgs.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps({"messages": msgs, "temperature": temperature, "max_tokens": max_tokens, "stream": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return str(data["choices"][0]["message"]["content"] or "").strip(), ms

    def compare(self, prompt: str, system_prompt: str, lora_scale: float = 1.0,
                temperature: float = 0.2, max_tokens: int = 140) -> dict[str, Any]:
        with self.lock:
            if not self.ready:
                return {"ok": False, "error": "Model not loaded yet. Click Load Model first."}
            if self.mode == "peft":
                if self.lora_path and hasattr(self.peft_model, "disable_adapter"):
                    with self.peft_model.disable_adapter():
                        base_text, base_ms = self._peft_generate(prompt, system_prompt, temperature, max_tokens)
                    slept_text, slept_ms = self._peft_generate(prompt, system_prompt, temperature, max_tokens)
                else:
                    base_text, base_ms = self._peft_generate(prompt, system_prompt, temperature, max_tokens)
                    slept_text, slept_ms = "(No PEFT adapter loaded)", 0.0
                return {
                    "ok": True,
                    "base": {"text": base_text, "ms": base_ms, "loraScale": 0.0},
                    "slept": {"text": slept_text, "ms": slept_ms, "loraScale": 1.0 if self.lora_path else 0.0},
                }
            self._set_lora_scale(0.0)
            base_text, base_ms = self._gguf_chat(prompt, system_prompt, temperature, max_tokens)
            if self.lora_path:
                self._set_lora_scale(lora_scale)
                slept_text, slept_ms = self._gguf_chat(prompt, system_prompt, temperature, max_tokens)
            else:
                slept_text, slept_ms = "(No LoRA GGUF loaded)", 0.0
            return {
                "ok": True,
                "base": {"text": base_text, "ms": base_ms, "loraScale": 0.0},
                "slept": {"text": slept_text, "ms": slept_ms, "loraScale": lora_scale if self.lora_path else 0.0},
            }


RUNNER = UnifiedRunner(port=8810)
QUIZ = load_quiz()

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Hikari SLEEP Lab — Zero-Context A/B Compare</title>
<style>
  :root { color-scheme: dark; --bg:#0f0f12; --card:#17171b; --card2:#1f1f25; --bdr:rgba(255,255,255,.08); --txt:#f3f3f6; --mut:#9898a6; --pink:#ff6a9e; --grn:#3dd68c; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif; background:var(--bg); color:var(--txt); min-height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--bdr); display:flex; align-items:center; justify-content:space-between; gap:12px; background:#121216; }
  header h1 { margin:0; font-size:15px; font-weight:600; display:flex; align-items:center; gap:8px; }
  .badge { font-size:11px; padding:2px 8px; border-radius:999px; background:rgba(255,106,158,.15); color:var(--pink); border:1px solid rgba(255,106,158,.35); }
  .pill { font-size:12px; color:var(--mut); }
  .wrap { max-width:1280px; width:100%; margin:0 auto; padding:16px 20px; display:flex; flex-direction:column; gap:14px; flex:1; }
  .card { background:var(--card); border:1px solid var(--bdr); border-radius:12px; padding:14px 16px; }
  .grid-cfg { display:grid; grid-template-columns:1fr 1fr auto; gap:10px; align-items:end; }
  label { display:block; font-size:11.5px; color:var(--mut); margin-bottom:4px; }
  input[type=text], select { width:100%; background:var(--card2); color:var(--txt); border:1px solid var(--bdr); border-radius:8px; padding:8px 10px; font:inherit; outline:none; }
  input:focus { border-color:rgba(255,106,158,.6); }
  button { background:#f3f3f6; color:#111; border:0; border-radius:8px; padding:8px 14px; font:600 13px/1 inherit; cursor:pointer; }
  button.sec { background:var(--card2); color:var(--txt); border:1px solid var(--bdr); font-weight:500; }
  button.pink { background:var(--pink); color:#111; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:14px; flex:1; min-height:320px; }
  .pane { background:var(--card); border:1px solid var(--bdr); border-radius:12px; display:flex; flex-direction:column; overflow:hidden; }
  .pane-hd { padding:10px 14px; border-bottom:1px solid var(--bdr); display:flex; justify-content:space-between; align-items:center; background:#141418; }
  .pane-hd strong { font-size:13px; }
  .pane-hd .tag { font-size:11px; color:var(--mut); }
  .pane.slept { border-color:rgba(255,106,158,.32); }
  .pane.slept .pane-hd strong { color:var(--pink); }
  .log { padding:14px; display:flex; flex-direction:column; gap:10px; overflow-y:auto; max-height:46vh; }
  .msg { padding:10px 12px; border-radius:10px; font-size:13.5px; white-space:pre-wrap; }
  .msg.u { background:#23232b; align-self:flex-end; max-width:88%; }
  .msg.a { background:#1c1c22; border:1px solid var(--bdr); align-self:flex-start; max-width:95%; }
  .meta { font-size:11px; color:var(--mut); margin-top:4px; }
  .composer { display:flex; gap:10px; align-items:center; }
  .quiz-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .ans-box { font-size:12px; color:var(--grn); background:rgba(61,214,140,.08); border:1px solid rgba(61,214,140,.25); padding:6px 10px; border-radius:8px; display:none; }
  @media (max-width:900px) { .cols, .grid-cfg { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Hikari SLEEP Lab <span class="badge">ZERO-CONTEXT A/B TEST</span></h1>
  <div id="status" class="pill">Checking models...</div>
</header>
<div class="wrap">
  <div class="card">
    <div class="grid-cfg">
      <div>
        <label>Base Model (HF ID like Qwen/Qwen3-0.6B OR local .gguf path for Qwen3-4B)</label>
        <input id="modelPath" type="text" />
      </div>
      <div>
        <label>Slept LoRA (PEFT adapter folder OR hikari-sleep-lora.gguf from Colab)</label>
        <input id="loraPath" type="text" />
      </div>
      <div style="display:flex;gap:8px;">
        <button id="loadBtn" class="pink">Load Model</button>
      </div>
    </div>
    <div style="display:flex;gap:14px;align-items:center;margin-top:10px;flex-wrap:wrap;font-size:12px;color:var(--mut);">
      <span>LoRA scale (GGUF): <strong id="scaleVal">1.0</strong> <input id="loraScale" type="range" min="0" max="1.5" step="0.05" value="1.0" style="vertical-align:middle;" /></span>
      <span>Temp: <strong id="tempVal">0.2</strong> <input id="temp" type="range" min="0" max="1.0" step="0.05" value="0.2" style="vertical-align:middle;" /></span>
      <label style="display:inline-flex;align-items:center;gap:6px;margin:0;cursor:pointer;"><input id="usePersona" type="checkbox" checked /> Include 1-line Hikari persona (still ZERO memories!)</label>
    </div>
  </div>

  <div class="card">
    <div class="quiz-row">
      <strong style="font-size:12.5px;">48-Question Real Stream Quiz:</strong>
      <select id="quizFilter" style="width:auto;">
        <option value="all">All Cycles</option>
        <option value="1">Cycle 1 (oldest — checks forgetting!)</option>
        <option value="2">Cycle 2 (newest)</option>
        <option value="A">Tier A (proper nouns / exact facts)</option>
      </select>
      <select id="quizPick" style="flex:1;min-width:260px;"></select>
      <button id="askQuizBtn" class="sec">Ask Both</button>
      <button id="revealBtn" class="sec">Show Answer</button>
    </div>
    <div id="ansBox" class="ans-box" style="margin-top:8px;"></div>
  </div>

  <div class="cols">
    <div class="pane">
      <div class="pane-hd"><strong>Normal Base Model</strong><span class="tag">LoRA OFF · Zero memories in prompt</span></div>
      <div id="logBase" class="log"></div>
    </div>
    <div class="pane slept">
      <div class="pane-hd"><strong>Slept Hikari (SLEEP LoRA)</strong><span class="tag">LoRA ON · Zero memories in prompt</span></div>
      <div id="logSlept" class="log"></div>
    </div>
  </div>

  <div class="card composer">
    <input id="prompt" type="text" placeholder="Ask Hikari something from your old streams (zero RAG, pure weight test)..." />
    <button id="sendBtn" class="pink">Compare</button>
    <button id="clearBtn" class="sec">Clear</button>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let QUIZ = [];
const SYS_HIKARI = """ + json.dumps(DEFAULT_SYSTEM_PROMPT) + """;

async function init() {
  const st = await (await fetch('/api/status')).json();
  $('modelPath').value = st.modelPath || '';
  $('loraPath').value = st.loraPath || '';
  $('status').textContent = st.statusMsg;
  QUIZ = await (await fetch('/api/quiz')).json();
  renderQuiz();
}
function renderQuiz() {
  const f = $('quizFilter').value;
  const list = QUIZ.filter(q => f === 'all' || String(q.cycle) === f || q.tier === f);
  $('quizPick').innerHTML = list.map(q =>
    `<option value="${QUIZ.indexOf(q)}">[C${q.cycle} · ${q.tier}] ${q.question.slice(0, 120)}</option>`
  ).join('');
  $('ansBox').style.display = 'none';
}
$('quizFilter').onchange = renderQuiz;
$('loraScale').oninput = e => $('scaleVal').textContent = Number(e.target.value).toFixed(2);
$('temp').oninput = e => $('tempVal').textContent = Number(e.target.value).toFixed(2);
$('revealBtn').onclick = () => {
  const q = QUIZ[Number($('quizPick').value)];
  if (!q) return;
  $('ansBox').textContent = `Expected (Cycle ${q.cycle}, Tier ${q.tier}): ${q.answer}`;
  $('ansBox').style.display = $('ansBox').style.display === 'block' ? 'none' : 'block';
};
$('loadBtn').onclick = async () => {
  $('loadBtn').disabled = true;
  $('status').textContent = 'Loading model...';
  const r = await (await fetch('/api/load', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ modelPath: $('modelPath').value.trim(), loraPath: $('loraPath').value.trim() })
  })).json();
  $('loadBtn').disabled = false;
  $('status').textContent = r.ok ? r.status : ('Error: ' + r.error);
};
function addMsg(el, cls, text, meta) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    d.appendChild(m);
  }
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}
async function runPrompt(text) {
  if (!text.trim()) return;
  addMsg($('logBase'), 'u', text);
  addMsg($('logSlept'), 'u', text);
  $('sendBtn').disabled = true;
  $('askQuizBtn').disabled = true;
  try {
    const r = await (await fetch('/api/compare', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: text,
        systemPrompt: $('usePersona').checked ? SYS_HIKARI : '',
        loraScale: Number($('loraScale').value),
        temperature: Number($('temp').value),
      })
    })).json();
    if (!r.ok) {
      addMsg($('logBase'), 'a', 'Error: ' + r.error);
      addMsg($('logSlept'), 'a', 'Error: ' + r.error);
    } else {
      addMsg($('logBase'), 'a', r.base.text, `${r.base.ms} ms · LoRA OFF`);
      addMsg($('logSlept'), 'a', r.slept.text, `${r.slept.ms} ms · LoRA ON (${r.slept.loraScale})`);
    }
  } finally {
    $('sendBtn').disabled = false;
    $('askQuizBtn').disabled = false;
  }
}
$('sendBtn').onclick = () => { const t = $('prompt').value; $('prompt').value = ''; runPrompt(t); };
$('prompt').onkeydown = e => { if (e.key === 'Enter') $('sendBtn').click(); };
$('askQuizBtn').onclick = () => { const q = QUIZ[Number($('quizPick').value)]; if (q) runPrompt(q.question); };
$('clearBtn').onclick = () => { $('logBase').innerHTML = ''; $('logSlept').innerHTML = ''; };
init();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/api/status":
            self._json(200, {
                "ready": RUNNER.ready,
                "mode": RUNNER.mode,
                "statusMsg": RUNNER.status_msg,
                "llamaExe": RUNNER.llama_exe,
                "modelPath": RUNNER.model_path or find_gguf("model"),
                "loraPath": RUNNER.lora_path or find_gguf("lora"),
            })
        elif path == "/api/quiz":
            self._json(200, QUIZ)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or 0)
        body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n > 0 else {}
        if path == "/api/load":
            res = RUNNER.start(str(body.get("modelPath") or ""), str(body.get("loraPath") or ""))
            self._json(200 if res.get("ok") else 400, res)
        elif path == "/api/compare":
            res = RUNNER.compare(
                prompt=str(body.get("prompt") or ""),
                system_prompt=str(body.get("systemPrompt") or ""),
                lora_scale=float(body.get("loraScale", 1.0)),
                temperature=float(body.get("temperature", 0.2)),
            )
            self._json(200 if res.get("ok") else 400, res)
        else:
            self._json(404, {"error": "not found"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Hikari SLEEP Lab Zero-Context A/B Compare Chat")
    ap.add_argument("--port", type=int, default=8809)
    ap.add_argument("--llama-port", type=int, default=8810)
    ap.add_argument("--model", default="", help="HF model ID (Qwen/Qwen3-0.6B) or GGUF path")
    ap.add_argument("--lora", default="", help="PEFT adapter folder or LoRA GGUF path")
    ap.add_argument("--autoload", action="store_true", help="Load model immediately on startup")
    args = ap.parse_args()

    RUNNER.port = args.llama_port
    RUNNER.base_url = f"http://127.0.0.1:{args.llama_port}"
    m = args.model or find_gguf("model")
    l = args.lora or find_gguf("lora")
    if args.autoload and m:
        print(f"[SLEEP Compare] Auto-loading model={m} lora={l or '(none)'}")
        print("[SLEEP Compare]", RUNNER.start(m, l))

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[SLEEP Compare] Ready at http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()


if __name__ == "__main__":
    main()
