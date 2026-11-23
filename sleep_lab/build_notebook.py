#!/usr/bin/env python3
"""Build `hikari_sleep_arxiv2606.ipynb` from the cell sources below.

The notebook is generated (not hand-edited) so it always stays valid nbformat-4 JSON and every
code cell can be compile-checked by `verify_sleep_lab.py`. Edit the cells here, then run:

    python sleep_lab/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "hikari_sleep_arxiv2606.ipynb"

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ======================================================================================
md(r'''
# Hikari-chan needs sleep 💤 — SLEEP (arXiv:2606.03979) on one Colab GPU

Implementation of **"Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"**
(Behrouz, Hashemi, Javanmard, Mirrokni — Google Research) on Hikari-chan's real Web Waifu memory logs.

| Paper (§) | This notebook |
|---|---|
| Wake / active phase (§3.1): fast memory = in-context learning | **Stage 1 — Wake Context Buffer**: `wake_cycle_k.jsonl` placed in the teacher's context window |
| Parameter expansion (§3.2): add a new low-rank expert `{A,B}` to the MLP block; pre-allocated + masked until its sleep | LoRA slots `sleep_expert_1`, `sleep_expert_2` (rank 16, `gate/up/down_proj`) created up front and **masked** until their cycle; earlier slots stay **frozen** |
| Compute → Consolidate → Update (§3.3) | teacher = model before update (*with* the wake buffer + earlier experts); student = same model, **no wake memories**, + new expert; after consolidation the wake buffer is cleared (synaptic pruning) and the new expert is activated |
| Knowledge Seeding objective (§3.3): GKD mixture `(1−λ)·E_D[F] + λ·E_{y∼student}[F]`, no backprop through sampling | `ks_loss`: λ-mixed on/off-policy rollouts, **top-K generalized JSD(β)** (β=1 → reverse KL) between teacher-with-context and student-without-context |
| Learning to Imitate (§3.3, Eq. 3–4): `r = γ·r_sem + (1−γ)·r_abs`, Levenshtein with threshold `z0`; `L_KS = (1−α)E[r] − αE[D]` | prefix of a teacher dream → student continues → frozen base judges semantic match + token Levenshtein → group-normalised policy gradient |
| Dreaming (§3.4): dreams `∼ LM_θ(·|C)` with random expert routing, gradient importance score → top-k + b random, isolated LoRA SFT per dream, reward 1 if `τ` improves, ReST-EM | `dream_cycle`: dream Q&A from wake + earlier-cycle memories, **grounded verifier gate** (teacher-with-context judge + lexical fact overlap), ‖∇L_SFT‖ ranking, isolated inner SFT + τ check, ReST-EM M-step on accepted dreams only |
| (not in paper) | **Stage 4**: PEFT `add_weighted_adapter(..., combination_type="ties")` merge, GGUF LoRA export, zero-context continual-retention benchmark |

**Honest single-GPU simplifications** (also in `README.md`): one 4B dense model instead of the Hope/CMS MoE
architecture; the in-context wake buffer plays the role of the "fast" memory block; experts are LoRA adapters;
the paper's λ/α/γ/z0/top-K/temperature values are not published, so the defaults below are ours.

**Run-2 fixes** (run 1 on a T4: recall 4% → 7%, 0 dreams kept, LTI reward always 0, 2 h 40 min per cycle):
D now covers every memory (teacher questions per group of 6 memories, ~40 per cycle instead of 24 global ones);
τ for dream rewards uses held-out *short factual* answers (the style dreams teach); LTI gets partial credit
(z0 = 0.9) and a fair semantic judge; 16-bit weights on ≥14 GB GPUs and KV-cached teacher generation (~3× faster);
results are archived after every cycle so a crash or time limit still leaves something to download.

**Why not Unsloth:** its fused LoRA kernels read only the *first* active adapter. SLEEP's anti-forgetting design
runs the frozen earlier expert and the new expert together, so Unsloth would silently drop one of them.

Runtime: **A100** (paper-like profile) · **L4** · **T4** (`t4` profile). `smoke` runs the full pipeline in minutes.
''')

code(r'''
# @title 0 · Environment + installs
import os, sys, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # must be set before torch loads

try:                                   # also true under `colab exec` (google-colab-cli), where nothing imported it yet
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and os.environ.get("SLEEP_SKIP_INSTALL") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "transformers>=4.51.3", "peft>=0.14.0", "bitsandbytes>=0.45.0",
                    "accelerate>=1.3.0", "sentencepiece", "huggingface_hub"], check=True)
print("python", sys.version.split()[0], "| colab:", IN_COLAB)
''')

md(r'''
## Config
Pick a `PROFILE`. Every paper hyper-parameter is named after its symbol in arXiv:2606.03979.
''')

code(r'''
# @title 1 · Config
import json, math, random, re, time, copy, gc, zipfile, shutil
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict

PROFILE  = os.environ.get("SLEEP_PROFILE", "t4")            # "smoke" | "t4" | "a100" | "cpu_test"
MODEL_ID = os.environ.get("SLEEP_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
#   "google/gemma-3-4b-it"  -> gated: run `from huggingface_hub import login; login()` first
GGUF_BASE_ID = os.environ.get("SLEEP_GGUF_BASE_ID", "")      # full-precision repo for the GGUF converter ("" = auto)
# "auto": plain 16-bit weights when the GPU has >= 14 GB (T4/L4/A100) - about 3x faster generation than NF4 and
# more exact; NF4 (QLoRA) only on smaller GPUs. "1" forces NF4, "0" forces 16-bit.
QUANT = os.environ.get("SLEEP_4BIT", "auto")
WORK_DIR = Path(os.environ.get("SLEEP_WORK_DIR", "/content/sleep_lab" if IN_COLAB else ".")).resolve()
OUT_DIR  = WORK_DIR / "outputs"
DRIVE_DIR = os.environ.get("SLEEP_DRIVE_DIR", "/content/drive/MyDrive/hikari_sleep_lab")
MOUNT_DRIVE = os.environ.get("SLEEP_MOUNT_DRIVE", "1") == "1"
SHOW_SAMPLES = os.environ.get("SLEEP_SHOW_SAMPLES", "0") == "1"   # off: keep private memories out of saved outputs
SEED = 2606

PAPER = dict(
    lora_rank=16, lora_alpha=32,                          # paper App.B uses d_low=64 / r=64, α=128; 16 keeps the GGUF ~33 MB
    lora_targets=["gate_proj", "up_proj", "down_proj"],   # §3.2: the new low-rank expert lives in the MLP block
    lambda_on_policy=0.5,                                 # λ  (§3.3) fraction of on-policy student rollouts
    beta_jsd=0.5,                                         # F = generalised JSD(β); 1.0 = reverse KL, 0.0 = forward KL
    alpha_ks=0.6,                                         # α  (§3.3) distillation weight vs Learning-to-Imitate
    gamma_lti=0.5,                                        # γ  (Eq. 3) semantic vs Levenshtein reward
    z0_frac=0.9,                                          # z0 (Eq. 4) as a fraction of max(|d̂|,|d|); 0.6 zeroed every reward in run 1
    topk=64, temperature=1.0,                             # top-K logit distillation (+ tail bucket)
    lr=2e-4,                                              # paper Table 5: 5e-6 with r=64/α=128, batch 32; 1e-4 moved recall too little
    inner_lr=3e-4, tau_eps=0.0,                           # SEAL-style isolated inner update; reward 1 iff τ strictly improves
    min_lex=0.5, verifier_p=0.5,                          # dream verifier: lexical fact overlap + teacher judge P(yes)
    random_expert_p=0.5,                                  # §3.4 random expert routing while dreaming
    rehearse_prev_cycle=True,                             # cross-session rehearsal dreams (cycle-1 memories in cycle-2 dreams)
    rollout_temp=0.8, teacher_temp=0.7, max_ctx_tokens=6000,
    group_size=6,                                         # D coverage: the teacher writes questions for every 6 memories
)
PROFILES = {
    "cpu_test": dict(qs_per_group=1, max_groups=1, n_replay=2, n_anchor=1, dev_per_task=2, max_new=12, ks_steps=2,
                     grad_accum=1, micro_bs=1, lti_every=1, lti_group=2, lti_max_new=8, dream_iters=1, dream_tasks=1,
                     dreams_per_task=2, dream_topk=1, dream_rand=1, inner_steps=1, dream_sft_epochs=1, eval_limit=4,
                     gen_bs=2, max_ctx_tokens=1500, topk=8),
    "smoke": dict(qs_per_group=2, max_groups=2, n_replay=2, n_anchor=1, dev_per_task=3, max_new=48, ks_steps=3,
                  grad_accum=1, micro_bs=2, lti_every=1, lti_group=2, lti_max_new=24, dream_iters=1, dream_tasks=1,
                  dreams_per_task=2, dream_topk=1, dream_rand=1, inner_steps=2, dream_sft_epochs=1, eval_limit=8, gen_bs=4),
    # T4: every memory gets questions (~40 per cycle), 40 KS steps x 8 samples, 2 dream rounds x 3 tasks x 6 dreams
    "t4": dict(qs_per_group=4, max_groups=None, n_replay=12, n_anchor=6, dev_per_task=6, max_new=96, ks_steps=40,
               grad_accum=4, micro_bs=2, lti_every=2, lti_group=4, lti_max_new=32, dream_iters=2, dream_tasks=3,
               dreams_per_task=6, dream_topk=2, dream_rand=1, inner_steps=3, dream_sft_epochs=2, eval_limit=None, gen_bs=8),
    # paper-like: 100 steps, effective batch 32 (Table 5); keep 3 of 12 dreams (App. B.3 keeps 15 of 60)
    "a100": dict(qs_per_group=6, max_groups=None, n_replay=999, n_anchor=12, dev_per_task=8, max_new=128, ks_steps=100,
                 grad_accum=8, micro_bs=4, lti_every=1, lti_group=4, lti_max_new=48, dream_iters=3, dream_tasks=4,
                 dreams_per_task=12, dream_topk=2, dream_rand=1, inner_steps=3, dream_sft_epochs=2, eval_limit=None,
                 gen_bs=16),
}
cfg = SimpleNamespace(**{**PAPER, **PROFILES[PROFILE]})
random.seed(SEED)
rng = random.Random(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
T_START = time.time()

def archive_outputs(tag="final"):
    """Zip the outputs so the launcher can download them at any point (after each cycle, or on failure)."""
    z, tmp = WORK_DIR / "sleep_outputs.zip", WORK_DIR / "sleep_outputs.tmp.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUT_DIR.rglob("*"):
            rel = f.relative_to(OUT_DIR).as_posix()
            if not f.is_file() or rel.startswith("_cpu_test"):
                continue
            if tag == "final" and rel.startswith("adapters/sleep_expert_"):
                continue                       # final archive: merged adapter + GGUF are enough
            zf.write(f, rel)
        zf.writestr("ARCHIVE_STAGE.txt", f"{tag} after {(time.time() - T_START) / 60:.0f} min\n")
    tmp.replace(z)
    print(f"archive [{tag}] -> {z} ({z.stat().st_size / 2**20:.1f} MB)")

print(f"profile={PROFILE} model={MODEL_ID} quant={QUANT} out={OUT_DIR}")
''')

md(r'''
## Data
Upload `sleep_lab_bundle.zip` (made by `run_colab_sleep.py`) next to the notebook, or put it / the `datasets/`
folder in Google Drive under `MyDrive/hikari_sleep_lab/`. The cell finds whichever exists.
''')

code(r'''
# @title 2 · Locate + load datasets (wake buffers, held-out quiz, persona anchors)
def load_jsonl(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

def _has_data(d):
    return d is not None and (Path(d) / "wake_cycle_1.jsonl").exists()

def _unzip(z):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(WORK_DIR)
    return WORK_DIR / "datasets"

def find_datasets():
    dirs = [os.environ.get("SLEEP_DATASETS"), WORK_DIR / "datasets", Path.cwd() / "datasets",
            Path("/content/datasets"), Path(DRIVE_DIR) / "datasets"]
    zips = [WORK_DIR / "sleep_lab_bundle.zip", Path.cwd() / "sleep_lab_bundle.zip",
            Path("/content/sleep_lab_bundle.zip"), Path(DRIVE_DIR) / "sleep_lab_bundle.zip"]
    for attempt in range(2):
        for d in dirs:
            if d and _has_data(d):
                return Path(d)
        for z in zips:
            if z.exists():
                return _unzip(z)
        if attempt == 0 and IN_COLAB and MOUNT_DRIVE and not Path("/content/drive/MyDrive").exists():
            from google.colab import drive
            drive.mount("/content/drive")
        else:
            break
    if IN_COLAB:
        from google.colab import files
        print("Upload sleep_lab_bundle.zip (from run_colab_sleep.py)")
        for name in files.upload():
            if name.endswith(".zip"):
                shutil.move(name, WORK_DIR / name)
                return _unzip(WORK_DIR / name)
    raise FileNotFoundError("datasets not found - run import_waifu_logs.py + run_colab_sleep.py --mode bundle")

DATA = find_datasets()
WAKE = {1: load_jsonl(DATA / "wake_cycle_1.jsonl"), 2: load_jsonl(DATA / "wake_cycle_2.jsonl")}
QUIZ = load_jsonl(DATA / "eval_recall_quiz.jsonl")
_anchor_rows = load_jsonl(DATA / "persona_anchor.jsonl")
PERSONA = next(r["system_prompt"] for r in _anchor_rows if r["kind"] == "persona_system")
ANCHORS = [r for r in _anchor_rows if r["kind"] == "style_anchor"]
if cfg.eval_limit:
    QUIZ = [q for c in (1, 2) for q in [p for p in QUIZ if p["cycle"] == c][: cfg.eval_limit // 2]]
for c, rows in WAKE.items():
    print(f"wake_cycle_{c}: {len(rows)} episodes ({Counter(r['kind'] for r in rows)})")
print("quiz probes:", dict(Counter(f"cycle{p['cycle']}/{p['tier']}" for p in QUIZ)), "| style anchors:", len(ANCHORS))
''')

md(r'''
## Model + pre-allocated, masked expert slots (§3.2 "Note on the Implementation")
''')

code(r'''
# @title 3 · Load the 4B model (16-bit, or NF4 on small GPUs) and pre-allocate the sleep experts
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    # native bf16 needs compute capability >= 8 (A100/L4). Newer torch reports True on a T4 too, via slow
    # emulation, so check the hardware instead: T4 (7.5) -> fp16 with tensor cores
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
    LOAD_IN_4BIT = QUANT == "1" or (QUANT == "auto" and VRAM_GB < 14)
    print(torch.cuda.get_device_name(0), f"{VRAM_GB:.1f} GB")
else:
    DTYPE, LOAD_IN_4BIT = torch.float32, False
print("weights:", "NF4 4-bit" if LOAD_IN_4BIT else str(DTYPE).replace("torch.", ""))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
import transformers
_tv = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
load_kw = {("dtype" if _tv >= (4, 56) else "torch_dtype"): DTYPE, "attn_implementation": "sdpa"}
if DEVICE == "cuda":
    load_kw["device_map"] = {"": 0}            # load straight onto the GPU (Colab T4 VMs have little system RAM)
if LOAD_IN_4BIT:
    load_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DTYPE)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kw)
if DEVICE != "cuda":
    base.to(DEVICE)
for p in base.parameters():
    p.requires_grad_(False)
base.config.use_cache = True

LORA_CFG = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=cfg.lora_targets, task_type="CAUSAL_LM")
EXPERTS = ["sleep_expert_1", "sleep_expert_2"]
model = get_peft_model(base, LORA_CFG, adapter_name=EXPERTS[0])
for name in EXPERTS[1:]:
    model.add_adapter(name, LORA_CFG)          # allocated now, masked (inactive) until its sleep cycle
LORA = model.base_model                         # peft LoraModel
model.eval()                                    # no dropout anywhere; gradients still flow into the active expert

_TRAINABLE: set = set()

def use_experts(active, trainable=None):
    """Activate a set of expert slots (others are masked) and mark which ones receive gradients."""
    global _TRAINABLE
    if trainable is not None:
        _TRAINABLE = set(trainable)
    active = list(active)
    if active:
        LORA.enable_adapter_layers()
        LORA.set_adapter(active if len(active) > 1 else active[0])
    else:
        LORA.disable_adapter_layers()
    # set_adapter() marks every active adapter trainable -> re-freeze all but the current sleep expert
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(any(f".{t}." in n for t in _TRAINABLE))

def expert_params(name):
    return [p for n, p in model.named_parameters() if "lora_" in n and f".{name}." in n]

use_experts([], trainable=[])
n_exp = sum(p.numel() for p in expert_params(EXPERTS[0]))
print(f"expert slot size: {n_exp / 1e6:.2f} M params ({n_exp * 2 / 2**20:.0f} MB fp16) x {len(EXPERTS)} slots")

EOS_IDS = set()
for t in [tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]:
    EOS_IDS |= set(t if isinstance(t, (list, tuple)) else [t] if t is not None else [])
for t in ["<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|endoftext|>"]:
    i = tok.convert_tokens_to_ids(t)
    if isinstance(i, int) and i != tok.unk_token_id and i >= 0:
        EOS_IDS.add(i)
EOT_ID = next((tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<end_of_turn>", "<|eot_id|>"]
               if tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id)), tok.eos_token_id)
# prefix-KV caching of the teacher's wake buffer; sliding-window caches (Gemma 3) cannot be cropped
USE_PREFIX_CACHE = "gemma" not in str(getattr(base.config, "model_type", "")).lower()
GEN_CACHE = USE_PREFIX_CACHE        # reuse that cache for teacher generation too (verified in the self-check)
JUDGE_BATCH = True                  # batched yes/no judging (verified in the self-check)
import inspect
CACHE_KW = "cache" if "cache" in inspect.signature(base.forward).parameters else "past_key_values"  # transformers 5.x renamed it
print("cache argument:", CACHE_KW)
''')

md(r'''
## Prompting, teacher context buffer, top-K distillation math
''')

code(r'''
# @title 4 · Prompt helpers + Stage-1 Wake Context Buffer
CHAT_KW = {"enable_thinking": False}     # Qwen3 hybrid templates; ignored by others

def chat_text(system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
        except Exception:
            msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + user}]
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (f"System: {system}\n" if system else "") + f"User: {user}\nAssistant:"

def enc(text):
    return tok(text, add_special_tokens=False)["input_ids"]

def dec(ids):
    return tok.decode([i for i in ids if i not in EOS_IDS], skip_special_tokens=True).strip()

def student_prompt(user):
    return enc(chat_text(PERSONA, user))

def memory_system(episodes):
    lines = "\n\n".join(e["text"] for e in episodes)
    return (PERSONA + "\n\n# Your private memories from recent streams (wake buffer)\n" + lines +
            "\n\nUse these memories naturally when they are relevant. Never say they were handed to you.")

def chunk_episodes(episodes, max_tokens):
    """Split a wake buffer into context windows that fit the teacher (chronological)."""
    budget = max_tokens - len(enc(PERSONA)) - 64
    chunks, cur, used = [], [], 0
    for e in episodes:
        n = len(enc(e["text"])) + 2
        if cur and used + n > budget:
            chunks.append(cur); cur, used = [], 0
        cur.append(e); used += n
    if cur:
        chunks.append(cur)
    return chunks

class WakeContext:
    """Teacher = model state before the update, *with* the wake buffer in its context window."""
    def __init__(self, name, episodes, experts):
        self.name, self.episodes, self.experts = name, list(episodes), list(experts)
        self.system = memory_system(self.episodes)
        self.text_lc = self.system.lower()
        self._pids, self._cache = None, None

    def prompt_ids(self, user):
        return enc(chat_text(self.system, user))

    def prefix_ids(self):
        if self._pids is None:   # longest common token prefix of two prompts = system + memories
            a, b = self.prompt_ids("Hey Hikari, question one?"), self.prompt_ids("zzz totally different words 42")
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            self._pids = a[: max(0, n - 2)]
        return self._pids

    @torch.no_grad()
    def cache(self):
        if self._cache is None:
            use_experts(self.experts)
            ids = torch.tensor([self.prefix_ids()], device=DEVICE)
            out = model(input_ids=ids, use_cache=True)
            self._cache = getattr(out, "cache", None) or out.past_key_values
        return self._cache

    def drop(self):
        """Synaptic pruning of the fast memory: the wake buffer is cleared after consolidation."""
        self._cache = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def forward_logits(ids, keep, past=None, **extra):
    """Logits for the last `keep` positions of a single sequence (optionally continuing a KV cache)."""
    kw = dict(input_ids=ids, use_cache=past is not None, **extra)
    if past is not None:
        kw[CACHE_KW] = past
    try:
        out = model(**kw, logits_to_keep=keep)
    except TypeError:
        out = model(**kw)
    return out.logits[0, -keep:]

@torch.no_grad()
def ctx_logits(ctx, full_ids, keep):
    """Teacher forward over [wake buffer + suffix], reusing the buffer's KV cache (then cropping it back)."""
    global USE_PREFIX_CACHE
    use_experts(ctx.experts)
    pre = ctx.prefix_ids()
    if USE_PREFIX_CACHE and full_ids[: len(pre)] == pre and len(full_ids) - len(pre) >= keep:
        try:
            cache = ctx.cache()
            logits = forward_logits(torch.tensor([full_ids[len(pre):]], device=DEVICE), keep, past=cache)
            extra = cache.get_seq_length() - len(pre)
            if extra > 0:
                cache.crop(-extra)               # negative = drop the newest tokens (transformers 4.x and 5.x)
            return logits
        except Exception as e:                   # older cache classes: fall back to full forwards
            print("prefix cache disabled:", type(e).__name__, e)
            USE_PREFIX_CACHE, ctx._cache = False, None
    return forward_logits(torch.tensor([full_ids], device=DEVICE), keep)
''')

code(r'''
# @title 5 · Top-K generalised JSD / KL (GKD F), log-probs, generation, frozen judge
def _log1mexp(x):
    x = x.clamp(max=-1e-6)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))

def topk_divergence(t_top, t_idx, s_logp, beta=None):
    """Per-token F(teacher || student) on the teacher's top-K support + one tail bucket (mass-preserving).
    beta=0 -> forward KL(p||q), beta=1 -> reverse KL(q||p), else generalised JSD(beta) as in GKD."""
    beta = cfg.beta_jsd if beta is None else beta
    s_top = s_logp.gather(-1, t_idx)
    p = torch.cat([t_top, _log1mexp(torch.logsumexp(t_top, -1, keepdim=True))], -1)
    q = torch.cat([s_top, _log1mexp(torch.logsumexp(s_top, -1, keepdim=True))], -1)
    if beta <= 0:
        return (p.exp() * (p - q)).sum(-1)
    if beta >= 1:
        return (q.exp() * (q - p)).sum(-1)
    m = torch.logsumexp(torch.stack([p + math.log(beta), q + math.log(1 - beta)]), 0)
    return beta * (p.exp() * (p - m)).sum(-1) + (1 - beta) * (q.exp() * (q - m)).sum(-1)

@torch.no_grad()
def teacher_topk(ctx, user, y_ids):
    """Teacher log-probs *with* the wake buffer in context (KV prefix cached), top-K per response token."""
    logits = ctx_logits(ctx, ctx.prompt_ids(user) + list(y_ids), len(y_ids) + 1)
    logp = F.log_softmax(logits[:-1].float() / cfg.temperature, -1)
    return logp.topk(min(cfg.topk, logp.shape[-1]), -1)

def student_logp(user, y_ids, system=None):
    """Student log-probs with ZERO wake memories in the prompt (persona only). Keeps the graph."""
    full = enc(chat_text(system or PERSONA, user)) + list(y_ids)
    logits = forward_logits(torch.tensor([full], device=DEVICE), len(y_ids) + 1)
    return F.log_softmax(logits[:-1].float() / cfg.temperature, -1)

def seq_logprob(prompt_ids, y_ids):
    """Mean log p(y | prompt) under the currently active experts (graph kept if grads enabled)."""
    logits = forward_logits(torch.tensor([list(prompt_ids) + list(y_ids)], device=DEVICE), len(y_ids) + 1)
    lp = F.log_softmax(logits[:-1].float(), -1)
    return lp.gather(-1, torch.tensor(y_ids, device=DEVICE)[:, None]).mean()

def _gen_kw(max_new, temperature, top_p=0.95):
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, eos_token_id=sorted(EOS_IDS))
    kw.update(dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else
              dict(do_sample=False, temperature=None, top_p=None, top_k=None))
    return kw

def _cut(row):
    k = next((i for i, t in enumerate(row) if t in EOS_IDS or t == tok.pad_token_id), len(row))
    return row[:k]

@torch.no_grad()
def generate(prompts, max_new, temperature, top_p=0.95):
    """Batched left-padded sampling; returns new token ids per prompt (stops at EOS)."""
    outs = []
    # long prompts (wake buffer in context) go one at a time: padded batches force SDPA's O(L^2) math kernel
    bs = 1 if max(len(p) for p in prompts) > 1024 else cfg.gen_bs
    for i in range(0, len(prompts), bs):
        batch = prompts[i: i + bs]
        L = max(len(p) for p in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for j, p in enumerate(batch):
            ids[j, L - len(p):] = torch.tensor(p)
            mask[j, L - len(p):] = 1
        gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                             **_gen_kw(max_new, temperature, top_p))[:, L:].tolist()
        outs += [_cut(r) for r in gen]
    return outs

@torch.no_grad()
def gen_cached(ctx, ids, max_new, temperature):
    """Generate from [wake buffer + suffix] starting from a copy of the buffer's KV cache (no 4k-token re-read)."""
    cache = copy.deepcopy(ctx.cache())
    x = torch.tensor([ids], device=DEVICE)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), **{CACHE_KW: cache},
                         **_gen_kw(max_new, temperature))
    return _cut(out[0, len(ids):].tolist())

def teacher_generate(ctx, users, max_new, temperature=None):
    global GEN_CACHE
    use_experts(ctx.experts)
    temp = cfg.teacher_temp if temperature is None else temperature
    pre, outs = ctx.prefix_ids(), []
    for u in users:
        ids = ctx.prompt_ids(u)
        if GEN_CACHE and USE_PREFIX_CACHE and ids[: len(pre)] == pre:
            try:
                outs.append(gen_cached(ctx, ids, max_new, temp))
                continue
            except Exception as e:
                print("cached generation disabled:", type(e).__name__, e)
                GEN_CACHE = False
        outs += generate([ids], max_new, temp)
    return outs

YES_IDS = sorted({i[0] for i in (enc(w) for w in ["Yes", " Yes", "yes", " yes"]) if i})
NO_IDS = sorted({i[0] for i in (enc(w) for w in ["No", " No", "no", " no"]) if i})

def _p_yes(lg):
    return torch.sigmoid(torch.logsumexp(lg[..., YES_IDS], -1) - torch.logsumexp(lg[..., NO_IDS], -1))

@torch.no_grad()
def judge_yes(system, question, experts=()):
    """P(Yes) from the frozen judge (base model unless experts are given)."""
    use_experts(list(experts))
    return _p_yes(forward_logits(torch.tensor([enc(chat_text(system, question))], device=DEVICE), 1)[-1].float()).item()

@torch.no_grad()
def judge_yes_batch(system, questions, experts=()):
    """Same judge, one left-padded batch (positions offset so every row sees normal RoPE positions)."""
    if not JUDGE_BATCH:
        return [judge_yes(system, q, experts) for q in questions]
    use_experts(list(experts))
    prompts = [enc(chat_text(system, q)) for q in questions]
    L = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, p in enumerate(prompts):
        ids[j, L - len(p):] = torch.tensor(p)
        mask[j, L - len(p):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    kw = dict(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE), position_ids=pos.to(DEVICE), use_cache=False)
    try:
        out = model(**kw, logits_to_keep=1)
    except TypeError:
        out = model(**kw)
    return _p_yes(out.logits[:, -1].float()).tolist()

@torch.no_grad()
def ctx_judge(ctx, question):
    """Teacher-with-context judge (the dream verifier), reusing the wake buffer's KV cache."""
    return _p_yes(ctx_logits(ctx, ctx.prompt_ids(question), 1)[-1].float()).item()

def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def r_abs(hat_ids, ref_ids):
    """Eq. 4: 1 - z/max(|d̂|,|d|) if the token Levenshtein distance z <= z0, else 0."""
    mx = max(len(hat_ids), len(ref_ids), 1)
    z = levenshtein(hat_ids, ref_ids)
    return 1 - z / mx if z <= cfg.z0_frac * mx else 0.0

SEM_SYS = "You are a fair grader. Wording can differ; judge only the facts and the main idea."

def lti_rewards(conts, ref_ids):
    """Eq. 3 for a group of student continuations: r = γ·r_sem + (1-γ)·r_abs. Returns (r, r_abs, r_sem) lists."""
    ra = [r_abs(c, ref_ids) for c in conts]
    rs = [0.0] * len(conts)
    if cfg.gamma_lti > 0:
        qs = [f"Text A: {dec(c)}\nText B: {dec(ref_ids)}\nDo Text A and Text B express the same facts or main idea? "
              "Answer only Yes or No." for c in conts]
        rs = [float(p > 0.5) for p in judge_yes_batch(SEM_SYS, qs)]
    return [cfg.gamma_lti * s + (1 - cfg.gamma_lti) * a for s, a in zip(rs, ra)], ra, rs

class Optim:
    """AdamW on the current sleep expert only (+ GradScaler when the base computes in fp16 on T4)."""
    def __init__(self, params, lr):
        self.params = [p for p in params]
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    def backward(self, loss):
        self.scaler.scale(loss).backward()
    def step(self, clip=1.0):
        if all(p.grad is None for p in self.params):
            return 0.0                              # nothing was back-propagated this step
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.params, clip).item()
        self.scaler.step(self.opt); self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        return gn
''')

code(r'''
# @title 6 · Self-check: cached teacher == uncached teacher, divergence sanity
def _selfcheck():
    global USE_PREFIX_CACHE
    # divergence identities
    lp = F.log_softmax(torch.randn(3, 50), -1)
    top, idx = lp.topk(8, -1)
    for b in (0.0, 0.5, 1.0):
        assert topk_divergence(top, idx, lp, beta=b).abs().max().item() < 1e-4, "F(p||p) must be 0"
    assert topk_divergence(top, idx, F.log_softmax(torch.randn(3, 50), -1), beta=0.5).min().item() >= -1e-5
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    # prefix cache equivalence on a real wake buffer
    ctx = WakeContext("selfcheck", WAKE[1][:6], [])
    y = enc(" Sure, I remember that.")
    if USE_PREFIX_CACHE:
        a_top, a_idx = teacher_topk(ctx, "What do you remember?", y)
        USE_PREFIX_CACHE = False
        b_top, b_idx = teacher_topk(ctx, "What do you remember?", y)
        USE_PREFIX_CACHE = True
        err = (a_top - b_top).abs().max().item()
        same = (a_idx[:, 0] == b_idx[:, 0]).float().mean().item()
        print(f"prefix-cache check: max |Δlogp| {err:.4f}, top-1 agreement {same:.2f}")
        if same < 0.95:
            USE_PREFIX_CACHE = False
            print("-> cache mismatch, using full teacher forwards")
    global GEN_CACHE, JUDGE_BATCH
    if USE_PREFIX_CACHE and GEN_CACHE:          # cached teacher generation must equal plain generation
        ids = ctx.prompt_ids("Say hi to SUBSECT in a few words.")
        try:
            a = gen_cached(ctx, ids, 8, 0.0)
            b = generate([ids], 8, 0.0)[0]
            GEN_CACHE = a[:4] == b[:4]
        except Exception as e:
            print("cached generation error:", type(e).__name__, e)
            GEN_CACHE = False
        print("cached generation check:", "match" if GEN_CACHE else "MISMATCH -> plain generation")
    qs = ["Is the sky blue on a clear day? Answer only Yes or No.", "Is 2 + 2 equal to 5? Answer only Yes or No."]
    single = [judge_yes(SEM_SYS, q) for q in qs]
    batched = judge_yes_batch(SEM_SYS, qs)
    JUDGE_BATCH = max(abs(x - y) for x, y in zip(single, batched)) < 0.05
    print(f"judge: P(yes) {single[0]:.2f}/{single[1]:.2f}, batched {'ok' if JUDGE_BATCH else 'MISMATCH -> one at a time'}")
    ctx.drop()
    # gradient isolation: only the new expert learns, the earlier expert stays frozen
    use_experts(EXPERTS, trainable=[EXPERTS[-1]])
    (-seq_logprob(student_prompt("hello?"), enc(" hi there"))).backward()
    g_new = sum(float(p.grad.abs().sum()) for p in expert_params(EXPERTS[-1]) if p.grad is not None)
    g_old = [p.grad for p in expert_params(EXPERTS[0])]
    assert g_new > 0, "new expert received no gradient"
    assert all(g is None for g in g_old), "frozen expert received gradient"
    for p in model.parameters():
        p.grad = None
    use_experts([], trainable=[])
    print(f"gradient isolation ok (new expert |grad| {g_new:.2e}, frozen expert untouched)")
    print("self-check ok")

_selfcheck()
''')

md(r'''
## Zero-Context Continual Retention Benchmark (defined here, run after each stage)
No RAG, no memories in the prompt: only the persona + the probe question. Metrics per cycle tag:
**hit** (all answer keywords produced), **partial** (keyword recall), **tier-A hit** (names/numbers only),
**logp** (mean log-prob of the reference answer), and persona **style logp** on real Hikari replies.
''')

code(r'''
# @title 7 · Benchmark
def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def score_answer(pred, probe):
    n, ns = _norm(pred), _norm(pred).replace(" ", "")
    kws = [_norm(k) for k in probe["answer_keywords"] if _norm(k)] or [_norm(probe["answer"])]
    hits = sum(1 for k in kws if k in n or k.replace(" ", "") in ns)
    return hits / len(kws)

@torch.no_grad()
def run_benchmark(label, experts, probes=None, verbose=True, ctx=None):
    """Zero-context benchmark (persona + question). With `ctx`, the same quiz *with* the wake buffer (upper bound)."""
    probes = QUIZ if probes is None else probes
    t0 = time.time()
    use_experts(experts)
    sys_prompt = ctx.system if ctx else PERSONA
    if ctx:
        preds = teacher_generate(ctx, [p["question"] for p in probes], 40, 0.0)
    else:
        preds = generate([enc(chat_text(sys_prompt, p["question"])) for p in probes], 40, 0.0)
    rows = []
    for p, ids in zip(probes, preds):
        part = score_answer(dec(ids), p)
        ans = enc(p["answer"])
        if ctx:
            lg = ctx_logits(ctx, ctx.prompt_ids(p["question"]) + ans, len(ans) + 1)
            lp = F.log_softmax(lg[:-1].float(), -1).gather(-1, torch.tensor(ans, device=DEVICE)[:, None]).mean().item()
        else:
            lp = seq_logprob(enc(chat_text(sys_prompt, p["question"])), ans).item()
        rows.append(dict(id=p["id"], cycle=p["cycle"], tier=p["tier"], partial=part, hit=float(part == 1.0), logp=lp))
        if SHOW_SAMPLES:
            print(f"   [{label}] c{p['cycle']} {p['answer']!r} <- {dec(ids)[:100]!r}")
    style = [seq_logprob(enc(chat_text(sys_prompt, a["user"])), enc(a["assistant"])[:96]).item() for a in ANCHORS[:12]]
    res = {"label": label, "experts": list(experts), "style_logp": sum(style) / max(1, len(style))}
    for c in (1, 2):
        rc = [r for r in rows if r["cycle"] == c]
        ra = [r for r in rc if r["tier"] == "A"]
        mean = lambda xs, k: sum(x[k] for x in xs) / len(xs) if xs else float("nan")
        res[f"c{c}"] = dict(n=len(rc), hit=mean(rc, "hit"), partial=mean(rc, "partial"),
                            tierA_hit=mean(ra, "hit"), logp=mean(rc, "logp"))
    res["seconds"] = round(time.time() - t0)
    if verbose:
        print(f"{label:<34} " + " | ".join(
            f"cycle{c}: hit {res[f'c{c}']['hit']:.2f} part {res[f'c{c}']['partial']:.2f} logp {res[f'c{c}']['logp']:.2f}"
            for c in (1, 2)) + f" | style logp {res['style_logp']:.2f}")
    res["rows"] = rows
    return res

RESULTS = {}
RESULTS["0_base_no_rag"] = run_benchmark("Base 4B (no RAG)", [])
''')

md(r'''
## Stage 2 — Memory Consolidation via Knowledge Seeding (§3.3)
`D` is sampled from the **teacher** (the model with the wake buffer in context): questions it writes for every
group of memories (so each memory is covered), replayed wake prompts, and persona prompts. Then the new expert is trained with
`α·[(1−λ)E_D F + λ E_{y∼student} F] + (1−α)·LTI`.
''')

code(r'''
# @title 8 · Teacher dataset D (covers every memory) + Knowledge Seeding (GKD top-K JSD + Learning-to-Imitate)
QFOCUS = ("Step out of chat mode for a moment and just write a plain list. Here are some of your memories:\n\n{block}\n\n"
          "Write {n} short, specific questions that SUBSECT or a viewer could ask you that can ONLY be answered from "
          "these memories (names, exact things said, preferences, what happened, how you feel about people). "
          "One question per line, no numbering, no answers.")
SHORT = "\n\n(Answer from memory in one short, factual sentence.)"
_QUIZ_KEYS = {_norm(p["question"]) for p in QUIZ}

def _questions(raw, seen, n):
    out = []
    for line in raw.splitlines():
        q = re.sub(r"^[\s\-*\d.)]+", "", line).strip().strip('"')
        k = _norm(q)
        if 10 <= len(q) <= 200 and q.endswith("?") and k not in _QUIZ_KEYS and k not in seen:
            seen.add(k)
            out.append(q)
        if len(out) >= n:
            break
    return out

def build_teacher_dataset(ctxs):
    """D ~ teacher (§3.3). Coverage: the teacher writes questions for every group of memories, so each memory is
    asked about; plus replayed wake prompts and persona prompts. DEV = held-out short-answer questions for τ."""
    D, DEV = [], {}
    for ctx in ctxs:
        t0 = time.time()
        groups = [ctx.episodes[i: i + cfg.group_size] for i in range(0, len(ctx.episodes), cfg.group_size)]
        if cfg.max_groups:
            groups = rng.sample(groups, min(cfg.max_groups, len(groups)))
        prompts = [QFOCUS.format(block="\n\n".join(e["text"] for e in g), n=cfg.qs_per_group) for g in groups]
        seen, qs = set(), []
        for raw in teacher_generate(ctx, prompts, 40 * cfg.qs_per_group, 0.9):
            qs += _questions(dec(raw), seen, cfg.qs_per_group)
        rng.shuffle(qs)
        dev_q, train_q = qs[: cfg.dev_per_task], qs[cfg.dev_per_task:]
        replay = [f"{e.get('speaker') or 'SUBSECT'}: {e['user']}" for e in ctx.episodes if e["kind"] == "exchange" and e.get("user")]
        anchors = [a["user"] for a in rng.sample(ANCHORS, min(cfg.n_anchor, len(ANCHORS)))]
        xs = ([("teacher_q", q) for q in train_q] +
              [("wake_replay", x) for x in rng.sample(replay, min(cfg.n_replay, len(replay)))] +
              [("persona", x) for x in anchors])
        for (src, x), y in zip(xs, teacher_generate(ctx, [x for _, x in xs], cfg.max_new)):
            if len(y) >= 4:
                D.append(dict(ctx=ctx.name, src=src, x=x, y_ids=y))
        dev_y = teacher_generate(ctx, [q + SHORT for q in dev_q], 40, 0.0)     # greedy concise references
        DEV[ctx.name] = [dict(x=q + SHORT, y_ids=y + [EOT_ID]) for q, y in zip(dev_q, dev_y) if len(y) >= 2]
        print(f"  D[{ctx.name}]: {len(groups)} memory groups -> {len(qs)} questions ({len(DEV[ctx.name])} held out for τ), "
              f"{len(xs)} prompts ({time.time() - t0:.0f}s)")
    print(f"  D: {len(D)} teacher samples {dict(Counter(d['src'] for d in D))}")
    if SHOW_SAMPLES and D:
        print("   e.g.", D[0]["x"][:80], "->", dec(D[0]["y_ids"])[:120])
    return D, DEV

def lti_term(batch, student_experts, opt, scale, st):
    """Learning to Imitate: student completes a random prefix of a teacher dream; group-normalised PG."""
    for it in batch:
        d = it["y_ids"]
        if len(d) < 8:
            continue
        cut = rng.randint(max(2, int(0.2 * len(d))), max(3, int(0.6 * len(d))))
        prefix, ref = d[:cut], d[cut: cut + cfg.lti_max_new]
        p_ids = student_prompt(it["x"]) + prefix
        use_experts(student_experts)
        conts = generate([p_ids] * cfg.lti_group, len(ref) + 4, cfg.rollout_temp)
        r, ra, rs = lti_rewards(conts, ref)
        st["lti_r"] += r; st["r_abs"] += ra; st["r_sem"] += rs
        r = torch.tensor(r)
        if r.std() < 1e-6:
            continue                                # no learning signal in this group
        adv = (r - r.mean()) / (r.std() + 1e-4)
        use_experts(student_experts)
        for c, a in zip(conts, adv.tolist()):
            if c:
                opt.backward(-(a * seq_logprob(p_ids, c)) * scale)

def knowledge_seeding(cycle, ctxs, D, new_expert, frozen):
    student = list(frozen) + [new_expert]
    use_experts(student, trainable=[new_expert])
    params = expert_params(new_expert)
    opt = Optim(params, cfg.lr)
    by = {c.name: c for c in ctxs}
    hist, t0 = [], time.time()
    for step in range(cfg.ks_steps):
        st = defaultdict(list)
        for micro in range(cfg.grad_accum):
            batch = rng.sample(D, min(cfg.micro_bs, len(D)))
            on_policy = rng.random() < cfg.lambda_on_policy          # GKD λ-mixture
            if on_policy:                                            # y ~ student(.|x) without context, no backprop through sampling
                use_experts(student)
                ys = generate([student_prompt(it["x"]) for it in batch], cfg.max_new, cfg.rollout_temp)
            else:                                                    # (x, y) ~ D from the teacher
                ys = [it["y_ids"] for it in batch]
            scale = cfg.alpha_ks / (cfg.grad_accum * max(1, len(batch)))
            for it, y in zip(batch, ys):
                if len(y) < 2:
                    continue
                t_top, t_idx = teacher_topk(by[it["ctx"]], it["x"], y)
                use_experts(student)
                div = topk_divergence(t_top, t_idx, student_logp(it["x"], y)).mean()
                opt.backward(div * scale)
                st["div_on" if on_policy else "div_off"].append(div.item())
            if cfg.alpha_ks < 1 and micro % cfg.lti_every == 0:
                lti_scale = (1 - cfg.alpha_ks) / (cfg.grad_accum * max(1, len(batch)) * cfg.lti_group)
                lti_term(batch, student, opt, lti_scale, st)
        gn = opt.step()
        row = {k: sum(v) / len(v) for k, v in st.items() if v}
        row.update(step=step, grad_norm=gn)
        hist.append(row)
        if step % max(1, cfg.ks_steps // 10) == 0 or step == cfg.ks_steps - 1:
            print(f"  KS{cycle} step {step:>3}/{cfg.ks_steps} " +
                  " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "step") + f" ({time.time() - t0:.0f}s)")
    return hist
''')

md(r'''
## Stage 3 — Dreaming (§3.4): self-rehearsal curriculum + verifier gate + ReST-EM
''')

code(r'''
# @title 9 · Dreaming
DREAM = ("Sleep mode: you are dreaming about your memories above. Rehearse them by writing {n} question-and-answer "
         "pairs that connect different memories (who said what, preferences, running jokes, how you feel about "
         "people, what happened when). Mix details from different moments. Use exactly this format, one pair per "
         "two lines:\nQ: ...\nA: ...\nAnswers must stay faithful to the memories.")
_STOP = set("the a an and or but to of in on at for with is are was were be been it its this that you your i me my "
            "he she they them his her their we our what who when where why how do does did so just like".split())

def parse_qa(text):
    pairs, q = [], None
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(Q|Question)\s*[:.]", line, re.I):
            q = re.sub(r"^(Q|Question)\s*[:.]\s*", "", line, flags=re.I)
        elif q and re.match(r"^(A|Answer)\s*[:.]", line, re.I):
            a = re.sub(r"^(A|Answer)\s*[:.]\s*", "", line, flags=re.I)
            if len(q) > 5 and len(a) > 2:
                pairs.append((q, a))
            q = None
    return pairs

def lexical_support(answer, ctx):
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9'\-]+", answer.lower()) if w not in _STOP and len(w) > 2]
    return sum(w in ctx.text_lc for w in words) / len(words) if words else 0.0

def verify(ctx, q, a):
    """Grounded verifier = lexical fact overlap with the wake text AND the teacher-with-context judge."""
    lex = lexical_support(a, ctx)
    if lex < cfg.min_lex:
        return False, lex, 0.0
    p = ctx_judge(ctx, f"Question: {q}\nProposed answer: {a}\nIs the proposed answer fully supported by your "
                  "memories above, with nothing invented? Answer only Yes or No.")
    return p >= cfg.verifier_p, lex, p

def sft_backward(pairs, scale=1.0, opt=None):
    """Accumulate grads of the mean SFT loss over Q/A pairs, one pair's graph at a time (≈1 GB each on a 4B
    16-bit model - building them all before backward ran a T4 out of memory). Returns the mean loss."""
    total = 0.0
    for q, a in pairs:
        loss = -seq_logprob(student_prompt(q), enc(a) + [EOT_ID]) * (scale / len(pairs))
        opt.backward(loss) if opt else loss.backward()
        total += loss.item() / scale
    return total

def importance(pairs, params):
    """g_DR = ||∇θ L_SFT(dream)|| on the current expert (gradient-based data selection)."""
    for p in params:
        p.grad = None
    sft_backward(pairs, scale=1024.0)            # scaled so fp16 activation grads do not underflow on T4
    g = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in params if p.grad is not None)) / 1024.0
    for p in params:
        p.grad = None
    return g

@torch.no_grad()
def tau(dev):
    """Task metric τ: mean zero-context log-prob of the teacher's SHORT factual answers to held-out questions
    (same answer style the dreams teach; run 1 used long chatty answers, so no dream could ever improve it)."""
    return sum(seq_logprob(student_prompt(d["x"]), d["y_ids"]).item() for d in dev) / len(dev)

def seal_reward(pairs, dev, params, student):
    """Isolated copy: SFT the dream into a snapshot of the expert, reward 1 iff τ improves, restore."""
    use_experts(student)
    snap = [p.detach().clone() for p in params]
    before = tau(dev)
    inner = Optim(params, cfg.inner_lr)
    for _ in range(cfg.inner_steps):
        sft_backward(pairs, opt=inner); inner.step()
    after = tau(dev)
    with torch.no_grad():
        for p, s in zip(params, snap):
            p.copy_(s)
    return int(after > before + cfg.tau_eps), before, after

class random_routing:
    """§3.4: while dreaming each router additionally picks a random expert -> random subset + random scale."""
    def __init__(self, student):
        self.student = student
    def __enter__(self):
        prev, cur = self.student[:-1], self.student[-1]
        act = [e for e in prev if rng.random() < cfg.random_expert_p] + [cur]
        self.pick, self.s = rng.choice(act), rng.uniform(0.5, 1.5)
        use_experts(act)
        self.mods = [m for m in model.modules() if isinstance(m, LoraLayer) and self.pick in m.scaling]
        for m in self.mods:
            m.scaling[self.pick] *= self.s
        return act
    def __exit__(self, *exc):
        for m in self.mods:
            m.scaling[self.pick] /= self.s
        use_experts(self.student)

def dream_contexts(cycle, ctxs, teacher_experts):
    tasks = []
    for _ in range(cfg.dream_tasks):
        ctx = rng.choice(ctxs)
        if cfg.rehearse_prev_cycle and cycle > 1:   # cross-session rehearsal: mix earlier-cycle memories in
            prev = [e for c in range(1, cycle) for e in WAKE[c]]
            k = min(len(prev), max(4, len(ctx.episodes) // 2))
            eps = sorted(ctx.episodes[-len(ctx.episodes) // 2:] + rng.sample(prev, k), key=lambda e: e["ts"])
            tasks.append(WakeContext(f"{ctx.name}+rehearse", eps, teacher_experts))
        else:
            tasks.append(ctx)
    return tasks

def dream_cycle(cycle, ctxs, D, DEV, new_expert, frozen):
    student = list(frozen) + [new_expert]
    params = expert_params(new_expert)
    use_experts(student, trainable=[new_expert])
    log = Counter()
    for it in range(cfg.dream_iters):
        accepted = []
        for task in dream_contexts(cycle, ctxs, list(frozen)):
            dev = DEV.get(task.name.split("+")[0]) or [d for v in DEV.values() for d in v]
            if not dev:
                continue
            cands = []
            for _ in range(cfg.dreams_per_task):            # DREAM(i) ~ LM_θ(.|C) with random expert routing
                with random_routing(student):
                    ids = generate([task.prompt_ids(DREAM.format(n=4))], 256, 0.9)[0]
                pairs = parse_qa(dec(ids))
                log["dreams"] += 1
                good = [(q, a) for q, a in pairs if verify(task, q, a)[0]]   # hallucination filter
                log["qa_generated"] += len(pairs); log["qa_verified"] += len(good)
                if good:
                    use_experts(student)
                    cands.append((importance(good, params), good))
            if not cands:
                continue
            cands.sort(key=lambda c: -c[0])                   # Top-k by importance + b random for diversity
            keep = cands[: cfg.dream_topk]
            rest = cands[cfg.dream_topk:]
            keep += rng.sample(rest, min(cfg.dream_rand, len(rest)))
            for g, pairs in keep:
                r, b, a = seal_reward(pairs, dev, params, student)
                log["selected"] += 1; log["rewarded"] += r
                log["tau_gain_x1000"] += round(1000 * (a - b))
                if r:
                    accepted += pairs
            task.drop()
        if accepted:                                          # ReST-EM M-step: SFT only on rewarded dreams
            use_experts(student)
            opt = Optim(params, cfg.lr)
            for ep in range(cfg.dream_sft_epochs):
                rng.shuffle(accepted)
                for i in range(0, len(accepted), cfg.micro_bs):
                    sft_backward(accepted[i: i + cfg.micro_bs], opt=opt); opt.step()
            log["tau_after_mstep_x1000"] = round(1000 * tau([d for v in DEV.values() for d in v]))
        log["accepted_qa"] += len(accepted)
        print(f"  dream iter {it}: {dict(log)}")
    return dict(log)
''')

md(r'''
## Run the sleep cycles
Each cycle = **Wake buffer → Knowledge Seeding (NREM) → reset wake buffer → Dreaming (REM)**, then the zero-context benchmark.
''')

code(r'''
# @title 10 · Sleep cycle 1  (teacher = base + wake buffer 1, student = base + sleep_expert_1, no context)
def sleep_cycle(cycle):
    new_expert, frozen = EXPERTS[cycle - 1], EXPERTS[: cycle - 1]
    t0 = time.time()
    print(f"=== SLEEP CYCLE {cycle}: train {new_expert}, frozen {frozen or 'none'} ===")
    ctxs = wake_buffer(cycle, teacher_experts=frozen)          # Stage 1 (teacher = state before the update)
    ctx_probe = [p for p in QUIZ if p["cycle"] == cycle]
    if ctxs and ctx_probe:                                     # sanity: the teacher really knows the facts in-context
        bench_ctx = ctxs[0] if len(ctxs) == 1 else WakeContext(f"c{cycle}all", WAKE[cycle], frozen)
        RESULTS[f"{cycle}_teacher_in_context"] = run_benchmark(
            f"Teacher with wake buffer {cycle} (RAG-like)", frozen, probes=ctx_probe, ctx=bench_ctx)
    t1 = time.time()
    D, DEV = build_teacher_dataset(ctxs)                       # Stage 2: D ~ teacher
    t2 = time.time()
    ks = knowledge_seeding(cycle, ctxs, D, new_expert, frozen)
    t3 = time.time()
    dr = dream_cycle(cycle, ctxs, D, DEV, new_expert, frozen)   # Stage 3 (verifier still reads the wake buffer)
    t4 = time.time()
    for c in ctxs:
        c.drop()                                               # update: clear the fast memory (synaptic pruning)
    for p in expert_params(new_expert):
        p.requires_grad_(False)                                # expert is now frozen long-term memory
    use_experts([], trainable=[])
    save_dir = OUT_DIR / "adapters"
    model.save_pretrained(str(save_dir), selected_adapters=[new_expert])
    timing = {"build_D": round((t2 - t1) / 60, 1), "knowledge_seeding": round((t3 - t2) / 60, 1),
              "dreaming": round((t4 - t3) / 60, 1), "total": round((time.time() - t0) / 60, 1)}
    with open(OUT_DIR / f"sleep_cycle_{cycle}_log.json", "w") as fh:
        json.dump({"ks": ks, "dream": dr, "D_size": len(D), "dev_size": sum(map(len, DEV.values())),
                   "minutes": timing}, fh, indent=1)
    print(f"=== cycle {cycle} done: {timing} (minutes) ===")

def wake_buffer(cycle, teacher_experts):
    chunks = chunk_episodes(WAKE[cycle], cfg.max_ctx_tokens)
    ctxs = [WakeContext(f"c{cycle}w{i}", ch, teacher_experts) for i, ch in enumerate(chunks)]
    for c in ctxs:
        print(f"  wake buffer {c.name}: {len(c.episodes)} episodes, {len(c.prefix_ids())} tokens in teacher context")
    return ctxs

sleep_cycle(1)
RESULTS["1_after_sleep_1"] = run_benchmark("After Sleep 1 (no RAG)", ["sleep_expert_1"])
(OUT_DIR / "results_partial.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                                           for k, v in RESULTS.items()}, indent=1, default=str))
archive_outputs("after_cycle_1")
''')

code(r'''
# @title 11 · Sleep cycle 2  (teacher = base + frozen expert_1 + wake buffer 2, student adds sleep_expert_2)
assert "1_after_sleep_1" in RESULTS, "sleep cycle 1 did not finish - fix the error above before cycle 2"
sleep_cycle(2)
RESULTS["2_after_sleep_2_stacked"] = run_benchmark("After Sleep 2 (no RAG, experts 1+2)", ["sleep_expert_1", "sleep_expert_2"])
(OUT_DIR / "results_partial.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                                           for k, v in RESULTS.items()}, indent=1, default=str))
archive_outputs("after_cycle_2")
''')

md(r'''
## Stage 4 — Modular multi-expert merge (TIES) + GGUF export
The paper keeps experts separate (that is the `stacked` row). For a single small file usable from
llama.cpp / Ollama we merge them with TIES (trim → elect sign → disjoint merge) and benchmark the merged adapter too.
''')

code(r'''
# @title 12 · TIES merge + benchmark
assert "2_after_sleep_2_stacked" in RESULTS, "sleep cycle 2 did not finish - not merging untrained experts"
MERGED = "hikari_sleep"
TIES_DENSITY = 0.5
if MERGED in model.peft_config:
    model.delete_adapter(MERGED)
model.add_weighted_adapter(adapters=["sleep_expert_1", "sleep_expert_2"], weights=[1.0, 1.0],
                           adapter_name=MERGED, combination_type="ties", density=TIES_DENSITY,
                           majority_sign_method="total")
for p in expert_params(MERGED):
    p.requires_grad_(False)
RESULTS["3_after_sleep_2_ties"] = run_benchmark("After Sleep 2 (no RAG, TIES merged)", [MERGED])
model.save_pretrained(str(OUT_DIR / "adapters"), selected_adapters=[MERGED])
''')

code(r'''
# @title 13 · Results: Zero-Context Continual Retention
order = [k for k in ["0_base_no_rag", "1_after_sleep_1", "2_after_sleep_2_stacked", "3_after_sleep_2_ties"] if k in RESULTS]
hdr = f"{'model':<38}{'C1 hit':>8}{'C1 part':>9}{'C1 logp':>9}{'C2 hit':>8}{'C2 part':>9}{'C2 logp':>9}{'style':>8}"
print(hdr); print("-" * len(hdr))
for k in order + [k for k in RESULTS if k.endswith("teacher_in_context")]:
    r = RESULTS[k]
    f = lambda c, m: r[f"c{c}"][m]
    print(f"{r['label'][:37]:<38}{f(1,'hit'):>8.2f}{f(1,'partial'):>9.2f}{f(1,'logp'):>9.2f}"
          f"{f(2,'hit'):>8.2f}{f(2,'partial'):>9.2f}{f(2,'logp'):>9.2f}{r['style_logp']:>8.2f}")
if "1_after_sleep_1" in RESULTS and "2_after_sleep_2_stacked" in RESULTS:
    a, b = RESULTS["1_after_sleep_1"]["c1"], RESULTS["2_after_sleep_2_stacked"]["c1"]
    print(f"\nCycle-1 forgetting after Sleep 2: Δpartial {b['partial'] - a['partial']:+.3f}, Δlogp {b['logp'] - a['logp']:+.3f}")
    base = RESULTS["0_base_no_rag"]
    print(f"Cycle-1 retention vs base: Δlogp {a['logp'] - base['c1']['logp']:+.3f}; "
          f"Cycle-2 gain after Sleep 2: Δlogp {RESULTS['2_after_sleep_2_stacked']['c2']['logp'] - base['c2']['logp']:+.3f}")

summary = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in RESULTS.items()}
with open(OUT_DIR / "retention_benchmark.json", "w") as fh:
    json.dump({"profile": PROFILE, "model": MODEL_ID, "config": vars(cfg), "results": summary}, fh, indent=1, default=str)
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = [RESULTS[k]["label"].replace(" (no RAG", "\n(no RAG") for k in order]
    for i, metric in enumerate(["partial", "logp"]):
        xs = range(len(order))
        ax[i].bar([x - 0.2 for x in xs], [RESULTS[k]["c1"][metric] for k in order], 0.4, label="cycle-1 facts")
        ax[i].bar([x + 0.2 for x in xs], [RESULTS[k]["c2"][metric] for k in order], 0.4, label="cycle-2 facts")
        ax[i].set_xticks(list(xs)); ax[i].set_xticklabels(labels, fontsize=7); ax[i].set_title(f"zero-context {metric}")
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT_DIR / "retention_benchmark.png", dpi=130); plt.show()
except Exception as e:
    print("plot skipped:", e)
''')

code(r'''
# @title 14 · Export hikari-sleep-lora.gguf (llama.cpp convert_lora_to_gguf.py)
assert "3_after_sleep_2_ties" in RESULTS, "merge did not finish - not exporting"
GGUF_OUTTYPE = os.environ.get("SLEEP_GGUF_OUTTYPE", "q8_0")    # q8_0 ≈ 33 MB at r=16 on MLP (rank-16 B matrices stay f16: narrower than q8_0 blocks); f16 ≈ 42 MB
EXPORT_GGUF = os.environ.get("SLEEP_EXPORT_GGUF", "1") == "1"

def export_gguf():
    adapter_dir = OUT_DIR / "adapters" / MERGED
    if not (adapter_dir / "adapter_config.json").exists():
        adapter_dir = OUT_DIR / "adapters"
    base_id = GGUF_BASE_ID or re.sub(r"^unsloth/(.+?)(-unsloth)?-bnb-4bit$", r"Qwen/\1", MODEL_ID)
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(base_id, allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                                                          "*.model", "special_tokens_map.json", "vocab*", "merges.txt"])
    lcpp = WORK_DIR / "llama.cpp"
    if not lcpp.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(lcpp)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(lcpp / "gguf-py"), "sentencepiece", "protobuf"], check=True)
    out = OUT_DIR / "hikari-sleep-lora.gguf"
    cmd = [sys.executable, str(lcpp / "convert_lora_to_gguf.py"), str(adapter_dir), "--base", base_dir,
           "--outtype", GGUF_OUTTYPE, "--outfile", str(out)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"GGUF LoRA: {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return out

GGUF_PATH = export_gguf() if EXPORT_GGUF else None
''')

code(r'''
# @title 15 · Save everything (and copy to Google Drive when mounted) - runs even if an earlier stage failed
_complete = "3_after_sleep_2_ties" in RESULTS and globals().get("GGUF_PATH") is not None
manifest = {
    "paper": "arXiv:2606.03979", "profile": PROFILE, "model": MODEL_ID, "experts": EXPERTS,
    "merged": globals().get("MERGED"), "complete": _complete, "results_so_far": sorted(RESULTS),
    "gguf": str(globals().get("GGUF_PATH")) if globals().get("GGUF_PATH") else None,
    "usage": "llama-server -m <Qwen3-4B-Instruct-2507 GGUF> --lora hikari-sleep-lora.gguf  |  Ollama: ADAPTER ./hikari-sleep-lora.gguf",
}
(OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
if Path("/content/drive/MyDrive").exists():
    dst = Path(DRIVE_DIR) / f"outputs_{time.strftime('%Y%m%d_%H%M')}"
    shutil.copytree(OUT_DIR, dst, dirs_exist_ok=True)
    print("copied outputs to", dst)
archive_outputs("final" if _complete else "incomplete")   # one file for `colab download`
print("done ->", OUT_DIR)
''')


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {}, "source": lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": "hikari_sleep_arxiv2606.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(nb['cells'])} cells ({sum(c['cell_type'] == 'code' for c in nb['cells'])} code)")
