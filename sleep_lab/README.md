# sleep_lab — Hikari-chan learns to sleep (arXiv:2606.03979)

A runnable, single-GPU implementation of Google Research's **SLEEP** framework,
*"Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"*
(Behrouz, Hashemi, Javanmard, Mirrokni, arXiv:2606.03979), trained on Hikari-chan's real
Web Waifu memory logs.

The goal: Hikari remembers what happened on stream **with zero memories in the prompt**
(no RAG), because the memories were moved into small LoRA "sleep experts". She also must not
forget cycle-1 memories when she sleeps on cycle-2 memories.

Everything lives in this folder. Nothing in `server.py`, `modules/`, `core/`, `web/` or
`desktop-tauri/` is touched. The memory logs are opened read-only.

## Files

| File | What it does |
|---|---|
| `import_waifu_logs.py` | Parses both `ladybug-memory.db.json` snapshot stores, removes duplicates, sorts by time, redacts secrets, writes `datasets/` |
| `hikari_sleep_arxiv2606.ipynb` | The Colab notebook: Wake → Knowledge Seeding → Dreaming → TIES merge → GGUF → benchmark |
| `build_notebook.py` | Source of the notebook. Edit cells here, then rebuild (keeps the `.ipynb` valid) |
| `run_colab_sleep.py` | Bundles `datasets/` + notebook and launches it (Colab CLI, Drive sync, or plain zip) |
| `verify_sleep_lab.py` | Verification gate (compile checks, notebook checks, dataset privacy checks, optional CPU run) |
| `datasets/` | **Private, gitignored.** `wake_cycle_1/2.jsonl`, `eval_recall_quiz.jsonl`, `persona_anchor.jsonl`, `manifest.json` |
| `dist/`, `outputs/` | Gitignored build outputs (bundle zip, adapters, GGUF, benchmark JSON/PNG) |

## Quick start

```powershell
cd C:\Users\SUBSECT\Documents\Codex\2026-11-21\hikari-2-5d\hikari-studio
python sleep_lab\import_waifu_logs.py          # real logs -> sleep_lab\datasets\ (prints counts only)
python sleep_lab\verify_sleep_lab.py           # static checks
python sleep_lab\run_colab_sleep.py            # -> sleep_lab\dist\sleep_lab_bundle.zip + notebook
```

Then pick one way to run it:

* **Plain upload:** open `dist/hikari_sleep_arxiv2606.ipynb` in Colab, choose a GPU runtime,
  upload `sleep_lab_bundle.zip` to `/content`, and Run all.
* **Google Drive:** `python sleep_lab\run_colab_sleep.py --mode drive` (auto-finds `G:\My Drive`
  etc., or pass `--drive-dir`). Open the notebook from `MyDrive/hikari_sleep_lab/` and Run all.
  The notebook mounts Drive, finds the bundle, and copies results back.
* **Colab CLI:** the launcher runs your CLI with an argument template:
  ```powershell
  python sleep_lab\run_colab_sleep.py --mode cli --dry-run                 # print the command only
  python sleep_lab\run_colab_sleep.py --mode cli --cli <your-colab-binary> `
      --cli-args "run {notebook} --gpu {gpu} --upload {bundle}" --gpu A100
  ```
  Placeholders: `{notebook} {bundle} {gpu} {profile} {dist} {datasets}`. Defaults can come from the
  `COLAB_CLI` / `COLAB_CLI_ARGS` environment variables. No Colab CLI was found on this PC, so the
  binary name and flags are settings, not hard-coded.

### Profiles (set by `--gpu`, or `--profile`)

| Profile | GPU | What it runs | Rough time |
|---|---|---|---|
| `smoke` | any | whole pipeline, a few steps each | ~10 min |
| `t4` | T4 / L4 | 40 KS steps × 8 samples, 3 dream tasks × 6 dreams | ~1–1.5 h on T4 |
| `a100` | A100 | paper-like: 100 steps, effective batch 32 (Table 5), 2 dream iterations × 4 tasks × 12 dreams | ~1 h |

Default model: `Qwen/Qwen3-4B-Instruct-2507` in 4-bit NF4. Options in the config cell:
`unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (Unsloth's pre-quantized weights, faster download) or
`google/gemma-3-4b-it` (gated: log in to Hugging Face first; the KV prefix cache is turned off
for Gemma's sliding-window cache).

## What the importer extracts

Snapshot schema: `{"reason", "snapshots": {key: {kind, scopeKey, updatedAt, value}}, ...}`.

| Snapshot `kind` | Used as |
|---|---|
| `grillo-records:turn_events` / `evidence_records` | conversation turns, deduped by turn id, paired user→Hikari |
| `semantic` | user/assistant exchanges (deduped against the turns) |
| `grillo` (`blocks`, `candidates`, `diaryEntries`) | memory facts, diary summaries/thoughts |
| `grillo-records:memory_blocks / memory_slots / memory_slot_patches / memory_candidates / diary_entries` | memory facts |
| `relationships:all` | user-said facts, relationship summaries/diaries, stage + mood |
| `grillo_activity_log`, `worker_context_traces`, `grillo-singleton` | skipped (worker internals and prompts) |

Test scopes (`test`, `restart`, `recovery`, `neuro-sama`) are dropped unless `--include-test`.
The bot's routing text (`Discord trusted controller NAME:` … `metadata: source=...`) is stripped.
API keys, tokens, JWTs, emails and IPs are redacted; long Discord/Twitch numeric IDs become short hashes.

**Real-data run (2026-11-22):** 252 turn rows → 134 unique turns, 61 semantic rows, 83 fact rows,
6 test rows skipped → **117 deduplicated episodes**:

* `wake_cycle_1.jsonl`: 58 episodes (29 exchanges, 29 facts), ~3.5k tokens, 2026-05-13 → 2026-07-14
* `wake_cycle_2.jsonl`: 59 episodes (38 exchanges, 21 facts), ~3.7k tokens, 2026-07-14
* `eval_recall_quiz.jsonl`: 48 held-out probes (cycle 1: 27, cycle 2: 21)
* `persona_anchor.jsonl`: Hikari's verbatim system prompt (parsed from `ai_companion.py` with `ast`,
  not imported) + 16 real in-voice replies

**Quiz probes.** Fill-in-the-blank questions over one specific detail of a memory, plus one
relationship-state question per cycle. Tier **A** blanks are names and numbers (for example a product
or model name); tier **B** blanks are rare content words and are a softer test. The probe wording never
appears in the wake buffers, so the only way to answer with zero context is through the weights.
The data is small (one evening of Discord voice plus a few local/Twitch sessions), so treat the scores as
a smoke-level signal, not a leaderboard.

## How the notebook maps to the paper

| Paper | Here |
|---|---|
| Wake: fast memory = in-context learning (§3.1–3.2) | **Stage 1.** The cycle's wake buffer goes into the teacher's system prompt, KV-cached once per buffer |
| Parameter expansion: add a low-rank expert `{A,B}` to the MLP block; allocate early, mask until used (§3.2) | LoRA slots `sleep_expert_1`, `sleep_expert_2` (rank 16, `gate/up/down_proj`), created at load and inactive until their sleep. Earlier slots are frozen |
| Compute → Consolidate → Update; teacher = state before update, student = expanded state (§3.3) | Teacher = frozen model + earlier experts **with** the wake buffer. Student = same model, **no memories in the prompt**, + the new expert. Afterwards the wake buffer is cleared (the paper's "synaptic pruning" of the sender) and the expert stays on |
| `L = (1−λ)E_{(x,y)∼D}[F] + λE_{x∼D}E_{y∼student}[F]`, D sampled from the teacher, no backprop through sampling (§3.3) | **Stage 2.** D = teacher-written memory questions + replayed wake prompts + persona prompts, answered by the teacher. Each step picks on-policy (student samples) with probability λ, otherwise the teacher's y. F = generalized JSD(β) on the teacher's top-K tokens plus one tail bucket. β=1 gives reverse KL |
| Learning to Imitate: random prefix of a teacher dream, student continues; `r = γ·r_sem + (1−γ)·r_abs`, Levenshtein with threshold z0; `L_KS = (1−α)E[r] − αE[D]` (Eq. 3–4) | `lti_term`: a frozen base-model judge gives the yes/no semantic reward, plus token-level Levenshtein. The update is a group-normalized policy gradient |
| Dreaming: dreams `∼ LM_θ(·|C)` with random expert routing; ‖∇L_SFT‖ importance → top-k + b random; isolated LoRA SFT per dream; reward 1 if τ improves; ReST-EM (§3.4) | **Stage 3.** Dream Q&A from the wake buffer (cycle 2 also mixes in cycle-1 memories: cross-session rehearsal). A **grounded verifier** (teacher-with-context yes/no judge + word overlap with the wake text) drops hallucinations. Dreams are ranked by gradient norm. Each kept dream is trained into a snapshot of the expert, and its reward is 1 if zero-context log-prob on held-in teacher Q&A improves. The snapshot is then restored. Only rewarded dreams are used in the ReST-EM SFT step |
| — (not in paper) | **Stage 4.** `add_weighted_adapter(["sleep_expert_1","sleep_expert_2"], combination_type="ties")` → `hikari_sleep`, exported with llama.cpp `convert_lora_to_gguf.py` (measured on a T4 run: 32.9 MB at q8_0; the rank-16 `B` matrices are narrower than q8_0 blocks and stay f16; f16 ≈ 42 MB) |

### Honest deviations (single GPU, 4B dense model)

* The paper's backbone is the Hope / Continuum-Memory-System architecture with chains of MoE MLPs
  at different update frequencies. Here the "fast" memory is the context window and the "slow"
  memories are LoRA experts on a standard dense 4B model.
* The paper does not publish λ, α, γ, z0, top-K, temperature or the reward model. Our defaults:
  λ=0.5, β=0.5, α=0.6, γ=0.5, z0=0.6·max length, K=64, T=1. The frozen base model acts as the reward model.
* Rank 16 (your spec) instead of the paper's 64, and lr 1e-4 instead of 5e-6: a rank-16 adapter
  trained for 40–100 steps learns nothing at 5e-6.
* Random expert routing has only two experts to choose from, so it is implemented as a random
  subset of earlier experts plus a random 0.5–1.5× scale on one active expert.
* The TIES merge is our addition. TIES works on `A` and `B` separately, and independently trained
  `A` matrices do not share a basis, so the merged adapter can score below the stacked experts. The
  benchmark reports both. For a lossless single file use `combination_type="cat"` (rank 32, about twice the size).
* Rehearsal dreams in cycle 2 do see cycle-1 memories, which is replay. For a pure no-replay
  forgetting test set `rehearse_prev_cycle=False`.

## Benchmark (cell 13)

All rows run with **no memories in the prompt** (persona + question only), except the two
"Teacher with wake buffer" rows, which are the in-context upper bound:

1. Base 4B (no RAG)
2. After Sleep 1 (no RAG): cycle-1 retention
3. After Sleep 2, experts 1+2 stacked (no RAG): cycle-1 forgetting check + cycle-2 retention
4. After Sleep 2, TIES merged (no RAG): the exported adapter

Metrics: `hit` (all answer keywords produced), `partial` (keyword recall), `tierA_hit`, `logp`
(mean log-prob of the reference answer, the most sensitive signal on small data), and `style logp`
on real Hikari replies (persona drift check). The notebook also prints Δ for cycle-1 forgetting.
Results are saved to `outputs/retention_benchmark.json` and `.png`.

## Using the result in Hikari Studio

`outputs/hikari-sleep-lora.gguf` loads on top of a Qwen3-4B-Instruct-2507 GGUF:

```text
llama-server -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf --lora hikari-sleep-lora.gguf
# Ollama Modelfile:  FROM qwen3:4b-instruct   ADAPTER ./hikari-sleep-lora.gguf
```

`modules/ai_companion.py` already has `localModel: "hikari-sleep-4b"` and a `localLoraPath` setting
for this file. This lab does not change it; set it from the app's own settings.

## Verification

```powershell
python sleep_lab\verify_sleep_lab.py --run-import        # compile + notebook + datasets (+ real import)
# full CPU execution of every notebook cell (needs torch/transformers/peft):
python sleep_lab\verify_sleep_lab.py --exec-notebook                                   # tiny random model
python sleep_lab\verify_sleep_lab.py --exec-notebook --tiny-model Qwen/Qwen3-0.6B --no-test-hooks
```

Checks: `py_compile` on every `.py`; notebook JSON + `nbformat.validate` + `ast.parse`/compile of every
code cell (IPython `!`/`%` lines become `pass`); dataset schema; chronological split; no secret or
raw-ID patterns left; probe wording held out of the wake buffers; persona prompt verbatim; `.gitignore`
covers `datasets/`. The notebook also checks itself on startup (cell 6): F(p‖p)=0, the cached teacher
matches the uncached one, the new expert receives gradient, and the frozen expert receives none.

## Privacy

* `datasets/`, `dist/`, `outputs/` and `*.gguf` are gitignored. The bundle zip contains your real
  memories: upload it only to your own Colab runtime or Drive.
* The importer, launcher and verifier print counts only. The notebook prints no memory text unless
  `SLEEP_SHOW_SAMPLES=1`.

## Run log

**Run 1 (T4, 2026-11-22, stopped in cycle 2):** recall without memories went from 4% to 7% (teacher with
memories: 78%), persona style improved, and one cycle took 2 h 40 min. The log showed three problems:

| Problem | Cause | Fix (run 2) |
|---|---|---|
| Recall barely moved | only 46 practice samples, built from 24 generic questions, so most memories were never asked about | the teacher writes questions for **every group of 6 memories** (about 40 per cycle), plus replay and persona prompts |
| 0 of 9 dreams rewarded | τ scored long chatty answers, while dreams teach short facts; the test update was tiny | τ = zero-context log-prob of the teacher's **short factual answers** to held-out questions; inner update 3 steps at 3e-4; reward when τ improves at all |
| Learning-to-Imitate reward always 0 | Levenshtein threshold z0 = 0.6 and a "strict, literal" semantic judge | z0 = 0.9 (partial credit), fair semantic judge, batched; `r_abs` and `r_sem` are logged separately |
| Slow | NF4 generation on a T4, and every teacher generation re-read the 4k-token wake buffer | 16-bit weights when the GPU has at least 14 GB, generation from a copy of the wake buffer's KV cache, batched judges (each checked against the slow path in cell 6) |

**Why not Unsloth:** its fused LoRA kernels read only the first active adapter
(`get_lora_parameters` → `active_adapters[0]`). SLEEP runs the frozen earlier expert and the new expert
together; that's what prevents forgetting. Unsloth would silently drop one of them. The 16-bit switch gives
the speedup without that risk.

**Safety for every Colab run:**
* The launcher stops the VM when the run ends, on success or failure.
* Each step has a hard time limit (T4 profile: 3.25 h).
* `exec` fails after 40 minutes with no output.
* The notebook archives results after each sleep cycle, and the launcher downloads them if the run fails.
* A detached `colab_watchdog.py` stops the VM (saving partial results first) if the launcher dies or overruns, then confirms with `colab sessions`.
