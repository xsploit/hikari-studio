"""Generate tools/hikari_sleep_colab_lora.ipynb — 1-click Google Colab SLEEP QLoRA for a 4B model.

Usage:  python tools/build_sleep_colab_notebook.py [--base-model unsloth/Qwen3-4B-unsloth-bnb-4bit]
The notebook: mounts Drive, reads sleep_batch.jsonl (from POST /api/memory/export-colab), optionally
resumes the previous adapter, trains a rank-16 QLoRA (~5 min on a T4) with persona-replay
anti-forgetting, and writes hikari-sleep-lora.gguf + an Ollama Modelfile / llama-server command
back to Drive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_BASE = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
OUT = Path(__file__).resolve().parent / "hikari_sleep_colab_lora.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build(base_model: str = DEFAULT_BASE) -> dict:
    cells = [
        md("""
# 🌙 Hikari SLEEP — 4B QLoRA in ~5 minutes (Google Colab)

Teaches Hikari-chan what she consolidated while "sleeping" (memory cards, inside jokes, your preferences)
without forgetting who she is.

1. In Hikari Studio run **Sleep** (or `POST /api/memory/export-colab`) → it writes `data/memory/sleep_batch.jsonl`.
2. Upload that file to Google Drive → `MyDrive/HikariSleep/sleep_batch.jsonl`.
3. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.
4. Download `MyDrive/HikariSleep/hikari-sleep-lora.gguf` and point Hikari Studio's `ai.localLoraPath` at it
   (brain mode `local_4b_fish`).

The batch already contains ~10% core-persona *replay* rows; this notebook re-checks the ratio and tops it up
so the adapter never drifts away from her personality (anti catastrophic forgetting).
"""),
        code("""
%%capture
# Unsloth (fast 4-bit QLoRA) + TRL. Colab already ships torch/transformers.
!pip install -q unsloth
!pip install -q --upgrade --no-deps "trl>=0.19" "peft>=0.15"
"""),
        code(f"""
#@title ⚙️ Settings
BASE_MODEL = "{base_model}"  #@param ["unsloth/gemma-3-4b-it-unsloth-bnb-4bit", "unsloth/Qwen3-4B-unsloth-bnb-4bit"]
DRIVE_DIR = "/content/drive/MyDrive/HikariSleep"  #@param {{type:"string"}}
BATCH_FILE = "sleep_batch.jsonl"  #@param {{type:"string"}}
RESUME_PREVIOUS_LORA = True  #@param {{type:"boolean"}}
MAX_STEPS = 60  #@param {{type:"integer"}}
LORA_RANK = 16  #@param {{type:"integer"}}
LEARNING_RATE = 2e-4  #@param {{type:"number"}}
REPLAY_RATIO = 0.10  #@param {{type:"number"}}
MAX_SEQ_LEN = 2048
GGUF_OUTTYPE = "q8_0"  # q8_0 keeps the rank-16 adapter around 25-35 MB (f16 is ~2x)
ADAPTER_DIR = f"{{DRIVE_DIR}}/hikari-sleep-adapter"     # PEFT adapter (resume point)
GGUF_PATH = f"{{DRIVE_DIR}}/hikari-sleep-lora.gguf"
IS_GEMMA = "gemma" in BASE_MODEL.lower()
CHAT_TEMPLATE = "gemma-3" if IS_GEMMA else "qwen3"
# Full-precision config twin of the 4-bit base (llama.cpp's adapter converter needs its config)
BASE_CONFIG_ID = "unsloth/gemma-3-4b-it" if IS_GEMMA else "unsloth/Qwen3-4B"
print(BASE_MODEL, "->", GGUF_PATH)
"""),
        code("""
from google.colab import drive
drive.mount("/content/drive")
import os
os.makedirs(DRIVE_DIR, exist_ok=True)
"""),
        code("""
import json, math, random

batch_path = os.path.join(DRIVE_DIR, BATCH_FILE)
assert os.path.exists(batch_path), f"Upload {BATCH_FILE} to {DRIVE_DIR} first (Hikari Studio: POST /api/memory/export-colab)"
rows = []
with open(batch_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        msgs = row.get("messages") or []
        if any(m.get("role") == "assistant" and m.get("content") for m in msgs):
            rows.append(row)

replay = [r for r in rows if r.get("replay")]
fresh = [r for r in rows if not r.get("replay")]
# Keep at least REPLAY_RATIO persona replay rows (anti catastrophic forgetting)
need = math.ceil(len(fresh) * REPLAY_RATIO / max(1e-6, 1 - REPLAY_RATIO))
if replay and len(replay) < need:
    replay = (replay * (need // len(replay) + 1))[:need]
data_rows = fresh + replay
random.Random(7).shuffle(data_rows)
print(f"{len(fresh)} new pairs + {len(replay)} persona replay = {len(data_rows)} rows "
      f"({100 * len(replay) / max(1, len(data_rows)):.1f}% replay)")
"""),
        code("""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

resume = RESUME_PREVIOUS_LORA and os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR if resume else BASE_MODEL,   # an adapter dir loads base + previous LoRA
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
if not resume:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
print("Resumed previous SLEEP adapter" if resume else "Fresh rank-%d adapter" % LORA_RANK)
tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
"""),
        code("""
from datasets import Dataset

def to_text(row):
    msgs = [m for m in row["messages"] if m.get("content")]
    if IS_GEMMA and msgs and msgs[0]["role"] == "system":
        # Gemma has no system role: fold the persona into the first user turn
        sys_txt = msgs[0]["content"]
        msgs = msgs[1:]
        if msgs and msgs[0]["role"] == "user":
            msgs[0] = {"role": "user", "content": sys_txt + "\\n\\n" + msgs[0]["content"]}
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if IS_GEMMA:
        text = text.removeprefix("<bos>")   # the trainer adds BOS itself
    return {"text": text}

dataset = Dataset.from_list(data_rows).map(to_text, remove_columns=list(data_rows[0].keys()))
print(dataset[0]["text"][:600])
"""),
        code("""
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="/content/outputs",
        report_to="none",
    ),
)
if IS_GEMMA:
    trainer = train_on_responses_only(trainer, instruction_part="<start_of_turn>user\\n",
                                      response_part="<start_of_turn>model\\n")
else:
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
                                      response_part="<|im_start|>assistant\\n")
stats = trainer.train()
print(f"Trained {stats.global_step} steps in {stats.metrics.get('train_runtime', 0):.0f}s")
"""),
        code("""
# Quick vibe check (should still sound like Hikari-chan)
FastLanguageModel.for_inference(model)
probe = [{"role": "user", "content": "Hey Hikari, do you remember anything about me?"}]
ids = tokenizer.apply_chat_template(probe, add_generation_prompt=True, return_tensors="pt").to("cuda")
out = model.generate(input_ids=ids, max_new_tokens=96, temperature=0.8, do_sample=True)
print(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
"""),
        code("""
# Save the PEFT adapter to Drive (next night's run resumes from it)
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Adapter saved:", ADAPTER_DIR)
"""),
        code("""
%%capture
# llama.cpp's converter turns the PEFT adapter into a tiny GGUF LoRA (no merge, no full model export)
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
!pip install -q -r /content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
"""),
        code("""
!python /content/llama.cpp/convert_lora_to_gguf.py "{ADAPTER_DIR}" --base-model-id "{BASE_CONFIG_ID}" --outtype {GGUF_OUTTYPE} --outfile "{GGUF_PATH}"
size_mb = os.path.getsize(GGUF_PATH) / 1e6
print(f"✅ {GGUF_PATH}  ({size_mb:.1f} MB)")
"""),
        code("""
ollama_base = "gemma3:4b" if IS_GEMMA else "qwen3:4b"
gguf_base = "gemma-3-4b-it-Q4_K_M.gguf" if IS_GEMMA else "Qwen3-4B-Q4_K_M.gguf"
modelfile = f'''FROM {ollama_base}
ADAPTER ./hikari-sleep-lora.gguf
PARAMETER temperature 0.82
'''
with open(os.path.join(DRIVE_DIR, "Modelfile"), "w") as f:
    f.write(modelfile)
print("---- Ollama ----")
print(modelfile)
print("ollama create hikari-sleep-4b -f Modelfile")
print()
print("---- llama-server ----")
print(f"llama-server -m {gguf_base} --lora hikari-sleep-lora.gguf -c 4096 --port 8081")
print()
print("Hikari Studio: ai.brainMode = local_4b_fish, ai.localModel = hikari-sleep-4b,")
print("               ai.localLoraPath = <path to hikari-sleep-lora.gguf>")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "hikari_sleep_colab_lora.ipynb", "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(build(a.base_model), indent=1, ensure_ascii=False), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
