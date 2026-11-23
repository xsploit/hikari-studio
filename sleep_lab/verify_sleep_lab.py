#!/usr/bin/env python3
"""Verification gate for sleep_lab.

Static checks (no GPU, no torch needed):
  1. py_compile every .py file in sleep_lab/
  2. the notebook is valid nbformat-4 JSON, and every code cell parses with ast
     (IPython `!shell` / `%magic` lines are replaced by `pass` at the same indent first)
  3. optional: run import_waifu_logs.py on the real logs and check the datasets
     (schema, counts, no leftover secrets / raw platform IDs, gitignore covers them)

Dynamic check (optional, needs torch + transformers + peft):
  --exec-notebook  executes every notebook code cell top to bottom on CPU with a tiny
                   random model (profile `cpu_test`, GGUF export off). Random weights produce
                   no parsable dream Q&A, so a test-only hook feeds wake-memory Q&A into the
                   dream verifier path; nothing in the notebook itself is changed.

Prints counts and pass/fail only, never dataset content.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import py_compile
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "hikari_sleep_arxiv2606.ipynb"
DATASETS = HERE / "datasets"
_MAGIC = re.compile(r"^(\s*)[!%]")
FAILS: list[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    FAILS.append(msg)


def strip_magics(src: str) -> str:
    return "\n".join((m.group(1) + "pass") if (m := _MAGIC.match(line)) else line for line in src.splitlines())


def check_py() -> None:
    print("[1] py_compile")
    for f in sorted(HERE.glob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True, cfile=os.devnull if os.name != "nt" else None)
            ok(f.name)
        except py_compile.PyCompileError as e:
            fail(f"{f.name}: {e.msg}")


def load_notebook() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def check_notebook() -> None:
    print("[2] notebook")
    try:
        nb = load_notebook()
    except Exception as e:  # noqa: BLE001
        fail(f"{NB.name} is not valid JSON: {e}")
        return
    ok(f"{NB.name} is valid JSON ({NB.stat().st_size // 1024} KB)")
    if nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
        fail("nbformat must be 4 with a cells list")
    for k in ("kernelspec", "language_info"):
        if k not in nb.get("metadata", {}):
            fail(f"metadata.{k} missing")
    try:
        import nbformat  # type: ignore
        nbformat.validate(nbformat.reads(NB.read_text(encoding="utf-8"), as_version=4))
        ok("nbformat.validate")
    except ImportError:
        print("  skip  nbformat not installed (structural checks only)")
    except Exception as e:  # noqa: BLE001
        fail(f"nbformat.validate: {e}")
    n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") not in ("code", "markdown"):
            fail(f"cell {i}: bad cell_type")
            continue
        if not isinstance(cell.get("source"), (list, str)):
            fail(f"cell {i}: source must be list/str")
            continue
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        if "outputs" not in cell or "execution_count" not in cell:
            fail(f"cell {i}: code cell needs outputs + execution_count")
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        try:
            compile(ast.parse(strip_magics(src)), f"<cell {i}>", "exec")
        except SyntaxError as e:
            fail(f"code cell {i} line {e.lineno}: {e.msg}")
    ok(f"{n_code} code cells parse + compile")


SECRET_RES = [re.compile(p) for p in (
    r"\bsk-[A-Za-z0-9_\-]{16,}", r"\bAIza[0-9A-Za-z_\-]{30,}", r"\bghp_[A-Za-z0-9]{20,}", r"\bhf_[A-Za-z0-9]{20,}",
    r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.", r"\b\d{15,21}\b", r"[\w.+\-]+@[\w\-]+\.[a-z]{2,}")]


def check_datasets(run_import: bool) -> None:
    print("[3] datasets")
    if run_import:
        r = subprocess.run([sys.executable, str(HERE / "import_waifu_logs.py")], capture_output=True, text=True)
        if r.returncode != 0:
            fail(f"import_waifu_logs.py exited {r.returncode}: {r.stderr.strip()[-300:]}")
            return
        for line in r.stdout.splitlines():
            print("        " + line)
        ok("import_waifu_logs.py ran on the real logs")
    need = {"wake_cycle_1.jsonl": ("id", "cycle", "kind", "ts", "text"),
            "wake_cycle_2.jsonl": ("id", "cycle", "kind", "ts", "text"),
            "eval_recall_quiz.jsonl": ("id", "cycle", "question", "answer", "answer_keywords", "tier"),
            "persona_anchor.jsonl": ("id", "kind")}
    rows = {}
    for name, keys in need.items():
        p = DATASETS / name
        if not p.exists():
            fail(f"{name} missing")
            continue
        data = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows[name] = data
        bad = [i for i, r in enumerate(data) if any(k not in r for k in keys)]
        if not data or bad:
            fail(f"{name}: {len(data)} rows, {len(bad)} missing keys")
        else:
            ok(f"{name}: {len(data)} rows, schema ok")
        text = p.read_text(encoding="utf-8")
        hits = sum(len(rx.findall(text)) for rx in SECRET_RES)
        (ok if hits == 0 else fail)(f"{name}: {hits} secret/ID pattern hits after redaction")
    if {"wake_cycle_1.jsonl", "wake_cycle_2.jsonl"} <= rows.keys():
        w1, w2 = rows["wake_cycle_1.jsonl"], rows["wake_cycle_2.jsonl"]
        if max(r["ts"] for r in w1) <= min(r["ts"] for r in w2):
            ok("cycle 1 strictly precedes cycle 2 (chronological split)")
        else:
            fail("chronological split overlaps")
        ids = [r["id"] for r in w1 + w2]
        (ok if len(ids) == len(set(ids)) else fail)(f"{len(ids)} episodes, no duplicates across cycles")
    if "eval_recall_quiz.jsonl" in rows:
        q = rows["eval_recall_quiz.jsonl"]
        cyc = {c: sum(1 for r in q if r["cycle"] == c) for c in (1, 2)}
        (ok if all(cyc.values()) else fail)(f"quiz covers both cycles {cyc}")
        wake_text = " ".join(r["text"] for r in rows.get("wake_cycle_1.jsonl", []) + rows.get("wake_cycle_2.jsonl", []))
        leaked = sum(1 for r in q if r["question"] in wake_text)
        (ok if leaked == 0 else fail)(f"probe wording held out of wake buffers ({leaked} leaks)")
    if "persona_anchor.jsonl" in rows:
        sysrow = [r for r in rows["persona_anchor.jsonl"] if r["kind"] == "persona_system"]
        src = (HERE.parent / "modules" / "ai_companion.py").read_text(encoding="utf-8")
        (ok if sysrow and sysrow[0]["system_prompt"] in src else fail)("persona system prompt is verbatim from ai_companion.py")
    gi = (HERE / ".gitignore").read_text(encoding="utf-8") if (HERE / ".gitignore").exists() else ""
    (ok if "datasets/" in gi else fail)(".gitignore keeps datasets/ out of git")


def exec_notebook(model_id: str, hooks: bool = True) -> None:
    print(f"[4] executing notebook on CPU with {model_id}")
    try:
        import torch  # noqa: F401
    except ImportError:
        fail("torch not importable in this interpreter")
        return
    work = HERE / "outputs" / "_cpu_test"
    import shutil
    shutil.rmtree(work, ignore_errors=True)        # never let an earlier run satisfy the checks
    os.environ.update({"SLEEP_PROFILE": "cpu_test", "SLEEP_MODEL_ID": model_id, "SLEEP_4BIT": "0",
                       "SLEEP_EXPORT_GGUF": "0", "SLEEP_DATASETS": str(DATASETS), "SLEEP_WORK_DIR": str(work),
                       "SLEEP_MOUNT_DRIVE": "0", "SLEEP_SKIP_INSTALL": "1", "MPLBACKEND": "Agg"})
    nb = load_notebook()
    g: dict = {"__name__": "__main__"}
    t0 = time.time()
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        title = next((l for l in src.splitlines() if l.startswith("# @title")), f"cell {i}")
        try:
            exec(compile(strip_magics(src), f"<cell {i}>", "exec"), g)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            fail(f"{title}: {type(e).__name__}: {e}")
            return
        ok(f"{title[9:]} ({time.time() - t0:.0f}s)")
        if hooks and "def parse_qa" in src:
            _install_test_hooks(g)
    res = g.get("RESULTS", {})
    expect = {"0_base_no_rag", "1_after_sleep_1", "2_after_sleep_2_stacked", "3_after_sleep_2_ties"}
    (ok if expect <= res.keys() else fail)(f"benchmark rows: {sorted(res)}")
    for c in (1, 2):
        log = work / "outputs" / f"sleep_cycle_{c}_log.json"
        if log.exists():
            d = json.loads(log.read_text())
            ok(f"cycle {c}: KS steps {len(d['ks'])}, dream log {d['dream']}")
        else:
            fail(f"sleep_cycle_{c}_log.json missing")
    for name in ("sleep_expert_1", "sleep_expert_2", "hikari_sleep"):
        cfgf = work / "outputs" / "adapters" / name / "adapter_config.json"
        (ok if cfgf.exists() else fail)(f"saved adapter {name}")


def _install_test_hooks(g: dict) -> None:
    """Random tiny models cannot write 'Q: / A:' dreams; feed wake-memory pairs so the dream path runs."""
    real_parse = g["parse_qa"]

    def parse_qa(text):
        pairs = real_parse(text)
        if pairs:
            return pairs
        eps = [e for e in g["WAKE"][1] + g["WAKE"][2] if e.get("assistant")]
        e = g["rng"].choice(eps)
        return [("What do you remember about this?", e["assistant"][:120])]

    g["parse_qa"] = parse_qa
    g["verify"] = lambda ctx, q, a: (True, 1.0, 1.0)
    print("        (test hook: synthetic dream Q&A + permissive verifier for random-weight model)")


def main() -> int:
    ap = argparse.ArgumentParser(description="sleep_lab verification gate")
    ap.add_argument("--run-import", action="store_true", help="re-run import_waifu_logs.py on the real logs first")
    ap.add_argument("--exec-notebook", action="store_true", help="execute the notebook on CPU with a tiny model")
    ap.add_argument("--tiny-model", default="trl-internal-testing/tiny-Qwen3ForCausalLM")
    ap.add_argument("--no-test-hooks", action="store_true", help="real dream parsing/verifier (use with a real model)")
    args = ap.parse_args()
    check_py()
    check_notebook()
    check_datasets(args.run_import)
    if args.exec_notebook:
        exec_notebook(args.tiny_model, hooks=not args.no_test_hooks)
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else f'{len(FAILS)} CHECK(S) FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
