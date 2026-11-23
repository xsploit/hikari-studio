#!/usr/bin/env python3
"""Bundle the SLEEP lab (datasets + notebook) and hand it to Google Colab.

Modes
  bundle  build dist/sleep_lab_bundle.zip + dist/hikari_sleep_arxiv2606.ipynb only (default)
  colab   bundle, then drive google-colab-cli end to end: new GPU session -> upload bundle ->
          exec the notebook (writes dist/hikari_sleep_arxiv2606_output.ipynb) -> download
          sleep_outputs.zip -> stop the session (always, unless --keep)
  cli     bundle, then run any other Colab command-line tool with a configurable argument template
  drive   bundle, then copy it into your Google Drive sync folder (MyDrive/hikari_sleep_lab/)

The notebook copy in dist/ has PROFILE baked in, so it runs the right size without env vars.
Nothing is uploaded unless you choose `cli` or `drive` without --dry-run. The bundle holds your
private Hikari memories: it goes only to your own Colab runtime / Drive.

Examples
  python sleep_lab/run_colab_sleep.py                                   # just build the bundle
  python sleep_lab/run_colab_sleep.py --mode colab --gpu T4 --profile smoke   # cheap end-to-end check
  python sleep_lab/run_colab_sleep.py --mode colab --gpu A100                 # paper-like run
  python sleep_lab/run_colab_sleep.py --mode cli --dry-run              # show the CLI command
  python sleep_lab/run_colab_sleep.py --mode cli --cli colab-cli \
      --cli-args "run {notebook} --gpu {gpu} --upload {bundle}" --gpu A100 --profile a100
  python sleep_lab/run_colab_sleep.py --mode drive --drive-dir "G:/My Drive"

Environment defaults: COLAB_CLI, COLAB_CLI_ARGS, SLEEP_DRIVE_SYNC.
Template placeholders: {notebook} {bundle} {gpu} {profile} {dist} {datasets}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
NOTEBOOK = HERE / "hikari_sleep_arxiv2606.ipynb"
DIST = HERE / "dist"
DATA_FILES = ["wake_cycle_1.jsonl", "wake_cycle_2.jsonl", "eval_recall_quiz.jsonl", "persona_anchor.jsonl",
              "manifest.json"]
DEFAULT_CLI = os.environ.get("COLAB_CLI", "colab")
DEFAULT_CLI_ARGS = os.environ.get("COLAB_CLI_ARGS", "run {notebook} --gpu {gpu} --upload {bundle}")
PROFILE_FOR_GPU = {"T4": "t4", "L4": "t4", "A100": "a100", "H100": "a100"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def drive_candidates() -> list[Path]:
    home = Path.home()
    cands = [os.environ.get("SLEEP_DRIVE_SYNC")]
    cands += [f"{d}:/My Drive" for d in "GHIJKLMNOPQRSTUVWXYZ"]
    cands += [home / "Google Drive" / "My Drive", home / "My Drive", home / "GoogleDrive" / "My Drive"]
    return [Path(c) for c in cands if c]


def patched_notebook(profile: str) -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    needle = 'PROFILE  = os.environ.get("SLEEP_PROFILE", "t4")'
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if needle in src:
            src = src.replace(needle, f'PROFILE  = os.environ.get("SLEEP_PROFILE", "{profile}")')
            cell["source"] = src.splitlines(keepends=True)
            hits += 1
    if hits != 1:
        raise RuntimeError("could not find the PROFILE line in the notebook config cell")
    return nb


def build_bundle(profile: str) -> tuple[Path, Path]:
    missing = [f for f in DATA_FILES if not (DATASETS / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing datasets {missing} - run import_waifu_logs.py first")
    DIST.mkdir(exist_ok=True)
    nb_out = DIST / NOTEBOOK.name
    nb_out.write_text(json.dumps(patched_notebook(profile), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    bundle = DIST / "sleep_lab_bundle.zip"
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": profile,
            "paper": "arXiv:2606.03979", "files": {}}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in DATA_FILES:
            zf.write(DATASETS / f, f"datasets/{f}")
            meta["files"][f"datasets/{f}"] = sha256(DATASETS / f)
        zf.write(nb_out, NOTEBOOK.name)
        meta["files"][NOTEBOOK.name] = sha256(nb_out)
        zf.writestr("bundle.json", json.dumps(meta, indent=1))
    return bundle, nb_out


def run_cli(args, bundle: Path, nb: Path) -> int:
    values = {"notebook": str(nb), "bundle": str(bundle), "gpu": args.gpu, "profile": args.profile,
              "dist": str(DIST), "datasets": str(DATASETS)}
    try:
        tokens = [t.format(**values) for t in shlex.split(args.cli_args)]
    except KeyError as e:
        print(f"[colab] unknown placeholder {e} in --cli-args", file=sys.stderr)
        return 2
    exe = shutil.which(args.cli) or args.cli
    cmd = [exe, *tokens]
    print("[colab] command:", subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd))
    if args.dry_run:
        print("[colab] dry run - not executed")
        return 0
    if not shutil.which(args.cli) and not Path(args.cli).exists():
        print(f"[colab] '{args.cli}' not found on PATH. Install your Colab CLI or pass --cli <path>.", file=sys.stderr)
        return 127
    env = dict(os.environ, SLEEP_PROFILE=args.profile)
    return subprocess.call(cmd, env=env)


def colab_base(args) -> list[str]:
    """google-colab-cli: `colab` on PATH, else the same package through uvx (how it is set up on this PC)."""
    if args.colab_bin:
        return shlex.split(args.colab_bin, posix=os.name != "nt")
    if shutil.which("colab"):
        return [shutil.which("colab")]
    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / ("uvx.exe" if os.name == "nt" else "uvx"))
    return [uvx, "--from", "google-colab-cli", "colab"]


def notebook_errors(executed: Path) -> list[str]:
    """`colab exec` keeps going after a failing cell, so read the executed notebook for error outputs."""
    if not executed.exists():
        return [f"{executed.name} was not written"]
    errs = []
    for i, cell in enumerate(json.loads(executed.read_text(encoding="utf-8")).get("cells", [])):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                title = next((l.strip() for l in cell.get("source", []) if l.startswith("# @title")), f"cell {i}")
                errs.append(f"{title}: {out.get('ename')}: {str(out.get('evalue'))[:160]}")
    return errs


def run_colab(args, bundle: Path, nb: Path) -> int:
    base = colab_base(args)
    name = args.session
    remote_bundle = "/content/sleep_lab_bundle.zip"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    envs = [f"SLEEP_PROFILE={args.profile}", "SLEEP_MOUNT_DRIVE=0", "SLEEP_WORK_DIR=/content/sleep_lab"]
    if args.model:
        envs.append(f"SLEEP_MODEL_ID={args.model}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    local_out = DIST / f"colab_outputs_{args.profile}_{stamp}.zip"
    steps = [
        ("new", [*base, "new", "-s", name, "--gpu", args.gpu]),
        ("upload", [*base, "upload", "-s", name, str(bundle), remote_bundle]),
        ("exec", [*base, "exec", "-s", name, "-f", str(nb), "--timeout", str(args.idle_timeout),
                  *[x for e in envs for x in ("--env", e)]]),
        ("download", [*base, "download", "-s", name, "/content/sleep_lab/sleep_outputs.zip", str(local_out)]),
    ]
    stop = [*base, "stop", "-s", name]
    for label, cmd in steps + [("stop", stop)]:
        print(f"[colab] {label}: {subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd)}")
    if args.dry_run:
        print("[colab] dry run - nothing started")
        return 0
    rc = 0
    started = False
    # safety net 1: a detached watchdog stops the VM if this process dies or the hard limit passes
    if not args.no_watchdog:
        wd = [sys.executable, str(HERE / "colab_watchdog.py"), "--pid", str(os.getpid()), "--session", name,
              "--max-hours", str(args.max_hours + 0.25)]   # backup: the launcher's own limit fires first
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        subprocess.Popen(wd, env=env, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=os.name != "nt")
        print(f"[colab] watchdog on: hard stop after {args.max_hours} h (log outputs/colab_watchdog.log)")
    deadline = time.time() + args.max_hours * 3600
    executed = nb.with_name(nb.stem + "_output.ipynb")
    executed.unlink(missing_ok=True)          # never judge this run by an older run's executed notebook
    try:
        for label, cmd in steps:
            print(f"[colab] >>> {label}", flush=True)
            # safety net 2: every step has a timeout; exec also fails after --idle-timeout s with no output
            try:
                rc = subprocess.call(cmd, env=env, timeout=max(60, deadline - time.time()))
            except subprocess.TimeoutExpired:
                print(f"[colab] {label} hit the {args.max_hours} h limit - stopping", file=sys.stderr)
                rc = 124
            started = started or label == "new" and rc == 0
            if rc != 0:
                print(f"[colab] {label} failed (exit {rc})", file=sys.stderr)
                if label == "new":
                    print("[colab] no GPU quota for this type? try --gpu T4", file=sys.stderr)
                break
        errs = notebook_errors(executed) if rc == 0 or executed.exists() else []
        if errs:
            print(f"[colab] {len(errs)} notebook cell(s) FAILED:", file=sys.stderr)
            for e in errs:
                print("   - " + e, file=sys.stderr)
            rc = rc or 3
        if rc == 0:
            print(f"[colab] all cells ran cleanly. results: {local_out}")
            print(f"[colab] executed notebook: {executed}")
    finally:
        if started and rc != 0 and not local_out.exists():
            # the notebook archives after every sleep cycle: grab whatever finished before the VM goes away
            partial = local_out.with_name(local_out.stem + "_partial.zip")
            print(f"[colab] >>> download partial results -> {partial.name}", flush=True)
            try:
                subprocess.call([*base, "download", "-s", name, "/content/sleep_lab/sleep_outputs.zip", str(partial)],
                                env=env, timeout=600)
            except subprocess.TimeoutExpired:
                print("[colab] partial download timed out", file=sys.stderr)
        if started and not args.keep:
            print("[colab] >>> stop (releasing the VM so it stops using compute units)", flush=True)
            subprocess.call(stop, env=env)
        elif started:
            print(f"[colab] --keep: session '{name}' is still running. Stop it with: colab stop -s {name}")
    return rc


def run_drive(args, bundle: Path, nb: Path) -> int:
    root = Path(args.drive_dir) if args.drive_dir else next((c for c in drive_candidates() if c.is_dir()), None)
    if root is None:
        print("[drive] no Google Drive sync folder found - pass --drive-dir \"G:/My Drive\"", file=sys.stderr)
        return 2
    dst = root / "hikari_sleep_lab"
    print(f"[drive] {bundle.name} + {nb.name} -> {dst}")
    if args.dry_run:
        print("[drive] dry run - nothing copied")
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle, dst / bundle.name)
    shutil.copy2(nb, dst / nb.name)
    print("[drive] copied. In Colab: open MyDrive/hikari_sleep_lab/" + nb.name + " and Run all "
          "(the notebook mounts Drive and finds the bundle).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bundle + launch the Hikari SLEEP notebook on Colab",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["bundle", "colab", "cli", "drive"], default="bundle")
    ap.add_argument("--session", default="hikari-sleep", help="colab session name (--mode colab)")
    ap.add_argument("--colab-bin", default=None, help="override the google-colab-cli command (default: colab or uvx)")
    ap.add_argument("--model", default=None, help="override MODEL_ID inside the notebook")
    ap.add_argument("--keep", action="store_true", help="do not stop the Colab session at the end")
    ap.add_argument("--max-hours", type=float, default=None, help="hard limit for the whole run (default per profile)")
    ap.add_argument("--idle-timeout", type=int, default=2400, help="fail exec after this many seconds with no output")
    ap.add_argument("--no-watchdog", action="store_true", help="do not start the detached safety watchdog")
    ap.add_argument("--gpu", default="T4", help="T4 | L4 | A100 (passed to {gpu})")
    ap.add_argument("--profile", default=None, help="smoke | t4 | a100 (default from --gpu)")
    ap.add_argument("--cli", default=DEFAULT_CLI, help="Colab CLI binary (env COLAB_CLI)")
    ap.add_argument("--cli-args", default=DEFAULT_CLI_ARGS, help="argument template (env COLAB_CLI_ARGS)")
    ap.add_argument("--drive-dir", default=None, help="Google Drive sync root, e.g. 'G:/My Drive'")
    ap.add_argument("--skip-import", action="store_true", help="reuse existing datasets/ instead of re-importing")
    ap.add_argument("--dry-run", action="store_true", help="print what would run/copy, do nothing external")
    args = ap.parse_args(argv)
    args.gpu = args.gpu.upper()
    args.profile = args.profile or PROFILE_FOR_GPU.get(args.gpu, "t4")
    if args.max_hours is None:
        args.max_hours = {"smoke": 0.75, "cpu_test": 0.5, "t4": 3.25, "a100": 2.0}.get(args.profile, 2.5)

    if not args.skip_import:
        sys.path.insert(0, str(HERE))
        import import_waifu_logs  # noqa: E402
        rc = import_waifu_logs.main([])
        if rc:
            return rc
    bundle, nb = build_bundle(args.profile)
    print(f"[bundle] {bundle} ({bundle.stat().st_size / 1024:.0f} KB, profile={args.profile})")
    print(f"[bundle] {nb}")
    if args.mode == "bundle":
        print("[bundle] upload both files to Colab (or use --mode drive / --mode cli). "
              "Contains private memories - keep it out of git and public shares.")
        return 0
    if args.mode == "colab":
        return run_colab(args, bundle, nb)
    return run_cli(args, bundle, nb) if args.mode == "cli" else run_drive(args, bundle, nb)


if __name__ == "__main__":
    raise SystemExit(main())
