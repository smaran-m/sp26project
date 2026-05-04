"""Local sweep orchestrator.

For each requested D value:
  1. resolve config (configs/sweep_<d>.py + base + profile overrides)
  2. translate `_total_kimg` / `_snapshot_kimg` into train.py's --duration / --tick
     / --snap / --dump
  3. resume if a training-state-*.pt already lives in the run dir; otherwise start fresh
  4. invoke `pfgmpp/train.py` via torchrun (foreground)
  5. run `eval_checkpoints.py` to FID every snapshot and append to sweep_results.csv
  6. (optional) hand off to plot_sweep.py for an immediate refresh

Crash + SIGINT safe: train.py owns its own state, eval_checkpoints owns the CSV.
A re-run picks up from the latest snapshot and only FIDs unevaluated snapshots.

Usage:
    python run_sweep.py --profile=local --d=2,32,inf
    python run_sweep.py --profile=cluster --d=128
    python run_sweep.py --profile=cluster --d=all
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PFGMPP_DIR = REPO_ROOT / "pfgmpp"
CONFIGS_DIR = REPO_ROOT / "configs"
TRAINING_RUNS = PFGMPP_DIR / "training-runs"

D_VALUES = ["edm", "2", "8", "32", "128", "512", "2048"]
D_TO_CONFIG = {d: f"sweep_d{d}" if d != "edm" else "sweep_edm" for d in D_VALUES}
D_LABEL = {d: ("inf" if d == "edm" else d) for d in D_VALUES}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [sweep] {msg}", flush=True)


def load_config_module(modname: str):
    path = CONFIGS_DIR / f"{modname}.py"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def find_resume(run_dir: Path) -> Path | None:
    """Newest training-state-*.pt in run_dir (or None)."""
    if not run_dir.is_dir():
        return None
    states = sorted(run_dir.glob("training-state-*.pt"))
    return states[-1] if states else None


def kimg_to_train_args(total_kimg: int, snap_kimg: int) -> dict:
    """Convert internal kimg knobs to train.py click flags.

    train.py accepts:
      --duration MIMG       # total = duration * 1000 kimg
      --tick KIMG           # progress print every N kimg
      --snap TICKS          # save snapshot every K ticks
      --dump TICKS          # save state every K ticks  (this is what we want for FID)

    We pick tick = snap_kimg so that --snap=1 / --dump=1 lines up with one snapshot
    per `_snapshot_kimg`.
    """
    return dict(
        duration=total_kimg / 1000.0,
        tick=snap_kimg,
        snap=1,
        dump=1,
    )


def build_train_cmd(cfg: dict, run_dir: Path, resume: Path | None) -> list[str]:
    """Compose torchrun + train.py invocation. Boolean flags are passed as 0/1 because
    train.py uses click's BOOL type which accepts integers."""
    train_kwargs = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "name"}
    train_kwargs.update(kimg_to_train_args(cfg["_total_kimg"], cfg["_snapshot_kimg"]))

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "train.py",
        f"--outdir={TRAINING_RUNS.relative_to(PFGMPP_DIR).as_posix()}",
        f"--name={cfg['name']}",
    ]
    for k, v in train_kwargs.items():
        # train.py's click options use the literal Python names (mostly underscored,
        # one `--batch-gpu` exception we don't pass here).
        flag = f"--{k}"
        if isinstance(v, bool):
            cmd.append(f"{flag}={int(v)}")
        else:
            cmd.append(f"{flag}={v}")
    if resume is not None:
        cmd.append(f"--resume={resume.relative_to(PFGMPP_DIR).as_posix()}")
    return cmd


def run_one_d(d: str, profile: str, args: argparse.Namespace) -> bool:
    """Train + eval for a single D value. Returns True on clean completion."""
    base = load_config_module("sweep_base")
    d_cfg_mod = load_config_module(D_TO_CONFIG[d])
    cfg = base.materialize(profile, d_cfg_mod.CONFIG)

    run_dir = TRAINING_RUNS / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    resume = find_resume(run_dir)
    if resume:
        log(f"D={d}: resuming from {resume.name}")
    else:
        log(f"D={d}: fresh start in {run_dir.name}")

    cmd = build_train_cmd(cfg, run_dir, resume)
    log(f"D={d}: train cmd: {' '.join(shlex.quote(c) for c in cmd)}")

    if args.dry_run:
        log("--dry-run: skipping subprocess.run for train")
    else:
        try:
            subprocess.run(cmd, cwd=PFGMPP_DIR, check=True)
        except KeyboardInterrupt:
            log("D={d}: interrupted; train.py should have flushed its latest state on SIGINT")
            return False
        except subprocess.CalledProcessError as e:
            log(f"D={d}: train.py exited non-zero ({e.returncode}); proceeding to eval anyway")

    # Eval whatever snapshots exist (resume case may have new ones, fresh-start case
    # will have whatever the train run produced).
    eval_cmd = [
        sys.executable, "eval_checkpoints.py",
        f"--run-dir={run_dir}",
        f"--d-value={D_LABEL[d]}",
        f"--pfgmpp={int(cfg['pfgmpp'])}",
        f"--aug-dim={cfg['aug_dim']}",
        f"--batch-size={cfg['batch']}",
        f"--num-samples={args.fid_samples}",
    ]
    log(f"D={d}: eval cmd: {' '.join(shlex.quote(c) for c in eval_cmd)}")
    if args.dry_run:
        log("--dry-run: skipping subprocess.run for eval")
    else:
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=False)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=("local", "cluster"), default="local")
    ap.add_argument("--d", default="all",
                    help="comma-separated list from {edm,2,8,32,128,512,2048} or 'all'")
    ap.add_argument("--fid-samples", type=int, default=10000,
                    help="number of generated samples for each FID evaluation")
    ap.add_argument("--plot-after", action="store_true",
                    help="run plot_sweep.py after each D completes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing torchrun")
    args = ap.parse_args()

    targets = D_VALUES if args.d == "all" else [d.strip() for d in args.d.split(",")]
    unknown = [d for d in targets if d not in D_VALUES]
    if unknown:
        raise SystemExit(f"unknown D value(s): {unknown}; expected subset of {D_VALUES}")

    if not PFGMPP_DIR.is_dir():
        raise SystemExit(f"missing upstream repo at {PFGMPP_DIR}")
    if not (PFGMPP_DIR / "datasets" / "cifar10-32x32.zip").exists():
        log("warning: pfgmpp/datasets/cifar10-32x32.zip not found — run prepare_dataset.py first")
    if not (PFGMPP_DIR / "fid-refs" / "cifar10-32x32.npz").exists():
        log("warning: pfgmpp/fid-refs/cifar10-32x32.npz not found — FID will fail until built")

    log(f"profile={args.profile} D values={targets}")
    for d in targets:
        try:
            run_one_d(d, args.profile, args)
        except KeyboardInterrupt:
            log("interrupted by user; partial results preserved in sweep_results.csv")
            return 130
        if args.plot_after:
            subprocess.run([sys.executable, "plot_sweep.py"], cwd=REPO_ROOT, check=False)

    log("sweep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
