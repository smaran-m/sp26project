"""Post-hoc FID + sample-grid evaluation for one sweep run directory.

For each `training-state-<kimg>.pt` not yet recorded in sweep_results.csv:
  1. invoke pfgmpp/generate.py to produce <run_dir>/ckpt_<kimg>/*.png
     (--ckpt and --end_ckpt narrow the glob to a single snapshot)
  2. invoke pfgmpp/fid.py calc to compute FID against the CIFAR-10 reference
  3. compose a 64-image grid PNG -> <run_dir>/samples/step_<kimg>.png
  4. atomically append (d_value, step, fid, wall_time) to sweep_results.csv

Skips snapshots already present in the CSV. Safe to re-run / interrupt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
PFGMPP_DIR = REPO_ROOT / "pfgmpp"
CSV_PATH = REPO_ROOT / "sweep_results.csv"
CSV_HEADER = ["d_value", "step", "kimg", "fid", "wall_time"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [eval] {msg}", flush=True)


def discover_snapshots(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(kimg, path), ...] sorted ascending."""
    pat = re.compile(r"^training-state-(\d{6})\.pt$")
    out = []
    for p in run_dir.glob("training-state-*.pt"):
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def already_evaluated(d_value: str, kimg: int) -> bool:
    if not CSV_PATH.exists():
        return False
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["d_value"] == d_value and int(row["kimg"]) == kimg:
                return True
    return False


def append_csv_atomic(row: dict) -> None:
    """Atomic append: read full file, rewrite to .tmp, rename. Survives crash mid-write."""
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.append(row)
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_HEADER})
    os.replace(tmp, CSV_PATH)


def run_generate(run_dir: Path, kimg: int, pfgmpp: bool, aug_dim: int,
                 num_samples: int, batch_size: int) -> Path:
    """Invoke pfgmpp/generate.py for a single snapshot. Returns the ckpt subdir."""
    out_subdir = run_dir / f"ckpt_{kimg:06d}"
    if out_subdir.exists() and any(out_subdir.glob("*.png")):
        log(f"  generate: {out_subdir.name} already populated, skipping")
        return out_subdir
    seeds = f"0-{num_samples - 1}"
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "generate.py",
        f"--outdir={run_dir.relative_to(PFGMPP_DIR).as_posix()}",
        f"--seeds={seeds}",
        f"--batch={batch_size}",
        f"--ckpt={kimg}",
        f"--end_ckpt={kimg}",
        f"--pfgmpp={int(pfgmpp)}",
        f"--aug_dim={aug_dim}",
        "--subdirs",
    ]
    log(f"  generate: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PFGMPP_DIR, check=True)
    return out_subdir


def run_fid(images_dir: Path, num_samples: int) -> float:
    ref = (PFGMPP_DIR / "fid-refs" / "cifar10-32x32.npz").relative_to(PFGMPP_DIR).as_posix()
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "fid.py", "calc",
        f"--images={images_dir.relative_to(PFGMPP_DIR).as_posix()}",
        f"--ref={ref}",
        f"--num={num_samples}",
    ]
    log(f"  fid: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PFGMPP_DIR, check=True, capture_output=True, text=True)
    # fid.py prints e.g. "FID = 12.345" on the last meaningful line. Be defensive.
    m = re.search(r"FID\s*[=:]\s*([0-9]+\.?[0-9]*)", res.stdout + res.stderr)
    if not m:
        log(f"  ! could not parse FID from output. stdout tail:")
        log(res.stdout[-500:])
        raise RuntimeError("FID parse failed")
    return float(m.group(1))


def make_sample_grid(images_dir: Path, dest: Path, n: int = 64) -> None:
    """Compose first n PNGs from images_dir into a sqrt(n) x sqrt(n) grid."""
    try:
        from PIL import Image
    except ImportError:
        log("  ! PIL unavailable, skipping sample grid")
        return
    pngs = sorted(images_dir.rglob("*.png"))[:n]
    if not pngs:
        log(f"  ! no PNGs in {images_dir}, skipping grid")
        return
    side = int(len(pngs) ** 0.5)
    if side * side > len(pngs):
        side -= 1
    pngs = pngs[: side * side]
    sample = Image.open(pngs[0])
    w, h = sample.size
    grid = Image.new(sample.mode, (w * side, h * side))
    for i, p in enumerate(pngs):
        r, c = divmod(i, side)
        grid.paste(Image.open(p), (c * w, r * h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest)
    log(f"  grid: {dest} ({side}x{side})")


def evaluate_run(run_dir: Path, d_value: str, pfgmpp: bool, aug_dim: int,
                 batch_size: int, num_samples: int, batch_per_step: int) -> int:
    """Evaluate every unevaluated snapshot in run_dir. Returns count of new evals."""
    if not run_dir.is_dir():
        log(f"run dir does not exist yet: {run_dir}")
        return 0
    snapshots = discover_snapshots(run_dir)
    if not snapshots:
        log(f"no snapshots in {run_dir}")
        return 0
    log(f"found {len(snapshots)} snapshot(s) in {run_dir.name}")
    n_new = 0
    for kimg, _path in snapshots:
        if already_evaluated(d_value, kimg):
            continue
        t0 = time.time()
        log(f"evaluating kimg={kimg}")
        try:
            images_dir = run_generate(run_dir, kimg, pfgmpp, aug_dim,
                                      num_samples, batch_per_step)
            fid = run_fid(images_dir, num_samples)
            grid_dest = run_dir / "samples" / f"step_{kimg:06d}.png"
            make_sample_grid(images_dir, grid_dest)
            wall = time.time() - t0
            # Steps from kimg requires batch_size; recorded for plotting convenience.
            step = (kimg * 1000) // batch_size
            append_csv_atomic({
                "d_value": d_value,
                "step": step,
                "kimg": kimg,
                "fid": f"{fid:.4f}",
                "wall_time": f"{wall:.1f}",
            })
            log(f"  -> FID={fid:.3f} step={step} kimg={kimg} ({wall:.0f}s)")
            n_new += 1
        except subprocess.CalledProcessError as e:
            log(f"  ! generate/fid failed for kimg={kimg}: {e}")
        except Exception as e:
            log(f"  ! unexpected error for kimg={kimg}: {e}")
    return n_new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="path to training-runs/<name> for one D value")
    ap.add_argument("--d-value", required=True,
                    help="label for the CSV (e.g. 'inf', '128', '2048')")
    ap.add_argument("--pfgmpp", type=int, choices=(0, 1), required=True)
    ap.add_argument("--aug-dim", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True,
                    help="training batch size (used to convert kimg<->step)")
    ap.add_argument("--num-samples", type=int, default=10000,
                    help="number of images to generate per FID eval (spec: 10k)")
    ap.add_argument("--gen-batch", type=int, default=64,
                    help="generate.py per-call batch size")
    args = ap.parse_args()

    n = evaluate_run(args.run_dir, args.d_value, bool(args.pfgmpp),
                     args.aug_dim, args.batch_size, args.num_samples, args.gen_batch)
    log(f"evaluated {n} new snapshot(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
