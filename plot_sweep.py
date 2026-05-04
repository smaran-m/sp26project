"""Plot the current contents of sweep_results.csv.

Two figures:
  * sweep_fid_curves.png : FID vs training step, one curve per D (convergence speed)
  * sweep_fid_vs_d.png   : FID vs D at the latest available step (the main result curve)

Safe to run at any time during a sweep — plots whatever rows exist so far. If a D value
has no rows yet, it's silently skipped.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "sweep_results.csv"

# Visual order: ascending D, with EDM (inf) last.
D_ORDER = ["2", "8", "32", "128", "512", "2048", "inf"]


def d_sort_key(d: str) -> float:
    return math.inf if d == "inf" else float(d)


def load(csv_path: Path) -> dict[str, list[dict]]:
    if not csv_path.exists():
        print(f"[plot] no CSV at {csv_path} yet — nothing to plot")
        return {}
    by_d: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["step"] = int(row["step"])
            row["kimg"] = int(row["kimg"])
            row["fid"] = float(row["fid"])
            by_d[row["d_value"]].append(row)
    for d in by_d:
        by_d[d].sort(key=lambda r: r["step"])
    return dict(by_d)


def plot_curves(by_d: dict[str, list[dict]], dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for d in sorted(by_d, key=d_sort_key):
        rows = by_d[d]
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        fids = [r["fid"] for r in rows]
        label = f"D = {d}" if d != "inf" else "D = ∞ (EDM)"
        ax.plot(steps, fids, marker="o", label=label, linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("FID (10k samples)")
    ax.set_title("PFGM++ D-sweep: FID vs training step")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f"[plot] wrote {dest}")


def plot_vs_d(by_d: dict[str, list[dict]], dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_finite, fids_finite = [], []
    fid_inf = None
    for d, rows in by_d.items():
        if not rows:
            continue
        last = rows[-1]
        if d == "inf":
            fid_inf = last["fid"]
        else:
            ds_finite.append(float(d))
            fids_finite.append(last["fid"])
    order = sorted(range(len(ds_finite)), key=lambda i: ds_finite[i])
    ds_finite = [ds_finite[i] for i in order]
    fids_finite = [fids_finite[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    if ds_finite:
        ax.plot(ds_finite, fids_finite, marker="o", linewidth=1.5, label="finite D")
    if fid_inf is not None:
        ax.axhline(fid_inf, linestyle="--", color="gray",
                   label=f"D = ∞ (EDM): {fid_inf:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Augmented dimension D")
    ax.set_ylabel("FID at final available step (10k samples)")
    ax.set_title("PFGM++ D-sweep: FID vs D")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f"[plot] wrote {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--curves-out", type=Path,
                    default=REPO_ROOT / "sweep_fid_curves.png")
    ap.add_argument("--vsd-out", type=Path,
                    default=REPO_ROOT / "sweep_fid_vs_d.png")
    args = ap.parse_args()

    by_d = load(args.csv)
    if not by_d:
        return 0
    n = sum(len(v) for v in by_d.values())
    print(f"[plot] loaded {n} rows across D values: {sorted(by_d, key=d_sort_key)}")
    plot_curves(by_d, args.curves_out)
    plot_vs_d(by_d, args.vsd_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
