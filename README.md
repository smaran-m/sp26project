# PFGM++ D-Sweep

Empirical sweep over the augmented dimension D in the PFGM++ generative-modeling framework
(Xu et al., ICML 2023), holding architecture and training budget fixed. Maps the
robustness–rigidity tradeoff curve on CIFAR-10.

D values: `{2, 8, 32, 128, 512, 2048, ∞}` (∞ = EDM baseline, `pfgmpp=False`).
Architecture: `ncsnpp`. Dataset: CIFAR-10 32x32, unconditional.

The upstream code lives in [`pfgmpp/`](pfgmpp/) (cloned from `Newbeeer/pfgmpp`).
Everything in this directory wraps that code into a sweep, eval, and plot pipeline.

---

## Layout

```
.
├── pfgmpp/                  # upstream repo — DO NOT modify
├── configs/
│   ├── sweep_base.py        # shared defaults
│   ├── sweep_edm.py         # D = inf  (pfgmpp=False)
│   ├── sweep_d2.py          # D = 2
│   ├── sweep_d8.py          # D = 8
│   ├── sweep_d32.py         # D = 32
│   ├── sweep_d128.py        # D = 128
│   ├── sweep_d512.py        # D = 512
│   └── sweep_d2048.py       # D = 2048
├── prepare_dataset.py       # download CIFAR-10, build zip, build FID ref
├── run_sweep.py             # local orchestrator (train + eval per D)
├── eval_checkpoints.py      # post-hoc FID + sample grid for one run dir
├── plot_sweep.py            # FID vs step / FID vs D from sweep_results.csv
├── submit_sweep.sbatch      # SLURM, single-job-per-D (submit 7 times)
├── submit_sweep_array.sbatch # SLURM, --array=0-6 over the 7 D values
├── environment.yml          # conda env spec (Python 3.9 + PyTorch 1.12)
├── requirements.txt         # pip fallback
└── sweep_results.csv        # appended per FID eval (created on first run)
```

---

## Setup

### 1. Conda env

The upstream repo pins PyTorch 1.12 / Python <3.10. Use the provided env:

```bash
conda env create -f environment.yml -n pfgmpp
conda activate pfgmpp
```

### 2. Prepare CIFAR-10

Downloads CIFAR-10, packs it into `pfgmpp/datasets/cifar10-32x32.zip`, and builds
the FID reference statistics in `pfgmpp/fid-refs/cifar10-32x32.npz`. Idempotent — re-runs
skip already-built artifacts.

```bash
python prepare_dataset.py
```

---

## Local validation (3080Ti, ~12 GB VRAM)

Smoke-tests the orchestrator without burning hours on training. By default uses the
`local` profile in `configs/sweep_base.py`: 32x32 res, batch 16, `--duration=0.08`
(~5k steps), three D values `{2, 32, inf}`.

```bash
python run_sweep.py --profile=local --d=2,32,inf
python plot_sweep.py
```

If anything OOMs at batch 16 on the 3080Ti, drop to batch 8 in `configs/sweep_base.py`
(`local_overrides.batch`), or pass `--batch-gpu=8` to enable gradient accumulation.

---

## Cluster (H100, full 200k-step training)

Per-D job (one sbatch per D — recommended for fault isolation):

```bash
for D in edm 2 8 32 128 512 2048; do
  sbatch submit_sweep.sbatch $D
done
```

Array variant (single submission, 7 array tasks):

```bash
sbatch submit_sweep_array.sbatch
```

Each job calls `run_sweep.py --profile=cluster --d=<D>`. On timeout or failure, just
re-submit — `run_sweep.py` resumes from the latest `training-state-*.pt`.

After the job completes (or times out cleanly via the trap), it auto-runs
`plot_sweep.py` so the latest plots are always in `./` even mid-sweep.

---

## Monitoring during a long run

Check `sweep_results.csv` directly, or just re-run plotting at any point:

```bash
python plot_sweep.py            # produces sweep_fid_curves.png + sweep_fid_vs_d.png
```

`sweep_results.csv` is written atomically (temp-file + rename) on every FID eval, so
a crash or timeout never corrupts partial results.

If `wandb` is available, `run_sweep.py --wandb` will additionally log loss + FID with
`resume="allow"`, so re-queued jobs continue the same wandb run.

---

## Notes / gotchas

- The upstream repo does **not** ship `hyper-parameters.py` or Python config files;
  it uses `train.py` argparse flags directly. Our `configs/sweep_*.py` are thin
  Python dicts mapping to CLI flags.
- `train.py --duration` is in **millions of training images**, not steps. We
  parameterize in kimg internally and convert.
- `generate.py` does not take a `--network` flag — it globs `training-state-*.pt`
  from `--outdir`. `eval_checkpoints.py` invokes it accordingly with
  `--ckpt`/`--end_ckpt` to limit which snapshot is processed.
- `r = σ * sqrt(D)` alignment is handled inside the loss/sampler — no extra
  calibration step is needed for CIFAR-10.
- `--rbatch` (reference batch for STF) is unused unless `--stf=True`. Ignore it.
