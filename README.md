# PFGM++ D-Sweep

Empirical sweep over the augmented dimension `D` in the PFGM++ generative-modeling
framework ([Xu et al., ICML 2023](https://arxiv.org/abs/2302.04265)), holding architecture
and training budget fixed. Maps the robustness–rigidity tradeoff curve on CIFAR-10.

- **D values:** `{2, 8, 32, 128, 512, 2048, ∞}` (∞ = EDM baseline)
- **Architecture:** `ncsnpp`
- **Dataset:** CIFAR-10 32×32, unconditional

Upstream code lives in [`pfgmpp/`](pfgmpp/) (cloned from
[Newbeeer/pfgmpp](https://github.com/Newbeeer/pfgmpp)) and is wrapped by this repo into
a sweep / eval / plot pipeline.

## Setup

```bash
conda env create -f environment.yml -n pfgmpp
conda activate pfgmpp
python prepare_dataset.py     # downloads CIFAR-10 + builds FID reference (idempotent)
```

The upstream repo pins PyTorch 1.12 / Python < 3.10; the env spec matches.

## Local smoke test

Three D values, ~5k steps, batch 16 — fits on a single 12 GB GPU:

```bash
python run_sweep.py --profile=local --d=2,32,inf
python plot_sweep.py
```

Drop `local_overrides.batch` in [`configs/sweep_base.py`](configs/sweep_base.py) or pass
`--batch-gpu=8` for gradient accumulation on tighter VRAM.

## Full sweep (SLURM)

One job per D (recommended — better fault isolation):

```bash
for D in edm 2 8 32 128 512 2048; do
  sbatch submit_sweep.sbatch $D
done
```

Array variant:

```bash
sbatch submit_sweep_array.sbatch     # --array=0-6
```

Each job runs `run_sweep.py --profile=cluster --d=<D>` for 200k steps. Re-submit on
timeout — runs resume from the latest `training-state-*.pt`. `plot_sweep.py` auto-runs
on completion so the latest curves are always available.

Add `--wandb` to log loss + FID; re-queued jobs continue the same run via
`resume="allow"`.

## Outputs

- `sweep_results.csv` — appended per FID eval, written atomically (safe under crash/timeout)
- `sweep_fid_curves.png` — FID vs. training step, per D
- `sweep_fid_vs_d.png` — best FID vs. D

## Layout

```
configs/             per-D sweep configs (thin dicts → train.py CLI flags)
pfgmpp/              upstream repo — do not modify
prepare_dataset.py   CIFAR-10 + FID reference
run_sweep.py         orchestrator (train + eval per D)
eval_checkpoints.py  post-hoc FID + sample grid for one run dir
plot_sweep.py        plots from sweep_results.csv
submit_sweep*.sbatch SLURM scripts (per-D and array)
```

## Citation

```bibtex
@inproceedings{xu2023pfgmpp,
  title     = {PFGM++: Unlocking the Potential of Physics-Inspired Generative Models},
  author    = {Xu, Yilun and Liu, Ziming and Tian, Yonglong and Tong, Shangyuan and Tegmark, Max and Jaakkola, Tommi},
  booktitle = {ICML},
  year      = {2023}
}
```