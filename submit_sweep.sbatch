#!/bin/bash
#SBATCH -J pfgmpp_d                       # Job name (override per-D below via --job-name)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1                 # avoid AMD MI210 (ROCm) and pre-Volta NVIDIAs (no PyTorch kernels). A100=sm_80, broadly available.
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=96GB                # Per hw2.sbatch — proven sufficient on this cluster
#SBATCH -t 16:00:00                       # PACE GPU partitions cap at 16h; resubmit to continue from checkpoint
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smishra327@gatech.edu

# ---------------------------------------------------------------------------
# Single-D submission. Pass D as the first positional arg:
#   sbatch submit_sweep.sbatch edm
#   sbatch submit_sweep.sbatch 128
#   sbatch submit_sweep.sbatch 2048
#
# Submit one of these per D value for fault isolation:
#   for D in edm 2 8 32 128 512 2048; do
#     sbatch -J pfgmpp_d${D} submit_sweep.sbatch $D
#   done
#
# Each job:
#   * trains one D to --duration=102.4 Mimg (200k steps @ batch 512)
#   * resumes from latest training-state-*.pt if one exists
#   * evaluates FID + sample grids on every checkpoint that lacks one
#   * regenerates plots so monitoring is always current
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

D="${1:-}"
if [[ -z "$D" ]]; then
  echo "usage: sbatch submit_sweep.sbatch <D>   where D in {edm,2,8,32,128,512,2048}" >&2
  exit 2
fi

# Module load — same as hw2.sbatch. Adjust if cluster modules change.
module load pytorch

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Always update plots on exit (even on timeout / failure) so the job log shows
# the latest state of the sweep.
trap 'python plot_sweep.py || true' EXIT

# Prepare dataset on first run; idempotent so subsequent runs are no-ops.
python prepare_dataset.py

# Train + evaluate.
python run_sweep.py --profile=cluster --d="$D"
