#!/bin/bash
#SBATCH -J pfgmpp_arr
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1                      # any GPU; H100 queue depth on PACE-ICE makes specific-type unrealistic
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=96GB
#SBATCH -t 16:00:00                       # PACE GPU partitions cap at 16h; resubmit to continue
#SBATCH --array=0-6                       # 7 D values: edm, 2, 8, 32, 128, 512, 2048
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smishra327@gatech.edu

# ---------------------------------------------------------------------------
# Array variant: one submission, 7 array tasks, one D each. To re-run only
# specific failed indices:
#   sbatch --array=3,5 submit_sweep_array.sbatch
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

# Index -> D mapping. Keep order in lockstep with run_sweep.py:D_VALUES.
D_VALUES=(edm 2 8 32 128 512 2048)
D="${D_VALUES[${SLURM_ARRAY_TASK_ID}]}"
echo "[array] task ${SLURM_ARRAY_TASK_ID} -> D=${D}"

module load pytorch
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Update plots on exit so partial sweep state is visible after each task finishes.
trap 'python plot_sweep.py || true' EXIT

# Idempotent — only the first task to reach this line actually does work; the rest
# see the artifacts already in place and short-circuit.
python prepare_dataset.py

python run_sweep.py --profile=cluster --d="${D}"
