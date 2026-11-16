#!/bin/bash
#SBATCH --job-name=humor-pref
#SBATCH --account=mscbdt2024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=humor-pref-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project directory}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

"$PWD/.venv/bin/python" \
  scripts/train_humor_preference_v1.py "$@"
