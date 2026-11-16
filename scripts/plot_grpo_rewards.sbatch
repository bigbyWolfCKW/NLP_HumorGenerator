#!/bin/bash
#SBATCH --job-name=grpo-plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=grpo-plot-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
python=".venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo "Run source setup.sh first." >&2
    exit 1
fi

exec "$python" scripts/plot_grpo_rewards.py "$@"
