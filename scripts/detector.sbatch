#!/bin/bash
#SBATCH --job-name=detector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=detector-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
PROJECT_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$PROJECT_PYTHON" || ! -f "$PWD/.venv/pyvenv.cfg" ]]; then
    echo "Run source setup.sh first." >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
if [[ "${HUMOR_CHECK_ENV:-0}" == "1" ]]; then
    :
fi
"$PROJECT_PYTHON" -m detector "$@"
