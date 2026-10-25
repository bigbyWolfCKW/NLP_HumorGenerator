#!/bin/bash
#SBATCH --job-name=humor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=humor-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
PROJECT_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$PROJECT_PYTHON" || ! -f "$PWD/.venv/pyvenv.cfg" ]]; then
    echo "Project environment missing. Run: source setup.sh" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
    :
"$PROJECT_PYTHON" main.py "$@"
