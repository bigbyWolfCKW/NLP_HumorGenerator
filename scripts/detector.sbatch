#!/bin/bash
#SBATCH --job-name=detector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
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
splits="data/source/processed/unbiased"

if [[ "${1:-}" == "prepare" ]]; then
    if (( $# != 1 )); then
        printf 'Usage: sbatch %s prepare\n' "$0" >&2
        exit 2
    fi

    if [[ ! -e "$splits" ]]; then
        "$PROJECT_PYTHON" scripts/download_formal_pqa.py
    fi

    eda_args=(--detector roberta --detector deberta --detector roberta-large --max-length 128)
    if [[ -e "$splits/test.csv" ]]; then
        eda_args+=(--include-test)
    fi
    exec "$PROJECT_PYTHON" -m detector eda "${eda_args[@]}"
fi

if [[ "${1:-}" == "train-qproduct" ]]; then
    if (( $# != 2 )); then
        printf 'Usage: sbatch %s train-qproduct {roberta|deberta|roberta-large}\n' "$0" >&2
        exit 2
    fi

    case "$2" in
        roberta) checkpoint="roberta_qproduct_seed42" ;;
        deberta) checkpoint="10_deberta_v3_base/qproduct_seed42" ;;
        roberta-large) checkpoint="11_roberta_large/qproduct_seed42" ;;
        *) printf 'Unknown QProduct detector: %s\n' "$2" >&2; exit 2 ;;
    esac

    exec "$PROJECT_PYTHON" -m detector train --detector "$2" \
        --train-csv "$splits/train.csv" --validation-csv "$splits/validation.csv" \
        --seed 42 --epochs 3 --batch-size 16 --max-length 128 --learning-rate 2e-5 \
        --output "models/detector/checkpoints/$checkpoint"
fi

if [[ "${1:-}" == "evaluate-qproduct" ]]; then
    if (( $# > 3 )); then
        printf 'Usage: sbatch %s evaluate-qproduct [test.csv] [output-dir]\n' "$0" >&2
        exit 2
    fi

    test_csv="${2:-$splits/test.csv}"
    output="${3:-results/detector/roberta_large_seed42_test}"
    "$PROJECT_PYTHON" -m detector rank --detector roberta-large --seed 42 --check
    exec "$PROJECT_PYTHON" -m detector evaluate --detector roberta-large --seed 42 \
        --csv "$test_csv" --output "$output"
fi

exec "$PROJECT_PYTHON" -m detector "$@"
