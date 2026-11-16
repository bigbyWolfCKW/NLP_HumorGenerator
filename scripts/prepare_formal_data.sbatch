#!/bin/bash
#SBATCH --job-name=formal-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=formal-data-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

member="${MEMBER_ID:-member_1}"
facts="${1:-}"
python=".venv/bin/python"
splits="data/source/processed/unbiased"
assignment="data/formal_v1"

case "$member" in
    member_1|member_2|member_3|member_4) ;;
    *) printf 'Invalid MEMBER_ID: %s\n' "$member" >&2; exit 2 ;;
esac

if (( $# > 1 )); then
    printf 'Usage: sbatch %s [reviewed_facts.csv]\n' "$0" >&2
    exit 2
fi

need() {
    for file in "$@"; do
        if [[ ! -s "$file" ]]; then
            printf 'Missing or empty input: %s\n' "$file" >&2
            exit 1
        fi
    done
}

if [[ -n "$facts" ]]; then
    need "$facts"
fi

if [[ ! -e "$splits" ]]; then
    "$python" scripts/download_formal_pqa.py
fi
need "$splits/train.csv" "$splits/validation.csv" "$splits/test.csv"
"$python" -m detector eda --include-test

if [[ ! -e "$assignment" ]]; then
    "$python" scripts/prepare_formal_product_list_v1.py
fi
need "$assignment/assignment_core.json"
for owner in member_1 member_2 member_3 member_4; do
    for split in train validation test; do
        need "$assignment/${owner}_${split}.csv"
    done
done

if [[ -n "$facts" ]]; then
    "$python" scripts/prepare_formal_enrichment.py \
        --input "$assignment/${member}_train.csv" \
        --facts "$facts" \
        --output "$assignment/enrichment/${member}_train.csv"
fi
