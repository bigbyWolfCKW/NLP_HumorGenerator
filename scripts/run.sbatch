#!/bin/bash
#SBATCH --job-name=humor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
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

case "${1:-}" in
    generate-ads) entry=scripts/generate_formal_ads_v1.py ;;
    prepare-sft) entry=scripts/prepare_formal_sft_v1.py ;;
    prepare-humor-pairs) entry=scripts/prepare_humor_pairs.py ;;
    prepare-reward-data) entry=scripts/prepare_humor_reward_data.py ;;
    train-sft) entry=scripts/train_formal_sft_v2.py ;;
    train-reward) entry=scripts/train_product_humor_reward.py ;;
    train-grpo) entry=scripts/train_formal_grpo_v2.py ;;
    evaluate) entry=scripts/evaluate_formal_ads_v1.py ;;
    prepare-final-pairs) entry=scripts/prepare_final_ad_pairs.py ;;
    *) exec "$PROJECT_PYTHON" main.py "$@" ;;
esac

shift
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec "$PROJECT_PYTHON" "$entry" "$@"
