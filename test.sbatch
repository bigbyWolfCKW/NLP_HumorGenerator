#!/bin/bash
#SBATCH --job-name=humor-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=humor-test-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the NLP_HumorGenerator directory}"

# EDIT SETTINGS HERE. Submit every mode with: sbatch test.sbatch
MODE="demo"                 # demo | full | custom | generate | rank | review | prepare | train | test
STEPS="generate rank"        # custom only; space-separated stages in pipeline order
RUN_DIR=""                  # blank: new run; resume: results/demo-709136
LLM="deepseek"               # deepseek | qwen | qwen3b | qwen14b | deepseek14b | mistral
REVIEW_BACKEND="api"          # api: remote reviewer, no download; transformers: local reviewer
REVIEW_LLM="gpt-4o-mini"   # API model ID; for local review use qwen / qwen14b etc.
REVIEW_API_BASE="https://hkust.azure-api.net/hkust-genai/v1"
REVIEW_API_KEY_ENV="HKUST_API_KEY"  # environment variable NAME only; never put the key here
REVIEW_API_MAX_TOKENS=1024
DETECTOR="roberta"           # e.g. roberta | deberta | roberta-large
DETECTOR_SEED=""             # blank: detector default; or 42, 43, 44
NUM_PRODUCTS=""             # blank: demo=10, otherwise=50; explicit PRODUCTS uses all its entries
EPOCHS=""                   # blank: demo=1, otherwise configs/training.json
SEED=42
PRODUCTS=""                 # blank: configured QA dataset; or data/products.json
RANKED_FILE="ranked.xlsx"    # use ranked_deberta.xlsx to add another detector's ranking
ADAPTER=""                  # test only: blank uses RUN_DIR/lora/final_adapter
TRAINING_CONFIG="configs/training.json"
DRY_RUN="no"                # yes: print commands only; no model execution or file writes
# API key: put HKUST_API_KEY=your_own_key in the private .env (copy .env.example once).
# END OF SETTINGS. No command editing needed below.

PROJECT_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$PROJECT_PYTHON" || ! -f "$PWD/.venv/pyvenv.cfg" ]]; then
    echo "Project environment missing. Run: source setup.sh" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1

ARGS=(--mode "$MODE" --llm "$LLM" --review-llm "$REVIEW_LLM"
      --review-backend "$REVIEW_BACKEND" --review-api-base "$REVIEW_API_BASE"
      --review-api-key-env "$REVIEW_API_KEY_ENV" --review-api-max-tokens "$REVIEW_API_MAX_TOKENS"
      --detector "$DETECTOR" --seed "$SEED" --ranked-file "$RANKED_FILE"
      --training-config "$TRAINING_CONFIG")
if [[ "$MODE" == "custom" ]]; then
    read -r -a SELECTED_STEPS <<< "$STEPS"
    ARGS+=(--steps "${SELECTED_STEPS[@]}")
fi
[[ -z "$RUN_DIR" ]] || ARGS+=(--run-dir "$RUN_DIR")
[[ -z "$DETECTOR_SEED" ]] || ARGS+=(--detector-seed "$DETECTOR_SEED")
[[ -z "$NUM_PRODUCTS" ]] || ARGS+=(--num-products "$NUM_PRODUCTS")
[[ -z "$EPOCHS" ]] || ARGS+=(--epochs "$EPOCHS")
[[ -z "$PRODUCTS" ]] || ARGS+=(--products "$PRODUCTS")
[[ -z "$ADAPTER" ]] || ARGS+=(--adapter "$ADAPTER")
[[ "$DRY_RUN" != "yes" ]] || ARGS+=(--dry-run)

echo "Job: $SLURM_JOB_ID | Mode: $MODE | LLM: $LLM | Detector: $DETECTOR"
date
"$PROJECT_PYTHON" run_pipeline.py "${ARGS[@]}" "$@"
date
