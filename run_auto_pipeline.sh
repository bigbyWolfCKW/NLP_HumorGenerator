#!/bin/bash
# Run the ProductPunch formal workflow, including AI review, from .env.
# Existing formal.sbatch commands are unchanged. This script only calls them.

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

DRY=0
FROM="data"

usage() {
    cat <<'EOF'
Usage: bash run_auto_pipeline.sh [--dry-run] [--from-stage STAGE]

Loads .env when present. GPU stages use sbatch; review and export stages run locally.
Completed outputs are skipped, so an interrupted run can be started again.

Stages, in order:
  data, download, generate-ads, sft-review, prepare-sft,
  prepare-humor-pairs, reward-review, prepare-reward-data,
  train, grpo, plot, evaluate, scorecard, final-pairs

Set REVIEW_FALLBACK_MODE=heuristic to review without an API key.
Set RUN_REWARD_PLOT=no to skip the optional reward-trace plot.
Set SKIP_FINAL_PAIRS=yes to skip the optional Arena A/B review.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --from-stage)
            [[ $# -ge 2 ]] || { echo "Missing stage name after --from-stage." >&2; exit 2; }
            FROM="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]]; then
    echo "Python 3 is required. Run: source setup.sh" >&2
    exit 1
fi

eval "$("$PY" - <<'PY'
import os, shlex, sys
sys.path.insert(0, ".")
from src.environment import load_local_env
load_local_env()
keys = """
MEMBER_ID MODEL SLURM_ACCOUNT SLURM_PARTITION
REVIEW_API_BASE_URL REVIEW_API_KEY REVIEW_MODEL REVIEW_FALLBACK_MODE
REVIEW_API_MAX_TOKENS REVIEW_BATCH_SIZE
HKUST_API_KEY GEMINI_API_KEY DEEPSEEK_API_KEY
TRAIN_PRODUCTS TEST_PRODUCTS RUN_REWARD_PLOT SKIP_FINAL_PAIRS CANDIDATE_RUN
REWARD_DATA REWARD_MODEL
""".split()
for key in keys:
    value = os.environ.get(key)
    if value:
        print(f"export {key}={shlex.quote(value)}")
PY
)"

MEMBER_ID="${MEMBER_ID:-member_4}"
MODEL="${MODEL:-qwen}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mscbdt2024}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
REVIEW_FALLBACK_MODE="${REVIEW_FALLBACK_MODE:-api}"
RUN_REWARD_PLOT="${RUN_REWARD_PLOT:-yes}"
SKIP_FINAL_PAIRS="${SKIP_FINAL_PAIRS:-no}"
export MEMBER_ID MODEL SLURM_ACCOUNT SLURM_PARTITION REVIEW_FALLBACK_MODE

case "$MEMBER_ID" in
    member_1|member_2|member_3|member_4|all) ;;
    *) echo "MEMBER_ID must be member_1, member_2, member_3, member_4, or all." >&2; exit 2 ;;
esac
case "$REVIEW_FALLBACK_MODE" in
    api|heuristic) ;;
    *) echo "REVIEW_FALLBACK_MODE must be api or heuristic." >&2; exit 2 ;;
esac
if [[ ! -f configs/llms.json ]] || ! "$PY" -c 'import json,sys; raise SystemExit(0 if sys.argv[1] in json.load(open("configs/llms.json")) else 1)' "$MODEL"; then
    echo "MODEL must be a key in configs/llms.json (qwen, mistral, or deepseek)." >&2
    exit 2
fi

if [[ "$MODEL" == "qwen" ]]; then
    DATA_SCOPE="$MEMBER_ID"
    REWARD_SCOPE="$MEMBER_ID"
else
    DATA_SCOPE="$MEMBER_ID/$MODEL"
    REWARD_SCOPE="${MEMBER_ID}_$MODEL"
fi
RUN_ROOT="results/formal_v2"
CANDIDATE_RUN="${CANDIDATE_RUN:-$RUN_ROOT/${MEMBER_ID}_${MODEL}_product_humor_v3}"
SFT_DATA="data/formal_v2/sft/$DATA_SCOPE"
SFT_FILE="$SFT_DATA/reviewed_product_ads_train.jsonl"
PAIR_DIR="data/formal_v2/reward/$DATA_SCOPE/humor_pairwise"
REWARD_DATA="${REWARD_DATA:-data/formal_v2/reward/$DATA_SCOPE/reward_data_review}"
REWARD_MODEL="${REWARD_MODEL:-models/detector/preferences/${REWARD_SCOPE}_product_humor_review}"
SFT_RUN="$RUN_ROOT/${MEMBER_ID}_${MODEL}_sft_pooled"
GRPO_PREFIX="$RUN_ROOT/${MEMBER_ID}_${MODEL}_grpo"
EVAL_ROOT="$RUN_ROOT/evaluation/$DATA_SCOPE"
SIX_DIR="$EVAL_ROOT/blind_six_models"
FINAL_DIR="$EVAL_ROOT/blind_grpo_original_arena_vs_grpo_adapted"
export CANDIDATE_RUN REWARD_DATA REWARD_MODEL

llm_args=()
if [[ "$MODEL" != "qwen" ]]; then
    llm_args=(--llm "$MODEL")
fi

STAGES=(
    data download generate-ads sft-review prepare-sft
    prepare-humor-pairs reward-review prepare-reward-data
    train grpo plot evaluate scorecard final-pairs
)
from_index=-1
stage_index=0
for name in "${STAGES[@]}"; do
    if [[ "$name" == "$FROM" ]]; then
        from_index=$stage_index
    fi
    stage_index=$((stage_index + 1))
done
if [[ "$from_index" -lt 0 ]]; then
    echo "Unknown stage: $FROM" >&2
    echo "Choose one of: ${STAGES[*]}" >&2
    exit 2
fi

should_run() {
    local index=0
    for name in "${STAGES[@]}"; do
        if [[ "$name" == "$1" ]]; then
            [[ "$index" -ge "$from_index" ]]
            return
        fi
        index=$((index + 1))
    done
    return 1
}

note() { printf '%s\n' "$*"; }

require_file() {
    local file="$1"
    if [[ -s "$file" ]]; then
        return
    fi
    if [[ "$DRY" -eq 1 ]]; then
        note "[dry-run] required file is not present yet: $file"
        return
    fi
    echo "Missing or empty input: $file" >&2
    exit 1
}

refuse_incomplete() {
    local directory="$1"
    local marker="$2"
    if [[ "$DRY" -eq 0 && -d "$directory" && ! -s "$marker" ]]; then
        echo "Incomplete output directory: $directory" >&2
        echo "Remove it before rerunning this stage, or restore $marker." >&2
        exit 1
    fi
}

check_api() {
    if [[ "$REVIEW_FALLBACK_MODE" != "api" ]]; then
        return
    fi
    local missing=0
    if [[ -z "${REVIEW_API_BASE_URL:-}" || -z "${REVIEW_MODEL:-}" ]]; then
        missing=1
    fi
    if [[ -z "${REVIEW_API_KEY:-}${HKUST_API_KEY:-}${GEMINI_API_KEY:-}${DEEPSEEK_API_KEY:-}" ]]; then
        missing=1
    fi
    if [[ "$missing" -eq 0 ]]; then
        return
    fi
    echo "API review needs REVIEW_API_BASE_URL, REVIEW_MODEL, and REVIEW_API_KEY." >&2
    echo "For an offline run, set REVIEW_FALLBACK_MODE=heuristic." >&2
    if [[ "$DRY" -eq 1 ]]; then
        note "Dry run will still print the plan."
        return
    fi
    exit 2
}

check_detectors() {
    local missing=0
    local path
    for path in \
        "models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model" \
        "models/detector/preferences/arena_v1_seed42/best_model" \
        "models/detector/preferences/mechanics_v1_seed42/best_model"
    do
        if [[ ! -s "$path/config.json" ]]; then
            echo "Missing detector checkpoint: $path" >&2
            missing=1
        fi
    done
    if [[ "$missing" -eq 0 ]]; then
        return 0
    fi
    echo "Train the detector and preference models before GRPO. See README.md." >&2
    if [[ "$DRY" -eq 1 ]]; then
        return 0
    fi
    return 1
}

model_ready() {
    "$PY" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from download_models import complete
name = ${MODEL@Q}
folder = Path(json.load(open("configs/llms.json"))[name]["path"])
raise SystemExit(0 if complete(folder) else 1)
PY
}

JOBS=()

submit() {
    local label="$1"
    shift
    if [[ "$DRY" -eq 1 ]]; then
        note "[dry-run] $label: sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION $*"
        return
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch was not found. Run: module load slurm" >&2
        exit 1
    fi
    local job_id
    job_id="$(sbatch --parsable --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" "$@")"
    note "Submitted $label as Slurm job $job_id"
    JOBS+=("$job_id|$label")
}

submit_formal() {
    local done="$1"
    local label="$2"
    shift 2
    if [[ -s "$done" ]]; then
        note "Skip $label: $done already exists."
        return
    fi
    local outdir
    outdir="$(dirname "$done")"
        if [[ "$DRY" -eq 1 ]]; then
            if [[ -d "$outdir" ]]; then
                note "[dry-run] $label: RESUME=yes sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION formal.sbatch $* ${llm_args[*]:-}"
            else
                note "[dry-run] $label: sbatch --account=$SLURM_ACCOUNT --partition=$SLURM_PARTITION formal.sbatch $* ${llm_args[*]:-}"
            fi
            return
        fi
    if [[ -d "$outdir" ]]; then
        RESUME=yes submit "$label" formal.sbatch "$@" ${llm_args[@]+"${llm_args[@]}"}
    else
        submit "$label" formal.sbatch "$@" ${llm_args[@]+"${llm_args[@]}"}
    fi
}

wait_jobs() {
    local expected=("$@")
    if [[ "$DRY" -eq 1 || ${#JOBS[@]} -eq 0 ]]; then
        JOBS=()
        return
    fi
    local -A seen=()
    while [[ ${#JOBS[@]} -gt 0 ]]; do
        local still=()
        local entry job_id label state queued
        for entry in "${JOBS[@]}"; do
            job_id="${entry%%|*}"
            label="${entry#*|}"
            queued="$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || true)"
            queued="${queued%%$'\n'*}"
            if [[ -n "$queued" ]]; then
                still+=("$entry")
                continue
            fi
            state="$(sacct -j "$job_id" -n -X -P --format=State 2>/dev/null || true)"
            state="${state%%$'\n'*}"
            state="${state// /}"
            case "$state" in
                COMPLETED) note "$label job $job_id completed." ;;
                ""|PENDING|RUNNING|COMPLETING)
                    if [[ -z "${seen[$job_id]:-}" ]]; then
                        seen["$job_id"]=1
                    elif [[ "${seen[$job_id]}" -ge 6 ]]; then
                        echo "Could not confirm Slurm state for $label job $job_id." >&2
                        if [[ "${ALLOW_FAILURE:-0}" -eq 1 ]]; then
                            JOBS=()
                            return 1
                        fi
                        exit 1
                    else
                        seen["$job_id"]=$((seen["$job_id"] + 1))
                    fi
                    still+=("$entry")
                    ;;
                *)
                    echo "$label job $job_id ended with state ${state:-unknown}." >&2
                    echo "See humor-formal-${job_id}.log or the matching Slurm log." >&2
                    if [[ "${ALLOW_FAILURE:-0}" -eq 1 ]]; then
                        JOBS=()
                        return 1
                    fi
                    exit 1
                    ;;
            esac
        done
        if [[ ${#still[@]} -eq 0 ]]; then
            JOBS=()
        else
            JOBS=("${still[@]}")
        fi
        if [[ ${#JOBS[@]} -gt 0 ]]; then
            note "Waiting for ${#JOBS[@]} Slurm job(s)..."
            sleep 30
        fi
    done
    if [[ ${#expected[@]} -gt 0 ]]; then
        local file
        for file in "${expected[@]}"; do
            [[ -z "$file" ]] && continue
            require_file "$file"
        done
    fi
}

run_local() {
    local label="$1"
    shift
    if [[ "$DRY" -eq 1 ]]; then
        note "[dry-run] $label: $*"
        return
    fi
    note "Running $label"
    "$@"
}

note "ProductPunch automated replication"
note "MEMBER_ID=$MEMBER_ID MODEL=$MODEL review=$REVIEW_FALLBACK_MODE account=$SLURM_ACCOUNT partition=$SLURM_PARTITION"
check_api
if should_run grpo; then
    check_detectors || true
fi

if should_run data; then
    prep_member="$MEMBER_ID"
    [[ "$MEMBER_ID" == "all" ]] && prep_member="member_1"
    need_data=0
    if [[ ! -s data/formal_v1/assignment_core.json ]]; then
        need_data=1
    fi
    if [[ "$MEMBER_ID" == "all" ]]; then
        for owner in member_1 member_2 member_3 member_4; do
            [[ -s "data/formal_v1/${owner}_train.csv" && -s "data/formal_v1/${owner}_test.csv" ]] || need_data=1
        done
    else
        [[ -s "data/formal_v1/${MEMBER_ID}_train.csv" && -s "data/formal_v1/${MEMBER_ID}_test.csv" ]] || need_data=1
    fi
    if [[ "$need_data" -eq 0 ]]; then
        note "Skip data preparation: member assignment files already exist."
    else
        note "Preparing Amazon PQA assignments with MEMBER_ID=$prep_member."
        MEMBER_ID="$prep_member" submit "prepare-data" scripts/prepare_formal_data.sbatch
        wait_jobs
    fi
fi

if should_run download; then
    if model_ready; then
        note "Skip model download: $MODEL is already available locally."
    else
        submit "download-$MODEL" --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=02:00:00 \
            --wrap "cd '$ROOT' && .venv/bin/python scripts/download_models.py --models '$MODEL'"
        wait_jobs
        if [[ "$DRY" -eq 0 ]] && ! model_ready; then
            echo "Base model download finished without a complete local checkpoint." >&2
            exit 1
        fi
    fi
fi

if [[ "$DRY" -eq 0 && ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "Project environment missing. Run: source setup.sh" >&2
    exit 1
fi
PY="$ROOT/.venv/bin/python"
review=("$PY" scripts/auto_review.py --mode "$REVIEW_FALLBACK_MODE" --resume)

if should_run generate-ads; then
    submit_formal "$CANDIDATE_RUN/completed.json" "generate-ads" generate-ads
    wait_jobs "$CANDIDATE_RUN/completed.json"
fi

if should_run sft-review; then
    require_file "$CANDIDATE_RUN/candidates.csv"
    run_local "sft-review" "${review[@]}" sft-review --candidate-run "$CANDIDATE_RUN"
fi

if should_run prepare-sft; then
    if [[ -s "$SFT_FILE" ]]; then
        note "Skip prepare-sft: $SFT_FILE already exists."
    else
        refuse_incomplete "$SFT_DATA" "$SFT_FILE"
        require_file "$CANDIDATE_RUN/reviewed_candidates.csv"
        run_local "prepare-sft" bash formal.sbatch prepare-sft ${llm_args[@]+"${llm_args[@]}"}
        [[ "$DRY" -eq 1 ]] || require_file "$SFT_FILE"
    fi
fi

if should_run prepare-humor-pairs; then
    if [[ -s "$PAIR_DIR/blind_pairs.jsonl" && -s "$PAIR_DIR/PRIVATE_pair_key.jsonl" ]]; then
        note "Skip prepare-humor-pairs: $PAIR_DIR already exists."
    else
        refuse_incomplete "$PAIR_DIR" "$PAIR_DIR/blind_pairs.jsonl"
        require_file "$CANDIDATE_RUN/completed.json"
        run_local "prepare-humor-pairs" bash formal.sbatch prepare-humor-pairs ${llm_args[@]+"${llm_args[@]}"}
        [[ "$DRY" -eq 1 ]] || require_file "$PAIR_DIR/blind_pairs.jsonl"
    fi
fi

if should_run reward-review; then
    require_file "$PAIR_DIR/blind_pairs.jsonl"
    run_local "reward-review" "${review[@]}" reward-review --pair-dir "$PAIR_DIR"
fi

if should_run prepare-reward-data; then
    if [[ -s "$REWARD_DATA/train.jsonl" && -s "$REWARD_DATA/validation.jsonl" ]]; then
        note "Skip prepare-reward-data: $REWARD_DATA already exists."
    else
        refuse_incomplete "$REWARD_DATA" "$REWARD_DATA/train.jsonl"
        require_file "$PAIR_DIR/reviewed_pairs.jsonl"
        run_local "prepare-reward-data" bash formal.sbatch prepare-reward-data ${llm_args[@]+"${llm_args[@]}"}
        [[ "$DRY" -eq 1 ]] || require_file "$REWARD_DATA/train.jsonl"
    fi
fi

if should_run train; then
    submit_formal "$SFT_RUN/completed.json" "train-sft" train-sft
    submit_formal "$REWARD_MODEL/completed.json" "train-reward" train-reward
    wait_jobs "$SFT_RUN/completed.json" "$REWARD_MODEL/completed.json"
fi

if should_run grpo; then
    check_detectors
    require_file "$SFT_RUN/completed.json"
    require_file "$REWARD_MODEL/completed.json"
    for variant in baseline arena mechanics adapted; do
        submit_formal "${GRPO_PREFIX}_${variant}_seed42/completed.json" "train-grpo-$variant" train-grpo "$variant"
    done
    wait_jobs \
        "${GRPO_PREFIX}_baseline_seed42/completed.json" \
        "${GRPO_PREFIX}_arena_seed42/completed.json" \
        "${GRPO_PREFIX}_mechanics_seed42/completed.json" \
        "${GRPO_PREFIX}_adapted_seed42/completed.json"
fi

if should_run plot; then
    plot_file="$RUN_ROOT/plots/${MEMBER_ID}_grpo_reward_traces_seed42.png"
    [[ "$MODEL" == "qwen" ]] || plot_file="$RUN_ROOT/plots/${MEMBER_ID}_${MODEL}_grpo_reward_traces_seed42.png"
    if [[ "$RUN_REWARD_PLOT" == "no" ]]; then
        note "Skip reward plot: RUN_REWARD_PLOT=no."
    elif [[ -s "$plot_file" ]]; then
        note "Skip reward plot: $plot_file already exists."
    else
        if [[ "$MODEL" == "qwen" ]]; then
            submit "reward-plot" scripts/plot_grpo_rewards.sbatch "$MEMBER_ID"
        else
            submit "reward-plot" scripts/plot_grpo_rewards.sbatch "$MEMBER_ID" --llm "$MODEL"
        fi
        if [[ "$DRY" -eq 0 ]]; then
            if ! ALLOW_FAILURE=1 wait_jobs; then
                note "Reward plot failed. The rest of the pipeline will continue."
            fi
        fi
    fi
fi

if should_run evaluate; then
    for variant in base sft_pooled grpo_baseline grpo_original_arena grpo_mechanics grpo_adapted; do
        submit_formal "$EVAL_ROOT/$variant/completed.json" "evaluate-$variant" evaluate "$variant"
    done
    wait_jobs \
        "$EVAL_ROOT/base/completed.json" \
        "$EVAL_ROOT/sft_pooled/completed.json" \
        "$EVAL_ROOT/grpo_baseline/completed.json" \
        "$EVAL_ROOT/grpo_original_arena/completed.json" \
        "$EVAL_ROOT/grpo_mechanics/completed.json" \
        "$EVAL_ROOT/grpo_adapted/completed.json"
fi

if should_run scorecard; then
    if [[ ! -s "$SIX_DIR/reviewer/blind_batch_01.jsonl" ]]; then
        refuse_incomplete "$SIX_DIR" "$SIX_DIR/reviewer/blind_batch_01.jsonl"
        run_local "six-model export" "$PY" scripts/prepare_six_model_review.py \
            --evaluation-dir "$EVAL_ROOT" \
            --output "$SIX_DIR" \
            --seed 42 --batch-size 20
    else
        note "Skip six-model export: $SIX_DIR already exists."
    fi
    run_local "scorecard-review" "${review[@]}" scorecard-review --review-dir "$SIX_DIR"
    summary="$SIX_DIR/six_model_summary.csv"
    if [[ -s "$summary" ]]; then
        note "Six-model summary already exists:"
        [[ "$DRY" -eq 1 ]] || cat "$summary"
    else
        run_local "scorecard summary" "$PY" scripts/summarize_six_model_review.py --review-dir "$SIX_DIR"
    fi
fi

if should_run final-pairs; then
    if [[ "$SKIP_FINAL_PAIRS" == "yes" ]]; then
        note "Skip final A/B review: SKIP_FINAL_PAIRS=yes."
    else
        if [[ ! -s "$FINAL_DIR/blind_pairs.jsonl" ]]; then
            refuse_incomplete "$FINAL_DIR" "$FINAL_DIR/blind_pairs.jsonl"
            run_local "prepare-final-pairs" bash formal.sbatch prepare-final-pairs ${llm_args[@]+"${llm_args[@]}"}
        else
            note "Skip prepare-final-pairs: $FINAL_DIR/blind_pairs.jsonl already exists."
        fi
        run_local "final-pairs-review" "${review[@]}" final-pairs-review --pair-dir "$FINAL_DIR"
        if [[ -s "$FINAL_DIR/final_pair_summary.csv" ]]; then
            note "Final pair summary already exists: $FINAL_DIR/final_pair_summary.csv"
            [[ "$DRY" -eq 1 ]] || cat "$FINAL_DIR/final_pair_summary.csv"
        else
            run_local "summarize-final-pairs" bash formal.sbatch summarize-final-pairs ${llm_args[@]+"${llm_args[@]}"}
        fi
    fi
fi

note "Automated replication finished for $MEMBER_ID ($MODEL)."
