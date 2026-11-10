#!/bin/bash
#SBATCH --job-name=humor-formal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=normal
#SBATCH --account=mscbdt2024
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=humor-formal-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SLURM_SUBMIT_DIR="$PWD"

# EDIT SETTINGS HERE
MEMBER_ID="${MEMBER_ID:-member_1}"
STAGE="${1:-generate-ads}"
VARIANT="${2:-arena}"
SEED=42
RESUME="${RESUME:-no}"

TRAIN_PRODUCTS=100
TRAIN_CANDIDATES=4
TEST_PRODUCTS=50
TEST_CANDIDATES=4
GRPO_STEPS=200
PROMPT_FILE="configs/prompts/product_humor_v3.txt"

RUN_ROOT="results/formal_v2"
CANDIDATE_RUN="${CANDIDATE_RUN:-$RUN_ROOT/${MEMBER_ID}_qwen_product_humor_v3}"
SFT_DATA="data/formal_v2/sft/$MEMBER_ID"
REWARD_ROOT="data/formal_v2/reward/$MEMBER_ID"
PAIR_DIR="$REWARD_ROOT/humor_pairwise"
REWARD_DATA="${REWARD_DATA:-$REWARD_ROOT/reward_data_review}"
REWARD_MODEL="${REWARD_MODEL:-models/detector/preferences/${MEMBER_ID}_product_humor_review}"

SFT_RUN="$RUN_ROOT/${MEMBER_ID}_qwen_sft_pooled"
GRPO_PREFIX="$RUN_ROOT/${MEMBER_ID}_qwen_grpo"
EVAL_ROOT="$RUN_ROOT/evaluation/$MEMBER_ID"
LEFT_ARM="${LEFT_ARM:-grpo_original_arena}"
RIGHT_ARM="${RIGHT_ARM:-grpo_adapted}"
FINAL_PAIR_DIR="$EVAL_ROOT/blind_${LEFT_ARM}_vs_${RIGHT_ARM}"
# END OF SETTINGS

case "$MEMBER_ID" in
    member_1|member_2|member_3|member_4) ;;
    *) echo "Invalid MEMBER_ID: $MEMBER_ID" >&2; exit 2 ;;
esac

need() {
    for file in "$@"; do
        if [[ ! -s "$file" ]]; then
            printf 'Missing or empty input: %s\n' "$file" >&2
            exit 1
        fi
    done
}

run() {
    exec bash scripts/run.sbatch "$@"
}

resume_args=()
if [[ "$RESUME" == "yes" ]]; then
    resume_args=(--resume)
fi

train_csv="data/formal_v1/${MEMBER_ID}_train.csv"
test_csv="data/formal_v1/${MEMBER_ID}_test.csv"
sft_file="$SFT_DATA/reviewed_product_ads_train.jsonl"

case "$STAGE" in
    generate-ads)
        need "$train_csv" "$PROMPT_FILE"
        run generate-ads --owner "$MEMBER_ID" --input "$train_csv" \
            --prompt-file "$PROMPT_FILE" --num-products "$TRAIN_PRODUCTS" \
            --candidates "$TRAIN_CANDIDATES" --seed "$SEED" \
            --output "$CANDIDATE_RUN" "${resume_args[@]}"
        ;;

    prepare-sft)
        need "$CANDIDATE_RUN/candidates.csv" "$CANDIDATE_RUN/reviewed_candidates.csv"
        run prepare-sft --owner "$MEMBER_ID" \
            --candidates "$CANDIDATE_RUN/candidates.csv" \
            --review "$CANDIDATE_RUN/reviewed_candidates.csv" \
            --output "$SFT_DATA" --train-filename reviewed_product_ads_train.jsonl
        ;;

    prepare-humor-pairs)
        need "$CANDIDATE_RUN/candidates.csv" "$CANDIDATE_RUN/completed.json"
        run prepare-humor-pairs --owner "$MEMBER_ID" \
            --run-dir "$CANDIDATE_RUN" --output "$PAIR_DIR" --seed "$SEED"
        ;;

    prepare-reward-data)
        need "$PAIR_DIR/blind_pairs.jsonl" "$PAIR_DIR/PRIVATE_pair_key.jsonl" \
            "$PAIR_DIR/reviewed_pairs.jsonl"
        run prepare-reward-data --pair-dir "$PAIR_DIR" \
            --reviews "$PAIR_DIR/reviewed_pairs.jsonl" --output "$REWARD_DATA"
        ;;

    train-sft)
        need "$sft_file"
        run train-sft --train-file "$sft_file" --output "$SFT_RUN" \
            "${resume_args[@]}"
        ;;

    train-reward)
        need "$REWARD_DATA/train.jsonl" "$REWARD_DATA/validation.jsonl"
        run train-reward --data-dir "$REWARD_DATA" --output "$REWARD_MODEL" \
            --seed "$SEED" "${resume_args[@]}"
        ;;

    train-grpo)
        need "$CANDIDATE_RUN/candidates.csv" "$CANDIDATE_RUN/completed.json" \
            "$SFT_RUN/completed.json"
        case "$VARIANT" in
            baseline|arena|mechanics) reward_args=(--reward "$VARIANT") ;;
            adapted)
                need "$REWARD_MODEL/completed.json"
                reward_args=(--reward-checkpoint "$REWARD_MODEL/best_model")
                ;;
            *) echo "Use baseline, arena, mechanics, or adapted." >&2; exit 2 ;;
        esac
        run train-grpo --owner "$MEMBER_ID" \
            --candidates "$CANDIDATE_RUN/candidates.csv" \
            --sft-adapter "$SFT_RUN/final_adapter" "${reward_args[@]}" \
            --steps "$GRPO_STEPS" --seed "$SEED" \
            --output "${GRPO_PREFIX}_${VARIANT}_seed${SEED}" "${resume_args[@]}"
        ;;

    evaluate)
        need "$test_csv" "$PROMPT_FILE"
        case "$VARIANT" in
            base) adapter_run="" ;;
            sft_pooled) adapter_run="$SFT_RUN" ;;
            grpo_baseline) adapter_run="${GRPO_PREFIX}_baseline_seed${SEED}" ;;
            grpo_original_arena) adapter_run="${GRPO_PREFIX}_arena_seed${SEED}" ;;
            grpo_mechanics) adapter_run="${GRPO_PREFIX}_mechanics_seed${SEED}" ;;
            grpo_adapted) adapter_run="${GRPO_PREFIX}_adapted_seed${SEED}" ;;
            *) echo "Invalid evaluation variant: $VARIANT" >&2; exit 2 ;;
        esac
        adapter_args=()
        if [[ -n "$adapter_run" ]]; then
            need "$adapter_run/completed.json"
            adapter_args=(--adapter "$adapter_run/final_adapter")
        fi
        run evaluate --member-id "$MEMBER_ID" --llm qwen \
            --model-variant "$VARIANT" "${adapter_args[@]}" \
            --products "$test_csv" --num-products "$TEST_PRODUCTS" \
            --candidates-per-product "$TEST_CANDIDATES" --seed "$SEED" \
            --prompt-file "$PROMPT_FILE" --output "$EVAL_ROOT/$VARIANT" \
            "${resume_args[@]}"
        ;;

    prepare-final-pairs)
        need "$EVAL_ROOT/$LEFT_ARM/completed.json" "$EVAL_ROOT/$RIGHT_ARM/completed.json"
        run prepare-final-pairs --left-run "$EVAL_ROOT/$LEFT_ARM" \
            --right-run "$EVAL_ROOT/$RIGHT_ARM" --output "$FINAL_PAIR_DIR" \
            --seed "$SEED"
        ;;

    *)
        echo "Unknown stage: $STAGE" >&2
        exit 2
        ;;
esac