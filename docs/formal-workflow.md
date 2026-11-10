# ProductPunch formal workflow

This workflow trains and compares product-grounded English advertising models. Each member uses their assigned product families. The six evaluation arms are Qwen2.5-7B-Instruct, pooled SFT, and four GRPO continuations from the same SFT adapter. The four rewards are the original binary detector, original Arena, Humor Mechanics, and Arena adapted to product-ad humor preferences.

## Set up on SuperPOD

Run from the repository root. Follow the project setup instructions to create `.venv`, then load Slurm with `module load slurm`. Have the local Qwen and Arena checkpoints available. The training and test CSVs and `data/formal_v1/assignment_core.json` must come from the same team assignment. Transfer data and checkpoints separately from code.

```bash
member=member_1                    # Choose member_1 through member_4.
account=mscbdt2024                  # Use your assigned Slurm account.
run="results/formal_v2"
candidate_run="$run/${member}_qwen_product_humor_v3"
reward_dir="data/formal_v2/reward/$member"
sft="$run/${member}_qwen_sft_pooled"
control="$run/${member}_qwen_grpo_arena_seed42"
baseline="$run/${member}_qwen_grpo_baseline_seed42"
mechanics="$run/${member}_qwen_grpo_mechanics_seed42"
adapted="$run/${member}_qwen_grpo_adapted_seed42"
reward_model="models/detector/preferences/${member}_product_humor"
train_file="data/formal_v2/sft/$member/reviewed_product_ads_train.jsonl"
prompt="configs/prompts/product_humor_v3.txt"
```

For member 1's current experiment, set `candidate_run=results/formal_v1/member_1_qwen_product_only_facts_v3` to reuse the archived candidate batch. Use the same `candidate_run` for both GRPO arms. If using the quickcheck reward dataset, set the training input to `$reward_dir/reward_data_quickcheck` and choose a distinct reward output path.

## Shared Slurm entry point

Use `scripts/run.sbatch` as the launcher. It routes the stage name to the corresponding script. The GPU examples below use the `normal` partition. The small JSONL preparation commands can run locally with Python 3.11 and the input files.

## 1. Generate training candidates

Skip this submission if a complete candidate run already exists for the intended prompts. A completed directory has `candidates.csv` and `completed.json`.

```bash
sbatch --account="$account" --partition=normal --gpus-per-node=1 \
  scripts/run.sbatch generate-ads --owner "$member" \
  --input "data/formal_v1/${member}_train.csv" --prompt-file "$prompt" \
  --num-products 100 --candidates 4 --seed 42 --output "$candidate_run"
```

## 2. Prepare supervision and preferences

For SFT, use a reviewed training JSONL. Each row has a `messages` list containing one user prompt followed by one assistant advertisement. If the file already exists, use it. The member 1 pooled run includes 35 legacy examples and 71 reviewed product ads. Other members supply their own reviewed examples. To prepare a new file, apply the [SFT selection prompt](formal-review-prompts.md#sft-example-selection) to a view of the candidates containing only the listed fields. Save the returned `reviewed_candidates.csv` in the candidate run, then export the approved rows:

```bash
.venv/bin/python scripts/prepare_formal_sft_v1.py \
  --owner "$member" --candidates "$candidate_run/candidates.csv" \
  --review "$candidate_run/reviewed_candidates.csv" \
  --output "data/formal_v2/sft/$member" \
  --train-filename reviewed_product_ads_train.jsonl
```

For reward adaptation, run the first command on the completed candidates. Give the AI reviewer only `blind_pairs.jsonl` and the [shared reward-label prompt](formal-review-prompts.md#reward-training-labels). Save the AI response as `reviewed_pairs.jsonl` alongside the blind file. Run the second command with the matching private key in the same directory.

```bash
.venv/bin/python scripts/prepare_humor_pairs.py \
  --owner "$member" --run-dir "$candidate_run" \
  --output "$reward_dir/humor_pairwise" --seed 42

.venv/bin/python scripts/prepare_humor_reward_data.py \
  --pair-dir "$reward_dir/humor_pairwise" \
  --reviews "$reward_dir/humor_pairwise/reviewed_pairs.jsonl" \
  --output "$reward_dir/reward_data"
```

The standard importer keeps decisive humor preferences where both ads pass. Product families stay together in either the training or validation split. Preserve the same A/B ordering and pair IDs throughout review and import.

## 3. Train the fresh SFT adapter

```bash
sbatch --account="$account" --partition=normal --gpus-per-node=1 \
  --job-name="sft-$member" scripts/run.sbatch train-sft \
  --train-file "$train_file" --output "$sft"
```

## 4. Adapt the humor reward

```bash
sbatch --account="$account" --partition=normal --gpus-per-node=1 \
  --job-name="reward-$member" scripts/run.sbatch train-reward \
  --data-dir "$reward_dir/reward_data" --output "$reward_model"
```

Read `completed.json` for the initial Arena and adapted validation accuracy and the validation sample count. The selected checkpoint is `best_model/`.

## 5. Compare GRPO rewards

Run all four GRPO arms with the same SFT adapter, candidate prompts, training steps, and seed. Baseline, Arena and Humor Mechanics use the existing checkpoints. The adapted arm uses the checkpoint from Step 4.

```bash
for reward in baseline arena mechanics; do
  sbatch --account="$account" --partition=normal --gpus-per-node=1 \
    --job-name="grpo-$reward-$member" scripts/run.sbatch train-grpo \
    --owner "$member" --candidates "$candidate_run/candidates.csv" \
    --sft-adapter "$sft/final_adapter" --reward "$reward" --steps 200 --seed 42 \
    --output "$run/${member}_qwen_grpo_${reward}_seed42" || break
done

sbatch --account="$account" --partition=normal --gpus-per-node=1 \
  --job-name="grpo-adapted-$member" scripts/run.sbatch train-grpo \
  --owner "$member" --candidates "$candidate_run/candidates.csv" \
  --sft-adapter "$sft/final_adapter" --reward-checkpoint "$reward_model/best_model" \
  --steps 200 --seed 42 --output "$adapted"
```

## 6. Evaluate on held-out products

Submit each arm with the same prompt, seed, test CSV and number of candidates. The following submits the six arms separately.

```bash
eval_dir="$run/evaluation/$member"
for variant in base sft_pooled grpo_baseline grpo_original_arena grpo_mechanics grpo_adapted; do
  case "$variant" in
    base) adapter="" ;;
    sft_pooled) adapter="$sft/final_adapter" ;;
    grpo_baseline) adapter="$baseline/final_adapter" ;;
    grpo_original_arena) adapter="$control/final_adapter" ;;
    grpo_mechanics) adapter="$mechanics/final_adapter" ;;
    grpo_adapted) adapter="$adapted/final_adapter" ;;
  esac
  adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  sbatch --account="$account" --partition=normal --gpus-per-node=1 \
    --job-name="eval-$member" scripts/run.sbatch evaluate \
    --member-id "$member" --model-variant "$variant" "${adapter_args[@]}" \
    --products "data/formal_v1/${member}_test.csv" --prompt-file "$prompt" \
    --seed 42 --candidates-per-product 4 --output "$eval_dir/$variant" || break
done
```

For a quick pilot, use the same smaller `--num-products` and `--candidates-per-product` on every arm and use separate output paths. Once all six runs complete, export matched scorecard batches:

```bash
.venv/bin/python scripts/prepare_six_model_review.py \
  --evaluation-dir "$eval_dir" --output "$eval_dir/blind_six_models" \
  --seed 42 --batch-size 20
```

Send the batch files in `reviewer/` with the [six-model scorecard prompt](formal-review-prompts.md#six-model-held-out-scorecard). Keep `PRIVATE_group_key.jsonl` with the run owner. To measure the main reward comparison directly, also export an A/B file:

```bash
.venv/bin/python scripts/prepare_final_ad_pairs.py \
  --left-run "$eval_dir/grpo_original_arena" --right-run "$eval_dir/grpo_adapted" \
  --output "$eval_dir/blind_arena_vs_adapted" --seed 42
```

Use the [held-out review prompt](formal-review-prompts.md#final-held-out-evaluation). Report humor, product-specific humor, ad suitability, individual compliance, and compliance-first preference separately. Keep the private key for decoding the review. The final held-out judgments are reserved for evaluation.

## 7. Team pooling

Team pooling is a separate stage. Merge only approved training examples and preference data, check product-family separation, and evaluate the shared model against the individual models on the same four test partitions. The pooled data builder is still pending.

## GitHub handoff

Add a link to this document from the root README after placing it at `docs/formal-workflow.md`. Commit the shared launcher, required Python scripts, and product-only prompt. Check the staged file list before committing. Keep weights, training examples, judge outputs, private mappings, and Slurm logs out of the code commit.
