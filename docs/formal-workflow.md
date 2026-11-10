# ProductPunch formal workflow

Run these commands from the repository root on SuperPOD. Each member uses
their assigned training and test products. The commands below use
`formal.sbatch` for all stages except the six-model scorecard export.

## 1. Set up

```bash
source setup.sh
module load slurm
export MEMBER_ID=member_2  # Replace with your assigned ID.
```

Use `member_1`, `member_2`, `member_3`, or `member_4`. Set the same ID again
if you start a new shell. Check that the Slurm account in `formal.sbatch`
matches your allocation.

Place the following files from the same team assignment in the checkout:

- `data/formal_v1/assignment_core.json`
- `data/formal_v1/${MEMBER_ID}_train.csv`
- `data/formal_v1/${MEMBER_ID}_test.csv`

The defaults use 100 training products, 50 test products, four ads per
product, and seed 42. Make sure your assigned files contain enough
products for those settings. If your assignment calls for fewer products,
set `TRAIN_PRODUCTS` and `TEST_PRODUCTS` in `formal.sbatch` before you begin.
Keep the same test settings across all six evaluations.

The jobs load models offline. Place the Qwen base and the three existing
reward checkpoints at these paths before running stages that need them:

| Model | Directory |
| --- | --- |
| Qwen2.5-7B-Instruct | `models/llm/Qwen2.5-7B-Instruct/` |
| Binary detector | `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model/` |
| Original Arena | `models/detector/preferences/arena_v1_seed42/best_model/` |
| Humor Mechanics | `models/detector/preferences/mechanics_v1_seed42/best_model/` |

Transfer the assigned data and existing reward checkpoints separately.
If the Qwen base is missing, download it before submitting GPU jobs with
`.venv/bin/python scripts/download_models.py --models qwen`.

## 2. Generate training candidates

```bash
export CANDIDATE_RUN="results/formal_v2/${MEMBER_ID}_qwen_product_humor_v3"
sbatch formal.sbatch generate-ads
```

Wait for `$CANDIDATE_RUN/completed.json` and
`$CANDIDATE_RUN/candidates.csv`. To reuse a compatible completed candidate
run, set `CANDIDATE_RUN` to that directory and skip the submission.
Use the same candidate run for the SFT review, reward pairs, and all four
GRPO runs. Re-export the override if you start a new shell.

A successful `sbatch` submission returns a job ID. Check progress with
`squeue -j JOB_ID` and read `humor-formal-JOB_ID.log`. Wait for the required
outputs before submitting a dependent stage.

## 3. Prepare the reviewed training data

### SFT examples

Give the AI the [SFT selection prompt](formal-review-prompts.md#sft-example-selection)
and a view of `$CANDIDATE_RUN/candidates.csv` containing `candidate_id`,
`ad_text`, supplied product information, and `target_audience`. Use
`verified_product_facts` when present and `product_facts` otherwise.
Keep each candidate ID intact. Save the returned CSV at
`$CANDIDATE_RUN/reviewed_candidates.csv`.

```bash
bash formal.sbatch prepare-sft
```

The output is
`data/formal_v2/sft/${MEMBER_ID}/reviewed_product_ads_train.jsonl`.
If you already have the intended reviewed JSONL at that path, use it
directly and skip `prepare-sft`.

### Reward preferences

```bash
bash formal.sbatch prepare-humor-pairs
```

In `data/formal_v2/reward/${MEMBER_ID}/humor_pairwise/`, give the AI only
`blind_pairs.jsonl` and the
[reward-label prompt](formal-review-prompts.md#reward-training-labels).
Save its response as `reviewed_pairs.jsonl` in that same directory.
Keep `PRIVATE_pair_key.jsonl` beside the blind file on SuperPOD.

```bash
bash formal.sbatch prepare-reward-data
```

The importer keeps decisive humor preferences where both ads pass and
keeps each product family in one split. It writes `train.jsonl` and
`validation.jsonl` under
`data/formal_v2/reward/${MEMBER_ID}/reward_data_review/`.

## 4. Train SFT and the adapted reward

Once their inputs are ready, these jobs can run in parallel:

```bash
sbatch formal.sbatch train-sft
sbatch formal.sbatch train-reward
```

| Job | Default output directory | Required before the next stage |
| --- | --- | --- |
| SFT | `results/formal_v2/${MEMBER_ID}_qwen_sft_pooled/` | `completed.json` and `final_adapter/` |
| Adapted reward | `models/detector/preferences/${MEMBER_ID}_product_humor_review/` | `completed.json` and `best_model/` |

The reward job starts from original Arena. Its `completed.json` reports
initial and adapted pairwise validation accuracy and the validation count.

## 5. Run GRPO with four rewards

After SFT completes, submit the first three jobs. Submit the adapted
run once the adapted reward also completes.

```bash
sbatch formal.sbatch train-grpo baseline
sbatch formal.sbatch train-grpo arena
sbatch formal.sbatch train-grpo mechanics
sbatch formal.sbatch train-grpo adapted
```

Each job uses the same candidate prompts and SFT adapter for 200 steps
at seed 42. It writes `completed.json` and `final_adapter/` under
`results/formal_v2/${MEMBER_ID}_qwen_grpo_<reward>_seed42/`.

## 6. Generate the six held-out runs

Submit `base` when the Qwen model and test CSV are ready. Submit each
other variant when its SFT or GRPO adapter is ready.

```bash
sbatch formal.sbatch evaluate base
sbatch formal.sbatch evaluate sft_pooled
sbatch formal.sbatch evaluate grpo_baseline
sbatch formal.sbatch evaluate grpo_original_arena
sbatch formal.sbatch evaluate grpo_mechanics
sbatch formal.sbatch evaluate grpo_adapted
```

The launcher uses the same test CSV, prompt, seed, and candidate count
for all six runs. Each variant writes `completed.json` and
`candidates.csv` under
`results/formal_v2/evaluation/${MEMBER_ID}/<variant>/`. The training
reward name `arena` corresponds to evaluation variant
`grpo_original_arena`.

## 7. Review the held-out ads

### Six-model scorecard

After all six evaluation runs complete:

```bash
eval_dir="results/formal_v2/evaluation/${MEMBER_ID}"
.venv/bin/python scripts/prepare_six_model_review.py \
  --evaluation-dir "$eval_dir" \
  --output "$eval_dir/blind_six_models" \
  --seed 42 --batch-size 20
```

Give the AI the `blind_batch_XX.jsonl` files from
`$eval_dir/blind_six_models/reviewer/` with the
[six-model scorecard prompt](formal-review-prompts.md#six-model-held-out-scorecard).
Save each response as `reviewed_batch_XX.jsonl`, using the input batch
number. Keep `$eval_dir/blind_six_models/PRIVATE_group_key.jsonl` on
SuperPOD for decoding the model slots.

### Original Arena versus adapted Arena

After `grpo_original_arena` and `grpo_adapted` evaluation complete:

```bash
bash formal.sbatch prepare-final-pairs
```

The default output directory is
`results/formal_v2/evaluation/${MEMBER_ID}/blind_grpo_original_arena_vs_grpo_adapted/`.
Give the AI only `blind_pairs.jsonl` and the
[final held-out evaluation prompt](formal-review-prompts.md#final-held-out-evaluation).
Save its response as `final_reviewed_pairs.jsonl` in that directory and
keep `PRIVATE_pair_key.jsonl` on SuperPOD for decoding. Use the final
reviews for evaluation only.

## Reusing outputs and resuming jobs

Keep `MEMBER_ID` and any path overrides consistent across dependent jobs.
`REWARD_DATA` points to the directory with `train.jsonl` and
`validation.jsonl`. `REWARD_MODEL` points to the reward run directory
containing `completed.json` and `best_model/`.

For an interrupted generation, training, or evaluation job, use
`RESUME=yes sbatch formal.sbatch <stage> [variant]` with the same settings
and a valid saved run or checkpoint. Reuse completed outputs for
downstream stages. The preparation and review-export commands require
new output directories.

## Team pooling

Cross-member pooling is separate from the six runs above. The pooled data
builder is still pending.
