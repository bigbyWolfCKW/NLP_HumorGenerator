# ProductPunch formal workflow

Run all commands from the repository root on SuperPOD. Use
`sbatch scripts/prepare_formal_data.sbatch` for initial data preparation,
`bash formal.sbatch` for local export steps, and `sbatch formal.sbatch`
for jobs that load or train a model.

## 1. Set up a fresh checkout

```bash
git clone https://github.com/bigbyWolfCKW/NLP_HumorGenerator.git
cd NLP_HumorGenerator
source setup.sh
module load slurm
export MEMBER_ID=member_4
```

Replace `member_4` with your assigned ID (`member_1` through `member_4`).
Set it again in each new terminal. The launcher defaults to `member_1`
when the variable is unset. Check that the Slurm account in both
`scripts/prepare_formal_data.sbatch` and `formal.sbatch` matches your
allocation. `setup.sh` creates the project's `.venv`.

### Prepare a new assignment

Submit one Slurm job to download Amazon PQA, create a deterministic
60/20/20 split by normalized product description, run detector EDA, and
build the four member assignments. The launcher requests one GPU to meet
the SuperPOD partition's Slurm requirements.
[Amazon PQA dataset and license](https://registry.opendata.aws/humor-detection/)

```bash
sbatch scripts/prepare_formal_data.sbatch
```

| Output | Location |
| --- | --- |
| Original Amazon PQA CSVs | `data/source/raw/amazon_pqa/` |
| Train, validation, and test CSVs and `split_manifest.json` | `data/source/processed/unbiased/` |
| EDA label counts, text lengths, and product overlap | `formal-data-<job-id>.log` |
| Shared assignment key and member train, validation, and test CSVs | `data/formal_v1/` |

If compute nodes cannot reach the dataset bucket, run
`.venv/bin/python scripts/download_formal_pqa.py` on the login node
before submitting the job. To build assignments from an existing processed
train CSV, run `.venv/bin/python scripts/prepare_formal_source.py --input /path/to/train.csv`
and then `.venv/bin/python scripts/prepare_formal_product_list_v1.py`.
EDA requires the matching validation CSV and also requires a test CSV
when run with `--include-test`. When reusing historical SFT examples,
pass their original split JSONLs to the assignment script with repeated
`--exclude-jsonl` arguments.

### Reproduce an existing assignment

Place the matching team files under `data/formal_v1/`. Substitute
your assigned ID in the CSV filenames.

| File under `data/formal_v1/` | Used by |
| --- | --- |
| `assignment_core.json` | Generation, SFT data export, GRPO and held-out evaluation |
| `${MEMBER_ID}_train.csv` | Training candidate generation |
| `${MEMBER_ID}_test.csv` | Held-out evaluation |

Check the uploaded files from the checkout root.

```bash
ls -lh data/formal_v1/assignment_core.json
ls -lh "data/formal_v1/${MEMBER_ID}_train.csv" "data/formal_v1/${MEMBER_ID}_test.csv"
```

The launcher uses 100 training products, 50 test products, four ads per
product and seed 42 by default. If the assigned CSVs contain fewer
eligible products, set `TRAIN_PRODUCTS` and `TEST_PRODUCTS` in the
job environment and the `formal.sbatch` settings block, respectively.
Keep the test settings identical for the six evaluation runs.

The GPU jobs load models offline. Download or train each model before
its first use and make its directory readable on the compute node.

| Model | How to obtain it | Directory relative to the checkout | Needed for |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | Download with `.venv/bin/python scripts/download_models.py --models qwen`. | `models/llm/Qwen2.5-7B-Instruct/` | Candidate generation, SFT, GRPO and evaluation |
| Binary detector | Fine-tune RoBERTa-large on the QProduct splits and export the best validation checkpoint. See the [Detector README](../detector/README.md#prepare-and-inspect-the-qproduct-inputs). | `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model/` | Baseline GRPO |
| Original Arena | Start from the binary detector and train on H.Arena human-voted joke pairs. See [humor preference training](../README.md#train-the-detector-and-humor-preference-model). | `models/detector/preferences/arena_v1_seed42/best_model/` | Reward adaptation and original Arena GRPO |
| Humor Mechanics | Start from the same binary detector and train on pairs derived from Humor Mechanics funniness ratings. Use the `mechanics` command in [humor preference training](../README.md#train-the-detector-and-humor-preference-model). | `models/detector/preferences/mechanics_v1_seed42/best_model/` | Mechanics GRPO |

## 2. Reuse existing training data

| Existing data | Where to put it | What to set |
| --- | --- | --- |
| Member 1 archived facts input | `data/formal_v1/member_1_train_first100_facts_v3.csv` | Use the matching original assignment when reproducing the archived run. |
| Members 2–4 archived facts-ready inputs | `data/formal_v1/enrichment/teammates_facts_full_v2_20261108_083357_977027/${MEMBER_ID}_train_facts_full_v2_ready.csv` | Use the original teammate generator for reproduction. These CSVs replace `product_facts`, which the current launcher checks against the assignment. |
| Training CSV from the current facts import | `data/formal_v1/enrichment/${MEMBER_ID}_train.csv` | Set `TRAIN_CSV` to this path and use the matching assignment. |
| Completed candidate run | Keep `candidates.csv`, `completed.json`, and any matching `reviewed_candidates.csv` together in the run directory | Set `CANDIDATE_RUN` to that directory and continue at Step 3. For the archived Member 1 run, use `results/formal_v1/member_1_qwen_product_only_facts_v3/`. |

## 3. Prepare the reviewed training data

### SFT examples

Give the AI the [SFT selection prompt](formal-review-prompts.md#sft-example-selection) and a view of `$CANDIDATE_RUN/candidates.csv` with these columns.

- `candidate_id` (keep intact)
- `ad_text`
- `verified_product_facts` if present, otherwise `product_facts`
- `target_audience`

Save the returned CSV as `$CANDIDATE_RUN/reviewed_candidates.csv`.

```bash
bash formal.sbatch prepare-sft
```

The launcher expects
`data/formal_v2/sft/${MEMBER_ID}/reviewed_product_ads_train.jsonl`.

- Received a ready-to-train JSONL? Place it at this path and go to Step 4.
- Received only the reviewed CSV? Put it next to `candidates.csv` in `$CANDIDATE_RUN` and run the export above.

### Reward preferences

```bash
bash formal.sbatch prepare-humor-pairs
```

Use the files under `data/formal_v2/reward/${MEMBER_ID}/humor_pairwise/`.

- Give the AI only `blind_pairs.jsonl` and the [reward-label prompt](formal-review-prompts.md#reward-training-labels).
- Save its response as `reviewed_pairs.jsonl` in the same directory.
- Keep `PRIVATE_pair_key.jsonl` beside the blind file on SuperPOD. Do not share it with the AI.

```bash
bash formal.sbatch prepare-reward-data
```

The importer keeps decisive humor preferences where both ads pass and
keeps each product family in one split. It writes `train.jsonl` and
`validation.jsonl` under
`data/formal_v2/reward/${MEMBER_ID}/reward_data_review/`.

## 4. Train SFT and the adapted reward

SFT uses the Qwen base and the exported SFT JSONL. Before submitting, check that the training file exists and is nonempty.

```bash
sft_file="data/formal_v2/sft/${MEMBER_ID}/reviewed_product_ads_train.jsonl"
if [[ -s "$sft_file" ]]; then
  sbatch formal.sbatch train-sft
else
  printf 'Missing or empty input: %s\n' "$sft_file" >&2
fi
```

If the check fails, verify `MEMBER_ID` and redo the SFT export or handoff in Step 3. Changing `CANDIDATE_RUN` only changes the source candidate folder. The SFT job always reads the JSONL path above.

Reward training can run in parallel once you have your reward train and validation JSONLs and the original Arena checkpoint:

```bash
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

Start here when Step 6 has written `run.json`, `completed.json`, and
`candidates.csv` for all six variants. Use the same `MEMBER_ID` as in
Step 6. Start with the scorecard to compare all six models. The separate
A/B review is for a head-to-head comparison of the two Arena variants.

### Six-model scorecard

Export the blind review batches from the repository root.

```bash
eval_dir="results/formal_v2/evaluation/${MEMBER_ID}"
.venv/bin/python scripts/prepare_six_model_review.py \
  --evaluation-dir "$eval_dir" \
  --output "$eval_dir/blind_six_models" \
  --seed 42 --batch-size 20
```

With 50 products and four candidates per product, the export has 200 matched
groups in ten batches. Each group shows one candidate from each variant
in shuffled slots A–F. If you already exported this run, reuse its
batches and key.

1. Give the reviewer one `blind_batch_XX.jsonl` at a time from
   `$eval_dir/blind_six_models/reviewer/` with the
   [six-model scorecard prompt](formal-review-prompts.md#six-model-held-out-scorecard).
2. Save each response next to its input as `reviewed_batch_XX.jsonl`
   with the same batch number. Each input `group_id` needs one review
   with ratings for all six slots.
3. Keep `$eval_dir/blind_six_models/PRIVATE_group_key.jsonl` on SuperPOD.
   After collecting the reviews, run the summary from the repo root.

   ```bash
   .venv/bin/python scripts/summarize_six_model_review.py \
     --review-dir "$eval_dir/blind_six_models"
   ```

The script uses `--review-dir` to locate the private key and reads the
reviewed batches from the `reviewer/` subdirectory. The shell expands
`$MEMBER_ID` when setting `eval_dir` above. To summarize Member 1 directly,
run the following command, including when you open a new terminal.

```bash
.venv/bin/python scripts/summarize_six_model_review.py \
  --review-dir results/formal_v2/evaluation/member_1/blind_six_models
```

The script matches each review to the key by `group_id` and slot letter.
It writes `six_model_summary.csv` in the review directory with PASS,
FAIL, and UNCLEAR counts, PASS percentage, and the three mean scores for
each model. It stops if a group is missing or duplicated, or a rating is
invalid. Use `--reviews-dir` for batches stored outside `reviewer/` and
`--output` to write to a different new CSV path.

### Optional Original Arena versus Adapted Arena

Run this command after the `grpo_original_arena` and `grpo_adapted` evaluations finish.

```bash
bash formal.sbatch prepare-final-pairs
```

The default output directory is
`results/formal_v2/evaluation/${MEMBER_ID}/blind_grpo_original_arena_vs_grpo_adapted/`.

1. Give the AI only `blind_pairs.jsonl` and the [final held-out evaluation prompt](formal-review-prompts.md#final-held-out-evaluation).
2. Save the response as `final_reviewed_pairs.jsonl` in the same directory. Check that each `pair_id` appears exactly once.
3. Keep `PRIVATE_pair_key.jsonl` on SuperPOD. Use it to map A/B choices back to model variants.
4. Report wins per model, ties, and neither for each criterion.

Use the final reviews for evaluation only.

## Reusing outputs and resuming jobs

Keep `MEMBER_ID` and any path overrides consistent across dependent jobs.

- `REWARD_DATA`: directory containing `train.jsonl` and `validation.jsonl`.
- `REWARD_MODEL`: reward run directory containing `completed.json` and `best_model/`.

To resume an interrupted generation, training, or evaluation job, rerun it with the same settings and a valid saved run or checkpoint.

```bash
RESUME=yes sbatch formal.sbatch <stage> [variant]
```

For example, resume Member 1's interrupted baseline GRPO run from the
latest checkpoint in `results/formal_v2/member_1_qwen_grpo_baseline_seed42/`.

```bash
MEMBER_ID=member_1 RESUME=yes sbatch formal.sbatch train-grpo baseline
```

Use the same `CANDIDATE_RUN` as in the original submission if you set
a custom path.

Completed outputs can be reused by downstream stages. Preparation and review-export commands need new output directories.

## Team pooling

The team experiment pools 215 reviewed SFT examples and 212 preference pairs (176 train, 36 validation) for reward adaptation. Its four GRPO variants share the pooled SFT adapter and candidate prompts.

This guide's commands run one assigned partition at a time. To reproduce the pooled experiment, request the pooled datasets and run configuration from the team.
