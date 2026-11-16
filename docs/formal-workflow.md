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

Replace `member_4` with your assigned ID (`member_1` through `member_4`),
or use `all` for a pooled experiment.
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

For an individual run, check the uploaded files from the checkout root.

```bash
ls -lh data/formal_v1/assignment_core.json
ls -lh "data/formal_v1/${MEMBER_ID}_train.csv" "data/formal_v1/${MEMBER_ID}_test.csv"
```

Individual runs use 100 training products, 50 test products, four ads per
product and seed 42 by default. Set `TRAIN_PRODUCTS` or `TEST_PRODUCTS`
in the job environment to use a smaller subset.
Keep the test settings identical for the six evaluation runs.

### Run one pooled experiment

Set `MEMBER_ID=all` on each formal stage to run one experiment using the
four member assignments. On first use, generation and evaluation build
`data/formal_v1/all_train.csv` and `data/formal_v1/all_test.csv` from the
four matching splits. Run paths use `all` in place of the member ID. The
combined CSVs preserve each product's owner and split for assignment checks.
The pooled run uses every product from the selected split by default.

```bash
export MEMBER_ID=all
sbatch formal.sbatch generate-ads
```

For SFT, either review the pooled candidates and run
`bash formal.sbatch prepare-sft`, or combine the four members' approved
training JSONLs manually and save the result as
`data/formal_v2/sft/all/reviewed_product_ads_train.jsonl`.
For another base model, save it under
`data/formal_v2/sft/all/<model>/reviewed_product_ads_train.jsonl`.
Use training samples only. Both routes use the same training command
and write to the same pooled adapter directory for the selected model.

```bash
sbatch formal.sbatch train-sft
```

Keep `MEMBER_ID=all` for GRPO, evaluation, and review summaries.

### Choose a base LLM

`formal.sbatch` defaults to `qwen` (Qwen2.5-7B-Instruct). Set `MODEL` to
any local Transformers model key in [`configs/llms.json`](../configs/llms.json)
and download it before submitting GPU jobs. Mistral is the example below.

```bash
MODEL=mistral
.venv/bin/python scripts/download_models.py --models "$MODEL"
```

When using the default Qwen model, run the stage commands as shown
throughout this workflow. If you use a different base model, append
`--llm "$MODEL"` to each stage command. With the example above, the
following commands select Mistral:

```bash
sbatch formal.sbatch generate-ads --llm "$MODEL"
sbatch formal.sbatch train-sft --llm "$MODEL"
sbatch formal.sbatch train-grpo arena --llm "$MODEL"
sbatch formal.sbatch evaluate grpo_original_arena --llm "$MODEL"
```

Pass the same `--llm` for review preparation, reward-data preparation, and
final pair summaries. Use a SFT adapter trained on the selected base model.
For a model other than Qwen, candidate and adapter directories include
its key in their names. The SFT data, reward data, and evaluation directories
have the key as a subdirectory under the member directory.

The GPU jobs load models offline. Download or train each model before
its first use and make its directory readable on the compute node.

| Model | How to obtain it | Directory relative to the checkout | Needed for |
| --- | --- | --- | --- |
| Selected base LLM | Download with `.venv/bin/python scripts/download_models.py --models "$MODEL"`. | The selected model's `path` in [`configs/llms.json`](../configs/llms.json) | Candidate generation, SFT, GRPO and evaluation |
| Binary detector | Fine-tune RoBERTa-large on the QProduct splits and export the best validation checkpoint. See the [Detector README](../detector/README.md#prepare-and-inspect-the-qproduct-inputs). | `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model/` | Baseline GRPO |
| Original Arena | Start from the binary detector and train on H.Arena human-voted joke pairs. See [humor preference training](../README.md#train-the-detector-and-humor-preference-model). | `models/detector/preferences/arena_v1_seed42/best_model/` | Reward adaptation and original Arena GRPO |
| Humor Mechanics | Start from the same binary detector and train on pairs derived from Humor Mechanics funniness ratings. Use the `mechanics` command in [humor preference training](../README.md#train-the-detector-and-humor-preference-model). | `models/detector/preferences/mechanics_v1_seed42/best_model/` | Mechanics GRPO |

## 2. Generate or reuse training candidates

After setting `MEMBER_ID`, submit the candidate generation job from the
repository root. This example uses the default Qwen base:

```bash
export CANDIDATE_RUN="results/formal_v2/${MEMBER_ID}_qwen_product_humor_v3"
sbatch formal.sbatch generate-ads
```

For another base model, use its key in the candidate directory and append
`--llm MODEL` to the submission. Wait for `$CANDIDATE_RUN/completed.json`
and `$CANDIDATE_RUN/candidates.csv` before starting Step 3. Check progress
with `squeue -j JOB_ID` and `humor-formal-JOB_ID.log`.

### Reuse existing training data

| Existing data | Where to put it | What to set |
| --- | --- | --- |
| Training CSV from the current facts import | `data/formal_v1/enrichment/${MEMBER_ID}_train.csv` | Set `TRAIN_CSV` to this path and use the matching assignment. |
| Completed candidate run | Keep `candidates.csv`, `completed.json`, and any matching `reviewed_candidates.csv` together in the run directory | Set `CANDIDATE_RUN` to that directory and continue at Step 3. |

Use the same candidate run for SFT review, reward pairs, and all four
GRPO runs. Re-export `CANDIDATE_RUN` if you start a new shell.

**Pooled run (`MEMBER_ID=all`).** By default, generation combines the four
member training CSVs into `data/formal_v1/all_train.csv`. Set `TRAIN_CSV`
for a custom pooled training CSV or `CANDIDATE_RUN` for a completed pooled
candidate run. Keep the original product IDs, owners, and split labels.

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

**Pooled run.** Run `prepare-sft` on the reviewed pooled candidates, or
combine the four approved training JSONLs and save the result as
`data/formal_v2/sft/all/reviewed_product_ads_train.jsonl` for Qwen.

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

**Pooled run.** Prepare and review the pairs from the pooled candidate
run. For Qwen, the reward datasets go under
`data/formal_v2/reward/all/reward_data_review/`.

## 4. Train SFT and the adapted reward

SFT uses the selected base model and the prepared SFT JSONL. Before
submitting, check that the training file exists and is nonempty.
The example below uses the default Qwen paths.

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

**Pooled run.** Submit both jobs with `MEMBER_ID=all`. For Qwen, the
SFT adapter is saved under `results/formal_v2/all_qwen_sft_pooled/` and
the adapted reward under
`models/detector/preferences/all_product_humor_review/`.

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

**Pooled run.** Submit each reward variant with `MEMBER_ID=all`. The
four jobs use the pooled training prompts and the same pooled SFT
adapter. For Qwen, each output directory starts with `all_qwen_grpo_`.

### Optional GRPO reward plots

After the four GRPO runs finish, submit a plot job from the repository root.
Install Matplotlib once if it is missing from the project environment.

```bash
uv pip install --python .venv/bin/python matplotlib
```

Set `SCOPE` to a member ID, `all` for the pooled workflow, or `team` for
the earlier pooled runs.

```bash
SCOPE=member_2
sbatch scripts/plot_grpo_rewards.sbatch "$SCOPE"
```

The job reads the four `trainer_state.json`
files and writes `results/formal_v2/plots/<scope>_grpo_reward_traces_seed42.png`.
The figure shows mean training reward and within-group reward standard
deviation using a trailing 10-step mean. Each curve uses its own reward model.
For another base LLM, append its key, for example
`sbatch scripts/plot_grpo_rewards.sbatch member_2 --llm mistral`.

## 6. Generate the six held-out runs

Submit `base` when the selected model and test inputs are ready. Submit each
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

**Pooled run.** Submit all six variants with `MEMBER_ID=all`. Evaluation
combines the four held-out CSVs into `data/formal_v1/all_test.csv`.
With the default Qwen base, outputs are under
`results/formal_v2/evaluation/all/`.

## 7. Review the held-out ads

Start here when Step 6 has written `run.json`, `completed.json`, and
`candidates.csv` for all six variants. Use the same `MEMBER_ID` as in
Step 6. Start with the scorecard to compare all six models. The separate
A/B review is for a head-to-head comparison of the two Arena variants.

**Pooled run.** Keep `MEMBER_ID=all` for A/B export and summary. For the
six-model scorecard, set `eval_dir` to
`results/formal_v2/evaluation/all` for Qwen, or append the selected model
key for another base model.

### Six-model scorecard

Export the blind review batches from the repository root.
For a model selected with `--llm`, append the model key to `eval_dir` below
when it differs from the default Qwen. For example, use
`results/formal_v2/evaluation/${MEMBER_ID}/mistral`.

```bash
eval_dir="results/formal_v2/evaluation/${MEMBER_ID}"
.venv/bin/python scripts/prepare_six_model_review.py \
  --evaluation-dir "$eval_dir" \
  --output "$eval_dir/blind_six_models" \
  --seed 42 --batch-size 20
```

With 50 products and four candidates per product, the export has 200 matched
groups in ten batches. With all 200 test products, it has 800 groups in
40 batches. Each group shows one candidate from each variant
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
reviewed batches from the `reviewer/` subdirectory. Set `eval_dir` again
when opening a new terminal.

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
3. Keep `PRIVATE_pair_key.jsonl` on SuperPOD and run the summary with the same `MEMBER_ID`.

```bash
bash formal.sbatch summarize-final-pairs
```

The command writes `final_pair_summary.csv` in the same directory. It reports
compliance by model, wins, ties, and neither for each criterion, and humor
results for pairs where both ads pass compliance.

Use the final reviews for evaluation only.

## Reusing outputs and resuming jobs

Keep `MEMBER_ID` and any path overrides consistent across dependent jobs.

- `REWARD_DATA`: directory containing `train.jsonl` and `validation.jsonl`.
- `REWARD_MODEL`: reward run directory containing `completed.json` and `best_model/`.

To resume an interrupted generation, training, or evaluation job, rerun it with the same settings and a valid saved run or checkpoint.

```bash
RESUME=yes sbatch formal.sbatch <stage> [variant]
```

For example, resume the pooled baseline GRPO run from the latest
checkpoint in `results/formal_v2/all_qwen_grpo_baseline_seed42/`.

```bash
MEMBER_ID=all RESUME=yes sbatch formal.sbatch train-grpo baseline
```

Use the same `CANDIDATE_RUN` as in the original submission if you set
a custom path.

Completed outputs can be reused by downstream stages. Preparation and review-export commands need new output directories.

## Team pooling

The team experiment pools 215 reviewed SFT examples and 212 preference pairs (176 train, 36 validation) for reward adaptation. Its four GRPO variants share the pooled SFT adapter and candidate prompts.

This guide's commands run one assigned partition at a time. To reproduce the pooled experiment, request the pooled datasets and run configuration from the team.
