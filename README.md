# ProductPunch

ProductPunch trains Qwen to write humorous English ads from product facts.
Each member uses an assigned set of products and evaluates six model
variants on the same held-out products.

## Start on SuperPOD

```bash
git clone https://github.com/bigbyWolfCKW/NLP_HumorGenerator.git
cd NLP_HumorGenerator
source setup.sh
module load slurm
export MEMBER_ID=member_1
```

`setup.sh` installs uv automatically if it is missing.
Set `MEMBER_ID` to your assigned member ID.
Check the Slurm account in `formal.sbatch` against your allocation.

### One-Command Automated Reproduction (For Grading / Replication)

The manual commands later in this README are unchanged. Finish the
preparation below before `bash run_auto_pipeline.sh`. The script then
submits each following stage itself and waits for Slurm; you do not start
the next stage by hand. It does not train the three shared detector
checkpoints. If those are still missing when GRPO starts, the run stops
there.

1. Train the shared detector checkpoints once, from the repository root.
   Wait until each job finishes and each `best_model/config.json` exists.
   These three paths are shared by every member:

   - `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model`
   - `models/detector/preferences/arena_v1_seed42/best_model`
   - `models/detector/preferences/mechanics_v1_seed42/best_model`

   Train the RoBERTa-large QProduct checkpoint first, using the
   [Detector README](detector/README.md#prepare-and-inspect-the-qproduct-inputs).
   Then submit the two preference jobs and wait for both to complete:

   ```bash
   sbatch scripts/humor_preference_v1.sbatch --source arena
   sbatch scripts/humor_preference_v1.sbatch --source mechanics
   ```

   Details are in [Train the detector and humor preference model](#train-the-detector-and-humor-preference-model).

2. Copy the environment file and edit it before starting the pipeline.

   ```bash
   cp .env.example .env
   ```

   Set `MEMBER_ID` (`member_1` through `member_4`, or `all`) and `MODEL`
   (`qwen`, `mistral`, or `deepseek`). For a real judge, set
   `REVIEW_API_BASE_URL`, `REVIEW_API_KEY`, and `REVIEW_MODEL` to any
   OpenAI-compatible endpoint (OpenAI, Gemini, Grok, or HKUST GenAI).
   The HKUST base URL is `https://hkust.azure-api.net/hkust-genai/v1`.
   For an offline check with no API key, set `REVIEW_FALLBACK_MODE=heuristic`.
   Use `REVIEW_FALLBACK_MODE=api` for the graded judge.

3. Keep this terminal session alive for the whole run. The script waits
   in the foreground, checks Slurm every 30 seconds, and submits the next
   stage when the current one finishes. Closing the SSH session or pressing
   Ctrl-C stops later submissions. Jobs already in Slurm keep running.
   A `tmux` session survives a disconnect:

   ```bash
   tmux new -s productpunch
   bash run_auto_pipeline.sh
   ```

   Detach with Ctrl-b, then d. Reattach with `tmux attach -t productpunch`.
   If the script stops, run the same command again. A stage whose output
   already exists is skipped, including reviews already saved.

After those three steps, this is the only pipeline command:

```bash
bash run_auto_pipeline.sh
```

The script follows `docs/formal-workflow.md`. It prepares the Amazon PQA
assignments when `data/formal_v1/` is missing, downloads the base LLM when
that model is missing, and trains this member's Adapted Arena reward model.
It generates ads, runs the four AI reviews, trains SFT, runs four GRPO
jobs, evaluates six variants, writes the six-model scorecard, and, unless
`SKIP_FINAL_PAIRS=yes`, writes the final Arena comparison. When the
scorecard is finished it prints `six_model_summary.csv`.

```bash
bash run_auto_pipeline.sh --dry-run
bash run_auto_pipeline.sh --from-stage evaluate
```

`--dry-run` prints the plan and does not submit jobs. `--from-stage` starts
at a later stage and still skips outputs that already exist. Heuristic
reviews are a deterministic stand-in for an offline check.

### Prepare data in one Slurm job

Submit from the repository root after setup. The launcher requests one GPU
to meet the SuperPOD partition's Slurm requirements.

Prefix the submission with `MEMBER_ID=member_1` as shown below. This sets
the member for this job only. Initial preparation creates all four member
assignments, including when the later training uses `MEMBER_ID=all`.
For a reviewed facts CSV, set the prefix to the member who owns its products.

| Use case | Command |
| --- | --- |
| Start from a fresh checkout | `MEMBER_ID=member_1 sbatch scripts/prepare_formal_data.sbatch` |
| Include a reviewed facts CSV for your member | `MEMBER_ID=member_1 sbatch scripts/prepare_formal_data.sbatch /path/to/member_1_reviewed_facts.csv` |

| Step | What the job writes |
| --- | --- |
| Download Amazon PQA and split by product | Original CSVs in `data/source/raw/amazon_pqa/` and train, validation, and test CSVs in `data/source/processed/unbiased/` |
| Run EDA | Label counts, text lengths, and product overlap in `formal-data-<job-id>.log` |
| Assign products to all four members | `data/formal_v1/assignment_core.json` and each member's train, validation, and test CSVs |
| Import reviewed facts when a CSV is supplied | `data/formal_v1/enrichment/${MEMBER_ID}_train.csv` |

For a new assignment:

1. Run the first command.
2. Review facts for the assigned product IDs.
3. Run the second command. It reuses the downloaded splits and assignments.

If you already have reviewed facts for the matching assignment, the second command runs the whole workflow in one job.

The reviewed facts CSV needs these columns:

- `product_id`
- `verified_product_facts`
- `fact_source_urls` (separate multiple URLs in one cell with whitespace)
- `verified_product_name` (optional)

The Slurm job downloads the dataset on the allocated compute node.

### Download your base LLM

To reproduce the original experiments, use `qwen` (Qwen2.5-7B-Instruct).
The example below downloads Qwen on a GPU compute allocation.
For experiments with another LLM, set `MODEL=mistral` or `MODEL=deepseek`.
Model keys and paths are listed in [`configs/llms.json`](configs/llms.json).

```bash
MODEL=qwen
module load slurm
srun --account=mscbdt2024 --partition=normal \
  --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=02:00:00 \
  .venv/bin/python scripts/download_models.py --models "$MODEL"
```

Wait for `Download complete.` or `Already available locally; skipped.`
before continuing. Complete local models are reused.

When using the default Qwen model, omit `--llm` from the Slurm commands.
If you use a different base model, append `--llm "$MODEL"` to each stage in
the [formal workflow](docs/formal-workflow.md#choose-a-base-llm), for example
`sbatch formal.sbatch generate-ads --llm "$MODEL"`.

To run one pooled experiment across all four assignments, use
`MEMBER_ID=all`. The [pooled workflow](docs/formal-workflow.md#run-one-pooled-experiment)
covers generation, review, training, and evaluation for that scope.
The reward-training and GRPO checkpoint paths are listed by stage in
the [full workflow](docs/formal-workflow.md#1-set-up-a-fresh-checkout).

## Train the detector and humor preference model

Do this before `bash run_auto_pipeline.sh`. The one-command script checks
the three `best_model` directories below and does not train them. A missing
checkpoint stops the run at GRPO.

1. Train and compare RoBERTa-base, DeBERTa-v3-base, and RoBERTa-large on
   the same QProduct splits using the
   [Detector README](detector/README.md#prepare-and-inspect-the-qproduct-inputs).
   Compare the best checkpoint from each model by validation macro-F1.
   For the downstream ProductPunch runs, use the RoBERTa-large QProduct
   checkpoint at `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model/`.
2. **First run.** From the repository root on SuperPOD, submit the source
   you want to train. Each job requests one GPU.

   ```bash
   sbatch scripts/humor_preference_v1.sbatch --source arena
   sbatch scripts/humor_preference_v1.sbatch --source mechanics
   ```

   The job fetches [H.Arena votes](https://github.com/SaveTheRbtz/humor/blob/main/data/humor_arena.choices.20250812.json)
   or [Humor Mechanics ratings](https://github.com/altsoph/humor-mechanics),
   prepares train and validation pairs, and trains from the QProduct
   RoBERTa-large checkpoint. The selected checkpoint is saved to
   `models/detector/preferences/arena_v1_seed42/best_model/` or
   `models/detector/preferences/mechanics_v1_seed42/best_model/`.

   For Adapted Arena, continue with the
   [member-specific reward workflow](docs/formal-workflow.md#3-prepare-the-reviewed-training-data).

## Train SFT

New Qwen runs write to
`results/formal_v2/${MEMBER_ID}_qwen_product_humor_v3/`.
For another model, replace `qwen` with its key.

`formal_v2` names the output layout, while `product_humor_v3` names
the prompt version. Both names stay the same when you rerun.

| Step | Run or save | Wait for this output |
| --- | --- | --- |
| Generate training ads | `sbatch formal.sbatch generate-ads` | `$CANDIDATE_RUN/candidates.csv` and `$CANDIDATE_RUN/completed.json` |
| Review candidates | Use the [SFT review prompt](docs/formal-review-prompts.md#sft-example-selection). Save the response as `$CANDIDATE_RUN/reviewed_candidates.csv`. | Reviewed CSV with the original candidate IDs |
| Export accepted examples | `bash formal.sbatch prepare-sft` | `data/formal_v2/sft/${MEMBER_ID}/reviewed_product_ads_train.jsonl` |
| Train the adapter | `sbatch formal.sbatch train-sft` | `results/formal_v2/${MEMBER_ID}_qwen_sft_pooled/final_adapter/` and the run's `completed.json` |

If `train-sft` reports `Missing or empty input`, check that the JSONL was exported for the selected `MEMBER_ID`. The reviewed CSV is the input to `prepare-sft`, not `train-sft`. `formal.sbatch` fills in the training and output paths automatically.

## Continue to reward training and evaluation

The [full workflow](docs/formal-workflow.md) covers A/B reward pairs,
reward training, four GRPO runs, and the six held-out evaluation runs.
Use the [review prompts](docs/formal-review-prompts.md) for the
corresponding stages.

For [optional GRPO reward plots](docs/formal-workflow.md#optional-grpo-reward-plots),
set `SCOPE` to a member ID, `all`, or `team`,
then submit `sbatch scripts/plot_grpo_rewards.sbatch "$SCOPE"`.

After all six variants finish generating held-out ads, go to
[Step 7: Review the held-out ads](docs/formal-workflow.md#7-review-the-held-out-ads)
to export the six-model blind scorecard, collect the reviewer JSONL files,
and use them with the private key to generate a summary for each model.

```bash
.venv/bin/python scripts/summarize_six_model_review.py \
  --review-dir "results/formal_v2/evaluation/${MEMBER_ID}/blind_six_models"
```

For the optional [Original Arena versus Adapted Arena review](docs/formal-workflow.md#optional-original-arena-versus-adapted-arena),
run `bash formal.sbatch summarize-final-pairs` after collecting the A/B ratings.
The command writes `final_pair_summary.csv` to the pair directory.

The command checks that every group has six ratings and writes
`six_model_summary.csv` in the review directory. Step 7 also covers the
optional Original Arena versus Adapted Arena A/B comparison.

## Reuse existing data or resume a job

The [reuse and resume guide](docs/formal-workflow.md#reusing-outputs-and-resuming-jobs)
lists where to place existing assignments, reviewed SFT data, candidates,
checkpoints, and review batches, and which step to continue from.

For preference training, the same job reuses valid pair files or prepares
them from existing raw data. If the source model has already completed
training, use its checkpoint for the downstream stages.

See the [archived README](README.legacy.md) for the earlier
`test.sbatch` workflow.
