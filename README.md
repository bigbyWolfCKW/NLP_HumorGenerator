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

Set `MEMBER_ID` to your assigned member ID.
Check the Slurm account in `formal.sbatch` against your allocation.

### Prepare data in one Slurm job

Submit from the repository root after setup. The launcher requests one GPU
to meet the SuperPOD partition's Slurm requirements.

| Use case | Command |
| --- | --- |
| Start from a fresh checkout | `sbatch scripts/prepare_formal_data.sbatch` |
| Include a reviewed facts CSV for your member | `MEMBER_ID=member_1 sbatch scripts/prepare_formal_data.sbatch /path/to/member_1_reviewed_facts.csv` |

| Step | What the job writes |
| --- | --- |
| Download Amazon PQA and split by product | Original CSVs in `data/source/raw/amazon_pqa/` and train, validation, and test CSVs in `data/source/processed/unbiased/` |
| Run EDA | Label counts, text lengths, and product overlap in `formal-data-<job-id>.log` |
| Assign products to all four members | `data/formal_v1/assignment_core.json` and each member's train, validation, and test CSVs |
| Import reviewed facts when a CSV is supplied | `data/formal_v1/enrichment/${MEMBER_ID}_train.csv` |

For a new assignment, run the first command, review facts for the assigned
product IDs, then run the second command. The second run reuses the
downloaded splits and assignments. If you already have reviewed facts for
the matching assignment, the second command runs the whole workflow in one job.
The reviewed facts CSV needs `product_id`, `verified_product_facts`, and
`fact_source_urls`. Add `verified_product_name` if available. Separate
multiple source URLs in a cell with whitespace.

If the compute node cannot reach the dataset bucket, run
`.venv/bin/python scripts/download_formal_pqa.py` on the login node first.

To reproduce an existing run, use the matching files from the team data
handoff under `data/formal_v1/`:

- `assignment_core.json`
- `${MEMBER_ID}_train.csv`
- `${MEMBER_ID}_test.csv`

Download Qwen2.5-7B-Instruct before generation, SFT, GRPO, or evaluation.
The download command is in [Step 1 of the full workflow](docs/formal-workflow.md#1-set-up-a-fresh-checkout).
The reward-training and GRPO checkpoint paths are listed by stage in
the [full workflow](docs/formal-workflow.md#1-set-up-a-fresh-checkout).

## Train the detector and humor preference model

1. Train and compare RoBERTa-base, DeBERTa-v3-base, and RoBERTa-large on
   the same QProduct splits using the
   [Detector README](detector/README.md#prepare-and-inspect-the-qproduct-inputs).
   Compare the best checkpoint from each model by validation macro-F1.
   For the downstream ProductPunch runs, use the RoBERTa-large QProduct
   checkpoint at `models/detector/checkpoints/11_roberta_large/qproduct_seed42/best_model/`.
2. Prepare the preference datasets under
   `data/source/preferences/humor_preference_v1/`. Download the
   [H.Arena votes](https://github.com/SaveTheRbtz/humor/blob/main/data/humor_arena.choices.20250812.json)
   as `h_arena.json`. Place the six `jokes-*.json` rating files from
   [Humor Mechanics](https://github.com/altsoph/humor-mechanics) in the
   `humor-mechanics/` subdirectory. From the repository root, run:

   ```bash
   .venv/bin/python data/source/preferences/humor_preference_v1/prepare_preferences.py
   ```
  It writes `train.jsonl` and `validation.jsonl` in
   `pairwise/arena/` and `pairwise/mechanics/`.
    The preparation script requires a new `pairwise/` directory. It splits
   the H.Arena votes by session and removes joke overlap across splits.

   For Humor Mechanics it splits the jokes before deriving pairs from
   their ratings.

3. Train each reward model starting from the QProduct RoBERTa-large
   checkpoint:

   ```bash
   sbatch scripts/humor_preference_v1.sbatch \
     --source arena --seed 42 --steps 300 --batch-size 8 --learning-rate 1e-5
   sbatch scripts/humor_preference_v1.sbatch \
     --source mechanics --seed 42 --steps 300 --batch-size 8 --learning-rate 1e-5
   ```

   Training selects the checkpoint with the lowest validation loss and
   exports `models/detector/preferences/arena_v1_seed42/best_model/` and
   `models/detector/preferences/mechanics_v1_seed42/best_model/` for the
   respective jobs. For Adapted Arena,
   continue with [reward data preparation and training](docs/formal-workflow.md#3-prepare-the-reviewed-training-data)
   using your member's reviewed pairs.

## Train SFT

New runs write to `results/formal_v2/${MEMBER_ID}_qwen_product_humor_v3/`
by default. Set `CANDIDATE_RUN` to that directory or to the directory
for your [existing candidate run](docs/formal-workflow.md#2-reuse-existing-training-data).

| Step | Run or save | Wait for this output |
| --- | --- | --- |
| Generate training ads | `sbatch formal.sbatch generate-ads` | `$CANDIDATE_RUN/candidates.csv` and `$CANDIDATE_RUN/completed.json` |
| Review candidates | Use the [SFT review prompt](docs/formal-review-prompts.md#sft-example-selection). Save the response as `$CANDIDATE_RUN/reviewed_candidates.csv`. | Reviewed CSV with the original candidate IDs |
| Export accepted examples | `bash formal.sbatch prepare-sft` | `data/formal_v2/sft/${MEMBER_ID}/reviewed_product_ads_train.jsonl` |
| Train the adapter | `sbatch formal.sbatch train-sft` | `results/formal_v2/${MEMBER_ID}_qwen_sft_pooled/final_adapter/` and the run's `completed.json` |

If `train-sft` reports `Missing or empty input`, check that the JSONL
was exported for the selected member. The reviewed CSV is the input
to `prepare-sft`. Use `formal.sbatch` to fill in the training and
output paths automatically.

## Continue to reward training and evaluation

The [full workflow](docs/formal-workflow.md) covers A/B reward pairs,
reward training, four GRPO runs, and the six held-out evaluation runs.
Use the [review prompts](docs/formal-review-prompts.md) for the
corresponding stages.

After all six variants finish generating held-out ads, go to
[Step 7: Review the held-out ads](docs/formal-workflow.md#7-review-the-held-out-ads)
to export the six-model blind scorecard, collect the reviewer JSONL files,
and use them with the private key to generate a summary for each model.

```bash
.venv/bin/python scripts/summarize_six_model_review.py \
  --review-dir "results/formal_v2/evaluation/${MEMBER_ID}/blind_six_models"
```

The command checks that every group has six ratings and writes
`six_model_summary.csv` in the review directory. Step 7 also covers the
optional Original Arena versus Adapted Arena A/B comparison.

See the [archived README](README.legacy.md) for the earlier
`test.sbatch` workflow.
