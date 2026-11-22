# Detector

## Detector code

The detector code is separate from LLM generation and LoRA training. It trains the best-performing binary humor/non-humor detector, and it shares the same `uv` environment and model/data folders as the rest of the project.

```text
detector/
├── __main__.py           # python -m detector
├── inference.py          # load checkpoints and score candidate sentences
├── models/              # architecture code, not model weights
│   └── roberta_sub.py
├── data.py               # labelled QA CSV input and binary metrics
├── eda.py                # QProduct split and tokenizer checks
├── training.py           # standard HF classifier training
├── evaluation.py         # labelled QA evaluation
└── experiments/          # named comparison experiments
```

## Prepare and inspect the QProduct inputs

Run from the repository root on SuperPOD:

```bash
source setup.sh
module load slurm
```

The preparation job downloads these three pretrained backbones and their
tokenizers to the configured paths. Complete local models are reused.
For each new download, `download_manifest.json` records the source repository
and resolved revision.

| Backbone source | Directory |
| --- | --- |
| [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) | `models/detector/models/roberta-base/` |
| [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | `models/detector/models/deberta-v3-base/` |
| [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) | `models/detector/models/roberta-large/` |

**First run.** Submit one job from the repository root. The job requests one GPU.

```bash
sbatch scripts/detector.sbatch prepare
```

The job downloads the three backbones to the paths above. It also downloads
the Amazon PQA CSVs to `data/source/raw/amazon_pqa/` and creates
product-disjoint train, validation, and test splits under
`data/source/processed/unbiased/`, and runs EDA. Check
`detector-<job-id>.log` for label counts, product overlap, and the number of
question–description pairs over 128 tokens for each of the three tokenizers.
Wait for the job to complete successfully before submitting detector training.

To download only the backbones on a GPU compute allocation, run:

```bash
srun --account=mscbdt2024 --partition=normal \
  --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=02:00:00 \
  .venv/bin/python scripts/download_detector_models.py
```

Use `--models roberta`, `--models deberta`, or `--models roberta-large`
to select one backbone. Repeating the same command skips complete models
and retries incomplete downloads.

**Existing data.** Use the same command. Existing splits go straight to EDA.
If a split is missing, empty, or invalid, the job reuses readable raw CSVs
and downloads missing or invalid raw files before rebuilding the splits.
It checks that surviving splits match the rebuilt data before filling in
the missing files. If they differ, it stops without changing the splits.
EDA checks the required columns and fails on product overlap.

## Compare the QProduct Transformer detectors

After preparation completes, submit one job per detector. The batch script
uses the same QProduct splits and training settings for all three jobs and
sets each model's checkpoint directory. Each output directory must be new.

```bash
sbatch scripts/detector.sbatch train-qproduct roberta
sbatch scripts/detector.sbatch train-qproduct deberta
sbatch scripts/detector.sbatch train-qproduct roberta-large
```

Under `models/detector/checkpoints/`, the jobs write to
`roberta_qproduct_seed42/`, `10_deberta_v3_base/qproduct_seed42/`, and
`11_roberta_large/qproduct_seed42/`, respectively. Each run saves its
highest-validation-macro-F1 epoch under `best_model/`,
including `model.safetensors`, the model config, and the tokenizer. Compare
`validation.macro_f1` in the three `best_validation.json` files to rank the
models. Baseline GRPO and the
[Original Arena training recipe](../README.md#train-the-detector-and-humor-preference-model)
use the RoBERTa-large output path above. To use a different detector in
those stages, configure its checkpoint path explicitly.

## Check and evaluate the trained detector

Submit one GPU job to check the trained RoBERTa-large checkpoint and evaluate
it on `data/source/processed/unbiased/test.csv`.

```bash
sbatch scripts/detector.sbatch evaluate-qproduct
```

The job writes `metrics.json` and `predictions.csv` under
`results/detector/roberta_large_seed42_test/`. To use a different test CSV
or a new output directory, pass the paths in that order after
`evaluate-qproduct`. Select models on validation and report accuracy and
macro-F1 on the held-out test.

Model selection
In our experiments, RoBERTa-large achieved the best performance. Since the three models produced very similar scores, we selected the highest-performing model, RoBERTa-large, to ensure the reproducibility of our results.
