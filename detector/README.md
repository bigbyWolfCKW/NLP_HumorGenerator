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

Obtain each pretrained backbone and tokenizer and put them at the
corresponding path. Record the source revision used for each run.

| Backbone source | Directory |
| --- | --- |
| [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) | `models/detector/models/roberta-base/` |
| [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | `models/detector/models/deberta-v3-base/` |
| [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large) | `models/detector/models/roberta-large/` |

**First run.** Submit one job from the repository root. The job requests one GPU.

```bash
sbatch scripts/detector.sbatch prepare
```

The job downloads the Amazon PQA CSVs to `data/source/raw/amazon_pqa/`,
creates product-disjoint train, validation, and test splits under
`data/source/processed/unbiased/`, and runs EDA. Check
`detector-<job-id>.log` for label counts, product overlap, and the number of
question–description pairs over 128 tokens for each of the three tokenizers.
Wait for the job to complete successfully before submitting detector training.

**Existing data.** Use the same command. Existing splits go straight to EDA.
If the splits have yet to be created, the job reuses readable raw CSVs and
downloads missing or invalid ones before splitting. EDA checks the required
columns and fails on product overlap.

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

Check that the trained RoBERTa-large checkpoint can be found, then evaluate
it with a held-out CSV at `data/source/processed/unbiased/test.csv` or use
`--csv` to pass its location:

```bash
.venv/bin/python -m detector rank --detector roberta-large --seed 42 --check
sbatch scripts/detector.sbatch evaluate \
  --detector roberta-large --seed 42 --output results/detector/roberta_large_seed42_test
```

It writes `metrics.json` and `predictions.csv`, with accuracy, macro-F1, the confusion matrix, input hash and resolved configuration. Select models using validation; use the held-out test for reporting rather than repeated tuning. Candidate ranking uses unlabelled generated sentences and is a separate command.

Model selection
In our experiments, RoBERTa-large achieved the best performance. Since the three models produced very similar scores, we selected the highest-performing model, RoBERTa-large, to ensure the reproducibility of our results.
