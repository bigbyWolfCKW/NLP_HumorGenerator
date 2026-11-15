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

Place the shared QProduct splits in `data/source/processed/unbiased/` as
`train.csv` and `validation.csv`.
Each CSV needs `question`, `product_description`, `product_group`, and
`label` (0 or 1). Each model receives the question and description as paired
input.

```bash
.venv/bin/python -m detector eda \
  --detector roberta --detector deberta --detector roberta-large --max-length 128
```

Review the label counts and product groups. A product appearing in both
splits causes the check to fail. The tokenizer checks verify the local
weight files, load each tokenizer, and show how many question–description
pairs exceed the training limit of 128 tokens. Run
`.venv/bin/python -m detector eda` to inspect just the CSVs while preparing
the backbones.

## Compare the QProduct Transformer detectors

Use the same train and validation splits and seed `42` for all three runs.
Each output directory must be new. If your allocation requires an explicit
Slurm account, add `--account=YOUR_SLURM_ACCOUNT` before the script name.

```bash
qproduct_args=(
  --train-csv data/source/processed/unbiased/train.csv
  --validation-csv data/source/processed/unbiased/validation.csv
  --seed 42 --epochs 3 --batch-size 16 --max-length 128 --learning-rate 2e-5
)

sbatch scripts/detector.sbatch train --detector roberta "${qproduct_args[@]}" \
  --output models/detector/checkpoints/roberta_qproduct_seed42

sbatch scripts/detector.sbatch train --detector deberta "${qproduct_args[@]}" \
  --output models/detector/checkpoints/10_deberta_v3_base/qproduct_seed42

sbatch scripts/detector.sbatch train --detector roberta-large "${qproduct_args[@]}" \
  --output models/detector/checkpoints/11_roberta_large/qproduct_seed42
```

Each run saves its highest-validation-macro-F1 epoch under `best_model/`,
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
