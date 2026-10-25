# Detector

Detector code is separate from LLM generation and LoRA training, but uses the same uv environment and model/data folders.

```text
detector/
├── __main__.py           # python -m detector
├── inference.py          # load checkpoints and score candidate sentences
├── models/              # architecture code, not model weights
│   └── roberta_sub.py
├── data.py               # labelled QA CSV input and binary metrics
├── training.py           # standard HF classifier training
├── evaluation.py         # labelled QA evaluation
└── experiments/          # named comparison experiments
```

Slurm defaults to account `mscbdt2024` and partition `normal`. Edit those directives in `scripts/detector.sbatch` or override them with `sbatch --account=YOUR_SLURM_ACCOUNT ...` when needed.

Weights remain in `models/detector/`. CSV datasets remain in `data/source/`. Existing `main.py rank` and demo commands continue to work through compatibility imports under `src/`.

## Use existing detectors

```bash
uv run --locked python -m detector list
uv run --locked python -m detector rank --detector roberta-large --check

sbatch scripts/detector.sbatch rank \
  --input results/demo-709136/sentences.json --detector roberta-large \
  --output results/ranked_roberta_large.xlsx
```

Inference support is unchanged by this folder separation: standard binary HF models and the inspected RoBERTa+64-dimensional SUB architecture are supported. The later 65-dimensional INC+SUB fusion checkpoints require an additional backend even when their INC branch is disabled. LSTM/CNN entries still require their original humor-classifier loaders.

## Train a new standard detector

The train command supports the HF entries for RoBERTa-base, RoBERTa-large and DeBERTa. It reads the corresponding `backbone` setting in `configs/detectors.json`. Supply the complete base model and tokenizer in `models/detector/models/`, plus both QA splits:

```text
data/source/processed/unbiased/train.csv
data/source/processed/unbiased/validation.csv
```

Each CSV requires `question`, `product_group`, `label` (0/1), and `product_description` for paired input. Train and validation products must be disjoint. Test data is not loaded by the training command.

```bash
# Example NEW seed/output; this command performs actual detector training.
sbatch scripts/detector.sbatch train \
  --detector roberta \
  --output models/detector/checkpoints/roberta_qproduct_seed45 \
  --seed 45
```

Defaults follow the standard classifier experiments: three epochs, batch size 16, maximum length 128, LR 2e-5, AdamW weight decay .01, warmup .1, gradient clipping 1, and best validation macro-F1 selection. Override them through the train CLI. `--seed` is independent of the detector name. Explicit `--train-csv` and `--validation-csv` overrides are available. Existing output directories are rejected, including partially completed runs.

Outputs: `config.json`, `history.json`, `best_validation.json`, and `best_model/` containing weights/config/tokenizer. Register a custom output path as a new detector entry; standard seed paths such as the example are already covered by the registry template. Keep registry `max_length` and `input_mode` consistent with training settings.

```bash
uv run --locked python -m detector rank --detector roberta --seed 45 --check
```

## Evaluate a labelled QA split

Supply `data/source/processed/unbiased/test.csv` or select an explicit labelled CSV. The evaluate command performs inference only:

```bash
sbatch scripts/detector.sbatch evaluate \
  --detector roberta --seed 45 --output results/detector/roberta_seed45_test
```

It writes `metrics.json` and `predictions.csv`, with accuracy, macro-F1, the confusion matrix, input hash and resolved configuration. Select models using validation; use the held-out test for reporting rather than repeated tuning. Candidate ranking uses unlabelled generated sentences and is a separate command.

## Comparison experiment code

`experiments/` retains descriptively named baseline and ablation sources, including random/fastText embeddings, product-QA incongruity, large-model learning-rate comparisons and biased/cross-dataset experiments. Lower-scoring variants remain useful comparisons. Sibling imports use the new names; checkpoint paths and experiment identifiers are unchanged. See its README and `filenames.json` for the mapping and execution boundaries.

The common trainer replaces four duplicate standard HF trainers. Two one-off diagnostic scripts are omitted. No historical source ZIP is kept in the new project. To remove unchanged old filenames after applying the update, run `.venv/bin/python scripts/clean_experiments.py`. Edited sources are retained; models, datasets and results are untouched. Research-path/cache/dependency adaptation remains separate from the supported portable standard CLI.

The new module's local tests cover CSV validation, metrics, compatibility imports and evaluation wiring. A tiny CPU RoBERTa classifier completed a training epoch and exported loadable weights/tokenizer locally. Real model training and GPU behaviour still require Superpod validation. Existing completed LLM/LoRA demo results are unaffected.
