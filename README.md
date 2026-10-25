# NLP HumorGenerator

Generate product-based humor, compare detectors, and fine-tune an LLM with selected sentences. **Hugging Face/Transformers is the default; Ollama is optional.**

## Pipeline

```text
QA dataset → product descriptions → 3 sentences per product
→ detector scores + ranked Excel → human/LLM selects 1 of 3
→ product-disjoint train/validation/test split → LoRA training
→ compare the base LLM and adapter on held-out products
```

## Folder structure

```text
NLP_HumorGenerator/
├── test.sbatch          # EDIT HERE: mode, models, sample size, epochs, run folder
├── run_pipeline.py      # executes the selected stages
├── main.py              # individual stage CLI
├── setup.sh             # one-time uv environment setup
├── configs/             # LLM, detector, dataset paths and training settings
├── src/                 # generation, review and LoRA code
├── detector/            # detector inference/training/evaluation + comparison experiments
├── data/
│   ├── products.json    # optional example product input
│   └── source/          # QA dataset: processed/unbiased/train.csv
├── models/
│   ├── llm/            # DeepSeek/Qwen model folders; adapters/ for existing LoRA
│   └── detector/       # checkpoints/ for trained detectors; models/ for backbones
└── results/             # one folder per run: sentences, Excel, split data and adapter
```

**Weights, full datasets and run results are not stored on GitHub.** Supply complete model/checkpoint folders and the QA CSV at the paths in `configs/`. Keep checkpoint subfolder names intact. Research experiment details are in [detector/README.md](detector/README.md).

## Setup once on Superpod

```bash
git clone https://github.com/bigbyWolfCKW/NLP_HumorGenerator.git
cd NLP_HumorGenerator
# Install uv if needed:
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
source setup.sh
```

In `test.sbatch`, adjust the `#SBATCH` account/resources for your Superpod allocation. Teammates need collaborator access to clone this private repository.

## Run: edit test.sbatch, then submit

**Only change the top settings block. Every mode uses the same submission command:**

```bash
sbatch test.sbatch
```

| What to run | Settings in `test.sbatch` |
| --- | --- |
| Small end-to-end demo | `MODE="demo"` — defaults to 10 products, 1 LoRA epoch |
| Full pipeline | `MODE="full"` — defaults to 50 products, epochs from training config |
| Selected stages | `MODE="custom"`, e.g. `STEPS="generate rank"` |
| One stage | `MODE="generate"`, `"rank"`, `"review"`, `"prepare"`, `"train"` or `"test"` |

**Main keywords:** `LLM`, `REVIEW_LLM`, `DETECTOR`, optional `DETECTOR_SEED`, `NUM_PRODUCTS`, `EPOCHS`, `RUN_DIR`. Blank `RUN_DIR` creates `results/MODE-JOB_ID/`; set an existing folder to continue. Full means all stages, not all dataset rows.

Example: rank existing sentences with another detector, without regenerating or training:

```bash
MODE="rank"
RUN_DIR="results/demo-709136"       # replace with your run folder
DETECTOR="deberta"                  # or roberta / roberta-large
RANKED_FILE="ranked_deberta_v2.xlsx" # choose a new filename
```

For human review, stop after `generate rank`, mark one `selected=1` per product and save as `reviewed.xlsx`; then run `prepare train test` with the same `RUN_DIR`. Partial runs need the preceding outputs.

**Remarks:** Existing outputs are not overwritten. `DRY_RUN="yes"` prints commands only; change back to `"no"` to execute. Jobs do not repeat setup checks.

## Outputs

Follow progress with `tail -f humor-test-JOB_ID.log`. In the run folder:

- `sentences.json` → `ranked.xlsx` → `reviewed.xlsx`
- `data/` — train/validation/test JSONL
- `lora/final_adapter/` — trained LoRA adapter
- `test.xlsx` — base-versus-adapter sentence comparison

