# NLP HumorGenerator

Product description → three humorous sentences → detector ranking → human or AI selects one → LoRA training → base/adapter comparison.

**The default workflow uses Hugging Face models through Transformers for DeepSeek/Qwen generation, AI selection, LoRA training and adapter testing. Ollama is optional. You do not need to install or start Ollama to run the default workflow.**

## Setup with uv

This project uses **uv, Python 3.11 and a project-local `.venv`**. Each teammate can start from a clean Linux x86-64 account without your conda environments. `pyproject.toml` declares dependencies and the committed `uv.lock` fixes the complete resolved package set, including PyTorch 2.8.0 from the explicit CUDA 12.8 index. `requirements.txt` is a compatibility reference, not the installation entry point.

Install uv once if it is not already available (uv 0.11.1 or newer is required):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Then install the project environment and validate actual package imports:

```bash
cd NLP_HumorGenerator
source setup.sh
uv run --locked python -m unittest discover -s tests -v
```

`setup.sh` installs uv-managed Python 3.11, runs `uv sync --locked`, checks dependencies and real model/trainer imports, and activates `.venv`. It does not download LLM weights or datasets. First installation still downloads Python packages; switching from conda does not automatically reuse its installed files. Setup must end with `Environment check: passed` and `Setup passed` before submitting a GPU job.

For subsequent sessions, use `uv run --locked python ...` for the documented Python commands, or `source .venv/bin/activate` to use plain `python`. No conda activation is needed. Ordinary `uv sync`/`uv run` can check package indexes; the sbatch scripts deliberately use the already-installed `.venv/bin/python` and do not install or resolve packages inside GPU jobs.

### Migrating an existing conda-created project environment

Wait until any current installation process has finished and returned to the prompt. Upload the uv update before running:

```bash
cd ~/BDT/NLP_HumorGenerator
source setup.sh --migrate-conda
```

If `.venv/conda-meta/history` exists, the migration deactivates it if needed and renames it to `.venv-conda-backup-TIMESTAMP`. It then creates a fresh uv environment. No files are deleted. The moved conda directory is a backup, not a runnable relocated environment; other conda environments such as `humor` and `humor-lora` remain in place. If no conda `.venv` exists, the command performs normal uv setup. An unrecognized `.venv` is preserved and setup stops for inspection.

Do not copy `.venv` or its backups to teammates or GitHub. Each teammate installs their own environment from `uv.lock`. If setup fails, keep its output and stop before submitting the demo.

### Teammate first run

```bash
git clone <REPOSITORY_URL> NLP_HumorGenerator
cd NLP_HumorGenerator
source setup.sh
uv run --locked python -m unittest discover -s tests -v
# Configure shared/local model and dataset paths, and Slurm settings.
sbatch test.sbatch
```

The code supports Linux x86-64 for this CUDA environment; the lockfile is not a Windows/macOS installation target. Every teammate needs access to the complete detector checkpoints, base LLM weights and QA training CSV. Copy them into the documented project folders before running. Default execution uses project-relative locations. Setup does not create weights or datasets, and Ollama remains optional.

All supplied Slurm templates default to partition `normal`, one GPU, and allocation account `mscbdt2024`. Edit `#SBATCH --account=...` in the file if your allocation differs, or override it for one submission with `sbatch --account=YOUR_SLURM_ACCOUNT test.sbatch`. Command-line sbatch options override the file directives. A Slurm compute allocation is independent of the GitHub account owning the repository.

The uv environment and an end-to-end demo have completed on Superpod with an NVIDIA H800: DeepSeek generation, RoBERTa-base paired-input scoring, Qwen one-of-three selection, one-epoch DeepSeek LoRA training and base/adapter comparison. This validates that configuration, not humor improvement or every detector backend. The new standalone HF detector trainer has a tiny CPU smoke test; its real GPU training is still to be validated. `.environment.json` records installed versions and is excluded from Git.

### Environment validation

Package/model imports are checked during setup only. Job scripts do not run `check_environment.py` or print package versions before each run. To diagnose an environment manually, run `uv run --locked python scripts/check_environment.py`; add `--gpu` only inside a GPU allocation. Routine jobs go directly to the requested stage.

## Folders

```text
NLP_HumorGenerator/
├── main.py
├── demo.py
├── setup.sh
├── test.sbatch             # choose mode/models/stages here; sbatch test.sbatch
├── run_pipeline.py         # routes the selected stages
├── pyproject.toml
├── uv.lock
├── .python-version
├── src/
├── detector/               # independent inference, training, evaluation and research code
├── configs/
│   ├── llms.json
│   ├── detectors.json
│   ├── datasets.json
│   └── training.json
├── models/
│   ├── detector/
│   │   ├── models/           # roberta-base/, roberta-large/, deberta-v3-base/, *.pt
│   │   └── checkpoints/     # detector checkpoint folders
│   └── llm/
│       ├── Qwen2.5-7B-Instruct/
│       ├── DeepSeek-R1-Distill-Qwen-7B/
│       └── adapters/        # optional existing LoRA adapters
├── data/
│   ├── products.json
│   └── lora/                # created after review: train/validation/test JSONL
├── results/                 # generated sentences, review Excel, trained adapter, test Excel
├── scripts/                 # run.sbatch and demo.sbatch
└── tests/
```

Copy existing detector backbone folders into `models/detector/models/` and trained detector checkpoint folders into `models/detector/checkpoints/`. Preserve subfolder names and include complete weights, config and tokenizer files. Weights are not included in this package. The old project is only a copy source; normal execution does not depend on it.

Copy complete LLM snapshots into the relative folders declared in `configs/llms.json`, including tokenizer/config files and all weight shards. Defaults do not fall back to a personal Hugging Face cache. Model loading is offline; missing project models produce an error. Explicit `--model` overrides remain available for advanced use. If copying from a Hugging Face snapshot cache, dereference its symlinks so the copied model no longer depends on the original cache.

```bash
python main.py models
```

## 1. Generate three sentences per product

### Dataset source

The default source is the **same processed QA training dataset used by the detector**, not a separately invented product dataset. Generation reads `product_group` and `product_description` from:

```text
data/source/processed/unbiased/train.csv
```

`configs/datasets.json` sets the dataset root and relative QA training path. Both normal generation and the one-job demo now read this CSV by default. Original QA questions and humor labels are not used as generation targets. The detector's QA validation/test splits are not read by this workflow.

| Source dataset folder (relative to the copied dataset root) | Role in this project |
|---|---|
| `processed/unbiased/train.csv` | Default product-description source for generation and the demo |
| `processed/humor_ad_lora_v1/` | Optional existing reviewed LoRA messages; can be used directly with `train --data` |
| `processed/biased/` | Existing alternative detector experiment data; not read by default |
| `processed/inc_pqa_200k/`, `subjectivity/`, `subjectivity_multicat_40k/` | Auxiliary detector training data; not needed when using existing checkpoints |
| `raw/humor_original/`, `raw/inc_pqa/`, `external/` | Original datasets, metadata and embedding resources; not needed for this inference/LoRA workflow |
| `features/` | Cached features for original QA rows; do not reuse these for newly generated sentences |

Datasets are **not bundled in this code ZIP** and are not reconstructed or downloaded automatically. Copy the QA training CSV into `data/source/processed/unbiased/train.csv`; `configs/datasets.json` already points to `data/source`. No old server directory or personal username is required. Other teammates must receive the model/dataset files separately and put them in the same relative folders. Before publishing, supply the actual team-accessible artifact download/storage location. Model and source-dataset files are excluded from ordinary Git commits.

The included `data/products.json` is an optional existing 50-product snapshot. Explicitly use it with `--products data/products.json`; a missing QA source does not silently fall back to the snapshot.

To extract and save products as a separate JSON file:

```bash
python main.py products \
  --csv data/source/processed/unbiased/train.csv \
  --limit 50 --output data/products_from_qa.json
```

It takes one nonempty description per `product_group` and samples products with a fixed seed. Use `--products data/products_from_qa.json` when generating from that saved list. The optional `--csv` argument overrides the configured source; omitting it uses `configs/datasets.json`.

Within a GPU allocation, run the default Hugging Face/Transformers backend. No Ollama service is needed:

```bash
python main.py generate --llm qwen --output results/qwen.json
python main.py generate --llm deepseek --output results/deepseek.json
```

The default is 50 products sampled from the QA training CSV and three short humorous advertising sentences per product. Change the sample size with `--num-products` and the sentences per product with `--per-product`. Generation uses temperature 0.8 and top-p 0.9. Model-specific token budgets are in `configs/llms.json`. DeepSeek reasoning text is removed from the final sentence output; increase its budget if it produces reasoning without a final answer.

The LoRA dataset is created **after review** from the generated sentences selected by humans or AI. It is a new product-to-sentence dataset, separate from the original detector QA labels. To reuse the existing LoRA dataset instead of generating new training examples:

```bash
python main.py train --llm qwen \
  --data data/existing_lora \
  --output results/qwen_existing_data_lora
```

That folder must contain the existing `train.jsonl`, `validation.jsonl`, and `test.jsonl` messages files.

To generate with a saved adapter:

```bash
python main.py generate --llm qwen \
  --adapter results/qwen_lora/final_adapter --output results/qwen_tuned.json
```

The adapter must match the selected base model. These commands load Hugging Face weights locally; they do not call a hosted inference API or automatically download missing weights.

### Optional: Ollama generation or AI selection

Skip this section when using the default Hugging Face workflow. If you choose to use an existing Ollama service and an installed Ollama model instead, explicitly add `--backend ollama`:

```bash
python main.py generate --llm deepseek --backend ollama \
  --model deepseek-r1:8b --output results/deepseek_ollama.json
```

Ollama defaults to `http://localhost:11434`; override with `--ollama-url`. It requires its own installed model; a Hugging Face model already in the cache is not automatically an Ollama model. In this project, LoRA training, `--adapter` generation and base/adapter testing use Transformers.

## 2. Select a detector and export the ranking

```bash
python main.py rank --input results/qwen.json \
  --detector roberta --output results/qwen_ranked.xlsx

python main.py rank --input results/deepseek.json \
  --detector deberta --output results/deepseek_ranked.xlsx --seed 43
```

The seed is a separate optional argument at the end. Omit it to use the detector's default (42 for the registered seed-based checkpoints). It is a checkpoint selection, not a new training seed. For file checks without loading a GPU model, add `--check`.

After copying the detector files, omit `--detector-root`; it defaults to this project's `models/detector/`. A storage override is optional, not required.

| Detector name | Model |
|---|---|
| `roberta` | RoBERTa-base, sentence + product |
| `roberta-qonly` | RoBERTa-base, sentence only |
| `roberta-large` | RoBERTa-large, sentence + product |
| `roberta-large-qonly` | RoBERTa-large, sentence only |
| `deberta` | DeBERTa-v3-base, sentence + product |
| `deberta-qonly` | DeBERTa-v3-base, sentence only |
| `roberta-sub` | RoBERTa-base + frozen single-layer SUB encoder |
| `lstm`, `cnn` | Existing FastText question-only checkpoints; matching inference loader must be configured |

Paths and input modes are in `configs/detectors.json`. `roberta-sub` loads the inspected 64-dimensional single-layer SUB architecture and finds the auxiliary encoder using the training checkpoint's hash. Newer INC/SUB fusion models require their own loader. Standard HF classifiers must have two output classes; confirm that `positive_label` denotes humor.

The Excel file contains `Human_review` and `All_scores`. Sentences are ranked **within each product** by detector probability; the review sheet contains the top three. It retains the original review columns: `product_id`, `product_description`, `candidate_id`, `tagline`, `class_1_score`, `rank`, `selected`, `comment`.

## 3. Choose one of three, then train and test

**Human selection:** open `results/qwen_ranked.xlsx`, put `selected=1` on the best sentence for each product, put 0 on the other two, and save as `results/qwen_reviewed.xlsx`.

**AI selection:** an LLM can instead choose exactly one candidate for each product:

```bash
python main.py review --llm qwen --input results/qwen_ranked.xlsx \
  --output results/qwen_reviewed.xlsx
```

Existing selections are preserved. To replace them deliberately, use `--overwrite-selection`. The AI chooses from the available candidate IDs and writes its reason in `comment`; it does not generate replacement sentences. The original `All_scores` sheet stays in the workbook.

Prepare data from either human or AI selections:

```bash
python main.py prepare --input results/qwen_reviewed.xlsx
```

Every reviewed product must have exactly one `selected=1`. The selected examples are split by product into approximately 70% train, 10% validation, 20% test; 50 products give 35/5/10 examples. At least three selected products are required. The split seed defaults to 42. For a separate DeepSeek dataset use `--output data/deepseek_lora` and pass it to `train --data` and `test --data`.

Train a new adapter and compare it with its base model on the held-out test products:

```bash
python main.py train --llm qwen
python main.py test --llm qwen --adapter results/qwen_lora/final_adapter \
  --output results/qwen_test.xlsx

python main.py train --llm deepseek
python main.py test --llm deepseek --adapter results/deepseek_lora/final_adapter \
  --output results/deepseek_test.xlsx
```

`configs/training.json` controls rank, alpha, target modules, epochs, batch size, learning rate and maximum sequence length. Training uses completion-only loss and selects the best validation loss; test data is not used by the trainer. `test` generates both base-model and adapter outputs for the same saved test prompts and writes the reference and both outputs into one Excel file. It does not calculate an automatic humor-quality score.

Each command runs only its own stage. Existing outputs are never overwritten. Use a new `--output` when repeating an experiment.

## Superpod submission

### Run demo, full pipeline or selected stages from test.sbatch

Edit only the settings block at the top of `test.sbatch`, then always submit from the project root:

```bash
sbatch test.sbatch
```

| Settings | Execution |
| --- | --- |
| `MODE="demo"` | All six stages, 10 products and 1 LoRA epoch by default |
| `MODE="full"` | All six stages, 50 products and the training config's epochs by default |
| `MODE="generate"` | Generate only |
| `MODE="rank"` | Rank existing sentences only |
| `MODE="custom"`, `STEPS="generate rank"` | Generate and rank, then stop |
| `MODE="custom"`, `STEPS="prepare train test"` | Continue after manual or AI review |

Pipeline order is `generate rank review prepare train test`. `full` runs every stage; it does not mean the entire dataset. Set `NUM_PRODUCTS` to your intended sample size. Optional `PRODUCTS` JSON uses every supplied product, so `NUM_PRODUCTS` does not limit that file. Full runs include AI review and new LLM LoRA training; detector training is separate (`python -m detector train`).

- Blank `RUN_DIR` creates `results/MODE-JOB_ID/`. To continue or rerank, set it to the existing run folder.
- Change `LLM`, `REVIEW_LLM`, `DETECTOR` and optional `DETECTOR_SEED` in the same settings block.
- Blank `NUM_PRODUCTS` and `EPOCHS` use the mode defaults above. Explicit values override those defaults.
- Files are `sentences.json`, `ranked.xlsx`, `reviewed.xlsx`, `data/`, `lora/final_adapter/` and `test.xlsx` inside the run folder. Each stage consumes the preceding stage's files; skipped stages must have existing outputs.
- For another detector on existing sentences: `MODE="rank"`, `RUN_DIR="results/demo-709136"`, `DETECTOR="deberta"`, `RANKED_FILE="ranked_deberta.xlsx"`. A later `review` consumes that same `RANKED_FILE`.
- For human review: stop after `generate rank`, edit the run's `ranked.xlsx`, save as `reviewed.xlsx`, then select `prepare train test` with the same `RUN_DIR`.
- Blank `ADAPTER` tests the run's `lora/final_adapter`. Set it only to test a different existing adapter. Generation uses the selected base LLM.
- Existing outputs are never overwritten. Use a new run or a different ranking filename for another experiment. Partial runs do not automatically skip completed stages.
- `DRY_RUN="yes"` prints stage commands without writing files or loading models. Set it back to `no` to execute.

The Slurm header holds account, partition, time, memory and GPU settings. Jobs use the project uv environment, perform no package/GPU import checks, and install no packages. Log: `humor-test-JOB_ID.log`. `demo.py` and `scripts/demo.sbatch` remain available for the original demo CLI.

This demo uses **Hugging Face/Transformers throughout; Ollama is not required**. It reads the configured QA training CSV, samples 10 products, generates 3 sentences each, runs detector ranking, lets an AI select one per product, prepares a 7/1/2 train/validation/test split, trains a **new LoRA adapter for 1 epoch**, then compares base and adapter outputs on the 2 test products. It tests the workflow rather than model quality.

Before submitting, configure the complete model/checkpoint locations and `configs/datasets.json`, run setup once, and confirm the account/partition directives in `test.sbatch` (or the advanced submission script you choose). The QA training CSV must exist, and complete model files must already be copied into the project model folders. The dataset is checked before any model stage starts.

```bash
cd /path/to/NLP_HumorGenerator
source .venv/bin/activate

# Optional: inspect all commands without generating, training or writing files.
python demo.py --llm deepseek --review-llm qwen --detector roberta --dry-run

# ONE submission uses the settings edited in test.sbatch.
sbatch test.sbatch
```

Advanced CLI example (the editable `test.sbatch` is the simpler default):

```bash
sbatch scripts/demo.sbatch --llm deepseek --review-llm qwen \
  --detector deberta --detector-seed 43
```

Use `--model /path/to/model_or_snapshot` for the generator/train/test model, and `--review-model /path/to/reviewer` if needed. If `--review-llm` is omitted, the selected LLM also performs the review. Change the demo size with `--num-products 20` or training length with `--epochs 2`.

Use `--qa-csv /absolute/path/to/train.csv` to select a specific QA training CSV, or explicitly choose the existing snapshot with `--products data/products.json`. The demo records the source path and SHA-256 in its summary.

Monitor the `test.sbatch` job using the ID returned by `sbatch` (advanced submissions via `scripts/demo.sbatch` instead use `humor-demo-JOB_ID.log`):

```bash
tail -f humor-test-JOB_ID.log
```

All demo outputs are grouped in **one folder**, `results/demo-JOB_ID/`:

```text
products.json       # sampled products
sentences.json      # 30 generated sentences
ranked.xlsx         # detector ranking
reviewed.xlsx       # AI one-of-three selections; inspect these
data/               # train, validation, test and product split assignments
training.json       # demo settings: 1 epoch
lora/               # new training checkpoints and final_adapter/
test.xlsx           # reference, base output and adapter output
summary.json        # completion status, or the stage that failed
```

Each stage runs in a separate Python process so its GPU memory is released before the next stage. A failed stage stops the job; later stages do not run. Previous models and ordinary stage outputs are preserved. Rerunning creates a new job folder; a partial failed run is not automatically resumed. This unattended demo uses **AI selection**; use the separate stage commands when human selection is required before training. The baseline demo has completed on Superpod; other model/configuration combinations still need runtime validation.

### Submit individual stages

Submit from the project root after setup has passed. `scripts/run.sbatch` selects `.venv/bin/python` itself; it does not use whichever `python` happens to be active in the submission shell. Set the account/partition for your cluster access.

```bash
sbatch scripts/run.sbatch generate --llm deepseek --output results/deepseek.json
sbatch scripts/run.sbatch rank --input results/deepseek.json --detector deberta \
  --output results/deepseek_ranked.xlsx --seed 42
sbatch scripts/run.sbatch review --llm qwen --input results/deepseek_ranked.xlsx \
  --output results/deepseek_reviewed.xlsx
# Run prepare after review is complete; then submit training and testing separately.
sbatch scripts/run.sbatch train --llm deepseek
sbatch scripts/run.sbatch test --llm deepseek --adapter results/deepseek_lora/final_adapter \
  --output results/deepseek_test.xlsx
```

Wait for each stage to finish before running the next. CPU stages `products` and `prepare` do not need a GPU; `rank --device cpu` is also supported.

## Additional detector architectures

### Independent detector module

Detector inference and architecture code now lives under `detector/`; `src/detector.py` and `src/networks.py` retain compatibility imports so existing pipeline commands are unchanged. The module also provides standard HF classifier training/evaluation and keeps descriptively named baseline/ablation research scripts under `detector/experiments/`. See `detector/README.md` for the supported commands and the remaining adaptation boundaries.

```bash
uv run --locked python -m detector list
uv run --locked python -m detector rank --detector roberta-large --check
```

Use `scripts/detector.sbatch` to submit detector-only work. It uses the existing uv environment and does not install dependencies or download weights. Training requires complete backbone files and both QA train/validation CSVs; evaluation requires a labelled QA split. This folder update does not retrain or overwrite any existing model.

### Inventory existing detector training code and artifacts

To consolidate another project before copying its assets, run the read-only inventory:

```bash
uv run --locked python scripts/inspect_existing.py \
  --source /path/to/existing/project \
  --output results/existing_inventory \
  --bundle-code
```


To inspect metadata on the server without training:

```bash
python main.py inspect-detector --detector lstm
python main.py inspect-detector --detector cnn
```

Add a module with `load(config, device, batch_size)` returning an object with `score(rows)`. Register `backend: custom` and `factory: src.my_detector:load` in `configs/detectors.json`. Scores must be humor probabilities in [0, 1], in input order. All custom settings and resolved model locations are passed to the factory. Use the exact training-time vocabulary, tokenizer, sequence construction and architecture.

## Checks

```bash
uv run --locked python -m unittest discover -s tests -v
```

