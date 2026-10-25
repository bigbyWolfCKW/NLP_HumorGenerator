# Detector comparison experiments

Baselines and lower-performing variants are retained for controlled comparisons. Filenames describe the task, architecture and feature variant. Seeds remain model settings, not script names.

## Standard classifiers

Use the common `python -m detector train/evaluate/rank` interface for RoBERTa-base, RoBERTa-large and DeBERTa. Their duplicated historical training scripts have been replaced by `detector/training.py`; both qonly and paired-input choices remain available.

## Auxiliary models and fusion comparisons

- `sub_lstm.py`: shared original subjectivity model/trainer definitions.
- `train_roberta_sub_baseline.py`, `train_roberta_sub_multiseed.py`: original 64-dimensional SUB baseline and its additional seeds.
- `train_roberta_fusion_baseline.py`: original INC + SUB ablations.
- `train_roberta_fusion_sub_multicategory_random.py`: multicategory SUB with random embeddings.
- `train_roberta_fusion_sub_multicategory_fasttext.py`: multicategory SUB with fastText.
- `train_roberta_fusion_product_qa.py`: product-QA incongruity features with multicategory fastText SUB.
- `train_roberta_large_fusion_lr2e_minus5.py`: large-model fusion comparison at LR 2e-5.
- `train_roberta_large_sub_lr1e_minus5.py`: large-model control/SUB comparison at LR 1e-5.
- `train_roberta_large_inc_fusion_lr1e_minus5.py`: large-model INC and INC+SUB comparison at LR 1e-5.
- `train_roberta_large_biased.py`: biased-data fusion ablations.

`prepare_*` scripts prepare auxiliary datasets/vectors. `extract_*` scripts create auxiliary features. Evaluation entry points are `evaluate_detector_suite.py`, `evaluate_roberta_large_sub.py`, and `evaluate_cross_dataset.py`; `freeze_test_plan.py` prepares the historical test plan. See `filenames.json` for the full old-to-new filename mapping.

## Execution scope

This update renames research sources and their sibling imports. It leaves algorithms, hyperparameters, feature dimensions, checkpoint folders, result identifiers and saved hash records unchanged. The supported standard detector CLI is unchanged.

The research scripts retain their original `PROJECT`, dataset/feature-cache layout and scikit-learn dependency. They need path/cache/dependency adaptation before execution in the new project; renaming does not make all historical training workflows portable. Execute research scripts as standalone files after that adaptation, rather than importing arbitrary trainers. In particular, 65-dimensional fusion and the portable 64-dimensional SUB backend remain different architectures.

## Update an existing checkout

After extracting the rename update, run `.venv/bin/python scripts/clean_experiments.py` once. It removes unchanged numeric-named source copies and the prior research ZIP/manifest only. Edited old sources are kept and reported. No duplicate history archive is created: the original project remains the historical source. Model files, datasets and results are untouched. Re-running is safe.
