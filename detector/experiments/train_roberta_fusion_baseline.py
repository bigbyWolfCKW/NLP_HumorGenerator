import copy
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup


# 沿用 091 的 RoBERTa、SUB 資料處理及訓練函數。
base = importlib.import_module("train_roberta_sub_baseline")

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data/processed/unbiased"
INC_DIR = PROJECT / "data/features/inc_siamese"
INC_CHECKPOINT = PROJECT / "models/inc_siamese.pt"

RUN_DIR = PROJECT / "checkpoints/094_roberta_inc_sub"
RESULTS_DIR = PROJECT / "results"

SEEDS = [42, 43, 44]
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5

EXPERIMENTS = [
    ("control", False, False),
    ("with_inc", True, False),
    ("with_sub", False, True),
    ("with_inc_sub", True, True),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ContextINCSUBModel(base.ContextSUBModel):
    def __init__(self, use_inc, use_sub):
        # 父類保留融合輸入；由本類控制哪些特徵設為零。
        super().__init__(use_sub=True)

        hidden = self.backbone.config.hidden_size

        # 1 維 INC + 64 維 SUB。
        # 四組實驗都有相同參數數量。
        self.dense = nn.Linear(hidden + 65, hidden)

        self.enable_inc = use_inc
        self.enable_sub = use_sub

    def forward(self, ids, mask, features):
        inc = features[:, :1]
        sub = features[:, 1:]

        if not self.enable_inc:
            inc = torch.zeros_like(inc)

        if not self.enable_sub:
            sub = torch.zeros_like(sub)

        combined = torch.cat([inc, sub], dim=1)

        return super().forward(ids, mask, combined)


def make_dataset(split, tokenizer):
    # 091 已檢查 SUB 的 CSV hash、維度與有限數值。
    df, sub_dataset, sub_hash = base.make_dataset(
        split, tokenizer
    )

    inc_saved = torch.load(
        INC_DIR / f"{split}.pt",
        map_location="cpu",
        weights_only=True,
    )

    csv_path = DATA_DIR / f"{split}.csv"

    if inc_saved["source_csv_sha256"] != sha256(csv_path):
        raise ValueError(f"{split}: INC 與目前 CSV 不一致。")

    if inc_saved["score_meaning"] != (
        "probability_of_original_product"
    ):
        raise ValueError("INC 分數方向不一致。")

    if inc_saved["inc_checkpoint_sha256"] != sha256(
        INC_CHECKPOINT
    ):
        raise ValueError("INC checkpoint 已變更，請核對特徵。")

    inc = inc_saved["features"].float()

    if inc.shape != (len(df), 1):
        raise ValueError(f"{split}: INC 特徵維度不正確。")

    if not torch.isfinite(inc).all():
        raise ValueError(f"{split}: INC 存在非有限數值。")

    if not ((inc >= 0) & (inc <= 1)).all():
        raise ValueError(f"{split}: INC 機率超出 0–1。")

    ids, mask, sub, labels = sub_dataset.tensors
    features = torch.cat([inc, sub.float()], dim=1)

    if features.shape != (len(df), 65):
        raise ValueError(f"{split}: 融合特徵維度不正確。")

    dataset = TensorDataset(
        ids, mask, features, labels
    )

    hashes = {
        "csv_sha256": sha256(csv_path),
        "inc_checkpoint_sha256": (
            inc_saved["inc_checkpoint_sha256"]
        ),
        "sub_checkpoint_sha256": sub_hash,
    }

    return df, dataset, hashes


def summarize(rows):
    results = pd.DataFrame(rows).sort_values(
        ["seed", "model"]
    )

    counts = results.groupby("model")["seed"].nunique()
    if (
        len(results) != len(SEEDS) * len(EXPERIMENTS)
        or len(counts) != len(EXPERIMENTS)
        or not counts.eq(len(SEEDS)).all()
        or results.duplicated(["seed", "model"]).any()
    ):
        raise ValueError("結果不完整或重複，停止彙總。")

    summary = results.groupby("model").agg(
        seeds=("seed", "nunique"),
        accuracy_mean=("validation_accuracy", "mean"),
        accuracy_std=("validation_accuracy", "std"),
        macro_f1_mean=("validation_macro_f1", "mean"),
        macro_f1_std=("validation_macro_f1", "std"),
    )

    differences = []

    for metric in [
        "validation_accuracy",
        "validation_macro_f1",
    ]:
        pivot = results.pivot(
            index="seed",
            columns="model",
            values=metric,
        )

        # 正值表示前者優於後者；單位為百分點。
        comparisons = [
            ("inc_minus_control", "with_inc", "control"),
            ("sub_minus_control", "with_sub", "control"),
            ("both_minus_control", "with_inc_sub", "control"),
            ("both_minus_inc", "with_inc_sub", "with_inc"),
            ("both_minus_sub", "with_inc_sub", "with_sub"),
        ]

        for name, left, right in comparisons:
            for seed in SEEDS:
                differences.append({
                    "seed": seed,
                    "metric": metric,
                    "comparison": name,
                    "delta_pp": 100 * (
                        pivot.loc[seed, left]
                        - pivot.loc[seed, right]
                    ),
                })

    paired = pd.DataFrame(differences)

    paired_summary = paired.groupby(
        ["metric", "comparison"]
    ).agg(
        delta_mean_pp=("delta_pp", "mean"),
        delta_std_pp=("delta_pp", "std"),
    )

    results.to_csv(
        RESULTS_DIR / "094_per_seed.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS_DIR / "094_summary.csv"
    )
    paired.to_csv(
        RESULTS_DIR / "094_paired_differences.csv",
        index=False,
    )
    paired_summary.to_csv(
        RESULTS_DIR / "094_paired_summary.csv"
    )

    print("\n=== Three-seed summary ===", flush=True)

    for name, row in summary.iterrows():
        print(
            f"{name}: "
            f"accuracy {row['accuracy_mean']:.2%} "
            f"+/- {100 * row['accuracy_std']:.2f} pp | "
            f"macro-F1 {row['macro_f1_mean']:.4f} "
            f"+/- {row['macro_f1_std']:.4f}",
            flush=True,
        )

    print(
        "\n=== Paired differences: percentage points ===",
        flush=True,
    )
    print(paired_summary.to_string(), flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    if RUN_DIR.exists():
        raise FileExistsError(
            f"實驗目錄已存在，停止以避免覆蓋：{RUN_DIR}"
        )

    torch.set_num_threads(4)

    base.MODEL_DIR = PROJECT / "models/roberta-base"
    base.DATA_DIR = DATA_DIR
    base.FEATURE_DIR = (
        PROJECT / "data/features/sub_lstm_seed42"
    )
    base.MAX_LENGTH = MAX_LENGTH

    tokenizer = base.AutoTokenizer.from_pretrained(
        base.MODEL_DIR,
        local_files_only=True,
    )

    train_df, train_dataset, train_hashes = make_dataset(
        "train", tokenizer
    )
    val_df, val_dataset, val_hashes = make_dataset(
        "validation", tokenizer
    )

    for key in [
        "inc_checkpoint_sha256",
        "sub_checkpoint_sha256",
    ]:
        if train_hashes[key] != val_hashes[key]:
            raise ValueError(f"Train / validation 的 {key} 不同。")

    if (
        set(train_df["product_group"])
        & set(val_df["product_group"])
    ):
        raise ValueError("Train / validation 存在 product overlap。")

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Seeds:", SEEDS, flush=True)
    print("Train rows:", len(train_df), flush=True)
    print("Validation rows:", len(val_df), flush=True)
    print("Features: INC 1 + SUB 64", flush=True)
    print("Both auxiliary extractors are fixed.", flush=True)

    rows = []

    for seed in SEEDS:
        set_seed(seed)

        initial_model = ContextINCSUBModel(
            use_inc=False,
            use_sub=False,
        )
        initial_state = copy.deepcopy(
            initial_model.state_dict()
        )
        del initial_model

        for name, use_inc, use_sub in EXPERIMENTS:
            print(
                f"\n=== {name} | seed {seed} ===",
                flush=True,
            )

            model = ContextINCSUBModel(
                use_inc=use_inc,
                use_sub=use_sub,
            )
            model.load_state_dict(initial_state)
            model.to("cuda")

            # 同一 seed 的四組匹配 dropout 與 batch 順序。
            set_seed(seed)

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                num_workers=0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR,
                weight_decay=0.01,
            )

            total_steps = EPOCHS * len(train_loader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * 0.1),
                num_training_steps=total_steps,
            )

            folder = RUN_DIR / f"seed{seed}" / name
            folder.mkdir(parents=True, exist_ok=False)

            config = {
                "model": name,
                "seed": seed,
                "use_inc": use_inc,
                "use_sub": use_sub,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "learning_rate": LR,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
                "inc_dimension": 1,
                "sub_dimension": 64,
                "input": "question + product_description",
                "backbone_path": str(base.MODEL_DIR),
                "train_sources": train_hashes,
                "validation_sources": val_hashes,
            }

            (folder / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

            history = []
            best_f1 = -1.0
            best = None

            for epoch in range(1, EPOCHS + 1):
                train_metrics = base.run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                )
                val_metrics = base.run_epoch(
                    model, val_loader
                )

                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                }

                if not all(
                    np.isfinite(value)
                    for value in row.values()
                ):
                    raise RuntimeError("出現非有限訓練或評估數值。")

                history.append(row)

                if row["validation_macro_f1"] > best_f1:
                    best_f1 = row["validation_macro_f1"]
                    best = row.copy()

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "validation": best,
                        },
                        folder / "best.pt",
                    )

                    (folder / "best_validation.json").write_text(
                        json.dumps(best, indent=2),
                        encoding="utf-8",
                    )

                    tokenizer.save_pretrained(
                        folder / "tokenizer"
                    )

                pd.DataFrame(history).to_csv(
                    folder / "history.csv",
                    index=False,
                )

                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Train loss: {row['train_loss']:.4f} | "
                    f"Val accuracy: "
                    f"{row['validation_accuracy']:.2%} | "
                    f"Val macro-F1: "
                    f"{row['validation_macro_f1']:.4f}",
                    flush=True,
                )

            rows.append({
                "model": name,
                "seed": seed,
                **best,
            })

            # 每完成一組立即保存結果。
            pd.DataFrame(rows).to_csv(
                RESULTS_DIR / "094_per_seed.csv",
                index=False,
            )

            print("Best:", best, flush=True)

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        del initial_state

    summarize(rows)
    print("\nAll 12 experiments completed.", flush=True)


if __name__ == "__main__":
    main()