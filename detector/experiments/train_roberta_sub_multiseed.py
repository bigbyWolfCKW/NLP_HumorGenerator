import argparse
import copy
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


# 沿用 091 的模型、資料處理及 epoch 訓練函數。
# import 不會執行 091 的 main()。
base = importlib.import_module("train_roberta_sub_baseline")

PROJECT = Path(__file__).resolve().parent

EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 43, 44],
        default=43,
    )
    args = parser.parse_args()
    seed = args.seed

    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU 後執行。")

    run_dir = (
        PROJECT / "checkpoints" / f"roberta_sub_seed{seed}"
    )

    if run_dir.exists():
        raise FileExistsError(
            f"實驗目錄已存在，停止以避免覆蓋：{run_dir}"
        )

    # 明確沿用目前 humorProj 的資料和模型。
    base.MODEL_DIR = PROJECT / "models/roberta-base"
    base.DATA_DIR = PROJECT / "data/processed/unbiased"
    base.FEATURE_DIR = PROJECT / "data/features/sub_lstm_seed42"
    base.MAX_LENGTH = MAX_LENGTH

    set_seed(seed)
    torch.set_num_threads(4)

    print("Seed:", seed, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("SUB extractor: fixed seed 42", flush=True)

    tokenizer = base.AutoTokenizer.from_pretrained(
        base.MODEL_DIR,
        local_files_only=True,
    )

    train_df, train_dataset, train_sub_hash = (
        base.make_dataset("train", tokenizer)
    )
    val_df, val_dataset, val_sub_hash = (
        base.make_dataset("validation", tokenizer)
    )

    if train_sub_hash != val_sub_hash:
        raise ValueError(
            "Train / validation SUB extractor hash 不一致。"
        )

    overlap = (
        set(train_df["product_group"])
        & set(val_df["product_group"])
    )
    if overlap:
        raise ValueError("Train / validation 存在 product overlap。")

    print("Train rows:", len(train_df), flush=True)
    print("Validation rows:", len(val_df), flush=True)

    # 同一 seed 的 control / with_sub 共用初始權重。
    initial_model = base.ContextSUBModel(use_sub=False)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    run_dir.mkdir(parents=True, exist_ok=False)

    results_dir = PROJECT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    comparison = []

    for name, use_sub in [
        ("control", False),
        ("with_sub", True),
    ]:
        print(
            f"\n=== {name} | seed {seed} ===",
            flush=True,
        )

        model = base.ContextSUBModel(use_sub=use_sub)
        model.load_state_dict(initial_state)
        model.to("cuda")

        # 匹配兩組的 dropout seed 和 batch 順序。
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

        folder = run_dir / name
        folder.mkdir()

        config = {
            "model": name,
            "seed": seed,
            "use_sub": use_sub,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LR,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "sub_dimension": 64,
            "sub_checkpoint_sha256": train_sub_hash,
            "backbone_path": str(base.MODEL_DIR),
            "input": "question + product_description",
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
                model,
                val_loader,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }

            if not all(np.isfinite(v) for v in row.values()):
                raise RuntimeError("訓練或 validation 出現非有限數值。")

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

        comparison.append({
            "model": name,
            "seed": seed,
            **best,
        })

        pd.DataFrame(comparison).to_csv(
            results_dir / f"roberta_sub_seed{seed}.csv",
            index=False,
        )

        print(
            "Best validation result:",
            best,
            flush=True,
        )

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    print("\nComparison:", flush=True)
    print(
        pd.DataFrame(comparison).to_string(index=False),
        flush=True,
    )
    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()