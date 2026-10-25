import copy
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


core = importlib.import_module("train_roberta_fusion_product_qa")

PROJECT = Path(__file__).resolve().parent
RUN_DIR = PROJECT / "checkpoints/20_large_sub_lr1e5"
RESULTS_DIR = PROJECT / "results/20_large_sub_lr1e5"

SEED = 42
LR = 1e-5
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    if RUN_DIR.exists():
        raise FileExistsError(
            f"實驗已存在，停止覆蓋：{RUN_DIR}"
        )

    torch.set_num_threads(4)

    core.base.MODEL_DIR = PROJECT / "models/roberta-large"
    core.base.DATA_DIR = PROJECT / "data/processed/unbiased"
    core.base.MAX_LENGTH = MAX_LENGTH
    core.base.FEATURE_DIR = (
        PROJECT
        / "data/features/sub_lstm_multicat40k_fasttext_seed42"
    )

    core.DATA_DIR = core.base.DATA_DIR
    core.INC_DIR = (
        PROJECT / "data/features/inc_siamese_pqa200k_seed42"
    )
    core.INC_CHECKPOINT = (
        PROJECT
        / "checkpoints/inc_siamese_pqa200k_seed42/best.pt"
    )

    tokenizer = core.base.AutoTokenizer.from_pretrained(
        core.base.MODEL_DIR,
        local_files_only=True,
    )

    train_df, train_dataset, train_hashes = core.make_dataset(
        "train", tokenizer
    )
    val_df, val_dataset, val_hashes = core.make_dataset(
        "validation", tokenizer
    )

    if (
        set(train_df["product_group"])
        & set(val_df["product_group"])
    ):
        raise ValueError("Train / validation 商品重疊。")

    set_seed()

    initial_model = core.ContextINCSUBModel(
        use_inc=False,
        use_sub=False,
    )
    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )
    del initial_model

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Seed:", SEED, flush=True)
    print("Learning rate:", LR, flush=True)
    print("Validation only; no test evaluation.", flush=True)

    results = []

    for name, use_sub in [
        ("control", False),
        ("with_sub", True),
    ]:
        print(f"\n=== {name} ===", flush=True)

        model = core.ContextINCSUBModel(
            use_inc=False,
            use_sub=use_sub,
        )
        model.load_state_dict(initial_state)
        model.to("cuda")

        set_seed()

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
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

        scheduler = core.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        folder = RUN_DIR / name
        folder.mkdir()

        config = {
            "model": name,
            "seed": SEED,
            "use_inc": False,
            "use_sub": use_sub,
            "backbone_path": str(core.base.MODEL_DIR),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LR,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "train_sources": train_hashes,
            "validation_sources": val_hashes,
            "selection_metric": "validation_macro_f1",
        }

        (folder / "config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        history = []
        best = None
        best_f1 = -1.0

        for epoch in range(1, EPOCHS + 1):
            train_metrics = core.base.run_epoch(
                model, train_loader, optimizer, scheduler
            )
            val_metrics = core.base.run_epoch(
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
                raise RuntimeError("出現非有限數值。")

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
                tokenizer.save_pretrained(folder / "tokenizer")

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

        results.append({
            "model": name,
            "seed": SEED,
            **best,
        })

        pd.DataFrame(results).to_csv(
            RESULTS_DIR / "validation.csv",
            index=False,
        )

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    print("\n=== Validation comparison ===", flush=True)
    print(
        pd.DataFrame(results).to_string(index=False),
        flush=True,
    )
    print("\nValidation-only check completed.", flush=True)


if __name__ == "__main__":
    main()