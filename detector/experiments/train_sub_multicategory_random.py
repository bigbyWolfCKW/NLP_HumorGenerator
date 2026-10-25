import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

# 沿用舊 SUB 的模型、tokenizer 和訓練函數。
import sub_lstm as base


PROJECT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT / "data/processed/subjectivity_multicat_40k"
)
RESULTS_DIR = PROJECT / "results"

SEEDS = [42, 43, 44]

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
MAX_LENGTH = 128
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3
LR = 0.001
MAX_VOCAB_WORDS = 10_000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請透過 sbatch 配置 GPU 執行。")

    torch.set_num_threads(4)
    base.MAX_LENGTH = MAX_LENGTH

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"

    if not (DATA_DIR / "manifest.json").is_file():
        raise FileNotFoundError("096 資料集尚未完整建立。")

    train = pd.read_csv(
        train_path,
        dtype={"parent_asin": str},
    )
    validation = pd.read_csv(
        val_path,
        dtype={"parent_asin": str},
    )

    for name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        required = {"text", "label", "parent_asin"}

        if not required.issubset(df.columns):
            raise ValueError(f"{name} 缺少必要欄位。")

        if df[list(required)].isna().any().any():
            raise ValueError(f"{name} 有缺失值。")

        if set(df["label"].unique()) != {0, 1}:
            raise ValueError(f"{name} 標籤不正確。")

    if len(train) != 32_000 or len(validation) != 8_000:
        raise ValueError("資料筆數與 096 設定不一致。")

    if (
        set(train["parent_asin"])
        & set(validation["parent_asin"])
    ):
        raise ValueError("Train / validation 商品重疊。")

    if set(train["text"]) & set(validation["text"]):
        raise ValueError("Train / validation 有完全相同文字。")

    # 詞彙表只使用 training text。
    counts = Counter(
        token
        for text in train["text"]
        for token in base.tokenize(text)
    )

    vocab = {"<pad>": 0, "<unk>": 1}

    for token, _ in counts.most_common(MAX_VOCAB_WORDS):
        if token not in vocab:
            vocab[token] = len(vocab)

    train_dataset = base.make_dataset(train, vocab)
    val_dataset = base.make_dataset(validation, vocab)

    shared_config = {
        "embedding_dim": EMBEDDING_DIM,
        "embedding_initialization": "random",
        "hidden_dim": HIDDEN_DIM,
        "dropout": 0.3,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "tokenizer_pattern": base.TOKEN_PATTERN,
        "lowercase": True,
        "pad_id": 0,
        "unk_id": 1,
        "max_vocab_words": MAX_VOCAB_WORDS,
        "vocab_size": len(vocab),
        "labels": {
            "0": "description proxy",
            "1": "review proxy",
        },
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_csv_sha256": sha256(train_path),
        "validation_csv_sha256": sha256(val_path),
        "manifest_sha256": sha256(DATA_DIR / "manifest.json"),
    }

    # 在訓練前檢查所有 seed 的目錄。
    # 完整且設定一致的結果可沿用；不完整結果不會覆蓋。
    completed = {}

    for seed in SEEDS:
        folder = (
            PROJECT / "checkpoints"
            / f"sub_lstm_multicat40k_seed{seed}"
        )

        if not folder.exists():
            continue

        marker = folder / "completed.json"

        if not marker.is_file():
            raise FileExistsError(
                f"發現未完成的實驗：{folder}\n"
                "請先檢查 log，將該目錄移到備份位置後再重跑。"
            )

        config = json.loads(
            (folder / "config.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {"seed": seed, **shared_config}

        if config != expected:
            raise ValueError(
                f"既有實驗設定或資料不同：{folder}"
            )

        if not (folder / "best.pt").is_file():
            raise FileNotFoundError(folder / "best.pt")

        best = json.loads(
            (folder / "best_validation.json").read_text(
                encoding="utf-8"
            )
        )

        completed[seed] = best

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Seeds:", SEEDS, flush=True)
    print("Train rows:", len(train), flush=True)
    print("Validation rows:", len(validation), flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Embedding: random, 100 dimensions", flush=True)

    rows = []

    for seed in SEEDS:
        if seed in completed:
            print(
                f"\nReuse completed seed {seed}",
                flush=True,
            )
            best = completed[seed]

        else:
            print(
                f"\n=== SUB multicategory | seed {seed} ===",
                flush=True,
            )

            set_seed(seed)

            folder = (
                PROJECT / "checkpoints"
                / f"sub_lstm_multicat40k_seed{seed}"
            )
            folder.mkdir(parents=True, exist_ok=False)

            config = {"seed": seed, **shared_config}

            (folder / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

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

            model = base.SubjectivityLSTM(
                vocab_size=len(vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
            ).to("cuda")

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR,
            )

            history = []
            best_f1 = -1.0
            best = None
            stale_epochs = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                train_metrics = base.run_epoch(
                    model, train_loader, optimizer
                )
                val_metrics = base.run_epoch(
                    model, val_loader
                )

                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_loss": val_metrics["loss"],
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                }

                if not all(
                    torch.isfinite(torch.tensor(value)).item()
                    for value in row.values()
                ):
                    raise RuntimeError("出現非有限訓練或評估數值。")

                history.append(row)

                if row["validation_macro_f1"] > best_f1:
                    best_f1 = row["validation_macro_f1"]
                    best = row.copy()
                    stale_epochs = 0

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "vocab": vocab,
                            "config": config,
                            "validation": best,
                        },
                        folder / "best.pt",
                    )

                    (folder / "best_validation.json").write_text(
                        json.dumps(best, indent=2),
                        encoding="utf-8",
                    )
                else:
                    stale_epochs += 1

                pd.DataFrame(history).to_csv(
                    folder / "history.csv",
                    index=False,
                )

                print(
                    f"Epoch {epoch}/{MAX_EPOCHS} | "
                    f"Train loss: {row['train_loss']:.4f} | "
                    f"Val loss: {row['validation_loss']:.4f} | "
                    f"Val accuracy: "
                    f"{row['validation_accuracy']:.2%} | "
                    f"Val macro-F1: "
                    f"{row['validation_macro_f1']:.4f} | "
                    f"Patience: {stale_epochs}/{PATIENCE}",
                    flush=True,
                )

                if stale_epochs >= PATIENCE:
                    print("Early stopping.", flush=True)
                    break

            (folder / "completed.json").write_text(
                json.dumps(
                    {"seed": seed, "status": "completed"},
                    indent=2,
                ),
                encoding="utf-8",
            )

            del model, optimizer
            torch.cuda.empty_cache()

        rows.append({"seed": seed, **best})

        # 每完成一個 seed 立即保存。
        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "097_sub_multicat_per_seed.csv",
            index=False,
        )

        print("Best:", best, flush=True)

    results = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "model": "sub_lstm_multicat40k_random100",
        "seeds": len(SEEDS),
        "accuracy_mean": results["validation_accuracy"].mean(),
        "accuracy_std": results["validation_accuracy"].std(),
        "macro_f1_mean": results["validation_macro_f1"].mean(),
        "macro_f1_std": results["validation_macro_f1"].std(),
    }])

    summary.to_csv(
        RESULTS_DIR / "097_sub_multicat_summary.csv",
        index=False,
    )

    print("\n=== Per-seed results ===", flush=True)
    print(results.to_string(index=False), flush=True)

    print("\n=== Three-seed summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    print("\nSUB training completed.", flush=True)


if __name__ == "__main__":
    main()