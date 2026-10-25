import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import sub_lstm as base


PROJECT = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT / "data/processed/subjectivity_multicat_40k"
)
CACHE_PATH = (
    PROJECT
    / "data/external/fasttext/sub_multicat40k_vectors.pt"
)
RESULTS_DIR = PROJECT / "results"

SEEDS = [42, 43, 44]
MAX_LENGTH = 128
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3
LR = 0.001


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    torch.set_num_threads(4)
    base.MAX_LENGTH = MAX_LENGTH

    cache = torch.load(
        CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    vocab = cache["vocab"]
    vectors = cache["embedding_matrix"].float()
    cache_config = cache["config"]

    if vectors.shape != (len(vocab), 300):
        raise ValueError("FastText matrix 維度不正確。")

    if not torch.isfinite(vectors).all():
        raise ValueError("FastText matrix 有非有限數值。")

    if vocab["<pad>"] != 0 or vocab["<unk>"] != 1:
        raise ValueError("PAD / UNK ID 不正確。")

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("詞彙 ID 不連續。")

    if cache_config["tokenizer_pattern"] != base.TOKEN_PATTERN:
        raise ValueError("Tokenizer 與快取設定不同。")

    if not cache_config["lowercase"]:
        raise ValueError("目前 tokenizer 預期 lowercase。")

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"

    train_hash = sha256(train_path)
    val_hash = sha256(val_path)

    if train_hash != cache_config["train_csv_sha256"]:
        raise ValueError("Training CSV 與 FastText 快取不一致。")

    train = pd.read_csv(
        train_path, dtype={"parent_asin": str}
    )
    validation = pd.read_csv(
        val_path, dtype={"parent_asin": str}
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
        raise ValueError("資料筆數與 0961 不一致。")

    if (
        set(train["parent_asin"])
        & set(validation["parent_asin"])
    ):
        raise ValueError("Train / validation 商品重疊。")

    if set(train["text"]) & set(validation["text"]):
        raise ValueError("Train / validation 有完全相同文字。")

    train_dataset = base.make_dataset(train, vocab)
    val_dataset = base.make_dataset(validation, vocab)

    shared_config = {
        "embedding_dim": 300,
        "embedding_initialization": "fasttext_wikipedia",
        "freeze_embeddings": False,
        "hidden_dim": 64,
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
        "vocab_size": len(vocab),
        "labels": {
            "0": "description proxy",
            "1": "review proxy",
        },
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_csv_sha256": train_hash,
        "validation_csv_sha256": val_hash,
        "embedding_cache_sha256": sha256(CACHE_PATH),
        "pretrained_word_type_coverage": (
            cache_config["word_type_coverage"]
        ),
    }

    # 完整且設定一致的 seed 可以沿用。
    # 不會覆蓋部分失敗或設定不同的實驗。
    completed = {}

    for seed in SEEDS:
        folder = (
            PROJECT / "checkpoints"
            / f"sub_lstm_multicat40k_fasttext_seed{seed}"
        )

        if not folder.exists():
            continue

        if not (folder / "completed.json").is_file():
            raise FileExistsError(
                f"發現未完成實驗：{folder}\n"
                "請先檢查 log，將該目錄移到備份位置後重跑。"
            )

        saved_config = json.loads(
            (folder / "config.json").read_text(
                encoding="utf-8"
            )
        )

        if saved_config != {"seed": seed, **shared_config}:
            raise ValueError(f"既有設定不同：{folder}")

        if not (folder / "best.pt").is_file():
            raise FileNotFoundError(folder / "best.pt")

        completed[seed] = json.loads(
            (folder / "best_validation.json").read_text(
                encoding="utf-8"
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Seeds:", SEEDS, flush=True)
    print("Train rows:", len(train), flush=True)
    print("Validation rows:", len(validation), flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Embedding: FastText 300, trainable", flush=True)

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
                f"\n=== SUB FastText | seed {seed} ===",
                flush=True,
            )

            set_seed(seed)

            folder = (
                PROJECT / "checkpoints"
                / f"sub_lstm_multicat40k_fasttext_seed{seed}"
            )
            folder.mkdir(parents=True, exist_ok=False)

            config = {"seed": seed, **shared_config}

            (folder / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

            model = base.SubjectivityLSTM(
                vocab_size=len(vocab),
                embedding_dim=300,
                hidden_dim=64,
            )

            with torch.no_grad():
                model.embedding.weight.copy_(vectors)
                model.embedding.weight[0].zero_()

            model.embedding.weight.requires_grad_(True)
            model.to("cuda")

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
                    raise RuntimeError("出現非有限數值。")

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

        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "0965_sub_fasttext_per_seed.csv",
            index=False,
        )

        print("Best:", best, flush=True)

    results = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "model": "sub_lstm_multicat40k_fasttext300",
        "seeds": len(SEEDS),
        "accuracy_mean": results["validation_accuracy"].mean(),
        "accuracy_std": results["validation_accuracy"].std(),
        "macro_f1_mean": results["validation_macro_f1"].mean(),
        "macro_f1_std": results["validation_macro_f1"].std(),
    }])

    summary.to_csv(
        RESULTS_DIR / "0965_sub_fasttext_summary.csv",
        index=False,
    )

    print("\n=== Per-seed results ===", flush=True)
    print(results.to_string(index=False), flush=True)

    print("\n=== Three-seed summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    print("\nFastText SUB training completed.", flush=True)


if __name__ == "__main__":
    main()