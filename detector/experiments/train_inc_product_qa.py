import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data/processed/inc_pqa_200k"
CACHE_PATH = (
    PROJECT / "data/external/fasttext/inc_pqa_vectors.pt"
)
RESULTS_DIR = PROJECT / "results"

SEEDS = [42, 43, 44]
MAX_LENGTH = 64
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 3
LR = 0.001


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IncongruityLSTM(nn.Module):
    def __init__(self, vectors, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            vectors.clone(),
            freeze=True,
            padding_idx=0,
        )
        self.encoder = nn.LSTM(
            vectors.shape[1],
            hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode(self, ids, lengths):
        packed = pack_padded_sequence(
            self.embedding(ids),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        return hidden[-1]

    def forward(self, q_ids, q_lengths, p_ids, p_lengths):
        q_rep = self.encode(q_ids, q_lengths)
        p_rep = self.encode(p_ids, p_lengths)

        difference = torch.abs(q_rep - p_rep)

        return self.classifier(
            self.dropout(difference)
        ).squeeze(-1)


def encode_texts(texts, vocab, config):
    pattern = re.compile(config["tokenizer_pattern"])
    sequences = []
    lengths = []

    for text in texts:
        tokens = pattern.findall(
            str(text).lower()
        )[:MAX_LENGTH]

        ids = [
            vocab.get(token, config["unk_id"])
            for token in tokens
        ] or [config["unk_id"]]

        lengths.append(len(ids))
        sequences.append(
            ids + [config["pad_id"]] * (MAX_LENGTH - len(ids))
        )

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def make_dataset(df, vocab, config):
    q_ids, q_lengths = encode_texts(
        df["question"], vocab, config
    )
    p_ids, p_lengths = encode_texts(
        df["product_description"], vocab, config
    )

    labels = torch.tensor(
        df["pair_label"].tolist(),
        dtype=torch.float32,
    )

    return TensorDataset(
        q_ids, q_lengths,
        p_ids, p_lengths,
        labels,
    )


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    actual = []
    predicted = []

    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, start=1):
            q_ids, q_lengths, p_ids, p_lengths, labels = batch

            q_ids = q_ids.to("cuda")
            p_ids = p_ids.to("cuda")
            labels = labels.to("cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(
                q_ids, q_lengths,
                p_ids, p_lengths,
            )
            loss = criterion(logits, labels)

            if (
                not torch.isfinite(loss)
                or not torch.isfinite(logits).all()
            ):
                raise RuntimeError("Non-finite loss or logits.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()

            total_loss += loss.item() * len(labels)

            actual.extend(
                labels.detach().cpu().tolist()
            )
            predicted.extend(
                (logits.detach() >= 0)
                .long()
                .cpu()
                .tolist()
            )

            if training and step % 500 == 0:
                print(
                    f"  Batch {step}/{len(loader)}",
                    flush=True,
                )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    torch.set_num_threads(4)

    cache = torch.load(
        CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    vocab = cache["vocab"]
    vectors = cache["embedding_matrix"].float()
    cache_config = cache["config"]

    if vectors.shape != (len(vocab), 300):
        raise ValueError("Embedding 維度不正確。")

    if not torch.isfinite(vectors).all():
        raise ValueError("Embedding 有非有限數值。")

    if cache_config["pad_id"] != 0:
        raise ValueError("PAD ID 與模型不一致。")

    if not cache_config["lowercase"]:
        raise ValueError("目前 tokenizer 預期 lowercase。")

    manifest_path = DATA_DIR / "manifest.json"

    if sha256(manifest_path) != (
        cache_config["dataset_manifest_sha256"]
    ):
        raise ValueError("Dataset manifest 與快取不一致。")

    if sha256(DATA_DIR / "train_source.csv") != (
        cache_config["train_source_sha256"]
    ):
        raise ValueError("Training source 與快取不一致。")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    frames = {}
    hashes = {}

    for split, expected_rows in [
        ("train", 200_000),
        ("validation", 50_000),
    ]:
        path = DATA_DIR / f"{split}_pairs.csv"
        file_hash = sha256(path)

        if file_hash != manifest[f"{split}_pairs_sha256"]:
            raise ValueError(f"{split} pairs 已變更。")

        # 保留 "NA"、"null" 等字面文字。
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        required = {
            "question",
            "product_description",
            "question_group",
            "product_group",
            "pair_label",
        }

        if not required.issubset(df.columns):
            raise ValueError(f"{split} 缺少必要欄位。")

        if len(df) != expected_rows:
            raise ValueError(f"{split} 配對筆數不正確。")

        empty = (
            df["question"].str.strip().eq("")
            | df["product_description"].str.strip().eq("")
        )
        if empty.any():
            raise ValueError(f"{split} 存在真正空白文字。")

        df["pair_label"] = pd.to_numeric(
            df["pair_label"], errors="raise"
        )

        counts = df["pair_label"].value_counts().to_dict()
        expected_counts = {
            0: expected_rows // 2,
            1: expected_rows // 2,
        }

        if counts != expected_counts:
            raise ValueError(f"{split} 標籤或比例不正確。")

        same_group = (
            df["question_group"] == df["product_group"]
        )
        if not same_group.eq(
            df["pair_label"].eq(1)
        ).all():
            raise ValueError(f"{split} 配對標籤與商品 group 不符。")

        frames[split] = df
        hashes[split] = file_hash

    train_groups = (
        set(frames["train"]["question_group"])
        | set(frames["train"]["product_group"])
    )
    val_groups = (
        set(frames["validation"]["question_group"])
        | set(frames["validation"]["product_group"])
    )

    if train_groups & val_groups:
        raise ValueError("INC train / validation 商品重疊。")

    datasets = {}

    for split, df in frames.items():
        print(f"Encoding {split}...", flush=True)
        datasets[split] = make_dataset(
            df, vocab, cache_config
        )

    shared_config = {
        "model_type": "siamese_lstm_absolute_difference",
        "embedding_dim": 300,
        "hidden_dim": 64,
        "dropout": 0.3,
        "max_length": MAX_LENGTH,
        "pad_id": cache_config["pad_id"],
        "unk_id": cache_config["unk_id"],
        "freeze_embeddings": True,
        "tokenizer_pattern": cache_config["tokenizer_pattern"],
        "lowercase": True,
        "pair_label_meaning": {
            "0": "mismatched_product",
            "1": "original_product",
        },
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "selection_metric": "validation_macro_f1",
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "train_pairs_sha256": hashes["train"],
        "validation_pairs_sha256": hashes["validation"],
        "embedding_cache_sha256": sha256(CACHE_PATH),
        "pretrained_word_type_coverage": (
            cache_config["word_type_coverage"]
        ),
    }

    completed = {}

    for seed in SEEDS:
        folder = (
            PROJECT / "checkpoints"
            / f"inc_siamese_pqa200k_seed{seed}"
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
    print("Train pairs:", len(frames["train"]), flush=True)
    print("Validation pairs:", len(frames["validation"]), flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Embedding: FastText 300, frozen", flush=True)

    rows = []

    for seed in SEEDS:
        if seed in completed:
            best = completed[seed]
            print(
                f"\nReuse completed seed {seed}",
                flush=True,
            )

        else:
            set_seed(seed)

            print(
                f"\n=== INC PQA 200K | seed {seed} ===",
                flush=True,
            )

            folder = (
                PROJECT / "checkpoints"
                / f"inc_siamese_pqa200k_seed{seed}"
            )
            folder.mkdir(parents=True, exist_ok=False)

            config = {"seed": seed, **shared_config}

            (folder / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

            model = IncongruityLSTM(vectors).to("cuda")

            train_loader = DataLoader(
                datasets["train"],
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                num_workers=0,
            )
            val_loader = DataLoader(
                datasets["validation"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            optimizer = torch.optim.Adam(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=LR,
            )

            history = []
            best_f1 = -1.0
            best = None
            stale_epochs = 0

            for epoch in range(1, MAX_EPOCHS + 1):
                train_metrics = run_epoch(
                    model, train_loader, optimizer
                )
                val_metrics = run_epoch(
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
            RESULTS_DIR / "14_inc_pqa_per_seed.csv",
            index=False,
        )

        print("Best:", best, flush=True)

    results = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "model": "inc_siamese_pqa200k",
        "seeds": len(SEEDS),
        "accuracy_mean": results["validation_accuracy"].mean(),
        "accuracy_std": results["validation_accuracy"].std(),
        "macro_f1_mean": results["validation_macro_f1"].mean(),
        "macro_f1_std": results["validation_macro_f1"].std(),
    }])

    summary.to_csv(
        RESULTS_DIR / "14_inc_pqa_summary.csv",
        index=False,
    )

    print("\n=== Per-seed results ===", flush=True)
    print(results.to_string(index=False), flush=True)

    print("\n=== Three-seed summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    print("\nINC training completed.", flush=True)


if __name__ == "__main__":
    main()