# 這版使用隨機初始化的 embedding、1,000 筆小樣本；尚未完整復現報告的 FastText 和較大資料規模。訓練後會保存 encoder，之後可凍結它並提取 64 維特徵。
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3

TOKEN_PATTERN = r"\w+|[^\w\s]"


def tokenize(text):
    return re.findall(TOKEN_PATTERN, text.lower())


class SubjectivityLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode(self, ids, lengths):
        embedded = self.embedding(ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]  # 64 維 SUB 特徵

    def forward(self, ids, lengths):
        features = self.encode(ids, lengths)
        return self.classifier(self.dropout(features)).squeeze(-1)


def make_dataset(df, vocab):
    sequences = []
    lengths = []

    for text in df["text"]:
        ids = [
            vocab.get(token, 1)
            for token in tokenize(text)[:MAX_LENGTH]
        ]
        if not ids:
            ids = [1]

        lengths.append(len(ids))
        sequences.append(ids + [0] * (MAX_LENGTH - len(ids)))

    return TensorDataset(
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(df["label"].tolist(), dtype=torch.float32),
    )


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    actual, predicted = [], []

    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            ids = ids.to("cuda")
            labels = labels.to("cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(ids, lengths)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            actual.extend(labels.long().cpu().tolist())
            predicted.extend(
                (logits.detach() >= 0).long().cpu().tolist()
            )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted, average="macro", zero_division=0
        ),
    }


def main():
    project = Path(__file__).resolve().parent
    data_dir = project / "data/processed/subjectivity"
    run_dir = project / "checkpoints/sub_lstm_seed42"

    if not torch.cuda.is_available():
        raise RuntimeError("Run through sbatch with a GPU.")
    if run_dir.exists():
        raise FileExistsError(f"Experiment already exists: {run_dir}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "validation.csv")

    for df in [train, val]:
        assert df["text"].notna().all()
        assert df["parent_asin"].notna().all()
        assert set(df["label"].unique()) == {0, 1}

    assert not (
        set(train["parent_asin"]) & set(val["parent_asin"])
    ), "SUB train/validation product overlap."

    # 詞彙表只從 SUB train 建立
    counts = Counter(
        token
        for text in train["text"]
        for token in tokenize(text)
    )
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(10000):
        vocab[token] = len(vocab)

    train_loader = DataLoader(
        make_dataset(train, vocab),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    val_loader = DataLoader(
        make_dataset(val, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SubjectivityLSTM(len(vocab)).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    config = {
        "seed": SEED,
        "embedding_dim": 100,
        "hidden_dim": 64,
        "dropout": 0.3,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.001,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "tokenizer_pattern": TOKEN_PATTERN,
        "lowercase": True,
        "pad_id": 0,
        "unk_id": 1,
        "labels": {"0": "description proxy", "1": "review proxy"},
        "train_rows": len(train),
        "validation_rows": len(val),
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Rows:", len(train), "/", len(val), flush=True)

    history = []
    best_f1 = -1.0
    best_result = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        val_metrics = run_epoch(model, val_loader)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_result = row.copy()
            stale = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "config": config,
                "epoch": epoch,
                "validation": val_metrics,
            }, run_dir / "best.pt")

            (run_dir / "best_validation.json").write_text(
                json.dumps(best_result, indent=2), encoding="utf-8"
            )
        else:
            stale += 1

        pd.DataFrame(history).to_csv(
            run_dir / "history.csv", index=False
        )

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val accuracy: {val_metrics['accuracy']:.2%} | "
            f"Val macro-F1: {val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if stale >= PATIENCE:
            print("Early stopping.", flush=True)
            break

    print("Best validation result:", best_result, flush=True)
    print("Saved checkpoint:", run_dir / "best.pt", flush=True)


if __name__ == "__main__":
    main()