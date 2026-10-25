import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "models/roberta-base"
DATA_DIR = PROJECT / "data/processed/unbiased"
FEATURE_DIR = PROJECT / "data/features/sub_lstm_seed42"
RUN_DIR = PROJECT / "checkpoints/roberta_sub_seed42"

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5


class ContextSUBModel(nn.Module):
    def __init__(self, use_sub):
        super().__init__()
        self.use_sub = use_sub
        self.backbone = AutoModel.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            add_pooling_layer=False,
        )
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(
            self.backbone.config.hidden_dropout_prob
        )
        self.dense = nn.Linear(hidden + 64, hidden)
        self.output = nn.Linear(hidden, 2)

    def forward(self, ids, mask, sub_features):
        encoded = self.backbone(
            input_ids=ids,
            attention_mask=mask,
        )
        context = encoded.last_hidden_state[:, 0, :]

        if not self.use_sub:
            sub_features = torch.zeros_like(sub_features)

        joined = torch.cat([context, sub_features], dim=1)
        hidden = torch.tanh(self.dense(self.dropout(joined)))
        return self.output(self.dropout(hidden))


def make_dataset(split, tokenizer):
    csv_path = DATA_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)

    saved = torch.load(
        FEATURE_DIR / f"{split}.pt",
        map_location="cpu",
        weights_only=True,
    )

    # 確認特徵對應目前 CSV，避免列順序錯配。
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert saved["source_csv_sha256"] == csv_hash
    assert saved["features"].shape == (len(df), 64)
    assert torch.isfinite(saved["features"]).all()

    assert df["question"].notna().all()
    assert df["product_group"].notna().all()
    assert set(df["label"].unique()) == {0, 1}

    encoded = tokenizer(
        df["question"].tolist(),
        df["product_description"].fillna("").astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        saved["features"].float(),
        torch.tensor(df["label"].tolist(), dtype=torch.long),
    )
    return df, dataset, saved["sub_checkpoint_sha256"]


def run_epoch(model, loader, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual, predicted = [], []

    with torch.set_grad_enabled(training):
        for ids, mask, features, labels in loader:
            ids = ids.to("cuda")
            mask = mask.to("cuda")
            features = features.to("cuda")
            labels = labels.to("cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(ids, mask, features)
            loss = nn.functional.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * len(labels)
            actual.extend(labels.cpu().tolist())
            predicted.extend(
                logits.detach().argmax(dim=1).cpu().tolist()
            )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted, average="macro", zero_division=0
        ),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run through sbatch with a GPU.")
    if RUN_DIR.exists():
        raise FileExistsError(f"Experiment already exists: {RUN_DIR}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    train_df, train_dataset, train_sub_hash = make_dataset(
        "train", tokenizer
    )
    val_df, val_dataset, val_sub_hash = make_dataset(
        "validation", tokenizer
    )

    assert train_sub_hash == val_sub_hash
    assert not (
        set(train_df["product_group"]) & set(val_df["product_group"])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 同一份初始權重供兩個實驗使用。
    initial_model = ContextSUBModel(use_sub=False)
    initial_state = copy.deepcopy(initial_model.state_dict())
    del initial_model

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    comparison = []

    for name, use_sub in [("control", False), ("with_sub", True)]:
        print(f"\n=== {name} | seed {SEED} ===", flush=True)

        model = ContextSUBModel(use_sub=use_sub)
        model.load_state_dict(initial_state)
        model.to("cuda")

        # 匹配 dropout 隨機種子和 batch 順序。
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=0.01
        )
        total_steps = EPOCHS * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        folder = RUN_DIR / name
        folder.mkdir()
        config = {
            "model": name,
            "seed": SEED,
            "use_sub": use_sub,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LR,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "sub_dimension": 64,
            "sub_checkpoint_sha256": train_sub_hash,
            "backbone_path": str(MODEL_DIR),
            "input": "question + product_description",
        }
        (folder / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        history = []
        best_f1 = -1.0
        best = None

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_epoch(
                model, train_loader, optimizer, scheduler
            )
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
                best = row.copy()

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "validation": best,
                }, folder / "best.pt")

                (folder / "best_validation.json").write_text(
                    json.dumps(best, indent=2), encoding="utf-8"
                )
                tokenizer.save_pretrained(folder / "tokenizer")

            pd.DataFrame(history).to_csv(
                folder / "history.csv", index=False
            )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train loss: {row['train_loss']:.4f} | "
                f"Val accuracy: {row['validation_accuracy']:.2%} | "
                f"Val macro-F1: {row['validation_macro_f1']:.4f}",
                flush=True,
            )

        comparison.append({
            "model": name,
            "seed": SEED,
            **best,
        })

        # 每完成一個模型，立即保存比較結果。
        results_dir = PROJECT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(comparison).to_csv(
            results_dir / "roberta_sub_seed42.csv",
            index=False,
        )

        print("Best validation result:", best, flush=True)

    print("\nComparison:", flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    print("Matched SUB comparison completed.", flush=True)


if __name__ == "__main__":
    main()