import argparse
import hashlib
import json
import math
import unicodedata
import html
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from humor_preference_data import ensure_pairs

PROJECT = Path(__file__).resolve().parents[1]
BASE = (
    PROJECT / "models/detector/checkpoints"
    / "11_roberta_large/qproduct_seed42/best_model"
)
DATA = (
    PROJECT / "data/source/preferences"
    / "humor_preference_v1/pairwise"
)


def normalize(text):
    return " ".join(
        unicodedata.normalize(
            "NFKC", html.unescape(text)
        ).casefold().split()
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pairs(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")

    for row in rows:
        for key in ("chosen", "rejected"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"Missing text field: {key}")
            if not normalize(row[key]):
                raise ValueError(f"Empty text field: {key}")
        if normalize(row["chosen"]) == normalize(row["rejected"]):
            raise ValueError("Identical chosen and rejected jokes")
        if not isinstance(row.get("context", ""), str):
            raise ValueError("Context must be text")
    return rows


def joke_keys(rows):
    return {
        normalize(row[key])
        for row in rows
        for key in ("chosen", "rejected")
    }


class Pairs(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        # Chosen first, rejected second. Joke first, context second.
        texts = [
            row[key]
            for key in ("chosen", "rejected")
            for row in rows
        ]
        contexts = [
            row.get("context", "")
            for _ in range(2)
            for row in rows
        ]
        encoded = self.tokenizer(
            texts,
            contexts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # One label per preference pair, not per encoded joke.
        encoded["labels"] = torch.ones(
            len(rows), dtype=torch.long
        )
        return encoded


class PreferenceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False,
        num_items_in_batch=None
    ):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        logits = model(**inputs).logits.float()

        if logits.shape != (2 * len(labels), 2):
            raise ValueError(f"Unexpected logits: {logits.shape}")

        scores = logits[:, 1] - logits[:, 0]
        n = len(labels)
        delta = scores[:n] - scores[n:]
        loss = torch.nn.functional.softplus(-delta).mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite preference loss")

        outputs = {"logits": delta.unsqueeze(-1)}
        return (loss, outputs) if return_outputs else loss


def compute_metrics(prediction):
    delta = np.asarray(prediction.predictions).reshape(-1)
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite evaluation scores")
    return {
        "pairwise_accuracy": float(np.mean(delta > 0)),
        "score_tie_rate": float(np.mean(delta == 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["arena", "mechanics"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume latest saved checkpoint in this run"
    )
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    if args.steps % 50:
        raise ValueError("Use steps divisible by 50")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError("Invalid learning rate")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit through the GPU Slurm script")

    ensure_pairs(args.source)
    train_path = DATA / args.source / "train.jsonl"
    validation_path = DATA / args.source / "validation.jsonl"
    train_rows = read_pairs(train_path)
    validation_rows = read_pairs(validation_path)

    if joke_keys(train_rows) & joke_keys(validation_rows):
        raise ValueError("Train/validation joke overlap")

    output = (
        PROJECT / "models/detector/preferences"
        / f"{args.source}_v1_seed{args.seed}"
    )
    metadata = {
        "source": args.source,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_checkpoint": str(BASE),
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
        "max_length": 128,
        "use_fast": False,
        "input_order": ["joke", "context"],
        "positive_label": 1,
        "score": "logit_1_minus_logit_0",
        "loss": "softplus_negative_score_difference",
        "precision": "float32",
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "selection": "minimum_validation_loss",
    }

    resume_checkpoint = None
    if args.resume:
        previous = json.loads(
            (output / "run.json").read_text(encoding="utf-8")
        )
        if previous != metadata:
            raise ValueError("Resume settings/data differ from original run")
        if (output / "completed.json").exists():
            raise ValueError("Run already completed; no retraining needed")

        checkpoints = [
            path for path in output.glob("checkpoint-*")
            if path.is_dir()
            and path.name.removeprefix("checkpoint-").isdigit()
        ]
        if not checkpoints:
            raise ValueError("No saved checkpoint available to resume")
        latest = max(
            checkpoints,
            key=lambda path: int(path.name.split("-")[-1])
        )
        required = [
            "trainer_state.json", "optimizer.pt", "scheduler.pt",
            "rng_state.pth",
        ]
        missing = [
            name for name in required
            if not (latest / name).is_file()
        ]
        if missing:
            raise ValueError(f"Incomplete checkpoint {latest}: {missing}")
        resume_checkpoint = str(latest)
        print("Resuming:", resume_checkpoint, flush=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, use_fast=False, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, local_files_only=True
    )
    if model.config.num_labels != 2:
        raise ValueError("Expected binary detector")

    training_args = TrainingArguments(
        output_dir=str(output),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        report_to="none",
        save_safetensors=True,
    )

    trainer = PreferenceTrainer(
        model=model,
        args=training_args,
        train_dataset=Pairs(train_rows),
        eval_dataset=Pairs(validation_rows),
        data_collator=PairCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(json.dumps(metadata, indent=2), flush=True)
    print(
        f"Pairs: train={len(train_rows)}, "
        f"validation={len(validation_rows)}",
        flush=True,
    )

    result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    # Trainer has reloaded the best source-validation checkpoint.
    trainer.save_model(str(output / "best_model"))

    completion = {
        "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "exported_model": str(output / "best_model"),
    }
    (output / "completed.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print("Completed:", json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
