import copy
import gc
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

core = importlib.import_module("train_roberta_fusion_product_qa")
evaluator = importlib.import_module("evaluate_detector_suite")

PROJECT = Path(__file__).resolve().parent

DATA = PROJECT / "data/processed/biased"
MODEL = PROJECT / "models/roberta-large"
RUN = PROJECT / "checkpoints/24_large_biased_lr1e5"
OUT = PROJECT / "results/24_large_biased_lr1e5"

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

EXPERIMENTS = [
    ("control", False, False),
    ("with_inc", True, False),
    ("with_sub", False, True),
    ("with_inc_sub", True, True),
]


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )


def save_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(split, tokenizer, plan, auxiliary_hashes):
    path = DATA / f"{split}.csv"

    df = pd.read_csv(
        path,
        keep_default_na=False,
    )

    required = [
        "question",
        "product_description",
        "product_group",
        "label",
    ]

    if df.empty or not set(required).issubset(df.columns):
        raise ValueError(f"資料或欄位不完整：{path}")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"標籤不正確：{path}")

    for column in required[:3]:
        if (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                f"文字欄位為空：{split}/{column}"
            )

    evaluator.OUTPUT = OUT / "aux_cache" / split
    evaluator.OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 沿用 18 的快取格式。
    # 雖然欄位名稱包含 test，但這裡記錄的是當前
    # train 或 validation CSV 的 SHA256。
    signature = {
        "test_csv_sha256": sha256(path),
        "inc_checkpoint_sha256": auxiliary_hashes["inc"],
        "sub_checkpoint_sha256": auxiliary_hashes["sub"],
    }

    features = evaluator.make_aux_features(
        df,
        plan,
        signature,
    )

    if (
        tuple(features.shape) != (len(df), 65)
        or not torch.isfinite(features).all()
    ):
        raise ValueError(
            f"輔助特徵維度或數值不正確：{split}"
        )

    features = features.detach().cpu().float()

    encoded = tokenizer(
        df["question"].astype(str).tolist(),
        df["product_description"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        features,
        torch.tensor(
            df["label"].to_numpy(),
            dtype=torch.long,
        ),
    )

    sources = {
        "csv_sha256": sha256(path),
        "inc_checkpoint_sha256": auxiliary_hashes["inc"],
        "sub_checkpoint_sha256": auxiliary_hashes["sub"],
    }

    print(
        f"{split}: {len(df)} rows | "
        f"features={tuple(features.shape)}",
        flush=True,
    )

    return dataset, sources


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "請使用 sbatch 配置 GPU。"
        )

    if RUN.exists():
        raise FileExistsError(
            f"實驗目錄已存在，停止避免覆蓋：{RUN}"
        )

    torch.set_num_threads(4)

    # 核對 23 建立的 biased 資料。
    manifest_path = DATA / "manifest.json"
    manifest = read_json(manifest_path)

    data_hashes = {
        split: sha256(DATA / f"{split}.csv")
        for split in ["train", "validation", "test"]
    }

    for split, current_hash in data_hashes.items():
        expected_hash = manifest["output_sha256"][
            f"{split}.csv"
        ]

        if current_hash != expected_hash:
            raise ValueError(
                f"Biased CSV 與 manifest 不一致：{split}"
            )

    # 確認原有 unbiased 切分沒有被改動。
    for split in ["train", "validation", "test"]:
        path = (
            PROJECT
            / "data/processed/unbiased"
            / f"{split}.csv"
        )

        expected_hash = manifest["source_sha256"][
            f"unbiased_{split}"
        ]

        if sha256(path) != expected_hash:
            raise ValueError(
                f"Unbiased CSV 已變更：{split}"
            )

    # 核對先前 20 的主要設定。
    reference_path = (
        PROJECT
        / "checkpoints/20_large_sub_lr1e5"
        / "control/config.json"
    )

    reference = read_json(reference_path)

    required_settings = {
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "backbone_path": str(MODEL),
    }

    for key, expected in required_settings.items():
        if reference.get(key) != expected:
            raise ValueError(
                f"與 20 的設定不同：{key}="
                f"{reference.get(key)!r}；"
                f"預期 {expected!r}"
            )

    optional_settings = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_fraction": WARMUP_RATIO,
    }

    for key, expected in optional_settings.items():
        if key in reference and reference[key] != expected:
            raise ValueError(
                f"與 20 的設定不同：{key}="
                f"{reference[key]!r}；"
                f"預期 {expected!r}"
            )

    # 固定原先 INC / SUB extractor checkpoint。
    plan = read_json(
        PROJECT / "results/17_test_plan.json"
    )

    auxiliary_hashes = {}

    for name in ["inc", "sub"]:
        record = plan["auxiliary_checkpoints"][name]
        current_hash = sha256(Path(record["path"]))

        if current_hash != record["sha256"]:
            raise ValueError(
                f"{name} checkpoint 已變更。"
            )

        auxiliary_hashes[name] = current_hash

    core.base.MODEL_DIR = MODEL
    evaluator.BATCH_SIZE = BATCH_SIZE

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
        use_fast=False,
    )

    train_dataset, train_sources = make_dataset(
        "train",
        tokenizer,
        plan,
        auxiliary_hashes,
    )

    val_dataset, val_sources = make_dataset(
        "validation",
        tokenizer,
        plan,
        auxiliary_hashes,
    )

    RUN.mkdir(parents=True)
    OUT.mkdir(parents=True, exist_ok=True)

    save_json(
        OUT / "preflight.json",
        {
            "data_sha256": data_hashes,
            "preparation_manifest_sha256": sha256(
                manifest_path
            ),
            "auxiliary_sha256": auxiliary_hashes,
            "script_sha256": sha256(Path(__file__)),
            "reference_20_config_sha256": sha256(
                reference_path
            ),
            "evaluation": "validation_only",
        },
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print("Training dataset: biased", flush=True)
    print(
        "Seed: 42 | Learning rate: 1e-5",
        flush=True,
    )
    print(
        "Validation only; no test evaluation.",
        flush=True,
    )

    # 四組使用相同的初始權重。
    set_seed()

    initial_model = core.ContextINCSUBModel(
        use_inc=False,
        use_sub=False,
    )

    initial_state = copy.deepcopy(
        initial_model.state_dict()
    )

    del initial_model

    results = []

    for name, use_inc, use_sub in EXPERIMENTS:
        set_seed()

        folder = RUN / name
        folder.mkdir()

        config = {
            "model": name,
            "seed": SEED,
            "dataset": "biased",
            "use_inc": use_inc,
            "use_sub": use_sub,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_RATIO,
            "backbone_path": str(MODEL),
            "inc_dimension": 1,
            "sub_dimension": 64,
            "train_sources": train_sources,
            "validation_sources": val_sources,
            "input": "question_and_product",
            "selection": "best_validation_macro_f1",
        }

        save_json(
            folder / "config.json",
            config,
        )

        tokenizer.save_pretrained(
            folder / "tokenizer"
        )

        model = core.ContextINCSUBModel(
            use_inc=use_inc,
            use_sub=use_sub,
        )

        model.load_state_dict(
            initial_state,
            strict=True,
        )

        model.to("cuda")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        total_steps = len(train_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                total_steps * WARMUP_RATIO
            ),
            num_training_steps=total_steps,
        )

        history = []
        best = None
        best_f1 = -1.0

        print(
            f"\n=== Biased training: {name} ===",
            flush=True,
        )

        for epoch in range(1, EPOCHS + 1):
            train_metrics = core.base.run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            val_metrics = core.base.run_epoch(
                model,
                val_loader,
            )

            row = {
                "epoch": epoch,
                "train_loss": float(
                    train_metrics["loss"]
                ),
                "train_accuracy": float(
                    train_metrics["accuracy"]
                ),
                "validation_loss": float(
                    val_metrics["loss"]
                ),
                "validation_accuracy": float(
                    val_metrics["accuracy"]
                ),
                "validation_macro_f1": float(
                    val_metrics["macro_f1"]
                ),
            }

            if not all(
                np.isfinite(value)
                for value in row.values()
            ):
                raise RuntimeError(
                    "出現非有限訓練或評估數值。"
                )

            history.append(row)

            pd.DataFrame(history).to_csv(
                folder / "history.csv",
                index=False,
            )

            if row["validation_macro_f1"] > best_f1:
                best = dict(row)
                best_f1 = row["validation_macro_f1"]

                torch.save(
                    {
                        "model_state_dict": (
                            model.state_dict()
                        ),
                        "config": config,
                        "validation": best,
                    },
                    folder / "best.pt",
                )

                save_json(
                    folder / "best_validation.json",
                    best,
                )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train loss: "
                f"{row['train_loss']:.4f} | "
                f"Val accuracy: "
                f"{row['validation_accuracy']:.2%} | "
                f"Val macro-F1: "
                f"{row['validation_macro_f1']:.4f}",
                flush=True,
            )

        results.append(
            {
                "model": name,
                "seed": SEED,
                **best,
            }
        )

        pd.DataFrame(results).to_csv(
            OUT / "validation.csv",
            index=False,
        )

        save_json(
            folder / "completed.json",
            {
                "best": best,
                "checkpoint_sha256": sha256(
                    folder / "best.pt"
                ),
            },
        )

        del model, optimizer, scheduler
        del train_loader, val_loader

        gc.collect()
        torch.cuda.empty_cache()

    del initial_state

    for split, original_hash in data_hashes.items():
        if sha256(DATA / f"{split}.csv") != original_hash:
            raise RuntimeError(
                f"訓練期間資料已變更：{split}"
            )

    print(
        "\n=== Biased validation comparison ===",
        flush=True,
    )

    print(
        pd.DataFrame(results).to_string(index=False),
        flush=True,
    )

    print(
        "\nAll four biased training runs completed. "
        "No test evaluation.",
        flush=True,
    )


if __name__ == "__main__":
    main()