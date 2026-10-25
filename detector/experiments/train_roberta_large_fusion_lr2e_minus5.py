import copy
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


core = importlib.import_module("train_roberta_fusion_product_qa")
evaluator = importlib.import_module("evaluate_detector_suite")

PROJECT = Path(__file__).resolve().parent

DATA_DIR = PROJECT / "data/processed/unbiased"
MODEL_DIR = PROJECT / "models/roberta-large"

INC_CHECKPOINT = (
    PROJECT / "checkpoints/inc_siamese_pqa200k_seed42/best.pt"
)
SUB_CHECKPOINT = (
    PROJECT
    / "checkpoints/sub_lstm_multicat40k_fasttext_seed42/best.pt"
)

RUN_DIR = (
    PROJECT / "checkpoints/19_roberta_large_inc_sub"
)
RESULTS_DIR = (
    PROJECT / "results/19_roberta_large_inc_sub"
)

SEED = 42
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
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請透過 sbatch 配置 GPU。")

    torch.set_num_threads(4)

    # 使用 RoBERTa-large backbone。
    core.base.MODEL_DIR = MODEL_DIR
    core.base.DATA_DIR = DATA_DIR
    core.base.MAX_LENGTH = MAX_LENGTH

    # 固定使用新版 INC / FastText SUB seed 42。
    core.base.FEATURE_DIR = (
        PROJECT
        / "data/features/sub_lstm_multicat40k_fasttext_seed42"
    )
    core.DATA_DIR = DATA_DIR
    core.INC_DIR = (
        PROJECT / "data/features/inc_siamese_pqa200k_seed42"
    )
    core.INC_CHECKPOINT = INC_CHECKPOINT

    tokenizer = core.base.AutoTokenizer.from_pretrained(
        MODEL_DIR,
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

    inc_hash = sha256(INC_CHECKPOINT)
    sub_hash = sha256(SUB_CHECKPOINT)

    for sources in [train_hashes, val_hashes]:
        if sources["inc_checkpoint_sha256"] != inc_hash:
            raise ValueError("INC 特徵與 checkpoint 不一致。")

        if sources["sub_checkpoint_sha256"] != sub_hash:
            raise ValueError("SUB 特徵與 checkpoint 不一致。")

    # 固定本輪 test 檔案，但先不計算 test 成績。
    test_path = DATA_DIR / "test.csv"
    test_hash = sha256(test_path)

    shared_config = {
        "seed": SEED,
        "backbone_path": str(MODEL_DIR),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LR,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "input": "question + product_description",
        "inc_dimension": 1,
        "sub_dimension": 64,
        "train_sources": train_hashes,
        "validation_sources": val_hashes,
        "test_csv_sha256": test_hash,
        "selection_metric": "validation_macro_f1",
    }

    # 完成的實驗可沿用，未完成的目錄不會覆蓋。
    completed = {}

    for name, use_inc, use_sub in EXPERIMENTS:
        folder = RUN_DIR / name

        if not folder.exists():
            continue

        if not (folder / "completed.json").is_file():
            raise FileExistsError(
                f"發現未完成實驗：{folder}\n"
                "請先檢查 log，將該目錄移至備份位置後重跑。"
            )

        config = json.loads(
            (folder / "config.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            **shared_config,
            "model": name,
            "use_inc": use_inc,
            "use_sub": use_sub,
        }

        if config != expected:
            raise ValueError(f"既有實驗設定不同：{folder}")

        if not (folder / "best.pt").is_file():
            raise FileNotFoundError(folder / "best.pt")

        completed[name] = json.loads(
            (folder / "best_validation.json").read_text(
                encoding="utf-8"
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Backbone: RoBERTa-large", flush=True)
    print("Downstream seed:", SEED, flush=True)
    print("INC / SUB seeds: fixed 42", flush=True)
    print("Train rows:", len(train_df), flush=True)
    print("Validation rows:", len(val_df), flush=True)

    # 同一份初始權重供四組使用。
    initial_state = None

    if len(completed) < len(EXPERIMENTS):
        set_seed()

        initial_model = core.ContextINCSUBModel(
            use_inc=False,
            use_sub=False,
        )
        initial_state = copy.deepcopy(
            initial_model.state_dict()
        )
        del initial_model

    validation_rows = []

    for name, use_inc, use_sub in EXPERIMENTS:
        if name in completed:
            best = completed[name]
            print(f"\nReuse {name}", flush=True)

        else:
            print(f"\n=== Training: {name} ===", flush=True)

            model = core.ContextINCSUBModel(
                use_inc=use_inc,
                use_sub=use_sub,
            )
            model.load_state_dict(initial_state)
            model.to("cuda")

            # 匹配四組的 dropout seed 和 batch 順序。
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
            folder.mkdir(parents=True, exist_ok=False)

            config = {
                **shared_config,
                "model": name,
                "use_inc": use_inc,
                "use_sub": use_sub,
            }
            write_json(folder / "config.json", config)

            history = []
            best_f1 = -1.0
            best = None

            for epoch in range(1, EPOCHS + 1):
                train_metrics = core.base.run_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
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

                    write_json(
                        folder / "best_validation.json",
                        best,
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
                    f"Val accuracy: "
                    f"{row['validation_accuracy']:.2%} | "
                    f"Val macro-F1: "
                    f"{row['validation_macro_f1']:.4f}",
                    flush=True,
                )

            write_json(
                folder / "completed.json",
                {"model": name, "seed": SEED},
            )

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        validation_rows.append({
            "model": name,
            "seed": SEED,
            **best,
        })

        pd.DataFrame(validation_rows).to_csv(
            RESULTS_DIR / "validation.csv",
            index=False,
        )

    del initial_state

    print("\nAll 4 training runs completed.", flush=True)
    print("Starting fixed test evaluation.", flush=True)

    # 四組完成後，才計算 test 成績。
    if sha256(test_path) != test_hash:
        raise ValueError("訓練期間 test CSV 已改變。")

    test_df = pd.read_csv(test_path)

    required = {
        "question", "product_description",
        "product_group", "label",
    }

    if not required.issubset(test_df.columns):
        raise ValueError("Test 缺少必要欄位。")

    if test_df.empty:
        raise ValueError("Test 沒有資料。")

    if test_df[
        ["question", "product_group", "label"]
    ].isna().any().any():
        raise ValueError("Test 有缺失的必要欄位。")

    if set(test_df["label"].unique()) != {0, 1}:
        raise ValueError("Test 標籤不正確。")

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        if (
            set(test_df["product_group"])
            & set(df["product_group"])
        ):
            raise ValueError(f"Test 與 {name} 商品重疊。")

    # 沿用 18 的輔助特徵推論函數，另存本輪 cache。
    evaluator.OUTPUT = RESULTS_DIR / "test_aux_cache"
    evaluator.OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluator.BATCH_SIZE = BATCH_SIZE

    auxiliary_plan = {
        "auxiliary_checkpoints": {
            "inc": {"path": str(INC_CHECKPOINT)},
            "sub": {"path": str(SUB_CHECKPOINT)},
        },
    }

    feature_signature = {
        "test_csv_sha256": test_hash,
        "inc_checkpoint_sha256": inc_hash,
        "sub_checkpoint_sha256": sub_hash,
    }

    features = evaluator.make_aux_features(
        test_df,
        auxiliary_plan,
        feature_signature,
    )

    encoded = tokenizer(
        test_df["question"].astype(str).tolist(),
        test_df["product_description"]
        .fillna("")
        .astype(str)
        .tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    test_dataset = evaluator.InputDataset(
        encoded,
        features=features,
    )
    actual = test_df["label"].to_numpy(dtype=int)

    test_rows = []

    for name, use_inc, use_sub in EXPERIMENTS:
        print(f"\n=== Test: {name} ===", flush=True)

        folder = RUN_DIR / name

        checkpoint = torch.load(
            folder / "best.pt",
            map_location="cpu",
            weights_only=True,
        )

        model = core.ContextINCSUBModel(
            use_inc=use_inc,
            use_sub=use_sub,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to("cuda")

        probabilities = evaluator.predict(
            model,
            test_dataset,
            custom=True,
        )

        if probabilities.shape != (len(test_df), 2):
            raise ValueError("Test 預測維度不正確。")

        predicted = probabilities.argmax(axis=1)

        row = {
            "model": name,
            "seed": SEED,
            "selected_epoch": checkpoint["validation"]["epoch"],
            "test_rows": len(test_df),
            "test_accuracy": float(
                evaluator.accuracy_score(actual, predicted)
            ),
            "test_macro_f1": float(
                evaluator.f1_score(
                    actual,
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

        test_rows.append(row)

        pd.DataFrame({
            "row_index": np.arange(len(test_df)),
            "actual": actual,
            "predicted": predicted,
            "probability_label_1": probabilities[:, 1],
        }).to_csv(
            RESULTS_DIR / f"test_predictions_{name}.csv",
            index=False,
        )

        pd.DataFrame(test_rows).to_csv(
            RESULTS_DIR / "test_results.csv",
            index=False,
        )

        print(
            f"Test accuracy: {row['test_accuracy']:.2%} | "
            f"macro-F1: {row['test_macro_f1']:.4f}",
            flush=True,
        )

        del model, checkpoint
        torch.cuda.empty_cache()

    results = pd.DataFrame(test_rows)
    control = results.loc[
        results["model"] == "control"
    ].iloc[0]

    results["accuracy_minus_control_pp"] = 100 * (
        results["test_accuracy"] - control["test_accuracy"]
    )
    results["macro_f1_minus_control"] = (
        results["test_macro_f1"] - control["test_macro_f1"]
    )

    results.to_csv(
        RESULTS_DIR / "test_results.csv",
        index=False,
    )

    print("\n=== Validation results ===", flush=True)
    print(
        pd.DataFrame(validation_rows).to_string(index=False),
        flush=True,
    )

    print("\n=== Test results: seed 42 ===", flush=True)
    print(results.to_string(index=False), flush=True)

    print(
        "\nAll 4 training and test evaluations completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()