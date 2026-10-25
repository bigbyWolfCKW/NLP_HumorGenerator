import gc
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


core = importlib.import_module("train_roberta_fusion_product_qa")
evaluator = importlib.import_module("evaluate_detector_suite")

PROJECT = Path(__file__).resolve().parent

RUN_DIR = PROJECT / "checkpoints/20_large_sub_lr1e5"
OUTPUT = PROJECT / "results/20_large_sub_lr1e5"

PLAN_PATH = PROJECT / "results/17_test_plan.json"
TEST_PATH = PROJECT / "data/processed/unbiased/test.csv"

EXPERIMENTS = [
    ("control", False),
    ("with_sub", True),
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    torch.set_num_threads(4)

    core.base.MODEL_DIR = PROJECT / "models/roberta-large"
    evaluator.BATCH_SIZE = 16

    plan = read_json(PLAN_PATH)
    test_hash = sha256(TEST_PATH)

    if test_hash != plan["data_sha256"]["test"]:
        raise ValueError("Test CSV 與之前鎖定的版本不同。")

    aux_hashes = {}

    for name, record in plan["auxiliary_checkpoints"].items():
        current_hash = sha256(Path(record["path"]))

        if current_hash != record["sha256"]:
            raise ValueError(f"{name} checkpoint 已變更。")

        aux_hashes[name] = current_hash

    # 在評估前固定兩份 checkpoint。
    runs = []

    for name, use_sub in EXPERIMENTS:
        folder = RUN_DIR / name
        config = read_json(folder / "config.json")
        best = read_json(folder / "best_validation.json")

        if (
            config["seed"] != 42
            or config["learning_rate"] != 1e-5
            or config["use_inc"] is not False
            or config["use_sub"] != use_sub
            or config["backbone_path"] != str(core.base.MODEL_DIR)
        ):
            raise ValueError(f"設定與預期不同：{name}")

        for split in ["train", "validation"]:
            sources = config[f"{split}_sources"]
            csv_path = (
                PROJECT / "data/processed/unbiased"
                / f"{split}.csv"
            )

            if sources["csv_sha256"] != sha256(csv_path):
                raise ValueError(f"{split} CSV 不一致。")

            if sources["inc_checkpoint_sha256"] != aux_hashes["inc"]:
                raise ValueError("INC 來源不同。")

            if sources["sub_checkpoint_sha256"] != aux_hashes["sub"]:
                raise ValueError("SUB 來源不同。")

        tokenizer_files = {
            path.name: sha256(path)
            for path in (folder / "tokenizer").iterdir()
            if path.is_file()
        }

        runs.append({
            "model": name,
            "use_sub": use_sub,
            "folder": str(folder),
            "checkpoint_sha256": sha256(folder / "best.pt"),
            "config": config,
            "validation": best,
            "tokenizer_files_sha256": tokenizer_files,
        })

    execution = {
        "seed": 42,
        "test_csv_sha256": test_hash,
        "auxiliary_sha256": aux_hashes,
        "backbone_config_sha256": sha256(
            core.base.MODEL_DIR / "config.json"
        ),
        "batch_size": 16,
        "runs": runs,
        "evaluation_type": "exploratory_test_extension",
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    execution_path = OUTPUT / "test_execution.json"

    if execution_path.exists():
        if read_json(execution_path) != execution:
            raise ValueError("既有 test 執行設定不同，停止覆蓋。")
    else:
        execution_path.write_text(
            json.dumps(execution, indent=2),
            encoding="utf-8",
        )

    df = pd.read_csv(TEST_PATH)

    if (
        df.empty
        or df[["question", "product_group", "label"]]
        .isna().any().any()
        or set(df["label"].unique()) != {0, 1}
    ):
        raise ValueError("Test 資料不完整或標籤不正確。")

    actual = df["label"].to_numpy(dtype=int)

    feature_signature = {
        "test_csv_sha256": test_hash,
        "inc_checkpoint_sha256": aux_hashes["inc"],
        "sub_checkpoint_sha256": aux_hashes["sub"],
    }

    # 優先沿用 19 的 test 特徵，但先核對來源。
    cache_path = (
        PROJECT
        / "results/19_roberta_large_inc_sub"
        / "test_aux_cache/test_aux_features.pt"
    )

    if cache_path.exists():
        saved = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if saved["signature"] != feature_signature:
            raise ValueError("19 的 test 特徵來源不同。")

        features = saved["features"]

    else:
        evaluator.OUTPUT = OUTPUT / "test_aux_cache"
        evaluator.OUTPUT.mkdir(parents=True, exist_ok=True)

        features = evaluator.make_aux_features(
            df, plan, feature_signature
        )

    if (
        features.shape != (len(df), 65)
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Test 特徵維度或數值不正確。")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Test rows:", len(df), flush=True)
    print("Seed: 42; no training.", flush=True)

    rows = []

    for run in runs:
        name = run["model"]
        folder = Path(run["folder"])
        config = run["config"]

        print(f"\n=== Test: {name} ===", flush=True)

        checkpoint = torch.load(
            folder / "best.pt",
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["config"] != config:
            raise ValueError("Checkpoint 與 config.json 不一致。")

        if checkpoint["validation"] != run["validation"]:
            raise ValueError("最佳 validation metadata 不一致。")

        model = core.ContextINCSUBModel(
            use_inc=False,
            use_sub=run["use_sub"],
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to("cuda")

        tokenizer = evaluator.AutoTokenizer.from_pretrained(
            folder / "tokenizer",
            use_fast=False,
            local_files_only=True,
        )

        encoded = tokenizer(
            df["question"].astype(str).tolist(),
            df["product_description"]
            .fillna("")
            .astype(str)
            .tolist(),
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )

        dataset = evaluator.InputDataset(
            encoded, features=features
        )

        probabilities = evaluator.predict(
            model, dataset, custom=True
        )

        if probabilities.shape != (len(df), 2):
            raise ValueError("Test 預測維度不正確。")

        predicted = probabilities.argmax(axis=1)

        row = {
            "model": name,
            "seed": 42,
            "learning_rate": 1e-5,
            "selected_epoch": run["validation"]["epoch"],
            "test_rows": len(df),
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
        rows.append(row)

        pd.DataFrame({
            "row_index": np.arange(len(df)),
            "actual": actual,
            "predicted": predicted,
            "probability_label_1": probabilities[:, 1],
        }).to_csv(
            OUTPUT / f"test_predictions_{name}.csv",
            index=False,
        )

        pd.DataFrame(rows).to_csv(
            OUTPUT / "test_results.csv",
            index=False,
        )

        print(
            f"Test accuracy: {row['test_accuracy']:.2%} | "
            f"macro-F1: {row['test_macro_f1']:.4f}",
            flush=True,
        )

        del model, checkpoint, tokenizer, encoded, dataset
        gc.collect()
        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
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
        OUTPUT / "test_results.csv",
        index=False,
    )

    print("\n=== Test comparison: seed 42 ===", flush=True)
    print(results.to_string(index=False), flush=True)
    print("\nBoth test evaluations completed.", flush=True)


if __name__ == "__main__":
    main()