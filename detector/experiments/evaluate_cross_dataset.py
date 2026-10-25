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
MODEL = PROJECT / "models/roberta-large"
OUT = PROJECT / "results/25_large_cross_dataset"

VARIANTS = [
    ("control", False, False),
    ("with_inc", True, False),
    ("with_sub", False, True),
    ("with_inc_sub", True, True),
]


def sha(path):
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
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def make_plan():
    old_plan = read_json(
        PROJECT / "results/17_test_plan.json"
    )
    manifest = read_json(
        PROJECT / "data/processed/biased/manifest.json"
    )

    sources = {}

    for dataset in ["unbiased", "biased"]:
        sources[dataset] = {}

        for split in ["train", "validation", "test"]:
            path = (
                PROJECT
                / f"data/processed/{dataset}/{split}.csv"
            )
            digest = sha(path)

            if dataset == "unbiased":
                expected = manifest["source_sha256"][
                    f"unbiased_{split}"
                ]
            else:
                expected = manifest["output_sha256"][
                    f"{split}.csv"
                ]

            if digest != expected:
                raise ValueError(
                    f"資料已變更：{dataset}/{split}"
                )

            sources[dataset][split] = digest

    if (
        sources["unbiased"]["test"]
        != old_plan["data_sha256"]["test"]
    ):
        raise ValueError(
            "Unbiased test 與 17 鎖定的版本不同。"
        )

    auxiliary = {}

    for name in ["inc", "sub"]:
        record = old_plan["auxiliary_checkpoints"][name]
        auxiliary[name] = sha(Path(record["path"]))

        if auxiliary[name] != record["sha256"]:
            raise ValueError(
                f"輔助 checkpoint 已變更：{name}"
            )

    runs = []

    for training_dataset in ["unbiased", "biased"]:
        for name, use_inc, use_sub in VARIANTS:
            if training_dataset == "biased":
                directory = "24_large_biased_lr1e5"
            elif use_inc:
                directory = "22_large_inc_sub_lr1e5"
            else:
                directory = "20_large_sub_lr1e5"

            folder = (
                PROJECT / "checkpoints"
                / directory / name
            )

            config = read_json(
                folder / "config.json"
            )
            best = read_json(
                folder / "best_validation.json"
            )

            expected_settings = {
                "seed": 42,
                "learning_rate": 1e-5,
                "use_inc": use_inc,
                "use_sub": use_sub,
                "backbone_path": str(MODEL),
                "max_length": 128,
            }

            for key, expected in expected_settings.items():
                if config.get(key) != expected:
                    raise ValueError(
                        f"設定不符：{folder}/{key}"
                    )

            for split in ["train", "validation"]:
                record = config[f"{split}_sources"]

                if (
                    record["csv_sha256"]
                    != sources[training_dataset][split]
                ):
                    raise ValueError(
                        f"訓練來源不符：{folder}/{split}"
                    )

                for feature in ["inc", "sub"]:
                    if (
                        record[
                            f"{feature}_checkpoint_sha256"
                        ]
                        != auxiliary[feature]
                    ):
                        raise ValueError(
                            f"輔助來源不符："
                            f"{folder}/{feature}"
                        )

            tokenizer_files = {
                path.name: sha(path)
                for path in sorted(
                    (folder / "tokenizer").iterdir()
                )
                if path.is_file()
            }

            if not tokenizer_files:
                raise ValueError(
                    f"Tokenizer 不存在：{folder}"
                )

            runs.append({
                "training_dataset": training_dataset,
                "model": name,
                "use_inc": use_inc,
                "use_sub": use_sub,
                "folder": str(folder),
                "config": config,
                "validation": best,
                "checkpoint_sha256": sha(
                    folder / "best.pt"
                ),
                "tokenizer_files_sha256": tokenizer_files,
            })

    execution = {
        "seed": 42,
        "batch_size": 16,
        "data_sha256": sources,
        "auxiliary_sha256": auxiliary,
        "runs": runs,
        "script_sha256": sha(Path(__file__)),
        "backbone_config_sha256": sha(
            MODEL / "config.json"
        ),
        "selection": (
            "Existing best validation checkpoints; "
            "no training or reselection"
        ),
        "evaluation": (
            "exploratory_bidirectional_cross_dataset"
        ),
    }

    return old_plan, execution


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

    torch.set_num_threads(4)
    core.base.MODEL_DIR = MODEL
    evaluator.BATCH_SIZE = 16

    # 先核對並固定全部八份 checkpoint，
    # 通過後才開始計算任何 test 分數。
    old_plan, execution = make_plan()

    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "execution.json"

    if lock.exists():
        if read_json(lock) != execution:
            raise ValueError(
                "既有執行設定不同，停止避免覆蓋。"
            )
    else:
        save_json(lock, execution)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )
    print(
        "Checkpoints locked: 8 | "
        "Test evaluations: 16 | Seed: 42",
        flush=True,
    )
    print(
        "No training; checkpoints selected "
        "by validation only.",
        flush=True,
    )

    results = []

    for test_dataset in ["unbiased", "biased"]:
        path = (
            PROJECT
            / f"data/processed/{test_dataset}/test.csv"
        )

        df = pd.read_csv(
            path,
            keep_default_na=False,
        )

        required = {
            "question",
            "product_description",
            "product_group",
            "label",
        }

        if df.empty or not required.issubset(df.columns):
            raise ValueError(
                f"Test 資料不完整：{path}"
            )

        if set(df["label"].unique()) != {0, 1}:
            raise ValueError(
                f"Test 標籤不正確：{path}"
            )

        for column in [
            "question",
            "product_description",
            "product_group",
        ]:
            if (
                df[column]
                .astype(str)
                .str.strip()
                .eq("")
                .any()
            ):
                raise ValueError(
                    f"Test 文字為空：{path}/{column}"
                )

        actual = df["label"].to_numpy(dtype=int)

        signature = {
            "test_csv_sha256": (
                execution["data_sha256"]
                [test_dataset]["test"]
            ),
            "inc_checkpoint_sha256": (
                execution["auxiliary_sha256"]["inc"]
            ),
            "sub_checkpoint_sha256": (
                execution["auxiliary_sha256"]["sub"]
            ),
        }

        evaluator.OUTPUT = (
            OUT / "aux_cache" / test_dataset
        )
        evaluator.OUTPUT.mkdir(
            parents=True,
            exist_ok=True,
        )

        features = evaluator.make_aux_features(
            df,
            old_plan,
            signature,
        )

        if (
            tuple(features.shape) != (len(df), 65)
            or not torch.isfinite(features).all()
        ):
            raise ValueError(
                f"Test 特徵不正確：{test_dataset}"
            )

        features = features.detach().cpu().float()

        for run in execution["runs"]:
            name = run["model"]
            training_dataset = run["training_dataset"]

            key = (
                f"{training_dataset}_to_"
                f"{test_dataset}_{name}"
            )

            metrics_path = OUT / f"metrics_{key}.json"
            predictions_path = (
                OUT / f"predictions_{key}.csv"
            )

            print(
                f"\n=== {training_dataset} -> "
                f"{test_dataset} | {name} ===",
                flush=True,
            )

            # 同一執行設定下可以接續已完成的評估。
            if metrics_path.exists():
                saved = read_json(metrics_path)

                if (
                    not predictions_path.is_file()
                    or saved["predictions_sha256"]
                    != sha(predictions_path)
                ):
                    raise ValueError(
                        f"已儲存的預測不一致：{key}"
                    )

                row = saved["metrics"]

                print(
                    "Verified saved result; "
                    "skipped recomputation.",
                    flush=True,
                )

            else:
                folder = Path(run["folder"])

                if (
                    sha(folder / "best.pt")
                    != run["checkpoint_sha256"]
                ):
                    raise ValueError(
                        f"鎖定後 checkpoint 已變更：{key}"
                    )

                checkpoint = torch.load(
                    folder / "best.pt",
                    map_location="cpu",
                    weights_only=True,
                )

                if (
                    checkpoint["config"]
                    != run["config"]
                    or checkpoint["validation"]
                    != run["validation"]
                ):
                    raise ValueError(
                        f"Checkpoint metadata 不符：{key}"
                    )

                model = core.ContextINCSUBModel(
                    use_inc=run["use_inc"],
                    use_sub=run["use_sub"],
                )

                model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=True,
                )

                del checkpoint
                model.to("cuda")

                tokenizer = (
                    evaluator.AutoTokenizer
                    .from_pretrained(
                        folder / "tokenizer",
                        use_fast=False,
                        local_files_only=True,
                    )
                )

                encoded = tokenizer(
                    df["question"].astype(str).tolist(),
                    df["product_description"]
                    .astype(str)
                    .tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=run["config"]["max_length"],
                    return_tensors="pt",
                )

                dataset = evaluator.InputDataset(
                    encoded,
                    features=features,
                )

                probabilities = evaluator.predict(
                    model,
                    dataset,
                    custom=True,
                )

                if (
                    probabilities.shape != (len(df), 2)
                    or not np.isfinite(
                        probabilities
                    ).all()
                ):
                    raise ValueError(
                        f"預測維度或數值不正確：{key}"
                    )

                predicted = probabilities.argmax(axis=1)

                row = {
                    "training_dataset": training_dataset,
                    "test_dataset": test_dataset,
                    "model": name,
                    "seed": 42,
                    "learning_rate": 1e-5,
                    "selected_epoch": (
                        run["validation"]["epoch"]
                    ),
                    "test_rows": len(df),
                    "test_accuracy": float(
                        evaluator.accuracy_score(
                            actual,
                            predicted,
                        )
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

                pd.DataFrame({
                    "row_index": np.arange(len(df)),
                    "actual": actual,
                    "predicted": predicted,
                    "probability_label_1": (
                        probabilities[:, 1]
                    ),
                }).to_csv(
                    predictions_path,
                    index=False,
                )

                save_json(
                    metrics_path,
                    {
                        "metrics": row,
                        "predictions_sha256": sha(
                            predictions_path
                        ),
                    },
                )

                del model, tokenizer, encoded
                del dataset, probabilities

                gc.collect()
                torch.cuda.empty_cache()

            results.append(row)

            pd.DataFrame(results).to_csv(
                OUT / "all_test_results.csv",
                index=False,
            )

            print(
                f"Accuracy: "
                f"{row['test_accuracy']:.2%} | "
                f"macro-F1: "
                f"{row['test_macro_f1']:.4f}",
                flush=True,
            )

    results = pd.DataFrame(results)

    for metric in [
        "test_accuracy",
        "test_macro_f1",
    ]:
        table = results.pivot(
            index=["model", "training_dataset"],
            columns="test_dataset",
            values=metric,
        )

        table.to_csv(
            OUT / f"matrix_{metric}.csv"
        )

        print(
            f"\n=== {metric}: "
            "rows=train, columns=test ===",
            flush=True,
        )

        if metric == "test_accuracy":
            display = (table * 100).round(2)
        else:
            display = table.round(4)

        print(
            display.to_string(),
            flush=True,
        )

    for dataset in ["unbiased", "biased"]:
        path = (
            PROJECT
            / f"data/processed/{dataset}/test.csv"
        )

        if (
            sha(path)
            != execution["data_sha256"][dataset]["test"]
        ):
            raise RuntimeError(
                f"評估期間 Test CSV 已變更：{dataset}"
            )

    print(
        "\nAll 16 test evaluations completed.",
        flush=True,
    )
    print("Saved:", OUT, flush=True)


if __name__ == "__main__":
    main()