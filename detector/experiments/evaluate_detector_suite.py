import gc
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT = Path(__file__).resolve().parent
PLAN_PATH = PROJECT / "results/17_test_plan.json"
OUTPUT = PROJECT / "results/final_test_v1"

BATCH_SIZE = 16


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


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class InputDataset(Dataset):
    def __init__(self, encoded, features=None):
        self.encoded = encoded
        self.features = features

    def __len__(self):
        return len(self.encoded["input_ids"])

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encoded.items()
        }

        if self.features is not None:
            item["features"] = self.features[index]

        return item


def make_aux_features(df, plan, signature):
    output_path = OUTPUT / "test_aux_features.pt"

    if output_path.exists():
        saved = torch.load(
            output_path,
            map_location="cpu",
            weights_only=True,
        )

        if saved["signature"] != signature:
            raise ValueError("既有 test 特徵來源不同。")

        features = saved["features"]

        if features.shape != (len(df), 65):
            raise ValueError("既有 test 特徵維度不正確。")

        if not torch.isfinite(features).all():
            raise ValueError("既有 test 特徵含非有限數值。")

        print("Reuse cached test auxiliary features.", flush=True)
        return features

    # 只 import 類別和函數，不執行各檔案的 main()。
    inc_lib = importlib.import_module("train_inc_product_qa")
    sub_lib = importlib.import_module("sub_lstm")

    inc_checkpoint = torch.load(
        plan["auxiliary_checkpoints"]["inc"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    sub_checkpoint = torch.load(
        plan["auxiliary_checkpoints"]["sub"]["path"],
        map_location="cpu",
        weights_only=True,
    )

    inc_config = inc_checkpoint["config"]
    sub_config = sub_checkpoint["config"]

    if inc_config["seed"] != 42 or sub_config["seed"] != 42:
        raise ValueError("預先固定的輔助模型應為 seed 42。")

    if inc_config["pair_label_meaning"]["1"] != "original_product":
        raise ValueError("INC 分數方向不同。")

    if sub_config["hidden_dim"] != 64:
        raise ValueError("SUB 應輸出 64 維。")

    inc_lib.MAX_LENGTH = inc_config["max_length"]
    sub_lib.MAX_LENGTH = sub_config["max_length"]

    if sub_config["tokenizer_pattern"] != sub_lib.TOKEN_PATTERN:
        raise ValueError("SUB tokenizer 與 checkpoint 不一致。")

    if not inc_config["lowercase"] or not sub_config["lowercase"]:
        raise ValueError("目前 encoder 預期 lowercase。")

    inc_state = inc_checkpoint["model_state_dict"]

    inc_model = inc_lib.IncongruityLSTM(
        inc_state["embedding.weight"],
        hidden_dim=inc_config["hidden_dim"],
    )
    inc_model.load_state_dict(inc_state, strict=True)
    inc_model.to("cuda").eval()

    sub_model = sub_lib.SubjectivityLSTM(
        vocab_size=len(sub_checkpoint["vocab"]),
        embedding_dim=sub_config["embedding_dim"],
        hidden_dim=sub_config["hidden_dim"],
    )
    sub_model.load_state_dict(
        sub_checkpoint["model_state_dict"],
        strict=True,
    )
    sub_model.to("cuda").eval()

    q_ids, q_lengths = inc_lib.encode_texts(
        df["question"],
        inc_checkpoint["vocab"],
        inc_config,
    )
    p_ids, p_lengths = inc_lib.encode_texts(
        df["product_description"].fillna(""),
        inc_checkpoint["vocab"],
        inc_config,
    )

    sub_dataset = sub_lib.make_dataset(
        df.assign(text=df["question"]),
        sub_checkpoint["vocab"],
    )
    sub_ids, sub_lengths, _ = sub_dataset.tensors

    loader = DataLoader(
        TensorDataset(
            q_ids, q_lengths,
            p_ids, p_lengths,
            sub_ids, sub_lengths,
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    batches = []

    with torch.inference_mode():
        for q, q_len, p, p_len, s, s_len in loader:
            inc_scores = inc_model(
                q.to("cuda"), q_len,
                p.to("cuda"), p_len,
            ).sigmoid().unsqueeze(1)

            sub_features = sub_model.encode(
                s.to("cuda"), s_len
            )

            batches.append(
                torch.cat(
                    [inc_scores, sub_features],
                    dim=1,
                ).cpu()
            )

    features = torch.cat(batches)

    if features.shape != (len(df), 65):
        raise ValueError("Test 輔助特徵維度不正確。")

    if not torch.isfinite(features).all():
        raise ValueError("Test 輔助特徵含非有限數值。")

    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "signature": signature,
            "features": features,
        },
        temporary,
    )
    temporary.replace(output_path)

    del inc_model, sub_model
    del inc_checkpoint, sub_checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Test auxiliary features:",
        tuple(features.shape),
        flush=True,
    )

    return features


def predict(model, dataset, custom):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    batches = []

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to("cuda")
                for key, value in batch.items()
            }

            if custom:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["features"],
                )
            else:
                logits = model(**batch).logits

            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("模型應輸出兩個分類 logits。")

            if not torch.isfinite(logits).all():
                raise ValueError("Test logits 含非有限數值。")

            batches.append(
                logits.softmax(dim=1).cpu().numpy()
            )

    return np.concatenate(batches, axis=0)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請透過 sbatch 配置 GPU。")

    torch.set_num_threads(4)

    plan = read_json(PLAN_PATH)
    plan_hash = sha256(PLAN_PATH)

    if len(plan["jobs"]) != 30:
        raise ValueError("預期固定評估 30 份 checkpoint。")

    # 在計算任何 test 成績之前，核對已鎖定的檔案。
    verified = set()

    for job in plan["jobs"]:
        print(
            "Verifying:",
            job["model"], job["seed"],
            flush=True,
        )

        for record in job["files"]:
            path = Path(record["path"])

            if str(path) in verified:
                continue

            if sha256(path) != record["sha256"]:
                raise ValueError(f"已鎖定檔案改變：{path}")

            verified.add(str(path))

    for record in plan["auxiliary_checkpoints"].values():
        path = Path(record["path"])

        if sha256(path) != record["sha256"]:
            raise ValueError(f"輔助 checkpoint 改變：{path}")

    for split, expected_hash in plan["data_sha256"].items():
        path = (
            PROJECT / "data/processed/unbiased"
            / f"{split}.csv"
        )

        if sha256(path) != expected_hash:
            raise ValueError(f"{split} CSV 已改變。")

    # 記錄 evaluator 使用的額外程式與融合 tokenizer。
    extra_files = {
        PROJECT / "14_train_inc_pqa.py",
        PROJECT / "16_train_roberta_inc_sub.py",
        PROJECT / "091_train_roberta_sub.py",
        PROJECT / "train_sub.py",
        PROJECT / "models/roberta-base/config.json",
    }

    for job in plan["jobs"]:
        if job["kind"] != "custom_inc_sub":
            continue

        tokenizer_dir = Path(job["folder"]) / "tokenizer"

        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(tokenizer_dir)

        extra_files.update(
            path
            for path in tokenizer_dir.iterdir()
            if path.is_file()
        )

    signature = {
        "plan_sha256": plan_hash,
        "test_csv_sha256": plan["data_sha256"]["test"],
        "batch_size": BATCH_SIZE,
        "torch_version": str(torch.__version__),
        "extra_files_sha256": {
            str(path): sha256(path)
            for path in sorted(extra_files)
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    execution_path = OUTPUT / "execution.json"

    if execution_path.exists():
        if read_json(execution_path) != signature:
            raise ValueError(
                "既有 test 執行來源不同，停止混合結果。"
            )
    else:
        write_json(execution_path, signature)

    test_path = Path(plan["test_csv"])
    df = pd.read_csv(test_path)

    required = {
        "question", "product_description",
        "product_group", "label",
    }

    if not required.issubset(df.columns):
        raise ValueError("Test 缺少必要欄位。")

    if df.empty:
        raise ValueError("Test 沒有資料。")

    if df[
        ["question", "product_group", "label"]
    ].isna().any().any():
        raise ValueError("Test 有缺失的必要資料。")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Test 標籤不正確。")

    # 再檢查 test 商品沒有出現在 train / validation。
    for split in ["train", "validation"]:
        other = pd.read_csv(
            PROJECT / "data/processed/unbiased"
            / f"{split}.csv",
            usecols=["product_group"],
        )

        if set(df["product_group"]) & set(other["product_group"]):
            raise ValueError(f"Test 與 {split} 商品重疊。")

    actual = df["label"].to_numpy(dtype=int)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Test rows:", len(df), flush=True)
    print("Fixed checkpoints:", len(plan["jobs"]), flush=True)

    features = make_aux_features(df, plan, signature)

    fusion_lib = importlib.import_module(
        "train_roberta_fusion_product_qa"
    )
    fusion_lib.base.MODEL_DIR = (
        PROJECT / "models/roberta-base"
    )

    rows = []

    for index, job in enumerate(plan["jobs"], start=1):
        name = job["model"]
        seed = job["seed"]
        run_name = f"{name}_seed{seed}"

        metrics_path = OUTPUT / f"{run_name}.json"
        predictions_path = OUTPUT / f"{run_name}.csv"

        print(
            f"\n[{index}/30] {name} | seed {seed}",
            flush=True,
        )

        if metrics_path.exists():
            saved = read_json(metrics_path)

            if saved["signature"] != signature:
                raise ValueError(
                    f"既有結果來源不一致：{run_name}"
                )

            if (
                saved["metrics"]["model"] != name
                or saved["metrics"]["seed"] != seed
                or saved["metrics"]["test_rows"] != len(df)
            ):
                raise ValueError("既有結果身份或筆數不一致。")

            if sha256(predictions_path) != (
                saved["predictions_sha256"]
            ):
                raise ValueError("既有逐筆預測已改變。")

            metrics = saved["metrics"]
            print("Reuse completed evaluation.", flush=True)

        else:
            custom = job["kind"] == "custom_inc_sub"

            if custom:
                folder = Path(job["folder"])

                checkpoint = torch.load(
                    folder / "best.pt",
                    map_location="cpu",
                    weights_only=True,
                )

                for flag in ["use_inc", "use_sub"]:
                    if checkpoint["config"][flag] != job[flag]:
                        raise ValueError(
                            f"Checkpoint {flag} 不一致。"
                        )

                model = fusion_lib.ContextINCSUBModel(
                    use_inc=job["use_inc"],
                    use_sub=job["use_sub"],
                )
                model.load_state_dict(
                    checkpoint["model_state_dict"],
                    strict=True,
                )
                del checkpoint

                tokenizer_path = folder / "tokenizer"

            else:
                tokenizer_path = Path(job["model_dir"])

                model = (
                    AutoModelForSequenceClassification
                    .from_pretrained(
                        tokenizer_path,
                        local_files_only=True,
                    )
                )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
                local_files_only=True,
            )

            questions = df["question"].astype(str).tolist()

            kwargs = {
                "padding": "max_length",
                "truncation": True,
                "max_length": job["max_length"],
                "return_tensors": "pt",
            }

            if job["input_mode"] == "qproduct":
                products = (
                    df["product_description"]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )
                encoded = tokenizer(
                    questions, products, **kwargs
                )
            else:
                encoded = tokenizer(questions, **kwargs)

            dataset = InputDataset(
                encoded,
                features=features if custom else None,
            )

            model.to("cuda")

            probabilities = predict(
                model, dataset, custom
            )

            if probabilities.shape != (len(df), 2):
                raise ValueError("預測筆數與 test 不一致。")

            predicted = probabilities.argmax(axis=1)

            metrics = {
                "model": name,
                "seed": seed,
                "test_rows": len(df),
                "selected_epoch": job["validation"]["epoch"],
                "test_accuracy": float(
                    accuracy_score(actual, predicted)
                ),
                "test_macro_f1": float(
                    f1_score(
                        actual,
                        predicted,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }

            predictions = pd.DataFrame({
                "row_index": np.arange(len(df)),
                "actual": actual,
                "predicted": predicted,
                "probability_label_1": probabilities[:, 1],
            })

            temporary = predictions_path.with_suffix(".csv.tmp")
            predictions.to_csv(temporary, index=False)
            temporary.replace(predictions_path)

            # JSON 最後寫入，作為此組評估完成的標記。
            write_json(
                metrics_path,
                {
                    "signature": signature,
                    "metrics": metrics,
                    "predictions_sha256": sha256(predictions_path),
                },
            )

            del model, tokenizer, encoded, dataset
            gc.collect()
            torch.cuda.empty_cache()

        rows.append(metrics)

        pd.DataFrame(rows).to_csv(
            OUTPUT / "per_seed.csv",
            index=False,
        )

        print(
            f"Test accuracy: {metrics['test_accuracy']:.2%} | "
            f"macro-F1: {metrics['test_macro_f1']:.4f}",
            flush=True,
        )

    results = pd.DataFrame(rows)

    if (
        len(results) != 30
        or results.duplicated(["model", "seed"]).any()
        or results["model"].nunique() != 10
        or not results.groupby("model")["seed"].nunique().eq(3).all()
    ):
        raise ValueError("結果未齊全或重複，停止彙總。")

    summary = results.groupby("model").agg(
        seeds=("seed", "nunique"),
        accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"),
        macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"),
    )

    summary.to_csv(OUTPUT / "summary.csv")

    # 同 seed 的配對差值，單位為百分點。
    comparisons = []

    for family in [
        "roberta_base",
        "deberta_v3_base",
        "roberta_large",
    ]:
        comparisons.append((
            f"{family}_product_minus_qonly",
            f"{family}_qproduct",
            f"{family}_qonly",
        ))

    prefix = "roberta_fusion16_"

    for label, left, right in [
        ("inc_minus_control", "with_inc", "control"),
        ("sub_minus_control", "with_sub", "control"),
        ("both_minus_control", "with_inc_sub", "control"),
        ("both_minus_inc", "with_inc_sub", "with_inc"),
        ("both_minus_sub", "with_inc_sub", "with_sub"),
    ]:
        comparisons.append((
            f"fusion16_{label}",
            prefix + left,
            prefix + right,
        ))

    paired_rows = []

    for metric in ["test_accuracy", "test_macro_f1"]:
        pivot = results.pivot(
            index="seed",
            columns="model",
            values=metric,
        )

        for label, left, right in comparisons:
            for seed in plan["seeds"]:
                paired_rows.append({
                    "metric": metric,
                    "comparison": label,
                    "seed": seed,
                    "delta_pp": float(
                        100 * (
                            pivot.loc[seed, left]
                            - pivot.loc[seed, right]
                        )
                    ),
                })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        OUTPUT / "paired_differences.csv",
        index=False,
    )

    paired_summary = paired.groupby(
        ["metric", "comparison"]
    ).agg(
        delta_mean_pp=("delta_pp", "mean"),
        delta_std_pp=("delta_pp", "std"),
    )
    paired_summary.to_csv(
        OUTPUT / "paired_summary.csv"
    )

    print("\n=== Final test summary ===", flush=True)

    for name, row in summary.iterrows():
        print(
            f"{name}: "
            f"accuracy {row['accuracy_mean']:.2%} "
            f"+/- {100 * row['accuracy_std']:.2f} pp | "
            f"macro-F1 {row['macro_f1_mean']:.4f} "
            f"+/- {row['macro_f1_std']:.4f}",
            flush=True,
        )

    print("\nAll 30 test evaluations completed.", flush=True)


if __name__ == "__main__":
    main()