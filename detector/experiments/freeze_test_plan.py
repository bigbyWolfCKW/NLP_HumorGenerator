import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
CHECKPOINTS = PROJECT / "checkpoints"
DATA_DIR = PROJECT / "data/processed/unbiased"
OUTPUT = PROJECT / "results/17_test_plan.json"

SEEDS = [42, 43, 44]


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


def fingerprint(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return {
        "path": str(path),
        "sha256": sha256(path),
    }


def main():
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.csv"

        if not path.is_file():
            raise FileNotFoundError(path)

    # 只計算檔案 hash，不解析 test 標籤或計算成績。
    data_hashes = {
        split: sha256(DATA_DIR / f"{split}.csv")
        for split in ["train", "validation", "test"]
    }

    auxiliary = {
        "inc": fingerprint(
            CHECKPOINTS
            / "inc_siamese_pqa200k_seed42/best.pt"
        ),
        "sub": fingerprint(
            CHECKPOINTS
            / "sub_lstm_multicat40k_fasttext_seed42/best.pt"
        ),
    }

    jobs = []

    for seed in SEEDS:
        for mode in ["qonly", "qproduct"]:
            for name, folder in [
                (
                    "roberta_base",
                    CHECKPOINTS / f"roberta_{mode}_seed{seed}",
                ),
                (
                    "deberta_v3_base",
                    CHECKPOINTS
                    / "10_deberta_v3_base"
                    / f"{mode}_seed{seed}",
                ),
                (
                    "roberta_large",
                    CHECKPOINTS
                    / "11_roberta_large"
                    / f"{mode}_seed{seed}",
                ),
            ]:
                jobs.append({
                    "model": f"{name}_{mode}",
                    "seed": seed,
                    "kind": "huggingface",
                    "input_mode": mode,
                    "folder": str(folder),
                })

        for name, use_inc, use_sub in [
            ("control", False, False),
            ("with_inc", True, False),
            ("with_sub", False, True),
            ("with_inc_sub", True, True),
        ]:
            jobs.append({
                "model": f"roberta_fusion16_{name}",
                "seed": seed,
                "kind": "custom_inc_sub",
                "input_mode": "qproduct",
                "use_inc": use_inc,
                "use_sub": use_sub,
                "folder": str(
                    CHECKPOINTS
                    / "16_roberta_inc_sub"
                    / f"seed{seed}"
                    / name
                ),
            })

    for index, job in enumerate(jobs, start=1):
        folder = Path(job["folder"])

        print(
            f"[{index}/{len(jobs)}] "
            f"{job['model']} | seed {job['seed']}",
            flush=True,
        )

        config_path = folder / "config.json"
        best_path = folder / "best_validation.json"

        config = read_json(config_path)
        best = read_json(best_path)

        if config.get("seed") != job["seed"]:
            raise ValueError(f"Seed mismatch: {folder}")

        accuracy = float(best["validation_accuracy"])
        macro_f1 = float(best["validation_macro_f1"])

        if not (
            0 <= accuracy <= 1
            and 0 <= macro_f1 <= 1
        ):
            raise ValueError(f"Invalid metrics: {folder}")

        if job["kind"] == "huggingface":
            model_dir = folder / "best_model"

            required = {
                model_dir / "config.json",
            }

            weights = (
                list(model_dir.glob("*.safetensors"))
                + list(model_dir.glob("pytorch_model*.bin"))
            )

            if not weights:
                raise FileNotFoundError(
                    f"No model weights: {model_dir}"
                )

            required.update(weights)

            for path in model_dir.glob("*.index.json"):
                required.add(path)
                weight_map = read_json(path)["weight_map"]

                required.update(
                    model_dir / filename
                    for filename in set(weight_map.values())
                )

            tokenizer_files = [
                model_dir / name
                for name in [
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "spm.model",
                    "vocab.json",
                    "merges.txt",
                    "added_tokens.json",
                ]
                if (model_dir / name).is_file()
            ]

            has_tokenizer = (
                (model_dir / "tokenizer.json").is_file()
                or (model_dir / "spm.model").is_file()
                or (
                    (model_dir / "vocab.json").is_file()
                    and (model_dir / "merges.txt").is_file()
                )
            )

            if not has_tokenizer:
                raise FileNotFoundError(
                    f"No tokenizer: {model_dir}"
                )

            required.update(tokenizer_files)

            for split in ["train", "validation"]:
                key = f"{split}_csv_sha256"

                if (
                    key in config
                    and config[key] != data_hashes[split]
                ):
                    raise ValueError(
                        f"{split} CSV mismatch: {folder}"
                    )

            job["model_dir"] = str(model_dir)

        else:
            required = {folder / "best.pt"}

            for flag in ["use_inc", "use_sub"]:
                if config[flag] != job[flag]:
                    raise ValueError(
                        f"{flag} mismatch: {folder}"
                    )

            for split in ["train", "validation"]:
                sources = config[f"{split}_sources"]

                if sources["csv_sha256"] != data_hashes[split]:
                    raise ValueError(
                        f"{split} CSV mismatch: {folder}"
                    )

                if sources["inc_checkpoint_sha256"] != (
                    auxiliary["inc"]["sha256"]
                ):
                    raise ValueError(
                        f"INC checkpoint mismatch: {folder}"
                    )

                if sources["sub_checkpoint_sha256"] != (
                    auxiliary["sub"]["sha256"]
                ):
                    raise ValueError(
                        f"SUB checkpoint mismatch: {folder}"
                    )

        required.update([config_path, best_path])

        job["files"] = [
            fingerprint(path)
            for path in sorted(
                required, key=lambda item: str(item)
            )
        ]
        job["validation"] = best
        job["max_length"] = config["max_length"]

    plan = {
        "version": 1,
        "selection_rule": "best validation macro-F1",
        "seeds": SEEDS,
        "classification_rule": "argmax over two class logits",
        "metrics": ["accuracy", "macro_f1"],
        "standard_deviation": "sample SD, ddof=1",
        "test_csv": str(DATA_DIR / "test.csv"),
        "data_sha256": data_hashes,
        "auxiliary_checkpoints": auxiliary,
        "jobs": jobs,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        if read_json(OUTPUT) != plan:
            raise FileExistsError(
                "已有不同的 test plan，停止覆蓋。"
            )
        print("\nExisting test plan matches.", flush=True)
    else:
        temporary = OUTPUT.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(plan, indent=2),
            encoding="utf-8",
        )
        temporary.replace(OUTPUT)

    print("Checkpoints verified:", len(jobs), flush=True)
    print("Saved:", OUTPUT, flush=True)
    print("No test scores were computed.", flush=True)


if __name__ == "__main__":
    main()