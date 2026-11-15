"""Summarize the QProduct training inputs using the detector's CSV reader."""
import json
from collections import Counter
from itertools import combinations
from math import ceil
from pathlib import Path

from src.artifacts import check_hf_weights
from src.config import select_detector
from src.data import sha256
from .data import read_examples, split_path


def summarize_lengths(lengths):
    lengths = sorted(lengths)
    return {"p95": lengths[ceil(len(lengths) * 0.95) - 1], "max": lengths[-1]}


def token_lengths(tokenizer, rows):
    lengths = []
    for start in range(0, len(rows), 256):
        batch = rows[start:start + 256]
        encoded = tokenizer(
            [row["question"] for row in batch],
            [row["product_description"] for row in batch],
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def inspect_splits(args):
    if args.max_length < 1:
        raise ValueError("Max length must be positive")
    splits = {}
    report = {"splits": {}, "tokenizers": {}}
    inputs = {"train": args.train_csv, "validation": args.validation_csv}
    if args.include_test or args.test_csv:
        inputs["test"] = args.test_csv
    product_groups = {}
    for split, override in inputs.items():
        path = split_path(split, override)
        rows = read_examples(path)
        splits[split] = rows
        products = Counter(row["product_group"] for row in rows)
        product_groups[split] = set(products)
        labels = Counter(row["label"] for row in rows)
        report["splits"][split] = {
            "csv": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
            "label_counts": {label: labels[label] for label in (0, 1)},
            "product_groups": len(products),
            "largest_product_group_rows": max(products.values()),
            "question_words": summarize_lengths(len(row["question"].split()) for row in rows),
            "description_words": summarize_lengths(len(row["product_description"].split()) for row in rows),
        }

    overlaps = {
        f"{left}/{right}": len(product_groups[left] & product_groups[right])
        for left, right in combinations(splits, 2)
    }
    report["product_overlap"] = overlaps["train/validation"]
    if "test" in splits:
        report["product_overlap_by_split"] = overlaps
    if any(overlaps.values()):
        print(json.dumps(report, indent=2))
        raise ValueError(f"Product groups overlap between splits: {overlaps}")

    for name in args.detector:
        config = select_detector(name)
        if config["backend"] != "hf" or config["input_mode"] != "pair":
            raise ValueError(f"EDA token lengths require a paired HF detector: {name}")
        backbone = Path(config["backbone"])
        check_hf_weights(backbone)
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Tokenizer checks require the project Transformers environment") from error

        tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
        report["tokenizers"][name] = {"backbone": str(backbone), "max_length": args.max_length}
        for split, rows in splits.items():
            lengths = token_lengths(tokenizer, rows)
            report["tokenizers"][name][split] = {
                **summarize_lengths(lengths),
                "rows_over_max_length": sum(length > args.max_length for length in lengths),
            }
    print(json.dumps(report, indent=2))
