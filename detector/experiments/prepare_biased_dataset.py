"""Prepare product-disjoint biased splits aligned with existing unbiased splits.

Run from the Superpod humorProj directory. Uses only the Python standard library.
Does not train models, score a test set, or modify existing unbiased data.
"""
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SEED = 42
SPLITS = ("train", "validation", "test")
COLUMNS = ("question", "product_description", "image_url", "label", "product_group")


def normalize(text):
    return re.sub(r"\s+", " ", html.unescape(text)).strip().casefold()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"question", "product_description", "label"} <= set(reader.fieldnames or []):
            raise ValueError(f"Missing required CSV columns: {path}")
        rows = list(reader)
    for row in rows:
        value = row["label"].strip()
        if value not in {"0", "1", "0.0", "1.0"}:
            raise ValueError(f"Unexpected label {value!r}: {path}")
        row["label"] = str(int(float(value)))
    return rows


def pair(row):
    return normalize(row["question"]), normalize(row["product_description"])


def prepare(project):
    raw_dir = project / "data/raw/humor_original"
    old_dir = project / "data/processed/unbiased"
    out = project / "data/processed/biased"
    staging = project / "data/processed/biased_preparing"
    if out.exists() or staging.exists():
        raise FileExistsError(f"Output or staging directory already exists: {out}, {staging}")

    paths = {f"unbiased_{s}": old_dir / f"{s}.csv" for s in SPLITS}
    paths["humorous_raw"] = raw_dir / "Humorous.csv"
    paths["biased_negative_raw"] = raw_dir / "Non-humours-biased.csv"
    before_hashes = {name: sha256(path) for name, path in paths.items()}
    old = {s: read_rows(paths[f"unbiased_{s}"]) for s in SPLITS}
    humorous = read_rows(paths["humorous_raw"])
    negatives = read_rows(paths["biased_negative_raw"])
    if any(r["label"] != "1" for r in humorous):
        raise ValueError("Humorous.csv contains a non-positive label.")
    if any(r["label"] != "0" for r in negatives):
        raise ValueError("Biased negative CSV contains a non-negative label.")
    raw_positive_pairs = {pair(r) for r in humorous}

    product_split = {}
    question_splits = defaultdict(set)
    positive_questions = set()
    result = {s: [] for s in SPLITS}
    for split in SPLITS:
        for row in old[split]:
            q, product = pair(row)
            if not q or not product:
                raise ValueError(f"Empty question/product in existing {split}.")
            previous = product_split.setdefault(product, split)
            if previous != split:
                raise ValueError("Normalized product text crosses existing unbiased splits; inspect first.")
            question_splits[q].add(split)
            if row["label"] == "1":
                if pair(row) not in raw_positive_pairs:
                    raise ValueError("An existing positive does not match Humorous.csv.")
                positive_questions.add(q)
                result[split].append({c: row.get(c, "") for c in COLUMNS})

    counts = Counter()
    seen_pairs = set()
    candidates = []
    for row in negatives:
        q, product = pair(row)
        if not q or not product:
            counts["empty_question_or_product"] += 1
            continue
        if (q, product) in seen_pairs:
            counts["duplicate_negative_pair"] += 1
            continue
        seen_pairs.add((q, product))
        # Conservative exclusion: do not introduce opposite labels for an
        # existing humorous question, even if its product context differs.
        if q in positive_questions:
            counts["negative_question_matches_retained_humorous_question"] += 1
            continue
        candidates.append(row)

    new_products = sorted({pair(r)[1] for r in candidates} - product_split.keys())
    random.Random(SEED).shuffle(new_products)
    train_end = round(len(new_products) * 0.6)
    val_end = round(len(new_products) * 0.8)
    for index, product in enumerate(new_products):
        product_split[product] = (
            "train" if index < train_end else "validation" if index < val_end else "test"
        )

    assigned = []
    new_question_splits = defaultdict(set)
    for row in candidates:
        q, product = pair(row)
        split = product_split[product]
        if question_splits.get(q, set()) - {split}:
            counts["negative_question_crosses_existing_unbiased_split"] += 1
            continue
        assigned.append((split, row))
        new_question_splits[q].add(split)
    for split, row in assigned:
        q, product = pair(row)
        if len(new_question_splits[q]) > 1:
            counts["negative_question_crosses_new_biased_splits"] += 1
            continue
        result[split].append({
            "question": row["question"].strip(),
            "product_description": row["product_description"].strip(),
            "image_url": row.get("image_url", ""),
            "label": "0",
            "product_group": product,
        })

    # Validate all final biased products against ALL existing unbiased rows,
    # including existing negatives, not just the shared humorous examples.
    observed = {}
    summary = {}
    for split in SPLITS:
        rows = result[split]
        labels = Counter(r["label"] for r in rows)
        if not all(labels[label] > 0 for label in ("0", "1")):
            raise ValueError(f"A class is missing in {split}.")
        for row in old[split] + rows:
            product = pair(row)[1]
            if observed.setdefault(product, split) != split:
                raise AssertionError("Product leakage across biased/unbiased splits.")
        expected = Counter(pair(r) for r in old[split] if r["label"] == "1")
        actual = Counter(pair(r) for r in rows if r["label"] == "1")
        if actual != expected:
            raise AssertionError("Shared humorous split membership changed.")
        summary[split] = {
            "rows": len(rows), "label_counts": dict(labels),
            "products": len({pair(r)[1] for r in rows}),
            "duplicate_pair_occurrences": len(rows) - len({pair(r) for r in rows}),
        }
        print(f"{split}: {summary[split]}", flush=True)

    manifest = {
        "seed": SEED,
        "purpose": "Aligned biased splits for within-dataset and cross-dataset evaluation",
        "positive_policy": "Keep exactly the existing cleaned unbiased positive rows in their original splits",
        "negative_policy": "Official biased negatives; conservative exclusions recorded below",
        "new_product_split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "split_note": "Fractions apply to newly introduced product groups, not exact row counts; no class downsampling",
        "normalization_for_matching": "HTML unescape, collapse whitespace, strip, casefold",
        "source_sha256": before_hashes,
        "source_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "raw_humorous_rows": len(humorous),
        "raw_biased_negative_rows": len(negatives),
        "retained_negative_rows": sum(r["label"] == "0" for rows in result.values() for r in rows),
        "excluded_negative_rows": dict(counts),
        "summary": summary,
        "checks": {
            "product_overlap_across_all_biased_and_unbiased_splits": 0,
            "positive_membership_preserved": True,
            "new_negative_question_cross_split_overlap": 0,
            "existing_unbiased_questions_in_multiple_splits": sum(len(v) > 1 for v in question_splits.values()),
            "note": "Existing positive question duplicates, if any, are preserved and reported, not silently repaired",
        },
        "test_policy": "No model inference or test-score-based selection performed during preparation",
    }
    if any(sha256(path) != before_hashes[name] for name, path in paths.items()):
        raise RuntimeError("Source file changed during preparation.")
    staging.mkdir(parents=True)
    for index, split in enumerate(SPLITS):
        rows = list(result[split])
        random.Random(SEED + index).shuffle(rows)
        path = staging / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest.setdefault("output_sha256", {})[f"{split}.csv"] = sha256(path)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    staging.rename(out)
    print("Excluded negatives:", dict(counts), flush=True)
    print("Existing unbiased question keys in multiple splits:",
          manifest["checks"]["existing_unbiased_questions_in_multiple_splits"], flush=True)
    print("Product overlap across all splits: 0", flush=True)
    print("Unbiased source CSVs unchanged.", flush=True)
    print("Saved:", out.resolve(), flush=True)
    print("Biased dataset preparation completed. No training or test scoring.", flush=True)


if __name__ == "__main__":
    prepare(Path.cwd())
