"""Download Amazon PQA humor data and build fresh product-disjoint splits."""

import argparse
import csv
import json
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
BUCKET = "https://humor-detection-pds.s3-us-west-2.amazonaws.com"
FILES = {"Humorous.csv": "1", "Non-humorous-unbiased.csv": "0"}
FIELDS = ("question", "product_description", "image_url", "label", "product_group")


def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw / name
        if target.is_file():
            continue
        with TemporaryDirectory(dir=raw) as temporary:
            staged = Path(temporary) / name
            with urlopen(f"{BUCKET}/{name}", timeout=60) as response, staged.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if not staged.stat().st_size:
                raise ValueError(f"Empty download: {name}")
            staged.rename(target)


def read_groups(raw):
    groups = defaultdict(list)
    skipped = {}
    for name, label in FILES.items():
        missing_text = Counter()
        with (raw / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(FIELDS[:-1]).issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected Amazon PQA columns in {name}: {reader.fieldnames}")
            for row in reader:
                if None in row or any(row[field] is None for field in FIELDS[:-1]):
                    raise ValueError(f"Invalid record in {name} at line {reader.line_num}")
                values = {field: row[field].strip() for field in FIELDS[:-1]}
                if values["label"] != label:
                    raise ValueError(f"Unexpected label in {name} at line {reader.line_num}")
                empty = [field for field in ("question", "product_description") if not values[field]]
                if empty:
                    missing_text[" and ".join(empty)] += 1
                    continue
                group = " ".join(
                    unicodedata.normalize("NFKC", values["product_description"]).casefold().split()
                )
                groups[group].append({**values, "product_group": group})
        skipped[name] = dict(missing_text)
    return groups, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data/source/raw/amazon_pqa")
    parser.add_argument("--offline", action="store_true", help="Use the two CSVs already in --raw-dir.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads((PROJECT / "configs/datasets.json").read_text(encoding="utf-8"))
    output = (PROJECT / config["root"] / config["qa_train"]).parent
    if output.exists():
        raise FileExistsError(f"Split directory already exists: {output}")
    if not args.offline:
        download(args.raw_dir)
    groups, skipped = read_groups(args.raw_dir)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    train_end = len(keys) * 60 // 100
    validation_end = len(keys) * 80 // 100
    splits = {
        "train": keys[:train_end],
        "validation": keys[train_end:validation_end],
        "test": keys[validation_end:],
    }
    if not all(splits.values()):
        raise ValueError("Too few product groups to create three nonempty splits.")
    summary = {
        "seed": args.seed,
        "group_split_percentages": [60, 20, 20],
        "group_key": "NFKC casefolded, whitespace-normalized product description",
        "sources": [f"{BUCKET}/{name}" for name in FILES],
        "skipped_missing_text_rows": skipped,
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / "unbiased"
        staged.mkdir()
        for split, selected in splits.items():
            rows = [row for key in selected for row in groups[key]]
            labels = {label: sum(row["label"] == label for row in rows) for label in ("0", "1")}
            if not all(labels.values()):
                raise ValueError(f"Both labels are required in the {split} split.")
            with (staged / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            summary["splits"][split] = {"product_groups": len(selected), "rows": len(rows), "labels": labels}
        (staged / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"Split directory already exists: {output}")
        staged.rename(output)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
