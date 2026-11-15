"""Create a fixed product assignment for four team members."""

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from tempfile import TemporaryDirectory
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SEED = 2026


def normalize(text):
    return " ".join(
        unicodedata.normalize("NFKC", str(text)).casefold().split()
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def historical_description(record):
    if record.get("product_description"):
        return record["product_description"]
    prompt = record["messages"][0]["content"]
    marker = "Product information:"
    ending = "Return only the tagline."
    if marker not in prompt or ending not in prompt:
        raise ValueError("Unrecognized historical prompt format")
    return prompt.split(marker, 1)[1].rsplit(ending, 1)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/formal_v1")
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[],
                        help="Earlier SFT split to exclude. Repeat for each archived split.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Assignment already exists: {output}")

    config = json.loads(
        (PROJECT / "configs/datasets.json").read_text(encoding="utf-8")
    )
    source = PROJECT / config["root"] / config["qa_train"]
    raw = read_csv(source)
    required = {"product_group", "product_description"}
    if not raw or not required.issubset(raw[0]):
        raise ValueError("Source needs product_group and product_description")

    excluded_ids = set()
    excluded_texts = set()
    historical_files = {}

    for path in dict.fromkeys(path.resolve() for path in args.exclude_jsonl):
        location = str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)
        historical_files[location] = sha256(path)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("product_id"):
                excluded_ids.add(normalize(record["product_id"]))
            excluded_texts.add(normalize(historical_description(record)))

    records = []
    parent = []
    seen = {}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for row in raw:
        identity = row["product_group"].strip()
        description = row["product_description"].strip()
        if not identity or not description:
            continue

        keys = {
            "id:" + normalize(identity),
            "description:" + normalize(description),
        }
        parent_id = next(
            (
                row[key].strip()
                for key in ("parent_asin", "parentASIN", "product_family_id")
                if row.get(key, "").strip()
            ),
            "",
        )
        if parent_id:
            keys.add("parent:" + normalize(parent_id))

        index = len(records)
        parent.append(index)
        records.append({
            "product_id": identity,
            "product_name": (
                row.get("product_name") or row.get("title") or description
            ).strip(),
            "product_facts": description,
            "parent_id": parent_id,
            "keys": keys,
        })

        for key in keys:
            if key in seen:
                parent[root(index)] = root(seen[key])
            seen[key] = index

    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(root(index), []).append(row)

    eligible = []
    excluded_groups = 0

    for group in groups.values():
        if any(
            normalize(row["product_id"]) in excluded_ids
            or normalize(row["product_facts"]) in excluded_texts
            for row in group
        ):
            excluded_groups += 1
            continue

        representative = min(
            group,
            key=lambda row: (
                normalize(row["product_id"]),
                normalize(row["product_facts"]),
            ),
        )
        family_keys = sorted(
            set().union(*(row["keys"] for row in group))
        )
        family_id = hashlib.sha256(
            json.dumps(family_keys, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24]

        eligible.append({
            "product_id": representative["product_id"],
            "product_family_id": family_id,
            "product_name": representative["product_name"],
            "product_facts": representative["product_facts"],
            "family_basis": (
                "source_parent_id"
                if any(row["parent_id"] for row in group)
                else "exact_id_and_description"
            ),
        })

    eligible.sort(key=lambda row: row["product_family_id"])
    random.Random(SEED).shuffle(eligible)

    if len(eligible) < 800:
        raise ValueError(
            f"Need 800 eligible product groups; found {len(eligible)}. "
            "No output was written."
        )

    assignments = []
    slices = {
        "train": eligible[:500],
        "validation": eligible[500:600],
        "test": eligible[600:800],
    }

    for split, products in slices.items():
        for index, product in enumerate(products):
            assignments.append({
                **product,
                "split": split,
                "owner": f"member_{index % 4 + 1}",
                "news_context": "",
                "news_source_url": "",
                "news_date": "",
                "target_audience": "Consumers appropriate for this product",
                "visual_context": "",
            })

    if len({row["product_family_id"] for row in assignments}) != 800:
        raise ValueError("Duplicate product-family assignments")

    fields = list(assignments[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "assignment"
        staging.mkdir()

        def write_csv(name, rows):
            with (staging / name).open(
                "x", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        write_csv("master_products.csv", assignments)

        for member in range(1, 5):
            for split in slices:
                selected = [
                    row for row in assignments
                    if row["owner"] == f"member_{member}"
                    and row["split"] == split
                ]
                write_csv(f"member_{member}_{split}.csv", selected)

        first100 = [
            row for row in assignments
            if row["owner"] == "member_1" and row["split"] == "train"
        ][:100]
        write_csv("member_1_train_first100.csv", first100)

        # Preserve immutable assignments separately from editable context fields.
        assignment_core = {
            row["product_id"]: {
                key: row[key]
                for key in (
                    "product_id", "product_family_id",
                    "product_facts", "split", "owner",
                )
            }
            for row in assignments
        }
        with (staging / "assignment_core.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(assignment_core, stream, ensure_ascii=False, indent=2)

        manifest = {
            "seed": SEED,
            "source": str(source),
            "source_sha256": sha256(source),
            "historical_sources": historical_files,
            "excluded_historical_groups": excluded_groups,
            "eligible_groups": len(eligible),
            "assignment_counts": dict(Counter(
                f"{row['owner']}/{row['split']}" for row in assignments
            )),
            "assignment_core_sha256": sha256(
                staging / "assignment_core.json"
            ),
            "prompt_sha256": sha256(
                PROJECT / "configs/prompts/product_humor_v3.txt"
            ),
            "first_batch": {
                "owner": "member_1",
                "split": "train",
                "products": len(first100),
                "file": "member_1_train_first100.csv",
            },
            "limitation": (
                "Variants without a shared source parent ID require manual "
                "review before formal evaluation. Exact-text normalization "
                "does not establish semantic uniqueness."
            ),
        }

        with (staging / "manifest.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        if output.exists():
            raise FileExistsError(f"Assignment already exists: {output}")
        staging.rename(output)

    print(json.dumps({
        "assignment_counts": manifest["assignment_counts"],
        "first_batch": manifest["first_batch"],
        "excluded_historical_groups": excluded_groups,
        "eligible_groups": len(eligible),
    }, indent=2))
    print("Assignment saved:", output)


if __name__ == "__main__":
    main()
