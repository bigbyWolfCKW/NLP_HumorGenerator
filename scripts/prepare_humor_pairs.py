"""Export blinded ad pairs with a private product-family split key."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def product_context(row):
    return {
        "product_name": row.get("verified_product_name", "").strip()
        or row["product_name"].strip(),
        "supplied_product_information": row.get("verified_product_facts", "").strip()
        or row["product_facts"].strip(),
        "target_audience": row["target_audience"].strip(),
    }


def prepare_pairs(rows, completion, owner, seed):
    products = defaultdict(list)
    candidate_ids = set()
    for row in rows:
        if row["owner"] != owner or row["split"] != "train":
            raise ValueError(f"Expected only {owner} training candidates")
        candidate_id = row["candidate_id"].strip()
        product_id = row["product_id"].strip()
        if not candidate_id or not product_id or candidate_id in candidate_ids:
            raise ValueError("Missing product ID or missing/duplicate candidate ID")
        candidate_ids.add(candidate_id)
        products[product_id].append(row)

    if (len(candidate_ids), len(products)) != (
        completion["candidates"], completion["products"],
    ):
        raise ValueError("Candidate or product count does not match completed.json")

    rng = random.Random(seed)
    selected = []
    for product_id, candidates in sorted(products.items()):
        family = candidates[0]["product_family_id"].strip()
        context = product_context(candidates[0])
        if not family or not context["supplied_product_information"]:
            raise ValueError(f"Missing product family or product information: {product_id}")

        eligible = {}
        for row in sorted(candidates, key=lambda item: item["candidate_id"]):
            if row["product_family_id"].strip() != family or product_context(row) != context:
                raise ValueError(f"Inconsistent product context: {product_id}")
            text = row["ad_text"].strip()
            if text and len(text.split()) <= 40 and not row["format_flags"].strip():
                eligible.setdefault(" ".join(text.casefold().split()), row)

        ads = list(eligible.values())
        rng.shuffle(ads)
        if len(ads) >= 2:
            selected.append((product_id, family, context, ads[0], ads[1]))
        if len(ads) >= 3:
            first = ads[3] if len(ads) >= 4 else ads[0]
            selected.append((product_id, family, context, first, ads[2]))

    families = sorted({family for _, family, _, _, _ in selected})
    if len(families) < 2:
        raise ValueError("Need at least two eligible product families")
    rng.shuffle(families)
    validation = set(families[:max(1, round(len(families) * 0.2))])
    rng.shuffle(selected)

    public, private = [], []
    for index, (product, family, context, a, b) in enumerate(selected, 1):
        if rng.randrange(2):
            a, b = b, a
        pair_id = f"{owner}_P{index:04d}"
        public.append({
            "pair_id": pair_id,
            **context,
            "A": a["ad_text"],
            "B": b["ad_text"],
        })
        private.append({
            "pair_id": pair_id,
            "product_id": product,
            "product_family_id": family,
            "reward_split": "validation" if family in validation else "train",
            "A_candidate_id": a["candidate_id"],
            "B_candidate_id": b["candidate_id"],
        })
    return public, private


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner", choices=("member_1", "member_2", "member_3", "member_4"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    completion = json.loads(
        (args.run_dir / "completed.json").read_text(encoding="utf-8")
    )
    with (args.run_dir / "candidates.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        public, private = prepare_pairs(
            csv.DictReader(stream), completion, args.owner, args.seed
        )

    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "PRIVATE_pair_key.jsonl", private)
    write_jsonl(args.output / "blind_pairs.jsonl", public)
    val_count = sum(row["reward_split"] == "validation" for row in private)
    print(
        f"Pair export completed: {len(public)} pairs, "
        f"{len(public) - val_count} train, {val_count} validation",
        flush=True,
    )


if __name__ == "__main__":
    main()
