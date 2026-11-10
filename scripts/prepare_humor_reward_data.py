"""Join blinded ad reviews to private split assignments for reward training."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_index(path):
    raw = path.read_bytes()
    records = {}
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip() or pair_id in records:
            raise ValueError(f"{path}:{number}: missing or duplicate pair_id")
        records[pair_id] = row
    if not records:
        raise ValueError(f"No records in {path}")
    return records, hashlib.sha256(raw).hexdigest()


def prepare(public, key, reviews):
    if public.keys() != key.keys() or public.keys() != reviews.keys():
        raise ValueError("Public pairs, private key, and reviews must have identical pair IDs")

    rows = {"train": [], "validation": []}
    counts = Counter()
    family_splits = {}
    for pair_id, pair in public.items():
        assignment = key[pair_id]
        review = reviews[pair_id]
        split = assignment["reward_split"]
        family = assignment["product_family_id"]
        if not family or split not in rows or family_splits.setdefault(family, split) != split:
            raise ValueError(f"Invalid or overlapping family assignment: {pair_id}")

        compliance = (review["A_compliance"], review["B_compliance"])
        if any(value not in ("PASS", "FAIL", "UNCLEAR") for value in compliance):
            raise ValueError(f"Invalid compliance label: {pair_id}")
        humor = review["humor_choice"]
        grounding = review["product_humor_choice"]
        if any(value not in ("A", "B", "TIE", "NEITHER") for value in (humor, grounding)):
            raise ValueError(f"Invalid preference label: {pair_id}")
        if compliance != ("PASS", "PASS"):
            counts["excluded_compliance"] += 1
            continue
        if humor not in ("A", "B"):
            counts["excluded_humor_tie_or_neither"] += 1
            continue

        loser = "B" if humor == "A" else "A"
        context = pair["supplied_product_information"]
        if any(not isinstance(text, str) or not text.strip()
               for text in (pair[humor], pair[loser], context)):
            raise ValueError(f"Empty ad or product information: {pair_id}")
        rows[split].append({
            "pair_id": pair_id,
            "product_family_id": family,
            "chosen": pair[humor],
            "rejected": pair[loser],
            "context": context,
            "product_humor_choice": grounding,
        })
    if not all(rows.values()):
        raise ValueError("No decisive, compliant pairs in one or both splits")
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public, public_hash = read_index(args.pair_dir / "blind_pairs.jsonl")
    key, key_hash = read_index(args.pair_dir / "PRIVATE_pair_key.jsonl")
    reviews, review_hash = read_index(args.reviews)
    rows, counts = prepare(public, key, reviews)
    manifest = {
        "blind_pairs_sha256": public_hash,
        "private_key_sha256": key_hash,
        "reviews_sha256": review_hash,
        "counts": {split: len(values) for split, values in rows.items()},
        "excluded": dict(counts),
        "training_label": "humor_choice among pairs where both ads pass compliance",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for split, values in rows.items():
        with (args.output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
