"""Summarize a complete six-model blind review as one CSV row per model."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from prepare_six_model_review import VARIANTS


SLOTS = set("ABCDEF")
COMPLIANCE = ("PASS", "FAIL", "UNCLEAR")
SCORES = ("humor", "product_humor", "ad_suitability")


def read_groups(paths):
    groups = {}
    for path in paths:
        with path.open(encoding="utf-8-sig") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: {error.msg}") from error
                group_id = row.get("group_id") if isinstance(row, dict) else None
                if not isinstance(group_id, str) or not group_id.strip():
                    raise ValueError(f"{path}:{number}: expected a group_id")
                if group_id in groups:
                    raise ValueError(f"{path}:{number}: duplicate group_id {group_id}")
                groups[group_id] = row
    return groups


def get_slots(row, field, group_id):
    slots = row.get(field)
    if not isinstance(slots, dict) or set(slots) != SLOTS or not all(
        isinstance(slot, dict) for slot in slots.values()
    ):
        raise ValueError(f"{group_id}: {field} must contain objects for A–F")
    return slots


def summarize(key, reviews):
    missing = sorted(key.keys() - reviews.keys())
    extra = sorted(reviews.keys() - key.keys())
    if not key:
        raise ValueError("The private key is empty")
    if missing or extra:
        raise ValueError(
            f"Review coverage does not match the private key. "
            f"Missing ({len(missing)}): {', '.join(missing[:10]) or 'none'}. "
            f"Unexpected ({len(extra)}): {', '.join(extra[:10]) or 'none'}."
        )

    counts = {variant: Counter() for variant in VARIANTS}
    totals = {variant: Counter() for variant in VARIANTS}
    for group_id, entry in key.items():
        model_slots = get_slots(entry, "slots", group_id)
        ratings = get_slots(reviews[group_id], "ratings", group_id)
        variants = [entry.get("variant") for entry in model_slots.values()]
        if any(variant not in VARIANTS for variant in variants) or len(set(variants)) != len(VARIANTS):
            raise ValueError(f"{group_id}: private key must assign each model to one slot")
        for slot, rating in ratings.items():
            variant = model_slots[slot]["variant"]
            compliance = rating.get("compliance")
            if compliance not in COMPLIANCE:
                raise ValueError(f"{group_id}/{slot}: expected PASS, FAIL, or UNCLEAR")
            counts[variant][compliance] += 1
            for field in SCORES:
                value = rating.get(field)
                if type(value) is not int or not 1 <= value <= 5:
                    raise ValueError(f"{group_id}/{slot}: {field} must be an integer from 1 to 5")
                totals[variant][field] += value

    n = len(key)
    return [
        {
            "model": variant,
            "n": n,
            **{status.lower(): counts[variant][status] for status in COMPLIANCE},
            "pass_pct": f"{100 * counts[variant]['PASS'] / n:.2f}",
            **{f"mean_{field}": f"{totals[variant][field] / n:.4f}" for field in SCORES},
        }
        for variant in VARIANTS
    ]


def write_summary(stream, rows):
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True, help="Exported blind_six_models directory")
    parser.add_argument("--reviews-dir", type=Path, help="Directory containing reviewed_batch_*.jsonl")
    parser.add_argument("--output", type=Path, help="New CSV path (default: <review-dir>/six_model_summary.csv)")
    args = parser.parse_args()

    reviews_dir = args.reviews_dir or args.review_dir / "reviewer"
    output = args.output or args.review_dir / "six_model_summary.csv"
    try:
        paths = sorted(reviews_dir.glob("reviewed_batch_*.jsonl"))
        if not paths:
            raise ValueError(f"No reviewed_batch_*.jsonl files in {reviews_dir}")
        key = read_groups([args.review_dir / "PRIVATE_group_key.jsonl"])
        rows = summarize(key, read_groups(paths))
        with output.open("x", encoding="utf-8", newline="") as stream:
            write_summary(stream, rows)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    write_summary(sys.stdout, rows)
    print(f"Saved {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
