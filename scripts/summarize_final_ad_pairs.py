"""Decode a completed held-out A/B review and write a CSV summary."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from prepare_humor_reward_data import read_index


COMPLIANCE = ("PASS", "FAIL", "UNCLEAR")
CRITERIA = ("humor_choice", "product_humor_choice", "ad_suitability_choice", "overall_choice")
SLOTS = ("A", "B")


def summarize(key, reviews):
    missing = sorted(key.keys() - reviews.keys())
    extra = sorted(reviews.keys() - key.keys())
    if missing or extra:
        raise ValueError(
            f"Review IDs differ from the private key. "
            f"Missing ({len(missing)}): {', '.join(missing[:5]) or 'none'}. "
            f"Unexpected ({len(extra)}): {', '.join(extra[:5]) or 'none'}."
        )
    variants = None
    compliance = Counter()
    choices = {criterion: Counter() for criterion in CRITERIA}
    both_pass_humor = Counter()
    for pair_id, entry in key.items():
        models = {slot: entry.get(f"{slot}_model_variant") for slot in SLOTS}
        if any(not isinstance(model, str) or not model.strip() for model in models.values()):
            raise ValueError(f"{pair_id}: missing model variant")
        pair_models = set(models.values())
        if len(pair_models) != 2 or (variants is not None and pair_models != variants):
            raise ValueError(f"{pair_id}: expected the same two distinct model variants")
        variants = pair_models

        review = reviews[pair_id]
        statuses = [review.get(f"{slot}_compliance") for slot in SLOTS]
        for slot, status in zip(SLOTS, statuses):
            if status not in COMPLIANCE:
                raise ValueError(f"{pair_id}/{slot}: invalid compliance status")
            compliance[models[slot], status] += 1
        for criterion in CRITERIA:
            choice = review.get(criterion)
            if choice not in (*SLOTS, "TIE", "NEITHER"):
                raise ValueError(f"{pair_id}: invalid {criterion}")
            winner = models[choice] if choice in SLOTS else choice
            choices[criterion][winner] += 1
            if criterion == "humor_choice" and statuses == ["PASS", "PASS"]:
                both_pass_humor[winner] += 1

    models = sorted(variants, reverse=True)
    total = len(key)
    for status in COMPLIANCE:
        yield {
            "metric": f"compliance_{status.lower()}",
            "scope": "all",
            "pairs": total,
            **{model: compliance[model, status] for model in models},
        }
    yield {
        "metric": "compliance_pass_pct",
        "scope": "all",
        "pairs": total,
        **{model: f"{100 * compliance[model, 'PASS'] / total:.2f}" for model in models},
    }
    for criterion in CRITERIA:
        counts = choices[criterion]
        yield {
            "metric": criterion,
            "scope": "all",
            "pairs": total,
            **{model: counts[model] for model in models},
            "tie": counts["TIE"],
            "neither": counts["NEITHER"],
        }
    yield {
        "metric": "humor_choice",
        "scope": "both_pass",
        "pairs": sum(both_pass_humor.values()),
        **{model: both_pass_humor[model] for model in models},
        "tie": both_pass_humor["TIE"],
        "neither": both_pass_humor["NEITHER"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.review_dir / "final_pair_summary.csv"
    try:
        key, _ = read_index(args.review_dir / "PRIVATE_pair_key.jsonl")
        reviews, _ = read_index(args.review_dir / "final_reviewed_pairs.jsonl")
        rows = list(summarize(key, reviews))
        with output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*rows[0], "tie", "neither"])
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {output} ({len(key)} pairs)")


if __name__ == "__main__":
    main()
