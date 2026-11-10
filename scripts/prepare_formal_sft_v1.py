"""Export reviewed ad candidates as SFT messages without review labels."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index_rows(rows, name):
    indexed = {}
    for row in rows:
        key = row["candidate_id"]
        if not key or key in indexed:
            raise ValueError(f"Missing or duplicate candidate ID in {name}: {key}")
        indexed[key] = row
    return indexed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner", choices=("member_1", "member_2", "member_3", "member_4"),
        default="member_1",
    )
    parser.add_argument(
        "--train-filename", choices=("train.jsonl", "reviewed_product_ads_train.jsonl"),
        default="train.jsonl",
    )
    parser.add_argument(
        "--candidates",
        default=(
            "results/formal_v1/member_1_qwen_product_only_facts_v3/"
            "candidates.csv"
        ),
    )
    parser.add_argument(
        "--review",
        default=(
            "results/formal_v1/member_1_qwen_product_only_facts_v3/"
            "detector_scores_v1/sft_ad_review_corrected_AI.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/formal_v1/sft/member_1_product_only_v1",
    )
    args = parser.parse_args()

    source_path = PROJECT / args.candidates
    review_path = PROJECT / args.review
    output = PROJECT / args.output
    core_path = PROJECT / "data/formal_v1/assignment_core.json"

    candidates = index_rows(read_csv(source_path), "candidates")
    reviews = index_rows(read_csv(review_path), "reviews")
    core = json.loads(core_path.read_text(encoding="utf-8"))

    if candidates.keys() != reviews.keys():
        raise ValueError("Candidate and review ID sets do not match.")

    allowed = {"keep", "reject", "needs_review"}
    recommendations = Counter()
    messages = []
    provenance = []

    for candidate_id, row in candidates.items():
        review = reviews[candidate_id]
        recommendation = review["recommendation"].strip().lower()
        if recommendation not in allowed:
            raise ValueError(f"Unknown recommendation: {candidate_id}")
        recommendations[recommendation] += 1

        if recommendation != "keep":
            continue

        assignment = core[row["product_id"]]
        for field in ("product_family_id", "owner", "split"):
            if row[field] != assignment[field]:
                raise ValueError(f"Assignment mismatch: {candidate_id}")
        if row["owner"] != args.owner or row["split"] != "train":
            raise ValueError(f"Unexpected owner or split: {candidate_id}")

        text = row["ad_text"]
        prompt = row["prompt"]
        if not prompt.strip() or not text.strip():
            raise ValueError(f"Empty prompt or ad: {candidate_id}")

        eligible = (
            review["format_pass"].strip().upper() == "TRUE"
            and review["factual_consistency"].strip() == "consistent"
            and int(review["humor_score"]) >= 3
            and int(review["product_relevance"]) >= 4
            and int(review["ad_suitability"]) >= 3
        )
        if not eligible:
            raise ValueError(f"Keep conflicts with review labels: {candidate_id}")
        if len(text.split()) > 40 or re.search(r"[\u4e00-\u9fff]", text):
            raise ValueError(f"Keep fails word/language check: {candidate_id}")
        if row.get("format_flags", "").strip().lower() not in ("", "none"):
            raise ValueError(f"Keep has generation flags: {candidate_id}")

        messages.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": text},
            ]
        })
        provenance.append({
            "candidate_id": candidate_id,
            "product_id": row["product_id"],
            "product_family_id": row["product_family_id"],
            "fact_status": row.get("fact_status", ""),
        })

    if not messages:
        raise ValueError("No eligible SFT examples.")

    output.mkdir(parents=True, exist_ok=False)

    train_path = output / args.train_filename
    with train_path.open("x", encoding="utf-8") as stream:
        for example in messages:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")

    with (output / "provenance.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)

    manifest = {
        "source_candidates": str(source_path),
        "source_candidates_sha256": sha256(source_path),
        "source_review": str(review_path),
        "source_review_sha256": sha256(review_path),
        "assignment_core_sha256": sha256(core_path),
        "script_sha256": sha256(Path(__file__)),
        "review_counts": dict(recommendations),
        "training_examples": len(messages),
        "training_products": len({x["product_id"] for x in provenance}),
        "fact_status_counts": dict(Counter(
            x["fact_status"] for x in provenance
        )),
        "split": "train",
        "validation_policy": (
            "Preserve existing product assignments. "
            "No random row-level train/validation split was created."
        ),
        "review_scope": (
            "AI-selected candidates; consistency with supplied information "
            "does not establish independent factual verification."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SFT export completed:", json.dumps({
        "examples": len(messages),
        "products": manifest["training_products"],
        "training_file": str(train_path),
    }), flush=True)


if __name__ == "__main__":
    main()
