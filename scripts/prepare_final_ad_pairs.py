"""Blind two completed evaluation runs on their shared held-out inputs."""

import argparse
import csv
import json
import random
from pathlib import Path

from prepare_humor_pairs import product_context, write_jsonl


def read_run(folder):
    settings = json.loads((folder / "run.json").read_text(encoding="utf-8"))["settings"]
    completed = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
    with (folder / "candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row["pair_key"]: row for row in rows}
    if not rows or len(index) != len(rows) or len(rows) != completed["candidates"]:
        raise ValueError(f"Missing or duplicate candidates: {folder}")
    return settings, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, required=True)
    parser.add_argument("--right-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    left, a_rows = read_run(args.left_run)
    right, b_rows = read_run(args.right_run)
    for field in (
        "member_id", "selected_products_sha256", "rendered_prompts_sha256",
        "base_model", "seed", "candidates_per_product", "temperature", "top_p",
        "max_new_tokens",
    ):
        if left[field] != right[field]:
            raise ValueError(f"Evaluation runs differ in {field}")
    if left["model_variant"] == right["model_variant"]:
        raise ValueError("Use different model variant names")
    if a_rows.keys() != b_rows.keys():
        raise ValueError("Evaluation candidate keys differ")

    rng = random.Random(args.seed)
    keys = sorted(a_rows)
    rng.shuffle(keys)
    public, private = [], []
    for index, key in enumerate(keys, 1):
        a, b = a_rows[key], b_rows[key]
        if rng.randrange(2):
            a, b = b, a
        pair_id = f'{left["member_id"]}_E{index:04d}'
        public.append({
            "pair_id": pair_id, **product_context(a),
            "A": a["ad_text"], "B": b["ad_text"],
        })
        private.append({
            "pair_id": pair_id, "pair_key": key,
            "product_family_id": a["product_family_id"],
            "A_model_variant": a["model_variant"], "B_model_variant": b["model_variant"],
            "A_candidate_id": a["candidate_id"], "B_candidate_id": b["candidate_id"],
        })

    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "PRIVATE_pair_key.jsonl", private)
    write_jsonl(args.output / "blind_pairs.jsonl", public)
    print(f"Final blind review exported: {len(public)} pairs", flush=True)


if __name__ == "__main__":
    main()
