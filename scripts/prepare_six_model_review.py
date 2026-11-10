"""Export matched, anonymous groups for six-arm evaluation."""

import argparse
import random
from pathlib import Path

from prepare_final_ad_pairs import read_run
from prepare_humor_pairs import product_context, write_jsonl


VARIANTS = (
    "base", "sft_pooled", "grpo_baseline", "grpo_original_arena",
    "grpo_mechanics", "grpo_adapted",
)
MATCH_FIELDS = (
    "member_id", "source_sha256", "selected_products_sha256",
    "rendered_prompts_sha256", "base_model", "seed",
    "candidates_per_product", "temperature", "top_p", "max_new_tokens",
)


def prepare(root, seed):
    runs = [read_run(root / variant) for variant in VARIANTS]
    reference, reference_rows = runs[0]
    for variant, (settings, rows) in zip(VARIANTS, runs):
        if settings["model_variant"] != variant:
            raise ValueError(f"Model variant does not match directory: {variant}")
        for field in MATCH_FIELDS:
            if settings[field] != reference[field]:
                raise ValueError(f"{variant} differs in {field}")
        if rows.keys() != reference_rows.keys():
            raise ValueError(f"{variant} candidate keys differ")
        if any(
            row["model_variant"] != variant or row["split"] != "test"
            or row["owner"] != reference["member_id"]
            for row in rows.values()
        ):
            raise ValueError(f"{variant} contains a candidate from the wrong arm or split")
    if len(reference_rows) != reference["products"] * reference["candidates_per_product"]:
        raise ValueError("Unexpected number of matched groups")

    rng = random.Random(seed)
    keys = sorted(reference_rows)
    rng.shuffle(keys)
    public, private = [], []
    for index, key in enumerate(keys, 1):
        ref_row = reference_rows[key]
        context = product_context(ref_row)
        rows = [candidates[key] for _, candidates in runs]
        for row in rows:
            if any(row[field] != ref_row[field] for field in (
                "product_id", "product_family_id", "candidate_index", "seed", "prompt",
            )) or product_context(row) != context:
                raise ValueError(f"Mismatched generation inputs for {key}")
        rng.shuffle(rows)
        group_id = f'{reference["member_id"]}_G{index:04d}'
        slots = dict(zip("ABCDEF", rows))
        public.append({
            "group_id": group_id,
            **context,
            "ads": {slot: row["ad_text"] for slot, row in slots.items()},
        })
        private.append({
            "group_id": group_id,
            "pair_key": key,
            "product_family_id": ref_row["product_family_id"],
            "candidate_index": int(ref_row["candidate_index"]),
            "slots": {
                slot: {
                    "variant": row["model_variant"], "candidate_id": row["candidate_id"],
                }
                for slot, row in slots.items()
            },
        })
    return public, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    public, private = prepare(args.evaluation_dir, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    reviewer = args.output / "reviewer"
    reviewer.mkdir()
    write_jsonl(args.output / "PRIVATE_group_key.jsonl", private)
    for batch, start in enumerate(range(0, len(public), args.batch_size), 1):
        write_jsonl(
            reviewer / f"blind_batch_{batch:02d}.jsonl",
            public[start:start + args.batch_size],
        )
    print(f"Exported {len(public)} groups and {len(public) * len(VARIANTS)} ads.")


if __name__ == "__main__":
    main()
