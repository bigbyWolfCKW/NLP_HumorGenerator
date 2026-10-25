"""Run a small end-to-end demo, with a separate process for each GPU stage."""
import argparse
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

from src.config import PROJECT, absolute, qa_dataset
from src.data import read_json, sha256, write_json


def parser():
    p = argparse.ArgumentParser(description="One-job Hugging Face demo: generate -> rank -> select -> train -> test")
    p.add_argument("--llm", default="qwen")
    p.add_argument("--model", help="Override generator/training/test model folder or cached HF repo ID")
    p.add_argument("--review-llm", help="AI selector; defaults to --llm")
    p.add_argument("--review-model", help="Optional separate model location for AI selection")
    p.add_argument("--detector", default="roberta")
    p.add_argument("--detector-root")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--products", help="Optional extracted products JSON instead of the QA training dataset")
    source.add_argument("--qa-csv", help="Explicit QA training CSV instead of configs/datasets.json")
    p.add_argument("--dataset-root", help="Root containing processed/unbiased/train.csv")
    p.add_argument("--num-products", type=int, default=10)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--training-config", default="configs/training.json")
    p.add_argument("--output", help="New demo folder; defaults to results/demo-JOB_ID or timestamp")
    p.add_argument("--dry-run", action="store_true", help="Show stages without writing files or running models")
    p.add_argument("--seed", type=int, default=42, help="Product sampling, generation and split seed")
    p.add_argument("--detector-seed", type=int, help="Optional detector checkpoint seed")
    return p


def stages(args, output):
    common = ["--llm", args.llm, "--backend", "transformers", "--seed", str(args.seed)]
    if args.model:
        common += ["--model", args.model]
    detector = ["--detector", args.detector]
    if args.detector_root:
        detector += ["--detector-root", str(absolute(args.detector_root))]
    if args.detector_seed is not None:
        detector += ["--seed", str(args.detector_seed)]
    reviewer = args.review_llm or args.llm
    review = ["--llm", reviewer, "--backend", "transformers", "--seed", str(args.seed)]
    review_model = args.review_model or (args.model if reviewer == args.llm else None)
    if review_model:
        review += ["--model", review_model]
    products = str(output / "products.json")
    sentences = str(output / "sentences.json")
    ranked, reviewed = str(output / "ranked.xlsx"), str(output / "reviewed.xlsx")
    data, adapter = str(output / "data"), str(output / "lora/final_adapter")
    return [
        ("Check detector files", ["rank", *detector, "--check"]),
        ("Generate three sentences per product", ["generate", *common, "--products", products,
                                                 "--per-product", "3", "--output", sentences]),
        ("Rank sentences and export Excel", ["rank", *detector, "--input", sentences, "--output", ranked]),
        ("AI selects one sentence per product", ["review", *review, "--input", ranked, "--output", reviewed]),
        ("Prepare product-disjoint LoRA data", ["prepare", "--input", reviewed, "--output", data,
                                               "--seed", str(args.seed)]),
        ("Train a new LoRA adapter", ["train", *common, "--data", data,
                                      "--config", str(output / "training.json"), "--output", str(output / "lora")]),
        ("Compare base and adapter on test products", ["test", *common, "--adapter", adapter,
                                                      "--data", str(output / "data/test.jsonl"),
                                                      "--output", str(output / "test.xlsx")]),
    ]


def run(args, runner=None):
    if args.num_products < 3 or args.epochs <= 0:
        raise ValueError("Demo requires at least 3 products and a positive epoch count.")
    if args.products:
        source = absolute(args.products)
        products = read_json(source)
    else:
        from src.products import from_qa_csv
        source = qa_dataset(args.qa_csv, args.dataset_root)
        products = from_qa_csv(source)
    if not isinstance(products, list) or len(products) < args.num_products:
        raise ValueError("Products file has fewer products than --num-products.")
    identities = [r.get("product_id") for r in products]
    if any(not isinstance(i, str) or not i.strip() for i in identities) or len(set(identities)) != len(identities):
        raise ValueError("Products file requires unique, nonempty product_id strings.")
    suffix = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = absolute(args.output or f"results/demo-{suffix}")
    plan = stages(args, output)
    if args.dry_run:
        print(f"Dataset: {source}")
        print(f"Demo: {args.num_products} products, 3 sentences each, {args.epochs:g} training epoch(s)")
        for i, (label, command) in enumerate(plan, 1):
            print(f"{i}. {label}: {subprocess.list2cmdline([sys.executable, 'main.py', *command])}")
        return
    if output.exists():
        raise FileExistsError(f"Demo output exists; choose a new --output: {output}")
    training = read_json(absolute(args.training_config))
    training.update(epochs=args.epochs, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "products.json", random.Random(args.seed).sample(products, args.num_products))
    write_json(output / "training.json", training)
    summary = {"status": "running", "products": args.num_products, "epochs": args.epochs,
               "llm": args.llm, "review_llm": args.review_llm or args.llm, "detector": args.detector,
               "dataset": str(source), "dataset_sha256": sha256(source),
               "completed_stages": []}
    runner = runner or subprocess.run
    for i, (label, command) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {label}", flush=True)
        try:
            runner([sys.executable, str(PROJECT / "main.py"), *command], cwd=str(PROJECT), check=True)
        except Exception:
            summary.update(status="failed", failed_stage=label)
            write_json(output / "summary.json", summary)
            raise
        summary["completed_stages"].append(label)
    summary["status"] = "complete"
    write_json(output / "summary.json", summary)
    print(f"\nDemo complete: {output}\nOpen reviewed.xlsx and test.xlsx to inspect the results.", flush=True)


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
