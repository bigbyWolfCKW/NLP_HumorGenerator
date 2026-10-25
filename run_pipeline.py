"""Run the stages selected in test.sbatch, one process per model stage."""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

from src.config import PROJECT, absolute
from src.data import read_json, write_json

STEPS = ("generate", "rank", "review", "prepare", "train", "test")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("demo", "full", "custom", *STEPS), default="demo")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=["generate", "rank"])
    p.add_argument("--run-dir", help="Existing run for partial stages; blank creates a new run")
    p.add_argument("--llm", default="deepseek")
    p.add_argument("--review-llm", default="qwen")
    p.add_argument("--review-backend", choices=["transformers", "api"], default="transformers")
    p.add_argument("--review-api-base")
    p.add_argument("--review-api-key-env", default="REVIEW_API_KEY")
    p.add_argument("--review-api-max-tokens", type=int, default=1024)
    p.add_argument("--detector", default="roberta")
    p.add_argument("--detector-seed", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-products", type=int, help="Default: demo=10, otherwise=50")
    p.add_argument("--epochs", type=float, help="Default: demo=1, otherwise training config")
    p.add_argument("--products", help="Optional product JSON; otherwise configured QA dataset")
    p.add_argument("--training-config", default="configs/training.json")
    p.add_argument("--ranked-file", default="ranked.xlsx", help="Filename inside run folder")
    p.add_argument("--adapter", help="Existing adapter for test only; generation uses the base LLM")
    p.add_argument("--dry-run", action="store_true", help="Print commands; no files or models loaded")
    return p


def plan(args):
    steps = list(STEPS) if args.mode in ("demo", "full") else (
        args.steps if args.mode == "custom" else [args.mode])
    if len(steps) != len(set(steps)) or steps != sorted(steps, key=STEPS.index):
        raise ValueError("STEPS must be unique and in generate rank review prepare train test order.")
    if not args.run_dir and steps[0] != "generate":
        raise ValueError("Set RUN_DIR to the existing run folder for partial stages.")
    count = args.num_products if args.num_products is not None else (10 if args.mode == "demo" else 50)
    if count < 1 or (args.epochs is not None and args.epochs <= 0):
        raise ValueError("NUM_PRODUCTS and EPOCHS must be positive.")
    if args.adapter and "train" in steps:
        raise ValueError("Leave ADAPTER blank when training a new adapter in this run.")
    suffix = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = absolute(args.run_dir or f"results/{args.mode}-{suffix}")
    ranked_name = args.ranked_file
    if not ranked_name or "/" in ranked_name or "\\" in ranked_name or ranked_name in (".", ".."):
        raise ValueError("RANKED_FILE must be a filename inside RUN_DIR.")
    sentences = str(run / "sentences.json")
    ranked = str(run / ranked_name)
    reviewed = str(run / "reviewed.xlsx")
    data = str(run / "data")
    common = ["--llm", args.llm, "--backend", "transformers", "--seed", str(args.seed)]
    detector = ["--detector", args.detector]
    if args.detector_seed is not None:
        detector += ["--seed", str(args.detector_seed)]
    source = ["--products", str(absolute(args.products))] if args.products else ["--num-products", str(count)]
    adapter = str(absolute(args.adapter)) if args.adapter else str(run / "lora/final_adapter")
    commands = {
        "generate": ["generate", *common, *source, "--per-product", "3", "--output", sentences],
        "rank": ["rank", *detector, "--input", sentences, "--output", ranked],
        "review": ["review", "--llm", args.review_llm, "--backend", args.review_backend, "--seed", str(args.seed),
                   "--input", ranked, "--output", reviewed],
        "prepare": ["prepare", "--input", reviewed, "--output", data, "--seed", str(args.seed)],
        "train": ["train", *common, "--data", data, "--config", str(run / "training.json"),
                  "--output", str(run / "lora")],
        "test": ["test", *common, "--adapter", adapter, "--data", str(run / "data/test.jsonl"),
                 "--output", str(run / "test.xlsx")],
    }
    if args.review_backend == "api":
        if "review" in steps and not args.review_api_base:
            raise ValueError("Set REVIEW_API_BASE when using API review.")
        commands["review"] += ["--api-base", args.review_api_base or "", "--api-key-env", args.review_api_key_env,
                               "--api-max-tokens", str(args.review_api_max_tokens)]
    return run, [(step, commands[step]) for step in steps]


def run(args, runner=None):
    folder, commands = plan(args)
    print(f"Run folder: {folder}", flush=True)
    if args.dry_run:
        for step, command in commands:
            print(subprocess.list2cmdline([sys.executable, str(PROJECT / "main.py"), *command]))
        return
    if args.review_backend == "api" and "review" in dict(commands):
        from src.api_review import APIReviewer
        # Validate credentials/URL before any generation; this makes no network request.
        APIReviewer(args.review_llm, args.review_api_base, args.review_api_key_env, args.review_api_max_tokens)
    # Existing runs are used only when explicitly selected. Stage outputs never overwrite.
    if not args.run_dir and folder.exists():
        raise FileExistsError(f"Run folder exists: {folder}")
    if "generate" not in dict(commands) and not folder.is_dir():
        raise FileNotFoundError(f"Existing RUN_DIR not found: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    runner = runner or subprocess.run
    for index, (step, command) in enumerate(commands, 1):
        if step == "train":
            config_path = folder / "training.json"
            if config_path.exists() or (folder / "lora").exists():
                raise FileExistsError(f"Training outputs already exist in {folder}; select a fresh run.")
            config = read_json(absolute(args.training_config))
            config["seed"] = args.seed
            if args.epochs is not None:
                config["epochs"] = args.epochs
            elif args.mode == "demo":
                config["epochs"] = 1
            write_json(config_path, config)
        print(f"[{index}/{len(commands)}] {step}", flush=True)
        runner([sys.executable, str(PROJECT / "main.py"), *command], cwd=str(PROJECT), check=True)
    print(f"Finished: {folder}", flush=True)


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
