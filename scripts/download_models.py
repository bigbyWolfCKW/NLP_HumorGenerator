"""Download selected Hugging Face LLMs into the configured project folders."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import struct
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.artifacts import check_hf_weights

# Transformers assets only; avoid duplicate original-format model weights.
FILES = ["*.json", "model*.safetensors", "*.model*", "*.txt", "*.jinja", "*.tiktoken",
         "LICENSE*", "README.md"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--list", action="store_true", help="Show model keywords; no download")
    choice.add_argument("--models", nargs="+", help="Model keywords from configs/llms.json")
    choice.add_argument("--from-batch", metavar="FILE", help="Read literal LLM and REVIEW_LLM settings (does not execute the file)")
    p.add_argument("--dry-run", action="store_true", help="Show selected downloads without network or writes")
    p.add_argument("--refresh", action="store_true", help="Contact Hub even when complete local weights exist")
    p.add_argument("--max-workers", type=int, default=2)
    return p


def batch_models(path):
    settings = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(LLM|REVIEW_LLM|REVIEW_BACKEND)=(.*)$", line)
        if not match:
            continue
        tokens = shlex.split(match[2], comments=True)
        if len(tokens) != 1 or not re.fullmatch(r"[A-Za-z0-9_./:-]+", tokens[0]):
            raise ValueError(f"{match[1]} must be a literal model keyword, without shell expressions.")
        settings[match[1]] = tokens[0]
    if "LLM" not in settings or "REVIEW_LLM" not in settings:
        raise ValueError("Batch file must define both LLM and REVIEW_LLM.")
    backend = settings.get("REVIEW_BACKEND", "transformers")
    if backend not in ("transformers", "api", "ollama"):
        raise ValueError("Unknown REVIEW_BACKEND in batch file.")
    return [settings["LLM"]] if backend in ("api", "ollama") else [settings["LLM"], settings["REVIEW_LLM"]]


def download_plan(names, registry, project=PROJECT):
    root = Path(project).resolve()
    plan = []
    seen = set()
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown model: {name}. Use --list to see choices.")
        config = registry[name]
        repo = config.get("repo_id")
        if not repo:
            raise ValueError(f"Model {name} has no repo_id in configs/llms.json.")
        destination = (root / config["path"]).resolve()
        if not destination.is_relative_to(root / "models/llm"):
            raise ValueError(f"Model destination must stay inside models/llm: {destination}")
        key = (repo, str(destination))
        if key not in seen:
            plan.append((name, repo, destination))
            seen.add(key)
    return plan


def complete(folder):
    try:
        check_hf_weights(folder)
        for name in ("config.json", "tokenizer_config.json"):
            json.loads((folder / name).read_text(encoding="utf-8"))
        return any((folder / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer.model.v3"))
    except (OSError, ValueError, KeyError, IndexError, StopIteration, struct.error):
        return False


def run(args, downloader=None):
    registry = json.loads((PROJECT / "configs/llms.json").read_text(encoding="utf-8"))
    if args.list:
        for name, config in registry.items():
            print(f"{name:14} {config.get('repo_id', '(no download source)')}")
        return
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive.")
    path = Path(args.from_batch) if args.from_batch else None
    if path is not None and not path.is_absolute():
        path = PROJECT / path
    names = batch_models(path) if path else args.models
    plan = download_plan(names, registry)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    for name, repo, destination in plan:
        print(f"{name}: {repo} -> {destination}", flush=True)
        if args.dry_run:
            continue
        if not args.refresh and complete(destination):
            print("  Already available locally; skipped.", flush=True)
            continue
        if downloader is None:
            from huggingface_hub import snapshot_download
            downloader = snapshot_download
        downloader(repo_id=repo, local_dir=str(destination), allow_patterns=FILES,
                   max_workers=args.max_workers)
        if not complete(destination):
            raise ValueError(f"Downloaded model is incomplete: {destination}; retry the same command.")
        print("  Download complete.", flush=True)


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
