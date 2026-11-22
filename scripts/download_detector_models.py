"""Download QProduct detector backbones into their configured local folders."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.artifacts import check_hf_weights

REPOSITORIES = {
    "roberta": "FacebookAI/roberta-base",
    "deberta": "microsoft/deberta-v3-base",
    "roberta-large": "FacebookAI/roberta-large",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=REPOSITORIES, default=list(REPOSITORIES))
    p.add_argument("--dry-run", action="store_true", help="Show sources and paths without network or writes.")
    p.add_argument("--max-workers", type=int, default=2)
    return p


def download_plan(names, project=PROJECT):
    project = Path(project).resolve()
    registry = json.loads((project / "configs/detectors.json").read_text(encoding="utf-8"))
    root = project / "models/detector"
    result = []
    for name in dict.fromkeys(names):
        destination = (root / registry[name]["backbone"]).resolve()
        if not destination.is_relative_to(root / "models"):
            raise ValueError(f"Backbone must be inside models/detector/models: {destination}")
        result.append((name, REPOSITORIES[name], destination))
    return result


def complete(folder):
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not config.get("model_type"):
            return False
        check_hf_weights(folder)
        # Also reject empty legacy PyTorch files, which the shared check accepts.
        for weights in folder.glob("*.bin"):
            if weights.stat().st_size == 0:
                return False
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(folder, local_files_only=True)
        return True
    except (OSError, ValueError, KeyError, IndexError, StopIteration, struct.error):
        return False


def run(args, downloader=None, model_info=None):
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    plan = download_plan(args.models)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    for name, repo, destination in plan:
        print(f"{name}: {repo} -> {destination}", flush=True)
        if args.dry_run:
            continue
        if complete(destination):
            print("  Already available locally; skipped.", flush=True)
            continue
        if downloader is None or model_info is None:
            from huggingface_hub import HfApi, snapshot_download
            downloader = downloader or snapshot_download
            model_info = model_info or HfApi().model_info
        info = model_info(repo_id=repo)
        if not info.sha:
            raise ValueError(f"Could not resolve source revision for {repo}")
        files = [item.rfilename for item in info.siblings]
        safe = any(f.startswith("model") and f.endswith(".safetensors") for f in files)
        weights = ["model*.safetensors"] if safe else ["pytorch_model*.bin"]
        downloader(repo_id=repo, revision=info.sha, local_dir=str(destination),
                   allow_patterns=["*.json", "*.txt", "*.model", "LICENSE*", "README.md", *weights],
                   max_workers=args.max_workers)
        if not complete(destination):
            raise ValueError(f"Incomplete backbone or tokenizer: {destination}; retry the same command.")
        provenance = dict(repo_id=repo, revision=info.sha, detector=name)
        (destination / "download_manifest.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"  Download complete. Source revision: {info.sha}", flush=True)


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
