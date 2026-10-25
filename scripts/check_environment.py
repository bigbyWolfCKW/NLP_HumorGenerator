"""Validate package files and real model/trainer imports without loading weights."""
import argparse
import importlib
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODULES = [
    ("torch", "torch"), ("requests", "requests"), ("urllib3", "urllib3"),
    ("certifi", "certifi"), ("charset_normalizer", "charset-normalizer"), ("idna", "idna"),
    ("tqdm", "tqdm"), ("huggingface_hub", "huggingface-hub"), ("transformers", "transformers"),
    ("tokenizers", "tokenizers"), ("peft", "peft"), ("trl", "trl"), ("datasets", "datasets"),
    ("accelerate", "accelerate"), ("bitsandbytes", "bitsandbytes"), ("safetensors", "safetensors"),
    ("openpyxl", "openpyxl"), ("sentencepiece", "sentencepiece"), ("google.protobuf", "protobuf"),
    ("numpy", "numpy"),
]


def checked_import(module_name, distribution):
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)
    if not origin or not Path(origin).is_file():
        raise ImportError(f"{module_name} has no package file; installation may be incomplete")
    version = importlib.metadata.version(distribution)
    if not version or version == "None":
        raise ImportError(f"{distribution} has no valid installed version")
    return {"version": version, "file": str(origin)}


def pins():
    result = {}
    with (PROJECT / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    for line in dependencies:
        if "==" in line:
            name, version = line.strip().split("==", 1)
            result[name.lower().replace("_", "-")] = version
    return result


def check(gpu=False):
    report = {"python": sys.executable, "python_version": sys.version, "packages": {}, "errors": []}
    if sys.version_info[:2] != (3, 11):
        report["errors"].append("Python 3.11 is required")
    expected = pins()
    for module, distribution in MODULES:
        try:
            info = checked_import(module, distribution)
            target = expected.get(distribution.lower().replace("_", "-"))
            if target and info["version"].split("+", 1)[0] != target:
                raise ImportError(f"expected {target}, found {info['version']}")
            report["packages"][distribution] = info
            print(f"OK   {distribution}: {info['version']}", flush=True)
        except Exception as error:
            report["errors"].append(f"{distribution}: {type(error).__name__}: {error}")
            print(f"FAIL {report['errors'][-1]}", flush=True)
    try:
        from requests.exceptions import HTTPError
        from tqdm._monitor import TMonitor
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel
        from trl import SFTConfig, SFTTrainer
        from datasets import load_dataset
        print("OK   model/trainer entry points", flush=True)
    except Exception as error:
        report["errors"].append(f"Model/trainer imports: {type(error).__name__}: {error}")
        print(f"FAIL {report['errors'][-1]}", flush=True)
    if gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable inside this job")
            result = (torch.ones(2, 2, device="cuda") @ torch.ones(2, 2, device="cuda")).sum().item()
            if result != 8:
                raise RuntimeError("GPU arithmetic failed")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Default training settings require bf16; configure fp16 for this GPU")
            report["gpu"] = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
            print(f"OK   GPU: {report['gpu']['name']}", flush=True)
        except Exception as error:
            report["errors"].append(f"GPU: {type(error).__name__}: {error}")
            print(f"FAIL {report['errors'][-1]}", flush=True)
    report["status"] = "passed" if not report["errors"] else "failed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run GPU checks inside a GPU allocation")
    parser.add_argument("--report", help="Optional JSON environment report path")
    args = parser.parse_args()
    result = check(args.gpu)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Environment check: {result['status']}")
    sys.exit(0 if result["status"] == "passed" else 1)
