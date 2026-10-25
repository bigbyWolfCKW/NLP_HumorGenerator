from pathlib import Path

from .artifacts import check_hf_weights, check_weights
from .data import read_json, sha256

PROJECT = Path(__file__).resolve().parents[1]


def absolute(path, root=PROJECT):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def qa_dataset(csv=None, root=None):
    config = read_json(PROJECT / "configs/datasets.json")
    source = absolute(csv) if csv else absolute(config["qa_train"], absolute(root or config["root"]))
    if not source.is_file():
        raise FileNotFoundError(
            f"QA training dataset not found: {source}. Set configs/datasets.json, "
            "place the QA CSV under data/source/processed/unbiased/, or select an explicit dataset root.")
    return source


def select_llm(name, backend=None, model=None, adapter=None):
    registry = read_json(PROJECT / "configs/llms.json")
    if name not in registry:
        raise ValueError(f"Unknown LLM: {name}. Add it to configs/llms.json.")
    config = dict(registry[name], name=name)
    config["backend"] = backend or config["backend"]
    if config["backend"] == "transformers":
        location = absolute(model or config["path"])
        # Default execution always uses this project's model folder.
        # Only an explicit --model override may select a cached HF repository ID.
        config["model"] = str(location) if not model or location.exists() else model
    else:
        config["model"] = model or config["ollama_model"]
    config["adapter"] = str(absolute(adapter)) if adapter else None
    if config["adapter"] and config["backend"] != "transformers":
        raise ValueError("--adapter requires the transformers backend.")
    return config


def select_detector(name, seed=None, root=None):
    registry = read_json(PROJECT / "configs/detectors.json")
    if name not in registry:
        raise ValueError(f"Unknown detector: {name}. Add it to configs/detectors.json.")
    config = dict(registry[name], name=name)
    root = absolute(root or "models/detector")
    config["model_root"] = str(root)
    templates = [config.get(key) or "" for key in ["checkpoint", "backbone", "sub_checkpoint"]]
    seeded = any("{seed}" in value for value in templates)
    if seed is not None and not seeded:
        raise ValueError(f"{name} has no seed template. Configure a {{seed}} path before selecting a seed.")
    selected_seed = seed if seed is not None else config.get("default_seed", 42)
    config["seed"] = selected_seed if seeded else None
    for key in ["checkpoint", "backbone", "sub_checkpoint"]:
        if config.get(key):
            config[key] = str(absolute(config[key].format(seed=selected_seed), root))
    return config


def find_sub(config, expected):
    if config.get("sub_checkpoint"):
        paths = [Path(config["sub_checkpoint"])]
    else:
        paths = sorted((Path(config["model_root"]) / "checkpoints").glob("sub_*/best.pt"))
    for path in paths:
        if path.is_file() and sha256(path) == expected:
            check_weights(path)
            return str(path)
    raise FileNotFoundError("Matching SUB encoder not found. Set sub_checkpoint in configs/detectors.json.")


def preflight(config):
    if config["backend"] == "custom":
        if ":" not in (config.get("factory") or ""):
            raise ValueError(f"{config['name']} needs its inference loader. Set factory to module:function in configs/detectors.json.")
        return config
    folder = Path(config["checkpoint"])
    saved = read_json(folder / "config.json")
    if config.get("positive_label") not in (0, 1):
        raise ValueError("positive_label must identify the humor class (0 or 1).")
    if config["backend"] == "hf":
        if config.get("input_mode") not in ["text", "pair"]:
            raise ValueError("input_mode must be text or pair.")
        check_hf_weights(folder)
    elif config["backend"] == "roberta-sub":
        check_weights(folder / "best.pt")
        if saved.get("use_inc"):
            raise ValueError("This checkpoint needs an INC fusion loader.")
        if saved["max_length"] != config["max_length"]:
            raise ValueError("max_length differs from training config.")
        check_hf_weights(config["backbone"])
        if saved["use_sub"]:
            config["sub_checkpoint"] = find_sub(config, saved["sub_checkpoint_sha256"])
    else:
        raise ValueError(f"Unsupported backend: {config['backend']}")
    return config
