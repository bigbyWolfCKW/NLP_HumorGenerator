"""Model registry and compatibility for formal Qwen runs."""

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
QWEN_SCRIPT_HASHES = {
    "generate_formal_ads_v1.py": "f3f33cca2b798b4eba4ca5845add2c905ef00e55dc3b7f8e5181c51ec9ed8781",
    "train_formal_sft_v2.py": "afba12b76ec2310f34490654e372267cd3d735326ff27e07d06aef22a1d6c6cb",
    "evaluate_formal_ads_v1.py": "35138bd68ce80b26b436a90efe6607804a37d1cb2e2a46cc20bde04251a1d190",
}


def model_registry():
    config = json.loads((PROJECT / "configs/llms.json").read_text(encoding="utf-8"))
    return {name: entry for name, entry in config.items() if entry["backend"] == "transformers"}


def compatible_resume(previous, current, llm, script):
    # The model-selection change preserves Qwen behavior at the preceding revision.
    prior_hash = QWEN_SCRIPT_HASHES[Path(script).name]
    if llm == "qwen" and previous.get("script_sha256") == prior_hash:
        previous = {**previous, "script_sha256": current["script_sha256"]}
    return previous == current
