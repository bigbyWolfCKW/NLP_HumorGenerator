"""Model registry and compatibility for formal Qwen runs."""

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
QWEN_SCRIPT_HASHES = {
    "generate_formal_ads_v1.py": "f3f33cca2b798b4eba4ca5845add2c905ef00e55dc3b7f8e5181c51ec9ed8781",
    "train_formal_sft_v2.py": "afba12b76ec2310f34490654e372267cd3d735326ff27e07d06aef22a1d6c6cb",
    "evaluate_formal_ads_v1.py": "35138bd68ce80b26b436a90efe6607804a37d1cb2e2a46cc20bde04251a1d190",
}
PRE_ALL_SCOPE_HASHES = {
    "generate_formal_ads_v1.py": "a0025d26d446a4545e2c374e4f4772ca051b1a01b0d23f78621539baa5c9444d",
    "evaluate_formal_ads_v1.py": "87f021dcab3e4d6698b076534cdb7bf547b00cfeb9a5a372a88d409d6b191db7",
    "train_formal_grpo_v2.py": "31b7f31af44c39f7649b1e0234d4c48096e6dbeb6680835524ea75a6ca93650c",
}


def model_registry():
    config = json.loads((PROJECT / "configs/llms.json").read_text(encoding="utf-8"))
    return {name: entry for name, entry in config.items() if entry["backend"] == "transformers"}


def compatible_resume(previous, current, llm, script):
    # Known earlier revisions preserve behavior for existing member runs.
    name = Path(script).name
    prior_hash = previous.get("script_sha256")
    if prior_hash and (
        prior_hash == PRE_ALL_SCOPE_HASHES.get(name)
        or (llm == "qwen" and prior_hash == QWEN_SCRIPT_HASHES.get(name))
    ):
        previous = {**previous, "script_sha256": current["script_sha256"]}
    return previous == current
