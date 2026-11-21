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
PRE_DEEPSEEK_EXTRACTION_HASHES = {
    # Candidate handling changed only for DeepSeek; other models keep their settings.
    "generate_formal_ads_v1.py": {
        "ffa6a8ac4d134ea86e1ea45f7ffd438d6355d903ec8bc8b51e2ac695923d18e9",
        "f722723dc9b037313c3cf6521bc80e30837f47a13b7345414913f3716a1d41e1",
    },
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
        or (llm not in {"deepseek", "deepseek14b"}
            and prior_hash in PRE_DEEPSEEK_EXTRACTION_HASHES.get(name, set()))
    ):
        previous = {**previous, "script_sha256": current["script_sha256"]}
    return previous == current
