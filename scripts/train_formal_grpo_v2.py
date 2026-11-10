"""Train GRPO from a selected SFT adapter and frozen humor reward."""

import argparse
import csv
import hashlib
import json
import os
import re
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer


PROJECT = Path(__file__).resolve().parents[1]
REWARDS = {
    "baseline": PROJECT / (
        "models/detector/checkpoints/11_roberta_large/"
        "qproduct_seed42/best_model"
    ),
    "arena": PROJECT / "models/detector/preferences/arena_v1_seed42/best_model",
    "mechanics": PROJECT / "models/detector/preferences/mechanics_v1_seed42/best_model",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_training_rows(path, core_path, owner, expected_candidates):
    core = read_json(core_path)
    unique = {}
    candidate_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            candidate_id = row["candidate_id"]
            if not candidate_id or candidate_id in candidate_ids:
                raise ValueError("Missing or duplicate candidate ID")
            candidate_ids.add(candidate_id)
            assignment = core[row["product_id"]]
            for field in ("product_family_id", "owner", "split"):
                if row[field] != assignment[field]:
                    raise ValueError(f"Assignment mismatch: {candidate_id}")
            if row["owner"] != owner or row["split"] != "train":
                raise ValueError(f"Only {owner} training products are allowed")
            prompt = row["prompt"]
            if not prompt.strip() or "News context" in prompt:
                raise ValueError("Expected a nonempty product-only prompt")
            context = (
                row.get("verified_product_facts", "").strip()
                or row["product_facts"].strip()
            )
            if not context:
                raise ValueError("Missing reward-model product context")
            item = {
                "prompt": [{"role": "user", "content": prompt}],
                "product_description": context,
                "product_id": row["product_id"],
                "product_family_id": row["product_family_id"],
                "fact_status": row.get("fact_status", ""),
            }
            previous = unique.setdefault(row["product_id"], item)
            if previous != item:
                raise ValueError("Conflicting prompts or contexts for one product")
    if len(candidate_ids) != expected_candidates:
        raise ValueError("Candidate count does not match the completed generation")
    rows = sorted(unique.values(), key=lambda item: item["product_id"])
    if not rows:
        raise ValueError("No training products")
    if len({row["product_family_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate product families in the training set")
    return rows


def invalid_format(text):
    if not text.strip():
        return "empty"
    words = text.split()
    if len(words) > 40:
        return "over_40_words"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "contains_cjk"
    if re.search(r"(?im)^\s*(user|assistant|system)\s*:?\s*$", text):
        return "conversation_marker"
    lowered = text.casefold()
    if any(marker in lowered for marker in (
        "<think>", "</think>", "let me think", "i need to come up with",
        "can you provide another", "here are some options",
    )):
        return "possible_analysis_or_dialogue"
    if len(words) >= 12:
        tokens = [word.casefold() for word in words]
        trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
        if len(set(trigrams)) / len(trigrams) < 0.6:
            return "repetitive"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner", choices=("member_1", "member_2", "member_3", "member_4"),
        required=True,
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--assignment-core", type=Path,
        default=Path("data/formal_v1/assignment_core.json"),
    )
    parser.add_argument("--sft-adapter", type=Path, required=True)
    reward_group = parser.add_mutually_exclusive_group(required=True)
    reward_group.add_argument("--reward", choices=tuple(REWARDS))
    reward_group.add_argument("--reward-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.steps < 1 or args.save_steps < 1:
        raise ValueError("Steps and save interval must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit through the GPU Slurm script")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This script supports one GPU process per job")
    if version("trl") != "0.23.1":
        raise RuntimeError("This script targets TRL 0.23.1; do not upgrade for this run")

    torch.set_num_threads(4)
    train_path = (PROJECT / args.candidates).resolve()
    core_path = (PROJECT / args.assignment_core).resolve()
    generation = read_json(train_path.parent / "completed.json")
    rows = load_training_rows(train_path, core_path, args.owner, generation["candidates"])
    if len(rows) != generation["products"]:
        raise ValueError("Product count does not match the completed generation")
    sft_adapter = (PROJECT / args.sft_adapter).resolve()
    sft_completed = read_json(sft_adapter.parent / "completed.json")
    if Path(sft_completed["final_adapter"]).resolve() != sft_adapter:
        raise ValueError("SFT completion record has a different adapter path")
    adapter_config = read_json(sft_adapter / "adapter_config.json")
    base_path = Path(adapter_config["base_model_name_or_path"])
    if not base_path.is_absolute():
        base_path = PROJECT / base_path
    reward_name = args.reward or "custom"
    reward_path = (
        REWARDS[args.reward] if args.reward
        else (PROJECT / args.reward_checkpoint).resolve()
    )
    output = (PROJECT / args.output).resolve()
    settings = {
        "owner": args.owner,
        "steps": args.steps,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "base_model": str(base_path),
        "sft_adapter": str(sft_adapter),
        "sft_adapter_sha256": file_hash(sft_adapter / "adapter_model.safetensors"),
        "sft_config_sha256": file_hash(sft_adapter / "adapter_config.json"),
        "reward_name": reward_name,
        "reward_checkpoint": str(reward_path),
        "reward_weights_sha256": file_hash(reward_path / "model.safetensors"),
        "reward_config_sha256": file_hash(reward_path / "config.json"),
        "train_file": str(train_path),
        "train_file_sha256": file_hash(train_path),
        "assignment_core_sha256": file_hash(core_path),
        "training_products": len(rows),
        "training_prompt_sha256": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "training_policy": "one prompt per assigned training product; no target ads used",
        "num_generations": 4,
        "learning_rate": 5e-6,
        "beta": 0.0,
        "reward": "clip(logit_1_minus_logit_0,-8,8); invalid_format=-10",
        "reward_context": "verified_product_facts_else_original_product_facts",
        "reward_max_length": 128,
        "max_prompt_length": 512,
        "max_completion_length": 128,
        "format_scope": (
            "empty, whitespace word count, CJK, role/analysis markers, repetition; "
            "not a factuality, full language, or sentence-count verifier"
        ),
        "versions": {
            name: version(name)
            for name in ("torch", "transformers", "trl", "peft")
        },
        "script_sha256": file_hash(Path(__file__)),
    }

    resume = None
    if args.resume:
        if read_json(output / "run.json") != settings:
            raise ValueError("Resume configuration or input files changed")
        if (output / "completed.json").exists():
            print("GRPO already completed:", output, flush=True)
            return
        required = (
            "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"
        )
        checkpoints = [
            path for path in output.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
            and all((path / name).is_file() for name in required)
        ]
        if not checkpoints:
            raise ValueError("No complete resumable checkpoint is available")
        resume = str(max(checkpoints, key=lambda path: int(path.name.split("-")[-1])))
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "run.json", settings)
        save_json(output / "training_prompts.json", rows)

    set_seed(args.seed)

    # Keep the frozen reward model on CPU, as in the working single-GPU run.
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_path, use_fast=False, local_files_only=True
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path, local_files_only=True
    ).float().to("cpu").eval()
    reward_model.requires_grad_(False)
    if reward_model.config.num_labels != 2:
        raise ValueError("Expected a binary humor reward model")

    def humor_reward(completions, product_description, product_id,
                     trainer_state=None, **kwargs):
        texts = [
            completion[0]["content"] if isinstance(completion, list) else completion
            for completion in completions
        ]
        if not (len(texts) == len(product_description) == len(product_id)):
            raise RuntimeError("Completion/context alignment mismatch")
        values = []
        for start in range(0, len(texts), 4):
            batch = reward_tokenizer(
                texts[start:start + 4], product_description[start:start + 4],
                padding="max_length", truncation=True, max_length=128,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = reward_model(**batch).logits.float()
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("Non-finite reward logits")
                values.extend((logits[:, 1] - logits[:, 0]).tolist())

        rewards = []
        with (output / "rollouts.jsonl").open("a", encoding="utf-8") as stream:
            for text, product, pid, raw in zip(
                texts, product_description, product_id, values
            ):
                reason = invalid_format(text)
                reward = -10.0 if reason else max(-8.0, min(8.0, raw))
                rewards.append(reward)
                stream.write(json.dumps({
                    "step": getattr(trainer_state, "global_step", None),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "reward_name": reward_name,
                    "product_id": pid,
                    "product_description": product,
                    "completion": text,
                    "raw_humor_score": raw,
                    "format_failure": reason,
                    "reward": reward,
                }, ensure_ascii=False) + "\n")
        return rewards

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_path, local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        model, sft_adapter, is_trainable=True, local_files_only=True
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    tokenizer_source = (
        sft_adapter if (sft_adapter / "tokenizer_config.json").is_file() else base_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for row in rows:
        tokens = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        if len(tokens) > settings["max_prompt_length"]:
            raise ValueError(f"Prompt would be truncated: {row['product_id']}")

    config = GRPOConfig(
        output_dir=str(output), max_steps=args.steps, learning_rate=5e-6,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_prompt_length=512, max_completion_length=128,
        temperature=0.8, top_p=0.9, beta=0.0, num_iterations=1,
        disable_dropout=True, use_vllm=False,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", max_grad_norm=1.0,
        warmup_steps=min(10, max(1, args.steps // 10)), logging_steps=1,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        save_only_model=False, eval_strategy="no", remove_unused_columns=False,
        dataloader_num_workers=0, report_to="none", seed=args.seed,
        data_seed=args.seed, save_safetensors=True,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=humor_reward, args=config,
        train_dataset=Dataset.from_list(rows), processing_class=tokenizer,
    )

    # Retain both fixes from the successful GRPO run.
    def align_output_head_input(module, inputs):
        hidden_states = inputs[0]
        if hidden_states.dtype != module.weight.dtype:
            hidden_states = hidden_states.to(module.weight.dtype)
        return (hidden_states, *inputs[1:])

    output_head = trainer.model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise RuntimeError("Cannot locate the Qwen output head")
    output_head.register_forward_pre_hook(align_output_head_input)
    trainer.model.enable_input_require_grads()
    trainable_names = [
        name for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Expected trainable LoRA parameters only")
    print(json.dumps(settings, indent=2), flush=True)

    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(output / "final_adapter")
    if file_hash(sft_adapter / "adapter_model.safetensors") != settings["sft_adapter_sha256"]:
        raise RuntimeError("Source SFT adapter changed")
    save_json(output / "completed.json", {
        "reward_name": reward_name,
        "global_step": trainer.state.global_step,
        "source_sft_adapter_unchanged": True,
        "final_adapter": str(output / "final_adapter"),
        "metrics": result.metrics,
    })
    print("GRPO completed:", output, flush=True)


if __name__ == "__main__":
    main()
