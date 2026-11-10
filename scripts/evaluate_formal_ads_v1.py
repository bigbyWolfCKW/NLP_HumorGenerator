"""Generate matched held-out ads with a selected base, SFT, or GRPO model."""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

VARIANTS = ("base", "sft", "grpo_baseline", "grpo_arena", "grpo_mechanics")
EXPLICIT_VARIANTS = ("sft_pooled", "grpo_original_arena", "grpo_adapted")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def absolute(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def select_products(source, core, member, limit, seed):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No products in the input CSV")
    seen_ids, seen_families = set(), set()
    for row in rows:
        pid = row["product_id"]
        family = row["product_family_id"]
        if pid in seen_ids or family in seen_families:
            raise ValueError("Duplicate product IDs or families")
        seen_ids.add(pid)
        seen_families.add(family)
        assignment = core[pid]
        for field in ("product_family_id", "owner", "split"):
            if row[field] != assignment[field]:
                raise ValueError(f"Assignment mismatch for {pid}: {field}")
        if row["owner"] != member or row["split"] != "test":
            raise ValueError("Input must contain only this member's test products")
        if not row["product_name"].strip() or not row["product_facts"].strip():
            raise ValueError("Missing product name or information")
    rows.sort(key=lambda row: (
        hashlib.sha256(f"{seed}|{row['product_family_id']}".encode()).hexdigest(),
        row["product_id"],
    ))
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise ValueError(f"NUM_PRODUCTS must be between 1 and {len(rows)}")
        rows = rows[:limit]
    return rows


def render_prompt(template, row):
    prompt = template.format(
        product_name=(row.get("verified_product_name", "").strip()
                      or row["product_name"]),
        product_facts=(row.get("verified_product_facts", "").strip()
                       or row["product_facts"]),
        target_audience=(row.get("target_audience", "").strip()
                         or "Consumers appropriate for this product"),
    )
    if "News context" in prompt:
        raise ValueError("Formal comparison requires the product-only prompt")
    return prompt


def candidate_seed(seed, product_id, index):
    return int(hashlib.sha256(
        f"{seed}|{product_id}|{index}".encode("utf-8")
    ).hexdigest()[:8], 16)


def format_flags(text, generated_ids, eos_ids, token_limit):
    flags = []
    if not text.strip():
        flags.append("empty")
    if len(text.split()) > 40:
        flags.append("over_40_words")
    if re.search(r"[\u4e00-\u9fff]", text):
        flags.append("contains_cjk")
    if re.search(r"(?im)^\s*(user|assistant|system)\s*:?\s*$", text):
        flags.append("conversation_marker")
    if any(marker in text.casefold() for marker in (
        "<think>", "</think>", "let me think", "i need to come up with",
        "can you provide another", "here are some options",
    )):
        flags.append("possible_analysis_or_dialogue")
    if len(generated_ids) >= token_limit and generated_ids[-1] not in eos_ids:
        flags.append("token_limit_reached")
    return ";".join(flags)


def adapter_for(variant):
    if variant == "base":
        return None
    if variant == "sft":
        return PROJECT / "results/formal_v1/member_1_qwen_sft_product_only_v1/final_adapter"
    reward = variant.removeprefix("grpo_")
    return PROJECT / f"results/formal_v1/member_1_qwen_grpo_{reward}_v1_seed42/final_adapter"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member-id", choices=[f"member_{i}" for i in range(1, 5)], required=True)
    p.add_argument("--llm", choices=["qwen"], default="qwen")
    p.add_argument("--model-variant", choices=VARIANTS + EXPLICIT_VARIANTS, default="sft")
    p.add_argument("--adapter", help="Path to a completed run's final_adapter directory")
    p.add_argument("--products")
    p.add_argument("--num-products", type=int)
    p.add_argument("--candidates-per-product", type=int, default=4)
    p.add_argument("--prompt-file", default="configs/prompts/product_only_v2.txt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(args=None):
    args = args or parser().parse_args()
    if args.candidates_per_product < 1:
        raise ValueError("Candidate count must be positive")
    if args.resume and not args.output:
        raise ValueError("Set RUN_DIR when resuming")
    source = absolute(args.products or f"data/formal_v1/{args.member_id}_test.csv")
    core_path = PROJECT / "data/formal_v1/assignment_core.json"
    prompt_path = absolute(args.prompt_file)
    template = prompt_path.read_text(encoding="utf-8")
    products = select_products(source, read_json(core_path), args.member_id,
                               args.num_products, args.seed)
    prompts = [render_prompt(template, row) for row in products]
    registry = read_json(PROJECT / "configs/llms.json")[args.llm]
    base_path = absolute(registry["path"]).resolve()
    if not (base_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing base model configuration: {base_path}")
    adapter_arg = getattr(args, "adapter", None)
    if args.model_variant == "base" and adapter_arg:
        raise ValueError("The base evaluation must run without an adapter")
    if args.model_variant in EXPLICIT_VARIANTS and not adapter_arg:
        raise ValueError("This model variant requires --adapter")
    adapter = absolute(adapter_arg).resolve() if adapter_arg else adapter_for(args.model_variant)
    if adapter is not None:
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing adapter weights: {adapter}")
        config = read_json(adapter / "adapter_config.json")
        adapter_base = absolute(config["base_model_name_or_path"]).resolve()
        if adapter_base != base_path:
            raise ValueError("The selected adapter belongs to a different base model")
        completed = read_json(adapter.parent / "completed.json")
        if absolute(completed["final_adapter"]).resolve() != adapter.resolve():
            raise ValueError("Adapter completion record mismatch")
    run_id = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = absolute(args.output or (
        f"results/evaluation_v1/{args.member_id}/{args.llm}/"
        f"{args.model_variant}_seed{args.seed}_{run_id}"
    ))
    summary = {
        "member": args.member_id,
        "model_variant": args.model_variant,
        "products": len(products),
        "candidates": len(products) * args.candidates_per_product,
        "adapter": str(adapter) if adapter else None,
        "output": str(output),
    }
    print("Evaluation plan:", json.dumps(summary), flush=True)
    if args.dry_run:
        print("Dry run complete: no models loaded or files written.", flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import set_seed
    from src.training import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("Submit through Slurm with a GPU")
    settings = {
        "member_id": args.member_id,
        "llm": args.llm,
        "model_variant": args.model_variant,
        "source_csv": str(source),
        "source_sha256": file_hash(source),
        "core_sha256": file_hash(core_path),
        "prompt_sha256": file_hash(prompt_path),
        "selected_products_sha256": digest(products),
        "rendered_prompts_sha256": digest(prompts),
        "base_model": str(base_path),
        "base_config_sha256": file_hash(base_path / "config.json"),
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors") if adapter else None,
        "script_sha256": file_hash(Path(__file__)),
        "loader_sha256": file_hash(PROJECT / "src/training.py"),
        "products": len(products),
        "candidates_per_product": args.candidates_per_product,
        "seed": args.seed,
        "seed_rule": "SHA256(seed|product_id|candidate_index); variant excluded",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 192,
        "selection_policy": "retain all outputs; no format retries",
    }
    if args.resume:
        if read_json(output / "run.json")["settings"] != settings:
            raise ValueError("Resume inputs, model or settings changed")
        if (output / "completed.json").exists():
            print("Evaluation already complete:", output, flush=True)
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "run.json", {
            "settings": settings,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save_json(output / "products_snapshot.json", products)
    samples = output / "samples"
    samples.mkdir(exist_ok=True)

    model, tokenizer = load_model({
        "model": str(base_path), "load_in_4bit": True,
        "bf16": torch.cuda.is_bf16_supported(),
    }, adapter=str(adapter) if adapter else None)
    if adapter:
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=False, local_files_only=True
        )
    model.eval()
    model.requires_grad_(False)
    tokenizer.padding_side = "left"

    def align_head(module, inputs):
        return (inputs[0].to(module.weight.dtype), *inputs[1:])

    model.get_output_embeddings().register_forward_pre_hook(align_head)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    records = []
    for position, (row, prompt) in enumerate(zip(products, prompts), 1):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        if encoded["input_ids"].shape[1] > 2048:
            raise ValueError("Prompt exceeds input limit; no truncation applied")
        for index in range(args.candidates_per_product):
            pair_key = digest([args.member_id, row["product_id"], args.seed, index])[:24]
            cid = "eval_" + digest([pair_key, args.model_variant])[:24]
            seed = candidate_seed(args.seed, row["product_id"], index)
            sample_path = samples / f"{cid}.json"
            if sample_path.exists():
                record = read_json(sample_path)
                if any(record[key] != expected for key, expected in (
                    ("candidate_id", cid), ("prompt", prompt), ("seed", seed),
                    ("model_variant", args.model_variant),
                )):
                    raise ValueError("Cached sample identity mismatch")
            else:
                set_seed(seed)
                with torch.inference_mode():
                    tokens = model.generate(
                        **encoded, max_new_tokens=192, do_sample=True,
                        temperature=0.8, top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated_ids = tokens[0, encoded["input_ids"].shape[1]:].tolist()
                text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                record = dict(row)
                record.update({
                    "candidate_id": cid, "pair_key": pair_key,
                    "model_variant": args.model_variant, "generator": args.llm,
                    "seed": seed, "candidate_index": index,
                    "prompt": prompt, "ad_text": text,
                    "raw_output": tokenizer.decode(generated_ids, skip_special_tokens=False),
                    "word_count": len(text.split()),
                    "generated_tokens": len(generated_ids),
                    "format_flags": format_flags(text, generated_ids, eos_ids, 192),
                })
                save_json(sample_path, record)
            records.append(record)
            print(f"Product {position}/{len(products)}, candidate "
                  f"{index + 1}/{args.candidates_per_product}, "
                  f"flags={record['format_flags'] or 'none'}", flush=True)

    fields = list(dict.fromkeys(key for record in records for key in record))
    temporary = output / "candidates.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "candidates.csv")
    temporary = output / "candidates.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output / "candidates.jsonl")
    summary["flagged_candidates"] = sum(bool(row["format_flags"]) for row in records)
    summary["candidates_file"] = str(output / "candidates.csv")
    save_json(output / "completed.json", summary)
    print("Formal evaluation generation completed:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
