"""Generate resumable advertising candidates from a fixed product assignment."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def format_flags(text, tokens, eos_id):
    flags = []
    if not text.strip():
        flags.append("empty")
    if len(text.split()) > 40:
        flags.append("over_40_words")
    lowered = text.casefold()
    if any(marker in lowered for marker in (
        "<think>", "</think>", "let me think",
        "i need to come up with", "here are some options",
    )):
        flags.append("possible_analysis_leakage")
    if len(tokens) == 192 and int(tokens[-1]) != eos_id:
        flags.append("token_limit_reached")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/formal_v1/member_1_train_first100.csv",
    )
    parser.add_argument("--owner", default="member_1")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/formal_v1/member_1_qwen_first100_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
    "--prompt-file", default="configs/prompts/humor_ad_v1.txt"
    )
    parser.add_argument("--num-products", type=int)
    args = parser.parse_args()

    if args.candidates < 1:
        raise ValueError("Candidate count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Submit through the GPU Slurm script")

    source = PROJECT / args.input
    output = PROJECT / args.output
    prompt_path = PROJECT / args.prompt_file
    core_path = PROJECT / "data/formal_v1/assignment_core.json"
    core = read_json(core_path)

    with source.open(encoding="utf-8-sig", newline="") as stream:
        products = list(csv.DictReader(stream))
    if not products:
        raise ValueError("Product list is empty")

    seen = set()
    for row in products:
        identity = row["product_id"]
        if identity in seen:
            raise ValueError(f"Repeated product ID: {identity}")
        seen.add(identity)

        expected = core[identity]
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Assignment field changed: {identity}, {key}"
                )
        if row["owner"] != args.owner or row["split"] != "train":
            raise ValueError("Only this owner's Train products are allowed")

        if row.get("news_context", "").strip():
            if not all(row.get(key, "").strip() for key in (
                "news_source_url", "news_date"
            )):
                raise ValueError(
                    f"News requires source URL and date: {identity}"
                )
    if args.num_products is not None:
        if not 1 <= args.num_products <= len(products):
            parser.error(f"--num-products must be between 1 and {len(products)}")
        products = products[:args.num_products]
    llm_config = read_json(PROJECT / "configs/llms.json")["qwen"]
    model_path = PROJECT / llm_config["path"]
    template = prompt_path.read_text(encoding="utf-8")

    settings = {
        "owner": args.owner,
        "products": len(products),
        "candidates_per_product": args.candidates,
        "seed": args.seed,
        "input_path": str(source),
        "input_sha256": sha256(source),
        "assignment_core_sha256": sha256(core_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256(prompt_path),
        "script_sha256": sha256(Path(__file__)),
        "model": str(model_path),
        "adapter": None,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_new_tokens": 192,
        "max_words": 40,
        "news_provided_products": sum(
            bool(row.get("news_context", "").strip())
            for row in products
        ),
        "selection_policy": "retain_all_outputs_for_review",
    }

    if args.resume:
        if read_json(output / "run.json") != settings:
            raise ValueError("Resume inputs or settings changed")
        if (output / "completed.json").exists():
            print("Generation already complete:", output)
            return
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run.json", settings)

    sample_dir = output / "samples"
    sample_dir.mkdir(exist_ok=True)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def align_head_input(module, inputs):
        return (
            inputs[0].to(module.weight.dtype),
            *inputs[1:],
        )

    model.get_output_embeddings().register_forward_pre_hook(
        align_head_input
    )

    all_records = []
    for index, product in enumerate(products):
        prompt = template.format(
            product_name=product.get("verified_product_name") or product["product_name"],
            product_facts=product.get("verified_product_facts") or product["product_facts"],
            news_context=product.get("news_context", "").strip() or "None",
            target_audience=product["target_audience"],
            visual_context=product.get("visual_context", "").strip() or "None",
        )
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        if encoded["input_ids"].shape[1] > 2048:
            raise ValueError(
                f"Prompt exceeds the configured safety limit: {product['product_id']}"
            )

        product_key = hashlib.sha256(
            product["product_id"].encode("utf-8")
        ).hexdigest()[:20]

        for candidate in range(args.candidates):
            candidate_id = (
                f"{args.owner}_{product_key}_c{candidate + 1:02d}"
            )
            sample_path = sample_dir / f"{candidate_id}.json"

            if sample_path.exists():
                record = read_json(sample_path)
                if (
                    record["candidate_id"] != candidate_id
                    or record["product_id"] != product["product_id"]
                    or record["prompt"] != prompt
                ):
                    raise ValueError("Saved generation does not match its input")
                all_records.append(record)
                continue

            seed = int(hashlib.sha256(
                f"{args.seed}|{product['product_id']}|{candidate}".encode()
            ).hexdigest()[:8], 16)
            set_seed(seed)

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=192,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            tokens = generated[0, encoded["input_ids"].shape[1]:].cpu()
            text = tokenizer.decode(
                tokens, skip_special_tokens=True
            ).strip()

            record = {
                **product,
                "candidate_id": candidate_id,
                "generator": "qwen_base",
                "seed": seed,
                "prompt": prompt,
                "ad_text": text,
                "word_count": len(text.split()),
                "generated_tokens": len(tokens),
                "format_flags": ";".join(
                    format_flags(text, tokens, tokenizer.eos_token_id)
                ),
                "selected": "",
                "comment": "",
            }
            atomic_json(sample_path, record)
            all_records.append(record)

            print(
                f"Product {index + 1}/{len(products)}, "
                f"candidate {candidate + 1}/{args.candidates}, "
                f"words={record['word_count']}, "
                f"flags={record['format_flags'] or 'none'}",
                flush=True,
            )

    # Rebuild derived exports from individually saved generations on resume.
    csv_temporary = output / "candidates.csv.tmp"
    with csv_temporary.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(all_records[0])
        )
        writer.writeheader()
        writer.writerows(all_records)
    csv_temporary.replace(output / "candidates.csv")

    jsonl_temporary = output / "candidates.jsonl.tmp"
    with jsonl_temporary.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    jsonl_temporary.replace(output / "candidates.jsonl")

    summary = {
        "products": len(products),
        "candidates": len(all_records),
        "flagged_candidates": sum(
            bool(record["format_flags"]) for record in all_records
        ),
        "news_provided_products": settings["news_provided_products"],
        "review_file": str(output / "candidates.csv"),
    }
    atomic_json(output / "completed.json", summary)
    print("Generation completed:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
