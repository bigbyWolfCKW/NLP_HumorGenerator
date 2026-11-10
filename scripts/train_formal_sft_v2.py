"""Train a fresh Qwen LoRA on one or more supervised ad datasets."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.training import load_model


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_examples(paths):
    examples = []
    sources = []
    seen = set()
    for path in paths:
        rows = 0
        kept = 0
        contents = path.read_bytes()
        for number, line in enumerate(contents.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                messages = json.loads(line)["messages"]
                if not isinstance(messages, list) or len(messages) != 2:
                    raise ValueError("Expected two messages.")
                pair = []
                for message, role in zip(messages, ("user", "assistant")):
                    content = message["content"]
                    if message["role"] != role or not isinstance(content, str) or not content.strip():
                        raise ValueError(f"Expected a nonempty {role} message.")
                    pair.append(" ".join(content.split()))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc

            rows += 1
            key = tuple(pair)
            if key in seen:
                continue
            seen.add(key)
            examples.append({"prompt": messages[:1], "completion": messages[1:]})
            kept += 1

        if not rows:
            raise ValueError(f"Training file is empty: {path}")
        sources.append({
            "path": str(path),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "rows": rows,
            "kept": kept,
            "duplicates": rows - kept,
        })
    return examples, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    paths = [(PROJECT / path).resolve() for path in args.train_file]
    if len(set(paths)) != len(paths):
        parser.error("Pass each training file once.")
    output = (PROJECT / args.output).resolve()
    if not args.resume and output.exists():
        raise FileExistsError(output)

    examples, sources = load_examples(paths)
    print(json.dumps({"sources": sources, "unique_examples": len(examples)}), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Submit through the GPU Slurm script.")

    base_model = PROJECT / read_json(PROJECT / "configs/llms.json")["qwen"]["path"]
    lora_config = {
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }

    settings = {
        "initialization": "fresh_lora",
        "base_model": str(base_model),
        "training_sources": sources,
        "script_sha256": sha256(Path(__file__)),
        "loader_sha256": sha256(PROJECT / "src/training.py"),
        "examples": len(examples),
        "epochs": 3,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "seed": 42,
        "lora": lora_config,
        "validation": None,
        "versions": {
            package: version(package)
            for package in ("torch", "transformers", "trl", "peft")
        },
    }

    resume = None
    if args.resume:
        if read_json(output / "run.json") != settings:
            raise ValueError("Resume inputs or settings changed.")
        if (output / "completed.json").exists():
            print("SFT already completed:", output, flush=True)
            return
        checkpoints = [
            path for path in output.glob("checkpoint-*")
            if (path / "trainer_state.json").is_file()
        ]
        if not checkpoints:
            raise RuntimeError("No resumable checkpoint was saved.")
        resume = str(max(
            checkpoints,
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        ))
    set_seed(42)
    use_bf16 = torch.cuda.is_bf16_supported()

    model, tokenizer = load_model(
        {
            "model": str(base_model),
            "load_in_4bit": True,
            "bf16": use_bf16,
        },
    )

    for index, example in enumerate(examples):
        tokens = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        if len(tokens) > settings["max_length"]:
            raise ValueError(
                f"Example {index} exceeds max_length; no truncation applied."
            )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(**lora_config))
    model.config.use_cache = False
    tokenizer.padding_side = "right"

    def align_head_input(module, inputs):
        return (inputs[0].to(module.weight.dtype), *inputs[1:])

    model.get_output_embeddings().register_forward_pre_hook(
        align_head_input
    )

    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Expected only LoRA parameters to train.")
    model.print_trainable_parameters()

    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / "run.json", settings)

    training_args = SFTConfig(
        output_dir=str(output),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=settings["learning_rate"],
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        max_length=settings["max_length"],
        completion_only_loss=True,
        eos_token=tokenizer.eos_token,
        packing=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=1,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        data_seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
    )
    trainer.model.enable_input_require_grads()

    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))
    trainer.save_state()
    save_json(output / "train_results.json", result.metrics)

    save_json(output / "completed.json", {
        "final_adapter": str(output / "final_adapter"),
        "initialization": "fresh_lora",
        "global_step": trainer.state.global_step,
        "examples": len(examples),
    })
    print("SFT completed:", output / "final_adapter", flush=True)


if __name__ == "__main__":
    main()
