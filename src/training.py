from pathlib import Path


def load_model(config, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Transformers generation and LoRA require a GPU allocation.")
    model_path = config["model"]
    if Path(model_path).is_absolute() and not Path(model_path).is_dir():
        raise FileNotFoundError(f"Local LLM folder not found: {model_path}. Copy the complete model and tokenizer into models/llm/.")
    dtype = torch.bfloat16 if config.get("bf16", True) else torch.float16
    kwargs = dict(local_files_only=True, device_map={"": 0}, torch_dtype=dtype,
                  attn_implementation="sdpa")
    if config.get("load_in_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def verify_adapter(config, adapter):
    """Load the existing adapter against the configured base; no generation/training."""
    from peft import PeftModel
    from .artifacts import check_weights
    check_weights(Path(adapter) / "adapter_model.safetensors")
    model, tokenizer = load_model(config, adapter)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
    return {"adapter_loaded": str(adapter), "base_model": config["model"],
            "vocab_size": len(tokenizer), "training": False}


def train(config, dataset_dir, output):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    set_seed(config.get("seed", 42))
    model, tokenizer = load_model(config)
    if config.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none",
        r=config["r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"]))
    dataset = load_dataset("json", data_files={
        split: str(Path(dataset_dir) / f"{split}.jsonl") for split in ["train", "validation"]})
    dataset = dataset.map(lambda x: {"prompt": x["messages"][:-1], "completion": x["messages"][-1:]},
                          remove_columns=["messages"])
    args = SFTConfig(output_dir=str(output), num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], warmup_ratio=.1, max_length=config["max_length"],
        completion_only_loss=True, eos_token=config.get("eos_token") or tokenizer.eos_token, packing=False,
        bf16=config.get("bf16", True), fp16=not config.get("bf16", True), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, dataloader_num_workers=0, report_to="none", seed=config.get("seed", 42))
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset["train"],
                         eval_dataset=dataset["validation"], processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    tokenizer.save_pretrained(str(output / "final_adapter"))

