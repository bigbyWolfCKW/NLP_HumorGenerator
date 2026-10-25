"""Train standard HF binary detectors from existing product-disjoint QA splits."""
import json
from pathlib import Path

from src.artifacts import check_hf_weights
from src.config import absolute, select_detector
from src.data import sha256, write_json
from .data import classification_metrics, ensure_disjoint, read_examples, split_path


def train_detector(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.max_length < 1:
        raise ValueError("Epochs, batch size, learning rate and max length must be positive")
    config = select_detector(args.detector, args.seed)
    if config["backend"] != "hf":
        raise ValueError("This training command supports standard HF classifiers. Fusion/auxiliary training is retained under detector/experiments/ for adaptation.")
    output = absolute(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new detector output: {output}")
    backbone = Path(config["backbone"])
    check_hf_weights(backbone)
    train_path, validation_path = split_path("train", args.train_csv), split_path("validation", args.validation_csv)
    train_rows = read_examples(train_path, config["input_mode"])
    validation_rows = read_examples(validation_path, config["input_mode"])
    ensure_disjoint(train_rows, validation_rows)
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Submit detector training through a GPU sbatch job")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    class Examples(Dataset):
        def __init__(self, rows):
            texts = [[r["question"] for r in rows]]
            if config["input_mode"] == "pair":
                texts.append([r["product_description"] for r in rows])
            self.encoded = tokenizer(*texts, padding="max_length", truncation=True,
                                     max_length=args.max_length, return_tensors="pt")
            self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{key: value[index] for key, value in self.encoded.items()}, "labels": self.labels[index]}
    train_loader = DataLoader(Examples(train_rows), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(Examples(validation_rows), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2, local_files_only=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * .1), steps)
    saved = {"detector": args.detector, "seed": args.seed, "backbone": str(backbone),
             "input_mode": config["input_mode"], "max_length": args.max_length, "epochs": args.epochs,
             "batch_size": args.batch_size, "learning_rate": args.learning_rate,
             "weight_decay": .01, "warmup_ratio": .1, "selection": "best_validation_macro_f1",
             "train_csv_sha256": sha256(train_path), "validation_csv_sha256": sha256(validation_path)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", saved)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = {}
        for name, loader, training in [("train", train_loader, True), ("validation", validation_loader, False)]:
            model.train(training)
            actual, predicted, loss_sum = [], [], 0.0
            with torch.set_grad_enabled(training):
                for batch in loader:
                    batch = {key: value.to(args.device) for key, value in batch.items()}
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    result = model(**batch)
                    if not torch.isfinite(result.loss) or not torch.isfinite(result.logits).all():
                        raise RuntimeError("Non-finite detector loss/logits")
                    if training:
                        result.loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                    actual.extend(batch["labels"].detach().cpu().tolist())
                    predicted.extend(result.logits.detach().argmax(dim=1).cpu().tolist())
                    loss_sum += result.loss.item() * len(batch["labels"])
            metrics = classification_metrics(actual, predicted)
            epoch_metrics[name] = {"loss": loss_sum / len(actual), **metrics}
        row = {"epoch": epoch, **epoch_metrics}
        history.append(row)
        # history is a run-local progress file; existing runs are never opened.
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch_metrics["validation"]["macro_f1"] > best:
            best = epoch_metrics["validation"]["macro_f1"]
            model.save_pretrained(output / "best_model", safe_serialization=True)
            tokenizer.save_pretrained(output / "best_model")
            (output / "best_validation.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"Epoch {epoch}/{args.epochs}: validation accuracy={epoch_metrics['validation']['accuracy']:.4f}, macro-F1={epoch_metrics['validation']['macro_f1']:.4f}", flush=True)
    print(f"Detector saved: {output / 'best_model'}", flush=True)
