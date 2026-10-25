import importlib
from pathlib import Path

from src.config import preflight
from src.data import read_json


def create_detector(config, device="cuda", batch_size=16):
    config = preflight(dict(config))
    if config["backend"] == "custom":
        module, function = config["factory"].split(":", 1)
        return getattr(importlib.import_module(module), function)(config, device=device, batch_size=batch_size)
    return HumorDetector(config, device, batch_size)


class HumorDetector:
    def __init__(self, config, device, batch_size):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.config, self.device, self.batch_size = torch, config, device, batch_size
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Submit a GPU job or use --device cpu.")
        folder = Path(config["checkpoint"])
        self.sub = None
        if config["backend"] == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True)
            if self.model.config.num_labels != 2:
                raise ValueError("A binary humor classifier is required.")
        else:
            from .models.roberta_sub import ContextSUBModel, SubjectivityLSTM
            saved = read_json(folder / "config.json")
            checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
            if checkpoint["config"] != saved:
                raise ValueError("Detector checkpoint/config mismatch.")
            self.model = ContextSUBModel(config["backbone"], saved["use_sub"])
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
            if saved["use_sub"]:
                sub = torch.load(config["sub_checkpoint"], map_location="cpu", weights_only=True)
                self.vocab, self.sub_config = sub["vocab"], sub["config"]
                if self.sub_config["hidden_dim"] != 64:
                    raise ValueError("This fusion model requires 64-dimensional SUB features.")
                self.sub = SubjectivityLSTM(len(self.vocab), self.sub_config["embedding_dim"], 64)
                # Different multiclass or multilayer architectures need a custom backend.
                self.sub.load_state_dict(sub["model_state_dict"], strict=True)
                self.sub.to(device).eval()
        self.model.to(device).eval()

    def score(self, rows):
        torch = self.torch
        scores = []
        with torch.inference_mode():
            for start in range(0, len(rows), self.batch_size):
                batch = rows[start:start + self.batch_size]
                texts = [r["question"] for r in batch]
                args = [texts]
                if self.config.get("input_mode", "pair") == "pair":
                    args.append([r["product_description"] for r in batch])
                encoded = self.tokenizer(*args, padding="max_length", truncation=True,
                    max_length=self.config["max_length"], return_tensors="pt").to(self.device)
                if self.config["backend"] == "hf":
                    logits = self.model(**encoded).logits
                else:
                    features = torch.zeros(len(batch), 64, device=self.device)
                    if self.sub is not None:
                        from .models.roberta_sub import sub_inputs
                        ids, lengths = sub_inputs(texts, self.vocab, self.sub_config)
                        features = self.sub.encode(ids.to(self.device), lengths)
                    logits = self.model(encoded["input_ids"], encoded["attention_mask"], features)
                scores.extend(logits.softmax(dim=1)[:, self.config["positive_label"]].cpu().tolist())
        return scores
