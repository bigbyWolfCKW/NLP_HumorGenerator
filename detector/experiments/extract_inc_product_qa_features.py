import hashlib
import re
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT / "checkpoints/inc_siamese_pqa200k_seed42/best.pt"
DATA_DIR = PROJECT / "data/processed/unbiased"
OUTPUT_DIR = PROJECT / "data/features/inc_siamese_pqa200k_seed42"

BATCH_SIZE = 256


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IncongruityLSTM(nn.Module):
    def __init__(self, vectors, config):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            vectors,
            freeze=True,
            padding_idx=config["pad_id"],
        )
        self.encoder = nn.LSTM(
            config["embedding_dim"],
            config["hidden_dim"],
            batch_first=True,
        )
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(config["hidden_dim"], 1)

    def encode(self, ids, lengths):
        packed = pack_padded_sequence(
            self.embedding(ids),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        return hidden[-1]

    def forward(self, q_ids, q_lengths, p_ids, p_lengths):
        q_rep = self.encode(q_ids, q_lengths)
        p_rep = self.encode(p_ids, p_lengths)
        difference = torch.abs(q_rep - p_rep)

        return self.classifier(
            self.dropout(difference)
        ).squeeze(-1)


def encode_texts(texts, vocab, config):
    sequences = []
    lengths = []

    pattern = re.compile(config["tokenizer_pattern"])
    max_length = config["max_length"]
    pad_id = config["pad_id"]
    unk_id = config["unk_id"]

    for text in texts:
        text = str(text)

        if config["lowercase"]:
            text = text.lower()

        tokens = pattern.findall(text)[:max_length]
        ids = [vocab.get(token, unk_id) for token in tokens]

        if not ids:
            ids = [unk_id]

        lengths.append(len(ids))
        sequences.append(
            ids + [pad_id] * (max_length - len(ids))
        )

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請透過 sbatch 配置 GPU 執行。")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"輸出目錄已存在，停止以避免覆蓋：{OUTPUT_DIR}"
        )

    torch.set_num_threads(4)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    state = checkpoint["model_state_dict"]

    if config["model_type"] != "siamese_lstm_absolute_difference":
        raise ValueError("不支援的 INC 模型類型。")

    if config["pair_label_meaning"].get("1") != "original_product":
        raise ValueError("INC 分數方向與預期不一致。")

    vectors = state["embedding.weight"]
    model = IncongruityLSTM(vectors, config)
    model.load_state_dict(state, strict=True)
    model.to("cuda")
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frames = {}
    for split in ["train", "validation"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")

        required = {
            "question",
            "product_description",
            "product_group",
        }
        if not required.issubset(df.columns):
            raise ValueError(f"{split} 缺少必要欄位。")

        if df.empty:
            raise ValueError(f"{split} 沒有資料。")

        if df[["question", "product_group"]].isna().any().any():
            raise ValueError(f"{split} 有缺失的 question/product_group。")

        frames[split] = df

    overlap = (
        set(frames["train"]["product_group"])
        & set(frames["validation"]["product_group"])
    )
    if overlap:
        raise ValueError("Train / validation 存在 product overlap。")

    checkpoint_hash = sha256(CHECKPOINT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("INC checkpoint SHA256:", checkpoint_hash, flush=True)
    print("Score: P(original_product | question, product)", flush=True)

    for split, df in frames.items():
        csv_path = DATA_DIR / f"{split}.csv"

        q_ids, q_lengths = encode_texts(
            df["question"].tolist(), vocab, config
        )
        p_ids, p_lengths = encode_texts(
            df["product_description"].fillna("").tolist(),
            vocab,
            config,
        )

        loader = DataLoader(
            TensorDataset(
                q_ids, q_lengths, p_ids, p_lengths
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        batches = []

        with torch.inference_mode():
            for q, q_len, p, p_len in loader:
                logits = model(
                    q.to("cuda"),
                    q_len,
                    p.to("cuda"),
                    p_len,
                )
                batches.append(logits.sigmoid().cpu())

        scores = torch.cat(batches)

        if scores.shape != (len(df),):
            raise ValueError("INC 分數數量與 CSV 列數不一致。")

        if not torch.isfinite(scores).all():
            raise ValueError("INC 分數出現非有限數值。")

        torch.save(
            {
                "features": scores.unsqueeze(1),
                "scores": scores,
                "source_csv_sha256": sha256(csv_path),
                "inc_checkpoint_sha256": checkpoint_hash,
                "config": config,
                "score_meaning": "probability_of_original_product",
                "split": split,
                "rows": len(df),
            },
            OUTPUT_DIR / f"{split}.pt",
        )

        print(
            f"{split}: "
            f"features={tuple(scores.unsqueeze(1).shape)} | "
            f"min={scores.min().item():.4f} | "
            f"mean={scores.mean().item():.4f} | "
            f"max={scores.max().item():.4f}",
            flush=True,
        )

    print("INC feature extraction completed.", flush=True)


if __name__ == "__main__":
    main()