import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sub_lstm import (
    SubjectivityLSTM,
    make_dataset,
    MAX_LENGTH,
)

PROJECT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT / "checkpoints/sub_lstm_seed42/best.pt"
OUTPUT_DIR = PROJECT / "data/features/sub_lstm_seed42"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Run through sbatch with a GPU.")
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Features already exist: {OUTPUT_DIR}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]

    assert MAX_LENGTH == config["max_length"]

    model = SubjectivityLSTM(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cuda")
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(CHECKPOINT)

    for split in ["train", "validation"]:
        csv_path = (
            PROJECT / "data/processed/unbiased" / f"{split}.csv"
        )
        df = pd.read_csv(csv_path)
        assert df["question"].notna().all()

        # 沿用 SUB 的 tokenizer／vocab。
        # make_dataset 的 labels 在這一步不參與運算。
        input_df = df.assign(text=df["question"])
        dataset = make_dataset(input_df, vocab)
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )

        feature_batches = []
        score_batches = []
        unknown_tokens = 0
        total_tokens = 0

        with torch.inference_mode():
            for ids, lengths, _ in loader:
                valid = (
                    torch.arange(ids.size(1)).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                unknown_tokens += ((ids == 1) & valid).sum().item()
                total_tokens += valid.sum().item()

                features = model.encode(ids.to("cuda"), lengths)
                logits = model.classifier(features).squeeze(-1)

                feature_batches.append(features.cpu())
                score_batches.append(logits.sigmoid().cpu())

        features = torch.cat(feature_batches)
        scores = torch.cat(score_batches)
        unknown_rate = unknown_tokens / max(total_tokens, 1)

        assert features.shape == (
            len(df), config["hidden_dim"]
        )
        assert torch.isfinite(features).all()
        assert torch.isfinite(scores).all()

        # 保存檔案雜湊，之後檢查 CSV 與特徵的列順序一致。
        torch.save({
            "features": features,
            "review_proxy_scores": scores,
            "source_csv_sha256": sha256(csv_path),
            "sub_checkpoint_sha256": checkpoint_hash,
            "row_count": len(df),
            "unknown_token_rate": unknown_rate,
        }, OUTPUT_DIR / f"{split}.pt")

        print(
            f"{split} | Features: {tuple(features.shape)} | "
            f"Unknown tokens: {unknown_rate:.2%} | "
            f"Mean review-proxy score: {scores.mean().item():.4f}",
            flush=True,
        )

    print("SUB feature extraction completed.", flush=True)
    print("Saved:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()