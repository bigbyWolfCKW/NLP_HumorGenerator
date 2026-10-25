import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import sub_lstm as base


PROJECT = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT
    / "checkpoints/sub_lstm_multicat40k_fasttext_seed42/best.pt"
)

DATA_DIR = PROJECT / "data/processed/unbiased"

OUTPUT_DIR = (
    PROJECT
    / "data/features/sub_lstm_multicat40k_fasttext_seed42"
)

BATCH_SIZE = 256


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("請使用 sbatch 配置 GPU。")

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

    if config["seed"] != 42:
        raise ValueError("本輪預先固定使用 SUB seed 42。")

    if config["hidden_dim"] != 64:
        raise ValueError("預期 SUB 特徵為 64 維。")

    if config["tokenizer_pattern"] != base.TOKEN_PATTERN:
        raise ValueError("Tokenizer 與訓練設定不同。")

    if not config["lowercase"]:
        raise ValueError("目前 tokenizer 預期使用 lowercase。")

    base.MAX_LENGTH = config["max_length"]

    model = base.SubjectivityLSTM(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.to("cuda")
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frames = {}

    for split in ["train", "validation"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")

        required = {"question", "product_group", "label"}

        if not required.issubset(df.columns):
            raise ValueError(f"{split} 缺少必要欄位。")

        if df.empty:
            raise ValueError(f"{split} 沒有資料。")

        if df[list(required)].isna().any().any():
            raise ValueError(f"{split} 有缺失值。")

        frames[split] = df

    if (
        set(frames["train"]["product_group"])
        & set(frames["validation"]["product_group"])
    ):
        raise ValueError("Humor train / validation 商品重疊。")

    checkpoint_hash = sha256(CHECKPOINT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("SUB checkpoint:", CHECKPOINT, flush=True)
    print("SUB seed: fixed 42", flush=True)
    print("Checkpoint SHA256:", checkpoint_hash, flush=True)

    split_summary = {}

    for split, df in frames.items():
        csv_path = DATA_DIR / f"{split}.csv"

        # SUB 只讀 question。
        # make_dataset 產生的 humor labels 不參與推論。
        input_df = df.assign(text=df["question"])

        dataset = base.make_dataset(input_df, vocab)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
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

                unknown_tokens += (
                    (ids == config["unk_id"]) & valid
                ).sum().item()
                total_tokens += valid.sum().item()

                features = model.encode(
                    ids.to("cuda"), lengths
                )

                logits = model.classifier(
                    features
                ).squeeze(-1)

                feature_batches.append(features.cpu())
                score_batches.append(
                    logits.sigmoid().cpu()
                )

        features = torch.cat(feature_batches)
        scores = torch.cat(score_batches)

        if features.shape != (len(df), 64):
            raise ValueError(f"{split}: 特徵維度不正確。")

        if scores.shape != (len(df),):
            raise ValueError(f"{split}: 分數維度不正確。")

        if (
            not torch.isfinite(features).all()
            or not torch.isfinite(scores).all()
        ):
            raise ValueError(f"{split}: 出現非有限數值。")

        unknown_rate = (
            unknown_tokens / max(total_tokens, 1)
        )

        saved = {
            "features": features,
            "scores": scores,
            "source_csv_sha256": sha256(csv_path),
            "sub_checkpoint_sha256": checkpoint_hash,
            "config": config,
            "score_meaning": "review_proxy_probability",
            "unknown_rate": unknown_rate,
            "split": split,
            "rows": len(df),
        }

        destination = OUTPUT_DIR / f"{split}.pt"
        temporary = OUTPUT_DIR / f"{split}.pt.tmp"

        torch.save(saved, temporary)
        temporary.replace(destination)

        split_summary[split] = {
            "rows": len(df),
            "feature_shape": list(features.shape),
            "source_csv_sha256": saved["source_csv_sha256"],
            "unknown_rate": unknown_rate,
            "mean_review_proxy_score": scores.mean().item(),
        }

        print(
            f"{split}: "
            f"features={tuple(features.shape)} | "
            f"UNK={unknown_rate:.2%} | "
            f"mean review-proxy score="
            f"{scores.mean().item():.4f}",
            flush=True,
        )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(
            {
                "sub_checkpoint": str(CHECKPOINT),
                "sub_checkpoint_sha256": checkpoint_hash,
                "sub_seed": 42,
                "input": "question",
                "splits": split_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nNew SUB feature extraction completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()