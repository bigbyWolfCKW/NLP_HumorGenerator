import hashlib
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent

VECTOR_FILE = (
    PROJECT / "data/external/fasttext/wiki.en.vec"
)

BASELINE_CHECKPOINT = (
    PROJECT
    / "checkpoints/sub_lstm_multicat40k_seed42/best.pt"
)

TRAIN_CSV = (
    PROJECT
    / "data/processed/subjectivity_multicat_40k/train.csv"
)

OUTPUT = (
    PROJECT
    / "data/external/fasttext/sub_multicat40k_vectors.pt"
)

DIMENSION = 300
FALLBACK_SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)

    if OUTPUT.exists():
        raise FileExistsError(
            f"快取已存在，停止以避免覆蓋：{OUTPUT}"
        )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    train_hash = sha256(TRAIN_CSV)

    if config["train_csv_sha256"] != train_hash:
        raise ValueError(
            "目前 training CSV 與 0961 checkpoint 不一致。"
        )

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("詞彙 ID 必須連續。")

    if vocab["<pad>"] != 0 or vocab["<unk>"] != 1:
        raise ValueError("PAD / UNK ID 與預期不同。")

    # 缺少 FastText 向量的詞使用固定隨機初始化。
    generator = torch.Generator().manual_seed(FALLBACK_SEED)

    vectors = torch.randn(
        len(vocab),
        DIMENSION,
        generator=generator,
    ) * 0.02

    vectors[vocab["<pad>"]].zero_()

    targets = {
        word.encode("utf-8"): index
        for word, index in vocab.items()
        if word not in {"<pad>", "<unk>"}
    }

    found_ids = set()
    digest = hashlib.sha256()

    print("Vector file:", VECTOR_FILE, flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Target words:", len(targets), flush=True)

    with VECTOR_FILE.open("rb") as handle:
        header = handle.readline()
        digest.update(header)

        total_words, dimension = map(
            int, header.split()
        )

        if dimension != DIMENSION:
            raise ValueError("FastText 維度不是 300。")

        print(
            f"Source: {total_words:,} words, "
            f"{dimension} dimensions",
            flush=True,
        )

        scanned = 0

        for raw_line in handle:
            digest.update(raw_line)

            if not raw_line.strip():
                continue

            scanned += 1

            word, separator, values = (
                raw_line.rstrip(b"\r\n").partition(b" ")
            )

            index = targets.get(word)

            if index is not None and index not in found_ids:
                if not separator:
                    raise ValueError(
                        f"向量格式不正確，row {scanned}"
                    )

                vector = np.fromstring(
                    values.decode("ascii"),
                    sep=" ",
                    dtype=np.float32,
                )

                if vector.shape != (DIMENSION,):
                    raise ValueError(
                        f"向量維度不正確，row {scanned}"
                    )

                if not np.isfinite(vector).all():
                    raise ValueError(
                        f"非有限向量，row {scanned}"
                    )

                vectors[index] = torch.from_numpy(vector)
                found_ids.add(index)

            if scanned % 250_000 == 0:
                print(
                    f"Scanned {scanned:,}/{total_words:,} | "
                    f"matched {len(found_ids):,}/"
                    f"{len(targets):,}",
                    flush=True,
                )

    if scanned != total_words:
        raise ValueError(
            f"檔案列數不完整："
            f"預期 {total_words:,}，實際 {scanned:,}"
        )

    if not found_ids:
        raise ValueError("沒有匹配到任何 FastText 向量。")

    if not torch.isfinite(vectors).all():
        raise ValueError("Embedding matrix 出現非有限數值。")

    missing_words = [
        word
        for word, index in vocab.items()
        if word not in {"<pad>", "<unk>"}
        and index not in found_ids
    ]

    coverage = len(found_ids) / len(targets)

    saved = {
        "vocab": vocab,
        "embedding_matrix": vectors,
        "pretrained_mask": torch.tensor(
            [
                index in found_ids
                for index in range(len(vocab))
            ],
            dtype=torch.bool,
        ),
        "config": {
            "embedding_dim": DIMENSION,
            "source": "fastText Wikipedia English",
            "source_url": (
                "https://dl.fbaipublicfiles.com/"
                "fasttext/vectors-wiki/wiki.en.vec"
            ),
            "source_vec_sha256": digest.hexdigest(),
            "baseline_checkpoint_sha256": sha256(
                BASELINE_CHECKPOINT
            ),
            "train_csv_sha256": train_hash,
            "tokenizer_pattern": config["tokenizer_pattern"],
            "lowercase": config["lowercase"],
            "pad_id": 0,
            "unk_id": 1,
            "matched_words": len(found_ids),
            "target_words": len(targets),
            "word_type_coverage": coverage,
            "fallback_initialization": "normal_std_0.02",
            "fallback_seed": FALLBACK_SEED,
        },
        "missing_words": missing_words,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    temporary = OUTPUT.with_suffix(".pt.tmp")
    torch.save(saved, temporary)
    temporary.replace(OUTPUT)

    print(
        "\nEmbedding shape:",
        tuple(vectors.shape),
        flush=True,
    )
    print(
        f"Pretrained word coverage: {coverage:.2%}",
        flush=True,
    )
    print(
        "Fallback words:", len(missing_words),
        flush=True,
    )
    print(
        "Missing examples:", missing_words[:30],
        flush=True,
    )
    print("Saved:", OUTPUT, flush=True)
    print("FastText cache completed.", flush=True)


if __name__ == "__main__":
    main()