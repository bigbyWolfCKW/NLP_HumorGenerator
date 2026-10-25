import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 沿用舊 INC 的 tokenize 規則。
import re


PROJECT = Path(__file__).resolve().parent

DATA_DIR = PROJECT / "data/processed/inc_pqa_200k"
TRAIN_SOURCE = DATA_DIR / "train_source.csv"

VECTOR_FILE = (
    PROJECT / "data/external/fasttext/wiki.en.vec"
)
OUTPUT = (
    PROJECT / "data/external/fasttext/inc_pqa_vectors.pt"
)

TOKEN_PATTERN = r"\w+|[^\w\s]"
MAX_VOCAB_WORDS = 10_000
DIMENSION = 300
SEED = 42


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def tokenize(text):
    return re.findall(
        TOKEN_PATTERN,
        str(text).lower(),
    )


def main():
    torch.set_num_threads(4)

    if OUTPUT.exists():
        raise FileExistsError(
            f"快取已存在，停止以避免覆蓋：{OUTPUT}"
        )

    manifest_path = DATA_DIR / "manifest.json"

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    train = pd.read_csv(
        TRAIN_SOURCE,
        usecols=["question", "product_description"],
        dtype=str,
        keep_default_na=False,
    )

    if len(train) != manifest["train_positive_pairs"]:
        raise ValueError("Training source 筆數與 manifest 不一致。")

    empty = (
        train["question"].str.strip().eq("")
        | train["product_description"].str.strip().eq("")
    )

    if empty.any():
        print(train.loc[empty].head(10).to_string())
        raise ValueError(
            f"Training source 有 {int(empty.sum())} 筆真正空白文字。"
        )

    # 計算原始問題及標題，不重複計算人工負配對。
    counts = Counter()

    for column in ["question", "product_description"]:
        for text in train[column]:
            counts.update(tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in counts.most_common(MAX_VOCAB_WORDS):
        if word not in vocab:
            vocab[word] = len(vocab)

    generator = torch.Generator().manual_seed(SEED)

    vectors = torch.randn(
        len(vocab),
        DIMENSION,
        generator=generator,
    ) * 0.02

    vectors[0].zero_()

    targets = {
        word.encode("utf-8"): index
        for word, index in vocab.items()
        if index >= 2
    }

    found_ids = set()
    digest = hashlib.sha256()

    print("Training source rows:", len(train), flush=True)
    print("Vocabulary size:", len(vocab), flush=True)
    print("Target words:", len(targets), flush=True)
    print("Reading:", VECTOR_FILE, flush=True)

    with VECTOR_FILE.open("rb") as handle:
        header = handle.readline()
        digest.update(header)

        total_words, dimension = map(
            int, header.split()
        )

        if dimension != DIMENSION:
            raise ValueError("FastText 維度不是 300。")

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
            f"向量檔筆數不完整：{scanned} / {total_words}"
        )

    if not found_ids:
        raise ValueError("沒有匹配到任何預訓練向量。")

    if not torch.isfinite(vectors).all():
        raise ValueError("Embedding matrix 有非有限數值。")

    missing_words = [
        word
        for word, index in vocab.items()
        if index >= 2 and index not in found_ids
    ]

    type_coverage = len(found_ids) / len(targets)

    total_tokens = sum(counts.values())

    vocab_tokens = sum(
        count
        for word, count in counts.items()
        if vocab.get(word, 1) >= 2
    )

    pretrained_tokens = sum(
        count
        for word, count in counts.items()
        if vocab.get(word, 1) in found_ids
    )

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
        "missing_words": missing_words,
        "config": {
            "embedding_dim": DIMENSION,
            "source": "fastText Wikipedia English",
            "source_vec_sha256": digest.hexdigest(),
            "train_source_sha256": sha256(TRAIN_SOURCE),
            "dataset_manifest_sha256": sha256(manifest_path),
            "train_source_rows": len(train),
            "tokenizer_pattern": TOKEN_PATTERN,
            "lowercase": True,
            "pad_id": 0,
            "unk_id": 1,
            "max_vocab_words": MAX_VOCAB_WORDS,
            "matched_words": len(found_ids),
            "target_words": len(targets),
            "word_type_coverage": type_coverage,
            "vocab_training_token_coverage": (
                vocab_tokens / max(total_tokens, 1)
            ),
            "pretrained_training_token_coverage": (
                pretrained_tokens / max(total_tokens, 1)
            ),
            "fallback_initialization": "normal_std_0.02",
            "fallback_seed": SEED,
        },
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
        f"Pretrained word coverage: {type_coverage:.2%}",
        flush=True,
    )
    print(
        "Vocabulary training-token coverage: "
        f"{vocab_tokens / max(total_tokens, 1):.2%}",
        flush=True,
    )
    print(
        "Pretrained training-token coverage: "
        f"{pretrained_tokens / max(total_tokens, 1):.2%}",
        flush=True,
    )
    print("Fallback words:", len(missing_words), flush=True)
    print("Missing examples:", missing_words[:30], flush=True)
    print("Saved:", OUTPUT, flush=True)
    print("INC FastText cache completed.", flush=True)


if __name__ == "__main__":
    main()