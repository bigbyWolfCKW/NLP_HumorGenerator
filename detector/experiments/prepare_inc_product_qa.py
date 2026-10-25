import ast
import gzip
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT = Path(__file__).resolve().parent

RAW_DIR = PROJECT / "data/raw/inc_pqa"
OUTPUT = PROJECT / "data/processed/inc_pqa_200k"
HUMOR_DIR = PROJECT / "data/processed/unbiased"

CATEGORIES = [
    "Beauty",
    "Electronics",
    "Home_and_Kitchen",
    "Toys_and_Games",
    "Sports_and_Outdoors",
]

TRAIN_POSITIVE_PER_CATEGORY = 20_000
VAL_POSITIVE_PER_CATEGORY = 5_000
SEED = 42


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def clean_text(value):
    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(value):
    return clean_text(value).casefold()


def read_records(path):
    # 舊 Amazon 檔案部分使用 Python dictionary 格式。
    # 使用 literal_eval，不使用 eval。
    with gzip.open(
        path, "rt", encoding="utf-8"
    ) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                try:
                    record = ast.literal_eval(line)
                except (ValueError, SyntaxError) as error:
                    raise ValueError(
                        f"無法解析 {path.name} 第 {number} 行"
                    ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name} 第 {number} 行不是 dictionary"
                )

            yield record


def load_exclusions():
    questions = set()
    products = set()
    hashes = {}

    # 只讀文字作去重排除，不使用 humor labels。
    for split in ["train", "validation", "test"]:
        path = HUMOR_DIR / f"{split}.csv"

        df = pd.read_csv(
            path,
            usecols=["question", "product_description"],
            keep_default_na=False,
        )

        questions.update(
            normalize(text) for text in df["question"]
        )
        products.update(
            normalize(text)
            for text in df["product_description"]
        )
        hashes[split] = sha256(path)

    questions.discard("")
    products.discard("")

    return questions, products, hashes


def make_pairs(source, seed):
    source = source.reset_index(drop=True)

    positive = pd.DataFrame({
        "question": source["question"],
        "product_description": source["product_description"],
        "question_asin": source["asin"],
        "product_asin": source["asin"],
        "question_group": source["product_group"],
        "product_group": source["product_group"],
        "category": source["category"],
        "pair_label": 1,
    })

    negative_parts = []
    rng = np.random.default_rng(seed)

    for category in CATEGORIES:
        subset = source.loc[
            source["category"] == category
        ].reset_index(drop=True)

        if subset["product_group"].nunique() < 2:
            raise ValueError(
                f"{category} 沒有足夠不同商品建立錯配。"
            )

        groups = subset["product_group"].to_numpy()
        indices = rng.integers(
            0, len(subset), size=len(subset)
        )

        same_group = groups[indices] == groups

        while same_group.any():
            indices[same_group] = rng.integers(
                0,
                len(subset),
                size=int(same_group.sum()),
            )
            same_group = groups[indices] == groups

        sampled = subset.iloc[indices]

        negative_parts.append(pd.DataFrame({
            "question": subset["question"].to_numpy(),
            "product_description": (
                sampled["product_description"].to_numpy()
            ),
            "question_asin": subset["asin"].to_numpy(),
            "product_asin": sampled["asin"].to_numpy(),
            "question_group": (
                subset["product_group"].to_numpy()
            ),
            "product_group": (
                sampled["product_group"].to_numpy()
            ),
            "category": category,
            "pair_label": 0,
        }))

    negative = pd.concat(
        negative_parts, ignore_index=True
    )

    if not (
        negative["question_group"]
        != negative["product_group"]
    ).all():
        raise ValueError("錯配中仍有相同商品 group。")

    if not (
        negative["question_asin"]
        != negative["product_asin"]
    ).all():
        raise ValueError("錯配中仍有相同 ASIN。")

    return pd.concat(
        [positive, negative], ignore_index=True
    ).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def main():
    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError(
            "完整資料集已存在，停止以避免覆蓋。"
        )

    paths = {}

    for category in CATEGORIES:
        for kind in ["qa", "meta"]:
            path = RAW_DIR / f"{kind}_{category}.json.gz"

            if not path.is_file():
                raise FileNotFoundError(path)

            paths[f"{kind}_{category}"] = path

    excluded_q, excluded_p, humor_hashes = (
        load_exclusions()
    )

    print(
        "Humor exclusion questions:",
        len(excluded_q),
        flush=True,
    )
    print(
        "Humor exclusion product texts:",
        len(excluded_p),
        flush=True,
    )

    # 同一 ASIN 在不同類別出現時，使用同一商品標題。
    titles = {}
    blocked_asins = set()
    raw_hashes = {}

    for category in CATEGORIES:
        path = paths[f"meta_{category}"]
        print(f"\nLoading metadata: {category}", flush=True)

        for record in read_records(path):
            asin = str(record.get("asin") or "").strip()
            title = clean_text(record.get("title"))

            if not asin or not title:
                continue

            if normalize(title) in excluded_p:
                blocked_asins.add(asin)

            titles.setdefault(asin, title)

        raw_hashes[path.name] = sha256(path)

        print(
            "Total indexed products:",
            len(titles),
            flush=True,
        )

    rows = []

    for category in CATEGORIES:
        path = paths[f"qa_{category}"]
        before = len(rows)

        print(f"\nLoading questions: {category}", flush=True)

        for record in read_records(path):
            asin = str(record.get("asin") or "").strip()

            if asin not in titles or asin in blocked_asins:
                continue

            question = clean_text(record.get("question"))
            question_key = normalize(question)

            if not question_key or question_key in excluded_q:
                continue

            title = titles[asin]
            title_key = normalize(title)

            if title_key in excluded_p:
                continue

            rows.append({
                "asin": asin,
                "question": question,
                "question_key": question_key,
                # 沿用既有欄位名；內容實際是 product title。
                "product_description": title,
                "product_group": title_key,
                "category": category,
            })

        raw_hashes[path.name] = sha256(path)

        print(
            "Eligible source rows:",
            len(rows) - before,
            flush=True,
        )

    source = pd.DataFrame(rows)

    if source.empty:
        raise ValueError("沒有匹配到合資格的 Q–Title 資料。")

    # 排除出現在多個不同商品上的相同問題，
    # 避免把已知也匹配另一商品的問題製成負例。
    group_counts = source.groupby(
        "question_key"
    )["product_group"].transform("nunique")

    ambiguous_rows = int((group_counts > 1).sum())

    source = source.loc[
        group_counts == 1
    ].drop_duplicates(
        "question_key"
    ).reset_index(drop=True)

    print(
        "\nRemoved ambiguous-question rows:",
        ambiguous_rows,
        flush=True,
    )
    print(
        "Deduplicated candidates by category:\n",
        source["category"].value_counts().to_string(),
        flush=True,
    )

    # 商品 group 是正規化標題。
    # 同 ASIN 使用一致標題，同標題的不同 ASIN 也不跨 split。
    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=0.2,
        random_state=SEED,
    )

    selected_train = None
    selected_val = None
    chosen_split = None

    for attempt, (train_indices, val_indices) in enumerate(
        splitter.split(
            source,
            groups=source["product_group"],
        ),
        start=1,
    ):
        train_pool = source.iloc[train_indices]
        val_pool = source.iloc[val_indices]

        train_counts = train_pool["category"].value_counts()
        val_counts = val_pool["category"].value_counts()

        enough = all(
            train_counts.get(category, 0)
            >= TRAIN_POSITIVE_PER_CATEGORY
            and val_counts.get(category, 0)
            >= VAL_POSITIVE_PER_CATEGORY
            for category in CATEGORIES
        )

        if not enough:
            continue

        selected_train = pd.concat([
            train_pool.loc[
                train_pool["category"] == category
            ].sample(
                n=TRAIN_POSITIVE_PER_CATEGORY,
                random_state=SEED + index,
            )
            for index, category in enumerate(CATEGORIES)
        ], ignore_index=True)

        selected_val = pd.concat([
            val_pool.loc[
                val_pool["category"] == category
            ].sample(
                n=VAL_POSITIVE_PER_CATEGORY,
                random_state=SEED + 100 + index,
            )
            for index, category in enumerate(CATEGORIES)
        ], ignore_index=True)

        chosen_split = attempt
        break

    if selected_train is None:
        raise RuntimeError(
            "去重與排除後，無法在商品分離下滿足各類配額。"
            "請貼出以上類別筆數，不要重複抽樣補足。"
        )

    if (
        set(selected_train["product_group"])
        & set(selected_val["product_group"])
    ):
        raise ValueError("Train / validation 商品 group 重疊。")

    if (
        set(selected_train["asin"])
        & set(selected_val["asin"])
    ):
        raise ValueError("Train / validation ASIN 重疊。")

    if (
        set(selected_train["question_key"])
        & set(selected_val["question_key"])
    ):
        raise ValueError("Train / validation 問題重疊。")

    train_pairs = make_pairs(
        selected_train, seed=SEED
    )
    val_pairs = make_pairs(
        selected_val, seed=SEED + 1
    )

    if len(train_pairs) != 200_000:
        raise ValueError("Train 配對數不正確。")

    if len(val_pairs) != 50_000:
        raise ValueError("Validation 配對數不正確。")

    train_groups = (
        set(train_pairs["question_group"])
        | set(train_pairs["product_group"])
    )
    val_groups = (
        set(val_pairs["question_group"])
        | set(val_pairs["product_group"])
    )

    if train_groups & val_groups:
        raise ValueError("配對後仍存在跨 split 商品重疊。")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    selected_train.to_csv(
        OUTPUT / "train_source.csv", index=False
    )
    selected_val.to_csv(
        OUTPUT / "validation_source.csv", index=False
    )
    train_pairs.to_csv(
        OUTPUT / "train_pairs.csv", index=False
    )
    val_pairs.to_csv(
        OUTPUT / "validation_pairs.csv", index=False
    )

    manifest = {
        "dataset": "Amazon Q&A and legacy product metadata",
        "categories": CATEGORIES,
        "product_text": "title",
        "labels": {
            "1": "original_product",
            "0": "mismatched_product",
        },
        "seed": SEED,
        "group_split_attempt": chosen_split,
        "split_group": "normalized_product_title",
        "negative_sampling": "within split and category",
        "train_positive_pairs": 100_000,
        "train_negative_pairs": 100_000,
        "validation_positive_pairs": 25_000,
        "validation_negative_pairs": 25_000,
        "humor_overlap_exclusion": (
            "normalized question and product text; "
            "labels not used"
        ),
        "humor_source_sha256": humor_hashes,
        "raw_source_sha256": raw_hashes,
        "train_pairs_sha256": sha256(
            OUTPUT / "train_pairs.csv"
        ),
        "validation_pairs_sha256": sha256(
            OUTPUT / "validation_pairs.csv"
        ),
    }

    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print(
        "\nTrain labels:",
        train_pairs["pair_label"].value_counts().to_dict(),
        flush=True,
    )
    print(
        "Validation labels:",
        val_pairs["pair_label"].value_counts().to_dict(),
        flush=True,
    )
    print("Saved:", OUTPUT, flush=True)
    print("INC PQA dataset completed.", flush=True)


if __name__ == "__main__":
    main()