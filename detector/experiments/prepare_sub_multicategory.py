import csv
import gzip
import html
import json
import random
import re
from contextlib import closing
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parent

OUTPUT = (
    PROJECT / "data/processed/subjectivity_multicat_40k"
)
OLD_DATA = PROJECT / "data/processed/subjectivity"

CATEGORIES = [
    "All_Beauty",
    "Electronics",
    "Home_and_Kitchen",
    "Toys_and_Games",
    "Sports_and_Outdoors",
]

PRODUCTS_PER_CATEGORY = 4000
VALIDATION_PRODUCTS_PER_CATEGORY = 800
MINIMUM_WORDS = 5
SEED = 42

# 限制掃描量，避免整個大型類別全部下載。
MAX_METADATA_ROWS = 200_000
MAX_REVIEW_ROWS = 500_000

BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/"
    "data/amazon_2023/raw"
)

SETTINGS = {
    "products_per_category": PRODUCTS_PER_CATEGORY,
    "minimum_words": MINIMUM_WORDS,
    "max_metadata_rows": MAX_METADATA_ROWS,
    "max_review_rows": MAX_REVIEW_ROWS,
    "seed": SEED,
    "sampling": (
        "random matched-product sample within bounded prefixes; "
        "first eligible review per product"
    ),
}


def clean_text(value):
    if isinstance(value, list):
        value = " ".join(
            str(item) for item in value if item is not None
        )

    if value is None:
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def eligible(text):
    return len(text.split()) >= MINIMUM_WORDS


def stream_records(url, limit):
    print("Opening:", url, flush=True)

    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urlopen(request, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            for index in range(1, limit + 1):
                line = compressed.readline()

                if not line:
                    break

                record = json.loads(line.decode("utf-8"))

                if index % 25_000 == 0:
                    print(
                        f"  scanned {index:,} rows",
                        flush=True,
                    )

                yield record


def read_old_groups():
    groups = {}

    for split in ["train", "validation"]:
        path = OLD_DATA / f"{split}.csv"

        with path.open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows = csv.DictReader(handle)

            for row in rows:
                product = row["parent_asin"]

                previous = groups.get(product)
                if previous is not None and previous != split:
                    raise ValueError(
                        "舊 SUB 資料存在跨 split 商品。"
                    )

                groups[product] = split

    return groups


def load_category(category, index, used_products, used_texts):
    cache_path = OUTPUT / f"{category}_pairs.json"

    if cache_path.exists():
        saved = json.loads(
            cache_path.read_text(encoding="utf-8")
        )

        if saved["settings"] != SETTINGS:
            raise ValueError(
                f"{category}: cache 設定不同，不能沿用。"
            )

        pairs = saved["pairs"]

        if len(pairs) != PRODUCTS_PER_CATEGORY:
            raise ValueError(
                f"{category}: cache 商品數不正確。"
            )

        print(
            f"\nReuse {category}: {len(pairs):,} products",
            flush=True,
        )
        return pairs, saved["scan"]

    print(f"\n=== {category} ===", flush=True)

    metadata_url = (
        f"{BASE_URL}/meta_categories/"
        f"meta_{category}.jsonl.gz"
    )
    review_url = (
        f"{BASE_URL}/review_categories/"
        f"{category}.jsonl.gz"
    )

    descriptions = {}
    metadata_scanned = 0

    with closing(
        stream_records(metadata_url, MAX_METADATA_ROWS)
    ) as records:
        for record in records:
            metadata_scanned += 1

            product = record.get("parent_asin")
            text = clean_text(record.get("description"))

            if (
                product
                and product not in used_products
                and product not in descriptions
                and text not in used_texts
                and eligible(text)
            ):
                descriptions[product] = text

    print(
        f"Eligible descriptions: {len(descriptions):,}",
        flush=True,
    )

    matches = {}
    reviews_scanned = 0

    with closing(
        stream_records(review_url, MAX_REVIEW_ROWS)
    ) as records:
        for record in records:
            reviews_scanned += 1
            product = record.get("parent_asin")

            if product not in descriptions or product in matches:
                continue

            review = clean_text(record.get("text"))
            description = descriptions[product]

            if (
                eligible(review)
                and review != description
                and review not in used_texts
            ):
                matches[product] = {
                    "parent_asin": product,
                    "description": description,
                    "review": review,
                    "category": category,
                }

    # 對掃描範圍內的匹配商品打亂，再選取。
    candidates = list(matches.values())
    random.Random(SEED + index).shuffle(candidates)

    selected = []
    selected_texts = set(used_texts)

    for pair in candidates:
        description = pair["description"]
        review = pair["review"]

        # 排除不同商品間完全相同的已選文字。
        if (
            description in selected_texts
            or review in selected_texts
        ):
            continue

        selected.append(pair)
        selected_texts.update([description, review])

        if len(selected) == PRODUCTS_PER_CATEGORY:
            break

    print(
        f"Matched products: {len(matches):,}; "
        f"selected: {len(selected):,}",
        flush=True,
    )

    if len(selected) != PRODUCTS_PER_CATEGORY:
        raise RuntimeError(
            f"{category} 目前只找到 {len(selected):,} 個"
            "符合條件且文字不重複的商品。"
            "停止，不會以不足樣本冒充完成。"
            "請把 log 貼出，再調整掃描範圍。"
        )

    scan = {
        "metadata_url": metadata_url,
        "review_url": review_url,
        "metadata_scanned": metadata_scanned,
        "reviews_scanned": reviews_scanned,
        "matched_products": len(matches),
    }

    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "settings": SETTINGS,
                "scan": scan,
                "pairs": selected,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(cache_path)

    return selected, scan


def write_csv(path, rows):
    columns = [
        "parent_asin", "text", "label",
        "source", "category",
    ]

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if (OUTPUT / "manifest.json").exists():
        raise FileExistsError(
            "完整資料集已存在，停止以避免覆蓋。"
        )

    old_groups = read_old_groups()

    used_products = set()
    used_texts = set()
    split_rows = {"train": [], "validation": []}
    scans = {}

    for index, category in enumerate(CATEGORIES):
        pairs, scan = load_category(
            category, index,
            used_products, used_texts,
        )
        scans[category] = scan

        # 同時驗證新抽取資料和重新載入的 cache。
        for pair in pairs:
            product = pair["parent_asin"]
            description = pair["description"]
            review = pair["review"]

            if (
                product in used_products
                or description in used_texts
                or review in used_texts
                or description == review
            ):
                raise ValueError(
                    f"{category}: 商品或文字重複。"
                )

            used_products.add(product)
            used_texts.update([description, review])

        # 選到舊商品時，保持舊 split 歸屬。
        fixed_train = [
            pair for pair in pairs
            if old_groups.get(pair["parent_asin"]) == "train"
        ]
        fixed_val = [
            pair for pair in pairs
            if old_groups.get(pair["parent_asin"]) == "validation"
        ]
        new_pairs = [
            pair for pair in pairs
            if pair["parent_asin"] not in old_groups
        ]

        random.Random(SEED + index).shuffle(new_pairs)

        needed_val = (
            VALIDATION_PRODUCTS_PER_CATEGORY
            - len(fixed_val)
        )

        if not 0 <= needed_val <= len(new_pairs):
            raise ValueError(
                "舊 split 歸屬與新 validation 配額衝突。"
            )

        category_splits = {
            "validation": fixed_val + new_pairs[:needed_val],
            "train": fixed_train + new_pairs[needed_val:],
        }

        for split, selected in category_splits.items():
            for pair in selected:
                for label, source in [
                    (0, "description"),
                    (1, "review"),
                ]:
                    split_rows[split].append({
                        "parent_asin": pair["parent_asin"],
                        "text": pair[source],
                        "label": label,
                        "source": source,
                        "category": category,
                    })

    train_groups = {
        row["parent_asin"] for row in split_rows["train"]
    }
    val_groups = {
        row["parent_asin"]
        for row in split_rows["validation"]
    }

    if train_groups & val_groups:
        raise ValueError("Train / validation 商品重疊。")

    train_texts = {
        row["text"] for row in split_rows["train"]
    }
    val_texts = {
        row["text"] for row in split_rows["validation"]
    }

    if train_texts & val_texts:
        raise ValueError("Train / validation 有完全相同文字。")

    expected_rows = {"train": 32_000, "validation": 8_000}

    for split, rows in split_rows.items():
        if len(rows) != expected_rows[split]:
            raise ValueError(f"{split}: 最終筆數不正確。")

        random.Random(SEED).shuffle(rows)
        write_csv(OUTPUT / f"{split}.csv", rows)

        print(
            f"{split}: {len(rows):,} rows",
            flush=True,
        )

    manifest = {
        "dataset": "Amazon Reviews 2023",
        "categories": CATEGORIES,
        "labels": {
            "0": "description proxy",
            "1": "review proxy",
        },
        "settings": SETTINGS,
        "validation_products_per_category": (
            VALIDATION_PRODUCTS_PER_CATEGORY
        ),
        "split_group": "parent_asin",
        "split_seed": SEED,
        "old_product_split_preserved_when_selected": True,
        "train_rows": len(split_rows["train"]),
        "validation_rows": len(split_rows["validation"]),
        "scans": scans,
    }

    # 最後才建立 manifest，表示全部完成。
    (OUTPUT / "manifest.json").write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print(
        "\nMulticategory SUB dataset completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()