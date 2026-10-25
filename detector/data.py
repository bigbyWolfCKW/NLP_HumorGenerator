import csv
from pathlib import Path

from src.config import absolute
from src.data import read_json


def split_path(split, override=None):
    if override:
        return absolute(override)
    config = read_json(absolute("configs/datasets.json"))
    return absolute(f"processed/unbiased/{split}.csv", absolute(config["root"]))


def read_examples(path, input_mode="pair"):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"question", "product_group", "label"}
        if input_mode == "pair":
            required.add("product_description")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing detector dataset columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for index, row in enumerate(reader, 2):
            question = (row.get("question") or "").strip()
            product = (row.get("product_group") or "").strip()
            description = (row.get("product_description") or "").strip()
            if not question or not product or (input_mode == "pair" and not description):
                raise ValueError(f"Empty text/product at CSV line {index}: {path}")
            if row.get("label") not in ["0", "1"]:
                raise ValueError(f"Expected binary label 0/1 at CSV line {index}: {path}")
            rows.append({"question": question, "product_id": product, "product_group": product,
                         "product_description": description, "label": int(row["label"]),
                         "candidate_id": f"row_{index-2}"})
    if not rows:
        raise ValueError(f"Empty detector dataset: {path}")
    return rows


def ensure_disjoint(train, validation):
    overlap = {r["product_group"] for r in train} & {r["product_group"] for r in validation}
    if overlap:
        raise ValueError(f"Train/validation product overlap: {len(overlap)} products")


def classification_metrics(actual, predicted):
    actual, predicted = list(actual), list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Expected equal, nonempty label/prediction lists")
    confusion = [[0, 0], [0, 0]]
    for label, prediction in zip(actual, predicted):
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Metrics require binary labels")
        confusion[label][prediction] += 1
    f1 = []
    for c in (0, 1):
        numerator = 2 * confusion[c][c]
        denominator = numerator + confusion[c][1-c] + confusion[1-c][c]
        f1.append(numerator / denominator if denominator else 0.0)
    return {"accuracy": (confusion[0][0] + confusion[1][1]) / len(actual),
            "macro_f1": sum(f1) / 2, "confusion_matrix": confusion, "rows": len(actual)}
