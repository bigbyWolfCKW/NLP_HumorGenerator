import csv
import random


def from_qa_csv(path, limit=None, seed=42):
    products = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"product_group", "product_description"}.issubset(reader.fieldnames or []):
            raise ValueError("QA CSV needs product_group and product_description columns.")
        for row in reader:
            identity, description = row["product_group"].strip(), row["product_description"].strip()
            if identity and description:
                products.setdefault(identity, {"product_id": identity, "product_description": description})
    result = list(products.values())
    if not result:
        raise ValueError("No products with descriptions found.")
    if limit is not None:
        if limit < 1 or limit > len(result):
            raise ValueError(f"limit must be between 1 and {len(result)}.")
        result = random.Random(seed).sample(result, limit)
    return result
