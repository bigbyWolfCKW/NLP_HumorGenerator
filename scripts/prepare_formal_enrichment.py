"""Attach reviewed product facts and evidence URLs to an assigned training CSV."""

import argparse
import csv
from pathlib import Path
from urllib.parse import urlsplit

ASSIGNMENT_FIELDS = {"product_id", "product_family_id", "product_facts", "owner", "split"}
FACT_FIELDS = ("verified_product_facts", "fact_source_urls", "verified_product_name")


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not required.issubset(fields):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(fields))}")
        rows = list(reader)
    ids = [row["product_id"] for row in rows]
    if not rows or any(not key or not key.strip() for key in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Expected nonempty, unique product IDs: {path}")
    return fields, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Assigned member_N_train.csv")
    parser.add_argument("--facts", type=Path, required=True, help="Reviewed facts CSV keyed by product_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-only", action="store_true", help="Include only products with reviewed facts.")
    args = parser.parse_args()
    fields, products = read_csv(args.input, ASSIGNMENT_FIELDS)
    if {row["split"] for row in products} != {"train"} or len({row["owner"] for row in products}) != 1:
        raise ValueError("Expected the training assignment for one member.")
    _, reviewed = read_csv(args.facts, {"product_id", "verified_product_facts", "fact_source_urls"})
    by_id = {row["product_id"]: row for row in reviewed}
    unknown = by_id.keys() - {row["product_id"] for row in products}
    if unknown:
        raise ValueError(f"Reviewed facts contain {len(unknown)} unassigned product IDs.")
    output = []
    for product in products:
        review = by_id.get(product["product_id"])
        if review is None and args.ready_only:
            continue
        added = {key: ((review or {}).get(key) or "").strip() for key in FACT_FIELDS}
        if review:
            urls = [urlsplit(value) for value in added["fact_source_urls"].split()]
            if not added["verified_product_facts"] or not urls or any(
                url.scheme not in ("http", "https") or not url.hostname for url in urls
            ):
                raise ValueError(f"Expected reviewed facts and HTTP(S) evidence: {product['product_id']}")
        output.append({**product, **added, "fact_status": "ready" if review else "catalog_only"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys([*fields, *FACT_FIELDS, "fact_status"])))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {len(output)} products to {args.output} ({len(reviewed)} with reviewed facts).")


if __name__ == "__main__":
    main()
