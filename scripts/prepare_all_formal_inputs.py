"""Combine a formal split across the four members while preserving assignment owners."""

import argparse
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from formal_scope import MEMBERS


def combine_split(directory, split):
    output = directory / f"all_{split}.csv"
    if output.exists():
        return output

    with TemporaryDirectory(dir=directory) as temporary:
        staged = Path(temporary) / output.name
        ids = set()
        with staged.open("w", encoding="utf-8", newline="") as destination:
            writer = None
            for member in MEMBERS:
                path = directory / f"{member}_{split}.csv"
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames or []
                    if not {"owner", "split", "product_id"}.issubset(fields):
                        raise ValueError(f"Missing assignment columns in {path}")
                    if writer is None:
                        writer = csv.DictWriter(destination, fieldnames=fields)
                        writer.writeheader()
                    if set(fields) != set(writer.fieldnames):
                        raise ValueError(f"CSV columns do not match in {path}")
                    count = 0
                    for row in reader:
                        identity = row["product_id"]
                        if row["owner"] != member or row["split"] != split:
                            raise ValueError(f"{path}: unexpected owner or split at line {reader.line_num}")
                        if not identity or identity in ids:
                            raise ValueError(f"{path}: missing or duplicate product ID {identity!r}")
                        ids.add(identity)
                        writer.writerow(row)
                        count += 1
                    if not count:
                        raise ValueError(f"No products in {path}")
        try:
            os.link(staged, output)
        except FileExistsError:
            # Another job may have prepared the same split concurrently.
            pass
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "validation", "test"))
    parser.add_argument("--directory", type=Path, default=Path("data/formal_v1"))
    args = parser.parse_args()
    try:
        print(f"Pooled {args.split} CSV: {combine_split(args.directory, args.split)}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
