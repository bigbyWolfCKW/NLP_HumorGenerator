"""Install the processed Amazon PQA training CSV at the configured source path."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]


def validate_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"product_group", "product_description"}
        missing = required - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        count = 0
        for row in rows:
            if None in row or any(row[key] is None for key in required):
                raise ValueError(f"Malformed CSV record ending at line {rows.line_num}")
            if all(row[key].strip() for key in required):
                count += 1
    if count == 0:
        raise ValueError("The CSV has no rows with a product group and description.")
    return count


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Processed unbiased train.csv")
    source = parser.parse_args().input.resolve()
    config = json.loads((PROJECT / "configs/datasets.json").read_text(encoding="utf-8"))
    destination = (PROJECT / config["root"] / config["qa_train"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "train.csv"
        shutil.copyfile(source, staged)
        count = validate_csv(staged)
        checksum = sha256(staged)
        if destination.exists():
            if sha256(destination) != checksum:
                raise FileExistsError(f"Different source CSV already exists: {destination}")
        else:
            try:
                target = destination.open("xb")
            except FileExistsError:
                if sha256(destination) != checksum:
                    raise FileExistsError(f"Different source CSV already exists: {destination}")
            else:
                try:
                    with target, staged.open("rb") as stream:
                        shutil.copyfileobj(stream, target)
                except BaseException:
                    destination.unlink()
                    raise

    print(f"Source ready: {destination}")
    print(f"Usable source rows: {count}")
    print(f"SHA256: {checksum}")


if __name__ == "__main__":
    main()
