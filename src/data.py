import hashlib
import json
import math
import random
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Candidates must be a nonempty JSON list.")
    result, seen = [], set()
    for i, row in enumerate(rows):
        r = dict(row)
        for key in ["product_id", "product_description", "question"]:
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"Row {i}: nonempty string required: {key}")
            r[key] = r[key].strip()
        r.setdefault("candidate_id", f"candidate_{i:06d}")
        if not isinstance(r["candidate_id"], str) or r["candidate_id"] in seen:
            raise ValueError(f"Duplicate or invalid candidate_id: {r['candidate_id']}")
        seen.add(r["candidate_id"])
        result.append(r)
    return result


def tagline_prompt(description):
    return ("Write one short humorous English advertising tagline for the following product.\n"
            f"Product information: {description}\nReturn only the tagline.")


def score_rows(rows, scores, threshold):
    scores = list(scores)
    if len(scores) != len(rows):
        raise ValueError("Detector returned the wrong number of scores.")
    if any(not math.isfinite(float(s)) or not 0 <= float(s) <= 1 for s in scores):
        raise ValueError("Detector scores must be finite probabilities in [0, 1].")
    scored = [dict(r, humor_score=float(s), approved=float(s) >= threshold)
              for r, s in zip(rows, scores)]
    return scored, [r for r in scored if r["approved"]]


def prepare_dataset(rows, output, seed=42):
    """Split by product, keeping all slogans for each product in one split."""
    groups = sorted({r["product_id"] for r in rows})
    if len(groups) < 3:
        raise ValueError("LoRA handoff needs approved candidates for at least 3 products.")
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * .2))
    n_val = max(1, round(len(groups) * .1))
    assignment = {g: ("test" if i < n_test else "validation" if i < n_test + n_val else "train")
                  for i, g in enumerate(groups)}
    output = Path(output)
    output.mkdir(exist_ok=False)
    counts = {}
    for split in ["train", "validation", "test"]:
        selected = [r for r in rows if assignment[r["product_id"]] == split]
        counts[split] = len(selected)
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for r in selected:
                record = {"messages": [
                    {"role": "user", "content": tagline_prompt(r["product_description"])},
                    {"role": "assistant", "content": r["question"]}]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(output / "split_manifest.json", {"seed": seed, "counts": counts, "products": assignment})
    return counts
