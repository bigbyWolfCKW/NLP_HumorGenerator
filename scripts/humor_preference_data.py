"""Auditable preference preparation; no torch dependency or implicit downloads."""
import html
import itertools
import json
import math
import random
import unicodedata
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from download_humor_preferences import DATA_DIR, download_source


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", html.unescape(text)).casefold().split())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def keys(row):
    result = {"joke:" + normalized(row[k]) for k in ("chosen", "rejected")}
    for field in ("group_id", "session_id"):
        if row.get(field):
            result.add(field + ":" + str(row[field]))
    if row.get("context"):
        result.add("context:" + normalized(row["context"]))
    return result


def write_pairs(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def ensure_disjoint(left, right):
    a = set().union(*(keys(r) for r in left))
    b = set().union(*(keys(r) for r in right))
    if a & b:
        raise ValueError(f"Split leakage: {len(a & b)} shared joke/context/group/session keys")


def arena_rows(records):
    rows = []
    skipped = 0
    for record in records:
        winner = record["winner"]
        if winner in ("BOTH", "NONE"):
            skipped += 1
            continue
        if winner not in ("LEFT", "RIGHT"):
            raise ValueError(f"Unknown winner: {winner}")
        a, b = record["left_joke"], record["right_joke"]
        if normalized(a) == normalized(b):
            skipped += 1
            continue
        rows.append({"chosen": a if winner == "LEFT" else b,
                     "rejected": b if winner == "LEFT" else a,
                     "context": "", "theme": record.get("theme", ""),
                     "session_id": record.get("session_id", ""),
                     "source": "h_arena", "label_type": "human_pairwise"})
    return rows, skipped


def split_arena(rows, fraction, seed):
    # Repeated jokes can connect almost every session into a giant component.
    # Split sessions first, then purge training votes containing validation jokes.
    groups = {}
    for index, row in enumerate(rows):
        session = ("session", row["session_id"]) if row.get("session_id") else ("missing", index)
        groups.setdefault(session, []).append(row)
    blocks = list(groups.values())
    random.Random(seed).shuffle(blocks)
    if len(blocks) < 2:
        raise ValueError("Need at least two independent sessions")
    validation, train = [], []
    target = max(1, round(len(rows) * fraction))
    for block in blocks[:-1]:
        if abs(len(validation) + len(block) - target) < abs(len(validation) - target):
            validation.extend(block)
        else:
            train.extend(block)
    train.extend(blocks[-1])
    held_out = {normalized(r[k]) for r in validation for k in ("chosen", "rejected")}
    cleaned = [r for r in train if not any(normalized(r[k]) in held_out for k in ("chosen", "rejected"))]
    ensure_disjoint(cleaned, validation)
    return cleaned, validation, len(train) - len(cleaned)


def mechanics_items(records):
    # Combine exact duplicate texts before splitting, never interpret missing scores as zero.
    grouped = {}
    skipped = 0
    for row in records:
        rating = row.get("funniness")
        if rating is None:
            skipped += 1
            continue
        rating = float(rating)
        if not math.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError("Mechanics funniness must be finite and in [1, 5]")
        text = row["text"]
        if not normalized(text):
            raise ValueError("Empty joke")
        item = grouped.setdefault(normalized(text), {"text": text, "ratings": []})
        item["ratings"].append(rating)
    return [{"text": r["text"], "rating": sum(r["ratings"]) / len(r["ratings"])}
            for r in grouped.values()], skipped


def rating_pairs(items, gap, cap, seed):
    # Reservoir sampling bounds output memory and prevents counting O(n^2) pairs as
    # independent human annotations. These pairs can have different joke topics.
    rng = random.Random(seed)
    result, eligible = [], 0
    for a, b in itertools.combinations(items, 2):
        if abs(a["rating"] - b["rating"]) < gap:
            continue
        high, low = (a, b) if a["rating"] > b["rating"] else (b, a)
        row = {"chosen": high["text"], "rejected": low["text"], "context": "",
               "source": "humor_mechanics", "label_type": "rating_derived",
               "rating_gap": high["rating"] - low["rating"]}
        eligible += 1
        if len(result) < cap:
            result.append(row)
        else:
            index = rng.randrange(eligible)
            if index < cap:
                result[index] = row
    rng.shuffle(result)
    return result


def prepare(args):
    if not 0 < args.validation_fraction < 1 or args.min_gap <= 0 or args.max_pairs < 1:
        raise ValueError("Invalid split fraction, rating gap or pair cap")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = []
    for path in args.input:
        raw = read_json(path)
        if not isinstance(raw, list):
            raise ValueError("Expected JSON array")
        records.extend(raw)
    if args.source == "h-arena":
        rows, _ = arena_rows(records)
        train, validation, _ = split_arena(rows, args.validation_fraction, args.seed)
    else:
        items, _ = mechanics_items(records)
        random.Random(args.seed).shuffle(items)
        cut = max(2, round(len(items) * args.validation_fraction))
        validation = rating_pairs(items[:cut], args.min_gap, args.max_pairs, args.seed)
        train = rating_pairs(items[cut:], args.min_gap, args.max_pairs, args.seed)
    if not train or not validation:
        raise ValueError("Empty split after filtering; inspect input size/rating gaps")
    ensure_disjoint(train, validation)
    output.mkdir(parents=True, exist_ok=False)
    write_pairs(output / "train.jsonl", train)
    write_pairs(output / "validation.jsonl", validation)


PAIR_FILES = ("train.jsonl", "validation.jsonl")


def pair_file_ready(path):
    try:
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (FileNotFoundError, ValueError):
        return False
    return bool(rows) and all(isinstance(row, dict) for row in rows)


def build_source_pairs(source, pair_dir):
    files = [pair_dir / name for name in PAIR_FILES]
    if all(pair_file_ready(path) for path in files):
        print(f"Using existing pairs in {pair_dir}", flush=True)
        return

    with TemporaryDirectory(dir=pair_dir.parent) as temporary:
        staged = Path(temporary) / source
        prepare(Namespace(
            source="h-arena" if source == "arena" else "mechanics",
            input=[str(path) for path in download_source(source)],
            output=str(staged),
            validation_fraction=0.2,
            min_gap=0.5,
            max_pairs=5000,
            seed=42,
        ))
        pair_dir.mkdir(exist_ok=True)
        # Replace readable files first so an interrupted repair still needs a retry.
        for path in sorted(files, key=pair_file_ready, reverse=True):
            (staged / path.name).replace(path)


def ensure_pairs(source):
    import fcntl

    pair_root = DATA_DIR / "pairwise"
    pair_root.mkdir(parents=True, exist_ok=True)
    with (pair_root / f".{source}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_source_pairs(source, pair_root / source)
