"""Download the raw H.Arena votes and Humor Mechanics ratings."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data/source/preferences/humor_preference_v1"
ARENA_URL = (
    "https://raw.githubusercontent.com/SaveTheRbtz/humor/main/data/"
    "humor_arena.choices.20250812.json"
)
MECHANICS_URL = "https://raw.githubusercontent.com/altsoph/humor-mechanics/main"
MECHANICS_FILES = (
    "jokes-0shot-scheme.json",
    "jokes-assocv1-scheme.json",
    "jokes-assocv2-scheme.json",
    "jokes-full-scheme.json",
    "jokes-noassoc-scheme.json",
    "jokes-reddit.json",
)


def valid_json(path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            json.load(stream)
    except (FileNotFoundError, ValueError):
        return False
    return True


def download_json(url, target):
    if valid_json(target):
        print(f"Using {target}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        with urlopen(url, timeout=60) as response, staged.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if not valid_json(staged):
            raise ValueError("Download is empty or invalid JSON")
        staged.replace(target)
    print(f"Downloaded {target}", flush=True)


def download_source(source):
    files = (
        [(ARENA_URL, DATA_DIR / "h_arena.json")]
        if source == "arena"
        else [
            (f"{MECHANICS_URL}/{name}", DATA_DIR / "humor-mechanics" / name)
            for name in MECHANICS_FILES
        ]
    )
    for url, target in files:
        try:
            download_json(url, target)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Could not prepare {target}: {error}") from error
    return [target for _, target in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("arena", "mechanics"))
    args = parser.parse_args()
    for source in (args.source,) if args.source else ("arena", "mechanics"):
        download_source(source)


if __name__ == "__main__":
    main()
