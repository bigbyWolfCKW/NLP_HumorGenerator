"""Remove unchanged old experiment filenames after installing renamed sources."""
import hashlib
import json
from pathlib import Path


def clean(project):
    project = Path(project).resolve()
    folder = project / "detector/experiments"
    manifest = json.loads((folder / "filenames.json").read_text(encoding="utf-8"))
    for name in manifest["renamed"].values():
        if Path(name).name != name or not (folder / name).is_file():
            raise ValueError(f"Renamed source missing or invalid: {name}; extract the complete update first.")
    for name in ("training.py", "evaluation.py"):
        if not (project / "detector" / name).is_file():
            raise ValueError(f"Shared detector implementation missing: {name}")
    candidates = dict(manifest["original_sha256"])
    candidates.update(manifest.get("obsolete_files", {}))
    removed = 0
    for name, expected in candidates.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid source filename: {name}")
        path = folder / name
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(project):
            raise ValueError(f"Source must stay inside project: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            print(f"Kept edited file: {path}")
            continue
        path.unlink()
        removed += 1
    print(f"Removed {removed} unchanged old files. No backup created; models/data/results untouched.")


if __name__ == "__main__":
    clean(Path(__file__).resolve().parents[1])
