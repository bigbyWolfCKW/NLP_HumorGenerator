"""Read a private local .env without executing shell commands."""
import os
from pathlib import Path
import re
import shlex

PROJECT = Path(__file__).resolve().parents[1]


def load_local_env(path=None):
    path = Path(path) if path is not None else PROJECT / ".env"
    if not path.is_file():
        return
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, raw = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env assignment at line {number}.")
        try:
            parts = shlex.split(raw.strip(), comments=True)
        except ValueError:
            raise ValueError(f"Invalid .env quoting at line {number}.") from None
        if len(parts) > 1:
            raise ValueError(f"Quote values containing spaces in .env, line {number}.")
        values[key] = parts[0] if parts else ""
    for key, value in values.items():
        if value:
            # Explicit exported variables take priority over the local file.
            os.environ.setdefault(key, value)
