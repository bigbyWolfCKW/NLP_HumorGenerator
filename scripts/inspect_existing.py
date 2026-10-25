"""Inventory an existing project and optionally bundle code without model/data payloads."""
import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXCLUDED = {"__pycache__", "node_modules", ".git", ".venv", ".cache", ".uv-cache"}
ARTIFACT_ROOTS = {"models", "checkpoints", "data", "results", "logs"}
CODE_SUFFIXES = {".py", ".ipynb", ".sbatch", ".sh", ".toml", ".yaml", ".yml"}


def code_candidate(relative):
    if relative.parts[0] in ARTIFACT_ROOTS:
        return False
    if any(part.startswith(".") for part in relative.parts):
        return False
    return (relative.suffix.lower() in CODE_SUFFIXES
            or relative.name.lower().startswith(("requirements", "readme"))
            or (relative.parts[0] == "configs" and relative.suffix == ".json"))


def source_copy(path):
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        code = []
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                code.append(source if isinstance(source, str) else "".join(source))
                cell["outputs"] = []
                cell["execution_count"] = None
        notebook.get("metadata", {}).pop("widgets", None)
        return json.dumps(notebook, ensure_ascii=False, indent=1).encode("utf-8"), "\n".join(code)
    return text.encode("utf-8"), text


def inspect(source, output, bundle=False):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the source project; source files are read-only.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    report = {"source": str(source), "files": [], "groups": {}, "code": [], "errors": [],
              "skipped_directories": [], "runtime_verified": False}
    groups = defaultdict(lambda: {"files": 0, "bytes": 0})
    sources = []
    def walk_error(error):
        report["errors"].append(str(error))
    for directory, children, filenames in os.walk(source, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        kept = []
        for child in sorted(children):
            path = parent / child
            if child in EXCLUDED or child.startswith(".venv-") or path.is_symlink():
                report["skipped_directories"].append(path.relative_to(source).as_posix())
            else:
                kept.append(child)
        children[:] = kept
        for filename in sorted(filenames):
            path = parent / filename
            relative = path.relative_to(source)
            try:
                size = path.stat().st_size
            except OSError as error:
                report["errors"].append(f"{relative}: {error}")
                continue
            report["files"].append({"path": relative.as_posix(), "bytes": size, "symlink": path.is_symlink()})
            # Summaries include artifact subfolders up to three levels deep.
            keys = [relative.parts[0] if len(relative.parts) > 1 else "(root files)"]
            if len(relative.parts) > 2:
                keys.extend("/".join(relative.parts[:depth]) for depth in range(2, min(3, len(relative.parts)-1)+1))
            for key in keys:
                groups[key]["files"] += 1
                groups[key]["bytes"] += size
            if code_candidate(relative):
                if not path.resolve().is_relative_to(source):
                    report["errors"].append(f"Code symlink outside source, not bundled: {relative}")
                    continue
                try:
                    payload, code = source_copy(path)
                    imports = sorted(set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", code, re.M)))
                    absolute_lines = [i for i, line in enumerate(code.splitlines(), 1)
                                      if re.search(r"/home/|[A-Za-z]:[\\/]", line)]
                    report["code"].append({"path": relative.as_posix(), "imports": imports,
                                            "absolute_path_lines": absolute_lines})
                    sources.append((relative.as_posix(), payload))
                except (OSError, ValueError) as error:
                    report["errors"].append(f"Cannot read source {relative}: {error}")
    report["groups"] = dict(sorted(groups.items()))
    report["total_files"] = len(report["files"])
    report["total_bytes"] = sum(f["bytes"] for f in report["files"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inventory.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    lines = [f"Source: {source}", f"Files: {report['total_files']}",
             f"Logical size: {report['total_bytes'] / 1024**3:.3f} GiB", f"Source/config files: {len(sources)}",
             "Counts exclude environment/cache directories listed in inventory.json.",
             "Sizes count file contents, not unique physical disk blocks. Runtime compatibility is not verified.",
             "", f"{'Folder':60} {'Files':>8} {'GiB':>12}"]
    lines += [f"{key:60} {value['files']:8} {value['bytes']/1024**3:12.3f}" for key, value in sorted(groups.items())]
    lines += ["", "Source/config files (imports and absolute-path line numbers are in inventory.json):"]
    lines += ["  " + name for name, _ in sources]
    if report["errors"]:
        lines += ["", "Inspection errors:", *report["errors"]]
    (output / "inventory.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bundle:
        with zipfile.ZipFile(output / "source_code.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sources:
                archive.writestr(name, payload)
        with zipfile.ZipFile(output / "source_code.zip") as archive:
            if archive.testzip():
                raise ValueError("Source archive integrity check failed")
    print("\n".join(lines[:7]))
    print(f"Inventory: {output / 'inventory.txt'}")
    print(f"Details: {output / 'inventory.json'}")
    if bundle:
        print(f"Code bundle: {output / 'source_code.zip'}")
        print("Notebook outputs cleared in the bundle only; original notebooks are unchanged.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing project to inspect, never modified")
    parser.add_argument("--output", help="New report directory outside the existing project")
    parser.add_argument("--bundle-code", action="store_true", help="Bundle code/configs; exclude model/data/result payloads")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    project = Path(__file__).resolve().parents[1]
    destination = Path(args.output).expanduser() if args.output else project / "results" / f"inventory-{stamp}"
    try:
        inspect(args.source, destination, args.bundle_code)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
