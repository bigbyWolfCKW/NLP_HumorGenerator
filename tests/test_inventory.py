import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.config import PROJECT

spec = importlib.util.spec_from_file_location("existing_inventory", PROJECT / "scripts/inspect_existing.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTests(unittest.TestCase):
    def test_counts_and_bundle_preserve_source(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            source, output = Path(tmp) / "old", Path(tmp) / "report"
            (source / "checkpoints/model").mkdir(parents=True)
            (source / "checkpoints/model/best.pt").write_bytes(b"weights")
            (source / "train.py").write_text("import torch\nROOT='/home/old'\n", encoding="utf-8")
            notebook = {"cells": [{"cell_type": "code", "source": ["import numpy"],
                         "execution_count": 1, "outputs": [{"text": "old output"}]}], "metadata": {}}
            original = json.dumps(notebook)
            (source / "train.ipynb").write_text(original, encoding="utf-8")
            (source / "__pycache__").mkdir()
            (source / "__pycache__/ignored.pyc").write_bytes(b"cache")
            report = inventory.inspect(source, output, bundle=True)
            self.assertEqual(report["total_files"], 3)
            self.assertEqual(report["groups"]["checkpoints"]["files"], 1)
            self.assertEqual(len(report["code"]), 2)
            self.assertEqual((source / "train.ipynb").read_text(encoding="utf-8"), original)
            with zipfile.ZipFile(output / "source_code.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"train.py", "train.ipynb"})
                self.assertEqual(json.loads(archive.read("train.ipynb"))["cells"][0]["outputs"], [])

    def test_output_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            source = Path(tmp)
            with self.assertRaisesRegex(ValueError, "outside the source"):
                inventory.inspect(source, source / "output")
            self.assertFalse((source / "output").exists())
