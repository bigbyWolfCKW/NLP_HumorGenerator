import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_formal_pqa.py"
SPEC = importlib.util.spec_from_file_location("download_formal_pqa", SCRIPT)
pqa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pqa)


class PrepareSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/datasets.json").write_text(json.dumps({
            "root": "data/source", "qa_train": "processed/unbiased/train.csv",
        }), encoding="utf-8")
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "data/source/processed/unbiased"
        for filename, label in pqa.FILES.items():
            with (self.raw / filename).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=pqa.FIELDS[:-1])
                writer.writeheader()
                for i in range(20):
                    writer.writerow(dict(question=f"Question {i} label {label}",
                                         product_description=f"Product {i}", image_url="", label=label))

    def run_prepare(self, *args):
        with patch.object(pqa, "PROJECT", self.root), patch("sys.argv", [str(SCRIPT), "--raw-dir", str(self.raw), *args]):
            with contextlib.redirect_stdout(io.StringIO()):
                pqa.main()

    def test_empty_directory_and_missing_splits_repaired_without_changing_train(self):
        self.output.mkdir(parents=True)
        self.run_prepare("--ensure", "--offline")
        original = {n: (self.output / n).read_bytes() for n in pqa.SPLIT_FILES}
        (self.output / "validation.csv").unlink()
        (self.output / "test.csv").write_text("", encoding="utf-8")
        self.run_prepare("--ensure", "--offline")
        self.assertEqual(original, {n: (self.output / n).read_bytes() for n in pqa.SPLIT_FILES})

    def test_complete_splits_do_not_download_or_rewrite(self):
        self.run_prepare("--offline")
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.iterdir()}
        with patch.object(pqa, "download", side_effect=AssertionError("Unexpected download")):
            self.run_prepare("--ensure")
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.iterdir()})

    def test_different_surviving_split_is_not_overwritten_or_mixed(self):
        self.run_prepare("--offline")
        train = self.output / "train.csv"
        train.write_text(train.read_text(encoding="utf-8").replace("Question", "Changed"), encoding="utf-8")
        (self.output / "validation.csv").unlink()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, "Existing split differs"):
            self.run_prepare("--ensure", "--offline")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_download_missing_raw_then_build_from_empty_split_directory(self):
        content = {n: (self.raw / n).read_bytes() for n in pqa.FILES}
        for n in content:
            (self.raw / n).unlink()
        self.output.mkdir(parents=True)
        with patch.object(pqa, "urlopen", side_effect=lambda url, timeout: io.BytesIO(content[url.rsplit("/", 1)[-1]])) as request:
            self.run_prepare("--ensure")
        self.assertEqual(request.call_count, 2)
        self.assertTrue(all(pqa.split_ready(self.output / n) for n in pqa.SPLIT_FILES))

    def test_fresh_only_default_still_refuses_existing_directory(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_prepare("--offline")


if __name__ == "__main__":
    unittest.main()
