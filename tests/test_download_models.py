import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_models as download


class DownloadModelsTests(unittest.TestCase):
    def test_batch_reads_generator_and_larger_reviewer_without_execution(self):
        with tempfile.TemporaryDirectory(dir=download.PROJECT) as tmp:
            path = Path(tmp) / "test.sbatch"
            path.write_text('LLM="deepseek" # 7B\nREVIEW_LLM="qwen14b" # 14B\necho do-not-execute\n')
            self.assertEqual(download.batch_models(path), ["deepseek", "qwen14b"])
            path.write_text('LLM="$(touch unwanted)"\nREVIEW_LLM="qwen14b"\n')
            with self.assertRaises(ValueError):
                download.batch_models(path)

    def test_plan_deduplicates_and_validates_all_selections(self):
        registry = json.loads((download.PROJECT / "configs/llms.json").read_text())
        plan = download.download_plan(["deepseek", "qwen14b", "deepseek"], registry)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[1][1], "Qwen/Qwen2.5-14B-Instruct")
        with self.assertRaises(ValueError):
            download.download_plan(["deepseek", "does-not-exist"], registry)
        with self.assertRaises(ValueError):
            download.download_plan(["bad"], {"bad": {"repo_id": "x/y", "path": "../outside"}})

    def test_dry_run_does_not_download_or_create_destination(self):
        args = download.parser().parse_args(["--models", "qwen14b", "--dry-run"])
        with patch.object(download, "complete", side_effect=AssertionError("No model access in dry run")):
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(args, downloader=lambda **kwargs: self.fail("Network used"))

    def test_complete_models_skipped(self):
        args = download.parser().parse_args(["--models", "deepseek", "qwen"])
        with patch.object(download, "complete", return_value=True):
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(args, downloader=lambda **kwargs: self.fail("Existing model downloaded again"))

    def test_download_parameters_and_completion(self):
        args = download.parser().parse_args(["--models", "qwen14b"])
        calls = []
        with patch.object(download, "complete", side_effect=[False, True]):
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(args, downloader=lambda **kwargs: calls.append(kwargs))
        self.assertEqual(calls[0]["repo_id"], "Qwen/Qwen2.5-14B-Instruct")
        self.assertTrue(calls[0]["local_dir"].endswith("Qwen2.5-14B-Instruct"))
        self.assertIn("model*.safetensors", calls[0]["allow_patterns"])
        self.assertEqual(calls[0]["max_workers"], 2)

    def test_local_completeness_requires_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory(dir=download.PROJECT) as tmp:
            folder = Path(tmp)
            self.assertFalse(download.complete(folder))
            header = json.dumps({"tensor": {"data_offsets": [0, 4]}}).encode()
            weights = struct.pack("<Q", len(header)) + header + b"1234"
            (folder / "model.safetensors").write_bytes(weights)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (folder / name).write_text("{}")
            self.assertTrue(download.complete(folder))
            (folder / "model.safetensors").write_bytes(b"broken")
            self.assertFalse(download.complete(folder))


if __name__ == "__main__":
    unittest.main()
