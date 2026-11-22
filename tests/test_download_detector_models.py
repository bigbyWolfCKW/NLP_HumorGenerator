import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import download_detector_models as download


class DetectorDownloadTests(unittest.TestCase):
    def test_default_plan_matches_detector_loader_paths(self):
        from src.config import select_detector
        plan = download.download_plan(download.parser().parse_args([]).models)
        self.assertEqual(len(plan), 3)
        for name, repo, destination in plan:
            self.assertEqual(destination, Path(select_detector(name)["backbone"]))
            self.assertEqual(repo, download.REPOSITORIES[name])

    def test_dry_run_never_checks_or_downloads(self):
        with patch.object(download, "complete", side_effect=AssertionError("Model accessed")):
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(download.parser().parse_args(["--dry-run"]),
                             downloader=Mock(side_effect=AssertionError("Network used")),
                             model_info=Mock(side_effect=AssertionError("Network used")))

    def test_complete_models_skip_hub_entirely(self):
        with patch.object(download, "complete", return_value=True):
            with contextlib.redirect_stdout(io.StringIO()):
                download.run(download.parser().parse_args([]),
                             downloader=Mock(side_effect=AssertionError("Network used")),
                             model_info=Mock(side_effect=AssertionError("Network used")))

    def test_download_pins_revision_and_supports_both_weight_formats(self):
        for filename in ("model.safetensors", "pytorch_model.bin"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                destination = Path(tmp)
                info = SimpleNamespace(sha="abc123", siblings=[SimpleNamespace(rfilename=filename)])
                fetch = Mock()
                with patch.object(download, "download_plan", return_value=[("deberta", "microsoft/deberta-v3-base", destination)]), \
                     patch.object(download, "complete", side_effect=[False, True]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    download.run(download.parser().parse_args(["--models", "deberta"]), fetch, Mock(return_value=info))
                kwargs = fetch.call_args.kwargs
                self.assertEqual(kwargs["revision"], "abc123")
                self.assertIn("*.model", kwargs["allow_patterns"])
                self.assertIn("model*.safetensors" if filename.endswith("safetensors") else "pytorch_model*.bin", kwargs["allow_patterns"])
                self.assertEqual(json.loads((destination / "download_manifest.json").read_text())["revision"], "abc123")

    def test_incomplete_download_fails_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            info = SimpleNamespace(sha="abc123", siblings=[SimpleNamespace(rfilename="model.safetensors")])
            with patch.object(download, "download_plan", return_value=[("roberta", "FacebookAI/roberta-base", destination)]), \
                 patch.object(download, "complete", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Incomplete backbone"):
                    download.run(download.parser().parse_args([]), Mock(), Mock(return_value=info))
            self.assertFalse((destination / "download_manifest.json").exists())

    def test_completeness_requires_weights_config_and_loadable_local_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertFalse(download.complete(folder))
            (folder / "config.json").write_text('{"model_type":"roberta"}')
            header = json.dumps({"tensor": {"data_offsets": [0, 4]}}).encode()
            (folder / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            tokenizer = Mock()
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)}):
                self.assertTrue(download.complete(folder))
                tokenizer.from_pretrained.assert_called_once_with(folder, local_files_only=True)
                tokenizer.from_pretrained.side_effect = OSError("Missing tokenizer")
                self.assertFalse(download.complete(folder))


if __name__ == "__main__":
    unittest.main()
