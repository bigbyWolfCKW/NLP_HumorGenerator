import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from detector.__main__ import main as detector_main, parser
from detector.data import classification_metrics, ensure_disjoint, read_examples
from src.config import PROJECT


class DetectorModuleTests(unittest.TestCase):
    def test_existing_imports_are_compatible(self):
        from src.detector import HumorDetector as old_detector, create_detector as old_create
        from detector.inference import HumorDetector, create_detector
        self.assertIs(old_detector, HumorDetector)
        self.assertIs(old_create, create_detector)

    def test_rank_forwards_existing_arguments(self):
        argv = ["rank", "--detector", "deberta", "--check"]
        with patch("main.main") as pipeline:
            detector_main(argv)
        pipeline.assert_called_once_with(argv)

    def test_classifier_metrics(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["macro_f1"], (2/3 + 4/5) / 2)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])

    def test_product_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "product overlap"):
            ensure_disjoint([{"product_group": "same"}], [{"product_group": "same"}])

    def test_qa_csv_input_modes_and_label_checks(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            path = Path(tmp) / "train.csv"
            path.write_text("question,product_group,label\nFunny?,p1,1\n", encoding="utf-8")
            self.assertEqual(read_examples(path, "text")[0]["label"], 1)
            with self.assertRaisesRegex(ValueError, "product_description"):
                read_examples(path, "pair")
            path.write_text("question,product_group,label\nFunny?,p1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binary label"):
                read_examples(path, "text")

    def test_train_requires_explicit_new_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["train", "--detector", "roberta"])

    def test_evaluation_writes_metrics_without_training(self):
        from detector.evaluation import evaluate_detector
        from src.data import read_json
        import types
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            path, output = Path(tmp) / "test.csv", Path(tmp) / "evaluation"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["question", "product_description", "product_group", "label"])
                writer.writerows([["Text A", "Product A", "a", 0], ["Text B", "Product B", "b", 1]])
            args = types.SimpleNamespace(detector="test", seed=42, csv=str(path), output=str(output), device="cpu", batch_size=2)
            config = {"input_mode": "pair", "positive_label": 1}
            with patch("detector.evaluation.select_detector", return_value=config), \
                 patch("detector.evaluation.create_detector", return_value=types.SimpleNamespace(score=lambda rows: [.1, .9])):
                evaluate_detector(args)
            self.assertEqual(read_json(output / "metrics.json")["accuracy"], 1.0)
            self.assertTrue((output / "predictions.csv").is_file())
