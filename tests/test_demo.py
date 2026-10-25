import subprocess
import tempfile
import unittest
from pathlib import Path

import demo
from src.config import PROJECT
from src.data import read_json


class DemoTests(unittest.TestCase):
    def test_complete_stage_order_and_isolated_outputs(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            output = Path(tmp) / "demo"
            args = demo.parser().parse_args(["--products", "data/products.json", "--output", str(output), "--llm", "deepseek",
                                            "--review-llm", "qwen", "--detector-seed", "43"])
            calls = []
            demo.run(args, runner=lambda command, **kwargs: calls.append(command))
            self.assertEqual([c[2] for c in calls], ["rank", "generate", "rank", "review", "prepare", "train", "test"])
            self.assertIn("--check", calls[0])
            self.assertEqual(calls[3][calls[3].index("--llm") + 1], "qwen")
            self.assertEqual(calls[5][calls[5].index("--llm") + 1], "deepseek")
            self.assertEqual(len(read_json(output / "products.json")), 10)
            self.assertEqual(read_json(output / "training.json")["epochs"], 1)
            self.assertEqual(read_json(output / "summary.json")["status"], "complete")
            for command in calls[1:]:
                if "--output" in command:
                    self.assertTrue(Path(command[command.index("--output") + 1]).is_relative_to(output))

    def test_failure_stops_before_review_or_training(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            output = Path(tmp) / "demo"
            args = demo.parser().parse_args(["--products", "data/products.json", "--output", str(output)])
            calls = []
            def runner(command, **kwargs):
                calls.append(command)
                if len(calls) == 3:
                    raise subprocess.CalledProcessError(1, command)
            with self.assertRaises(subprocess.CalledProcessError):
                demo.run(args, runner=runner)
            self.assertEqual(len(calls), 3)
            summary = read_json(output / "summary.json")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(len(summary["completed_stages"]), 2)

    def test_dry_run_has_no_writes_or_model_execution(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            output = Path(tmp) / "demo"
            args = demo.parser().parse_args(["--products", "data/products.json", "--output", str(output), "--dry-run"])
            def forbidden(*args, **kwargs):
                self.fail("Dry run must not launch any stage")
            demo.run(args, runner=forbidden)
            self.assertFalse(output.exists())

    def test_demo_reads_qa_training_dataset_without_labels(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            root = Path(tmp)
            csv = root / "train.csv"
            csv.write_text("product_group,product_description,question,label\n" +
                           "\n".join(f"p{i},Product {i},Original QA question,{i % 2}" for i in range(10)), encoding="utf-8")
            output = root / "demo"
            args = demo.parser().parse_args(["--qa-csv", str(csv), "--output", str(output)])
            demo.run(args, runner=lambda *args, **kwargs: None)
            products = read_json(output / "products.json")
            self.assertEqual(len(products), 10)
            self.assertEqual(set(products[0]), {"product_id", "product_description"})
            self.assertEqual(read_json(output / "summary.json")["dataset"], str(csv))

    def test_missing_dataset_stops_before_any_model_stage(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            output = Path(tmp) / "demo"
            args = demo.parser().parse_args(["--dataset-root", str(Path(tmp) / "missing"), "--output", str(output)])
            with self.assertRaisesRegex(FileNotFoundError, "QA training dataset not found"):
                demo.run(args, runner=lambda *args, **kwargs: self.fail("No stage should run"))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
