import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import main
import run_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def args(self, *argv):
        return pipeline.parser().parse_args(argv)

    def test_all_modes_produce_valid_stage_commands(self):
        for mode in ("demo", "full", *pipeline.STEPS):
            args = self.args("--mode", mode, "--run-dir", "results/example")
            _, commands = pipeline.plan(args)
            expected = list(pipeline.STEPS) if mode in ("demo", "full") else [mode]
            self.assertEqual([step for step, _ in commands], expected)
            for _, command in commands:
                main.parser().parse_args(command)
                self.assertNotIn("--check", command)

    def test_partial_ranking_uses_existing_folder_and_detector_seed(self):
        args = self.args("--mode", "rank", "--run-dir", "results/demo-709136",
                         "--detector", "deberta", "--detector-seed", "43",
                         "--ranked-file", "ranked_deberta.xlsx")
        folder, commands = pipeline.plan(args)
        parsed = main.parser().parse_args(commands[0][1])
        self.assertEqual(parsed.detector, "deberta")
        self.assertEqual(parsed.seed, 43)
        self.assertEqual(Path(parsed.input), folder / "sentences.json")
        self.assertEqual(Path(parsed.output), folder / "ranked_deberta.xlsx")

    def test_custom_does_not_add_unselected_stages(self):
        _, commands = pipeline.plan(self.args("--mode", "custom", "--steps", "generate", "rank"))
        self.assertEqual([s for s, _ in commands], ["generate", "rank"])

    def test_invalid_resume_and_order_fail_before_execution(self):
        for argv in [("--mode", "rank"), ("--mode", "custom", "--steps", "rank", "generate"),
                     ("--mode", "full", "--adapter", "old_adapter")]:
            with self.assertRaises(ValueError):
                pipeline.plan(self.args(*argv))

    def test_dry_run_does_not_write_or_call_models(self):
        with tempfile.TemporaryDirectory(dir=pipeline.PROJECT) as tmp:
            folder = Path(tmp) / "new_run"
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(self.args("--mode", "full", "--run-dir", str(folder), "--dry-run"),
                             runner=lambda *a, **kw: self.fail("Dry run started a process"))
            self.assertFalse(folder.exists())

    def test_demo_and_full_sizes(self):
        for mode, count in [("demo", 10), ("full", 50)]:
            _, commands = pipeline.plan(self.args("--mode", mode))
            parsed = main.parser().parse_args(commands[0][1])
            self.assertEqual(parsed.num_products, count)

    def test_train_uses_config_and_epoch_override(self):
        for mode, override, expected in [("demo", [], 1), ("full", [], 3),
                                         ("full", ["--epochs", "2"], 2)]:
            with tempfile.TemporaryDirectory(dir=pipeline.PROJECT) as tmp:
                calls = []
                folder = Path(tmp) / "run"
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.run(self.args("--mode", mode, "--run-dir", str(folder), *override),
                                 runner=lambda command, **kw: calls.append(command))
                config = pipeline.read_json(folder / "training.json")
                self.assertEqual(config["epochs"], expected)
                self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main()
