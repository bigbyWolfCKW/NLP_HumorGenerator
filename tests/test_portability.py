import os
import tempfile
import unittest

from src.config import PROJECT, absolute, select_detector, select_llm
from src.data import read_json


class PortabilityTests(unittest.TestCase):
    def test_dataset_default_is_inside_project(self):
        config = read_json(PROJECT / "configs/datasets.json")
        self.assertFalse(os.path.isabs(config["root"]))
        path = absolute(config["qa_train"], absolute(config["root"]))
        self.assertTrue(path.is_relative_to(PROJECT / "data"))
        self.assertEqual(path.name, "train.csv")

    def test_default_llm_does_not_fall_back_to_personal_cache(self):
        for name in read_json(PROJECT / "configs/llms.json"):
            from pathlib import Path
            self.assertTrue(Path(select_llm(name)["model"]).is_relative_to(PROJECT / "models/llm"))

    def test_working_directory_does_not_change_model_paths(self):
        expected = select_detector("roberta")["checkpoint"]
        old = os.getcwd()
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(select_detector("roberta")["checkpoint"], expected)
            finally:
                os.chdir(old)
