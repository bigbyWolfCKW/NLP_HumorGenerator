import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.environment import PROJECT, load_local_env
from src.api_review import APIReviewer


class PrivateEnvironmentTests(unittest.TestCase):
    def test_private_key_loads_automatically_for_reviewer(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            path = Path(tmp) / ".env"
            path.write_text('GEMINI_API_KEY="test-only-key" # private\n')
            with patch.dict(os.environ, {}, clear=True), patch("src.environment.PROJECT", Path(tmp)):
                reviewer = APIReviewer("gemini-3.8-flash", "https://generativelanguage.googleapis.com/v1beta/openai",
                                       "GEMINI_API_KEY")
                self.assertEqual(reviewer.key, "test-only-key")
                self.assertNotIn("test-only-key", str(reviewer.config))

    def test_exported_key_takes_priority_and_no_shell_evaluation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            path = Path(tmp) / ".env"
            path.write_text('export GEMINI_API_KEY="file-key"\nOTHER="$(echo do-not-run)"\nEMPTY=\n')
            with patch.dict(os.environ, {"GEMINI_API_KEY": "exported-key"}, clear=True):
                load_local_env(path)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "exported-key")
                self.assertEqual(os.environ["OTHER"], "$(echo do-not-run)")
                self.assertNotIn("EMPTY", os.environ)

    def test_bad_line_error_hides_value(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            path = Path(tmp) / ".env"
            path.write_text('GEMINI_API_KEY="private-test-value\n')
            with self.assertRaises(ValueError) as caught:
                load_local_env(path)
            self.assertNotIn("private-test-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
