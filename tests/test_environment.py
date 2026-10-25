import importlib.util
import tempfile
import types
import unittest
import tomllib
from pathlib import Path
from unittest.mock import patch

from src.config import PROJECT

spec = importlib.util.spec_from_file_location("environment_check", PROJECT / "scripts/check_environment.py")
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)


class EnvironmentTests(unittest.TestCase):
    def test_namespace_package_is_not_accepted_as_working_install(self):
        with patch.object(environment.importlib, "import_module", return_value=types.SimpleNamespace(__file__=None)):
            with self.assertRaisesRegex(ImportError, "installation may be incomplete"):
                environment.checked_import("torch", "torch")

    def test_package_needs_real_file_and_version(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            package = Path(tmp) / "__init__.py"
            package.write_text("", encoding="utf-8")
            with patch.object(environment.importlib, "import_module", return_value=types.SimpleNamespace(__file__=str(package))), \
                 patch.object(environment.importlib.metadata, "version", return_value="2.8.0+cu128"):
                self.assertEqual(environment.checked_import("torch", "torch")["version"], "2.8.0+cu128")
            with patch.object(environment.importlib, "import_module", return_value=types.SimpleNamespace(__file__=str(package))), \
                 patch.object(environment.importlib.metadata, "version", return_value="None"):
                with self.assertRaisesRegex(ImportError, "no valid installed version"):
                    environment.checked_import("torch", "torch")

    def test_core_pins_include_torch_and_huggingface_dependencies(self):
        pins = environment.pins()
        self.assertEqual(pins["torch"], "2.8.0")
        self.assertEqual(pins["transformers"], "4.57.1")
        self.assertEqual(pins["huggingface-hub"], "0.36.2")

    def test_lock_matches_core_dependencies_and_cuda_source(self):
        with (PROJECT / "uv.lock").open("rb") as stream:
            lock = tomllib.load(stream)
        packages = {package["name"]: package for package in lock["package"]}
        for name, version in environment.pins().items():
            self.assertEqual(packages[name]["version"].split("+", 1)[0], version)
        self.assertEqual(packages["torch"]["version"], "2.8.0+cu128")
        self.assertEqual(packages["torch"]["source"]["registry"], "https://download.pytorch.org/whl/cu128")


if __name__ == "__main__":
    unittest.main()
