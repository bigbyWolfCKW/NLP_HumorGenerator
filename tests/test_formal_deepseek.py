import ast
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from formal_deepseek import DEEPSEEK_MODELS, extract_ad
from formal_models import compatible_resume, PRE_DEEPSEEK_EXTRACTION_HASHES


class DeepSeekCandidateTests(unittest.TestCase):
    def test_complete_reasoning_keeps_only_final_ad(self):
        for prefix in ("<think>Private thoughts", "Thoughts continued from chat template"):
            self.assertEqual(extract_ad(prefix + "</think>\n\nBuy this funny mug!"),
                             ("Buy this funny mug!", []))

    def test_unfinished_reasoning_is_not_an_ad(self):
        for text in ("<think>Still thinking", "Okay, I need to write an ad", ""):
            self.assertEqual(extract_ad(text), ("", ["missing_final_answer"]))

    def test_empty_or_repeated_reasoning_is_flagged(self):
        self.assertEqual(extract_ad("thoughts</think>  "), ("", ["missing_final_answer"]))
        self.assertEqual(extract_ad("thoughts</think><think>more</think>ad"),
                         ("", ["invalid_final_answer"]))

    def test_extraction_does_not_rewrite_or_truncate_ads(self):
        ad = " ".join(["word"] * 45)
        self.assertEqual(extract_ad("thinking</think>" + ad), (ad, []))

    def test_only_deepseek_models_opt_in(self):
        self.assertEqual(DEEPSEEK_MODELS, {"deepseek", "deepseek14b"})

    def test_qwen_resume_preserves_settings_and_deepseek_rejects_old_run(self):
        for old_hash in PRE_DEEPSEEK_EXTRACTION_HASHES["generate_formal_ads_v1.py"]:
            previous = {"script_sha256": old_hash, "max_new_tokens": 192}
            current = {"script_sha256": "new", "max_new_tokens": 192}
            self.assertTrue(compatible_resume(previous, current, "qwen", "generate_formal_ads_v1.py"))
            self.assertFalse(compatible_resume(previous, current, "deepseek", "generate_formal_ads_v1.py"))
            self.assertFalse(compatible_resume(previous, {**current, "max_new_tokens": 1024},
                                               "qwen", "generate_formal_ads_v1.py"))

    def test_token_limit_checks_keep_qwen_default_and_accept_deepseek_budget(self):
        tree = ast.parse((SCRIPTS / "generate_formal_ads_v1.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "format_flags")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "flags", "exec"), namespace)
        flags = namespace["format_flags"]
        self.assertIn("token_limit_reached", flags("Final ad", [1] * 192, 2))
        self.assertNotIn("token_limit_reached", flags("Final ad", [1] * 192, 2, 1024))
        self.assertIn("token_limit_reached", flags("Final ad", [1] * 1024, 2, 1024))
        self.assertNotIn("token_limit_reached", flags("Final ad", [1] * 1023 + [2], 2, 1024))
        self.assertIn("over_40_words", flags("word " * 41, [2], 2))


if __name__ == "__main__":
    unittest.main()
