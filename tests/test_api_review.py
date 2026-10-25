import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import main
import run_pipeline
from scripts.download_models import batch_models
from src.api_review import APIReviewer, NoRedirect
from src.config import PROJECT


class APIReviewTests(unittest.TestCase):
    def reviewer(self, base="https://api.deepseek.com"):
        with patch.dict(os.environ, {"TEST_REVIEW_KEY": "test-only-key"}):
            return APIReviewer("deepseek-flash", base, "TEST_REVIEW_KEY")

    def test_review_request_and_text_without_gpu(self):
        reviewer = self.reviewer()
        response = io.StringIO(json.dumps({"choices": [{"message": {"content": '{"candidate_id":"c1"}'}, "finish_reason": "stop"}]}))
        with patch.object(reviewer.opener, "open", return_value=response) as call:
            self.assertEqual(reviewer.answer("Return JSON"), '{"candidate_id":"c1"}')
        request = call.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertNotIn("test-only-key", json.dumps(reviewer.config))

    def test_missing_key_and_insecure_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Missing API key"):
                APIReviewer("model", "https://example.org/v1")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            APIReviewer("model", "http://example.org/v1")
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example"))

    def test_google_compatibility_request(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}):
            reviewer = APIReviewer("gemini-3.8-flash", "https://generativelanguage.googleapis.com/v1beta/openai/",
                                   "GEMINI_API_KEY", 4096)
        response = io.StringIO(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}))
        with patch.object(reviewer.opener, "open", return_value=response) as call:
            reviewer.answer("Return JSON")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertNotIn("thinking", payload)

    def test_hkust_uses_gateway_endpoint_and_api_key_header(self):
        with patch.dict(os.environ, {"HKUST_API_KEY": "test-school-key"}):
            reviewer = APIReviewer("gpt-4o-mini", "https://hkust.azure-api.net/hkust-genai/v1",
                                   "HKUST_API_KEY", 1024)
        response = io.StringIO(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}))
        with patch.object(reviewer.opener, "open", return_value=response) as call:
            reviewer.answer("Return JSON")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://hkust.azure-api.net/hkust-genai/v1/chat/completions")
        self.assertEqual(request.get_header("Api-key"), "test-school-key")
        self.assertIsNone(request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertNotIn("thinking", payload)
        self.assertNotIn("reasoning_effort", payload)

    def test_error_does_not_print_body_or_key(self):
        reviewer = self.reviewer()
        error = urllib.error.HTTPError(reviewer.endpoint, 401, "test-only-key", {}, None)
        with patch.object(reviewer.opener, "open", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                reviewer.answer("private input")
        self.assertNotIn("test-only-key", str(caught.exception))
        self.assertIn("HTTP 401", str(caught.exception))

    def test_api_review_pipeline_routes_only_review_to_api(self):
        args = run_pipeline.parser().parse_args(["--mode", "full", "--review-backend", "api",
            "--review-llm", "deepseek-flash", "--review-api-base", "https://api.deepseek.com"])
        _, commands = run_pipeline.plan(args)
        for step, command in commands:
            parsed = main.parser().parse_args(command)
            if step == "review":
                self.assertEqual(parsed.backend, "api")
                self.assertEqual(parsed.llm, "deepseek-flash")
            elif step in ("generate", "train", "test"):
                self.assertEqual(parsed.backend, "transformers")

    def test_structured_error_explains_bad_parameter_and_redacts_key(self):
        reviewer = self.reviewer()
        body = io.BytesIO(json.dumps({"error": {"message": "Unsupported reasoning_effort; key=test-only-key"}}).encode())
        error = urllib.error.HTTPError(reviewer.endpoint, 400, "bad request", {}, body)
        with patch.object(reviewer.opener, "open", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                reviewer.answer("Return JSON")
        self.assertIn("Unsupported reasoning_effort", str(caught.exception))
        self.assertNotIn("test-only-key", str(caught.exception))

    def test_non_json_error_exposes_proxy_hint_without_secret(self):
        reviewer = self.reviewer()
        error = urllib.error.HTTPError(reviewer.endpoint, 400, "bad request", {"Content-Type": "text/html"},
                                      io.BytesIO(b"Proxy rejected request test-only-key"))
        with patch.object(reviewer.opener, "open", side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                reviewer.answer("Return JSON")
        self.assertIn("text/html", str(caught.exception))
        self.assertIn("Proxy rejected", str(caught.exception))
        self.assertNotIn("test-only-key", str(caught.exception))

    def test_downloader_does_not_download_api_reviewer(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as tmp:
            batch = Path(tmp) / "test.sbatch"
            batch.write_text('LLM="deepseek"\nREVIEW_BACKEND="api"\nREVIEW_LLM="provider/large-model"\n')
            self.assertEqual(batch_models(batch), ["deepseek"])

    def test_invalid_response_stops_instead_of_selecting_arbitrarily(self):
        reviewer = self.reviewer()
        response = io.StringIO(json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))
        with patch.object(reviewer.opener, "open", return_value=response):
            with self.assertRaisesRegex(ValueError, "token limit"):
                reviewer.answer("Return JSON")


if __name__ == "__main__":
    unittest.main()
