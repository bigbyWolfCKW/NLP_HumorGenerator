"""Remote chat-completions reviewer; credentials are read only from the environment."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from .environment import load_local_env


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer token to a redirected destination.


class APIReviewer:
    def __init__(self, model, base_url, key_env="REVIEW_API_KEY", max_tokens=1024):
        parsed = urllib.parse.urlsplit(base_url or "")
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Review API base URL must be an HTTPS URL without credentials/query/fragment.")
        if not model or max_tokens < 1:
            raise ValueError("An API model name and positive token limit are required.")
        load_local_env()
        self.key = os.environ.get(key_env, "").strip()
        if not self.key:
            raise ValueError(f"Missing API key: set {key_env} in the private .env file or export it before sbatch.")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.max_tokens = max_tokens
        self.config = {"name": f"api:{model}", "model": model, "backend": "api"}
        self.opener = urllib.request.build_opener(NoRedirect())

    def answer(self, prompt, sample=False):
        payload = {"model": self.config["model"], "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "max_tokens": self.max_tokens}
        if urllib.parse.urlsplit(self.endpoint).hostname == "api.deepseek.com":
            payload.update(thinking={"type": "disabled"}, response_format={"type": "json_object"})
        elif urllib.parse.urlsplit(self.endpoint).hostname == "generativelanguage.googleapis.com":
            payload.update(reasoning_effort="low", response_format={"type": "json_object"})
        headers = {"Content-Type": "application/json"}
        if urllib.parse.urlsplit(self.endpoint).hostname == "hkust.azure-api.net":
            headers["api-key"] = self.key
        else:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with self.opener.open(request, timeout=180) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            # Prefer structured messages; identify empty/non-JSON proxy responses too.
            raw = error.read(65536).decode("utf-8", errors="replace")
            content_type = error.headers.get("Content-Type", "unknown") if error.headers else "unknown"
            try:
                problem = json.loads(raw)
                detail = problem.get("error", problem)
                if isinstance(detail, dict):
                    detail = detail.get("message", "No structured error message.")
                if not isinstance(detail, str):
                    detail = "No structured error message."
            except (ValueError, AttributeError, TypeError):
                detail = f"Non-JSON response ({content_type}, {len(raw)} characters): {raw[:1200] or '[empty body]'}"
            detail = detail.replace(self.key, "[REDACTED]").replace(prompt, "[INPUT]")
            detail = " ".join(detail.split())[:800]
            raise RuntimeError(f"Review API returned HTTP {error.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("Review API connection failed. Check the compute node's outbound network access.") from None
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Review API reached the token limit; raise REVIEW_API_MAX_TOKENS.")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Review API returned no text; check model/output settings.")
            return content.strip()
        except (KeyError, IndexError, TypeError):
            raise ValueError("Review API response has no chat-completions text result.") from None
