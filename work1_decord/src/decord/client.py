"""OpenAI-compatible chat client. Failures raise; they are not stored as blank answers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class ChatError(RuntimeError):
    pass


class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        extra_body: dict | None = None,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("API_KEY", "")
        self.timeout_s = timeout_s
        self.extra_body = dict(extra_body or {})

    def request_payload(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_body)
        return payload

    def complete_details(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> dict:
        if not self.api_key:
            raise ChatError("API_KEY is empty")
        if not self.model:
            raise ChatError("model is empty")
        payload = json.dumps(
            self.request_payload(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ChatError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ChatError(f"transport error: {exc.reason}") from exc
        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = message["content"] if isinstance(message, dict) else None
            finish_reason = choice.get("finish_reason") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"response has no message content: {body}") from exc
        return {
            "content": "" if content is None else str(content),
            "finish_reason": str(finish_reason),
        }

    def complete(self, prompt: str, *, temperature: float, top_p: float, max_tokens: int) -> str:
        return self.complete_details(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)["content"]
