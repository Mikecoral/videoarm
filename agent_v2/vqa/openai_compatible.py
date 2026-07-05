from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def image_to_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class OpenAICompatibleClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        api = config.get("api", {})
        self.base_url = str(api.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.model = str(api.get("model", "gpt-4o-mini"))
        self.api_key = str(api.get("api_key", ""))
        self.timeout = float(api.get("timeout", 120))
        self.temperature = float(api.get("temperature", 0.2))
        self.max_tokens = int(api.get("max_tokens", 1200))
        self.use_response_format = bool(api.get("use_response_format", True))

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def chat_json(self, system_prompt: str, user_text: str, image_paths: list[str] | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths or []:
            if image_path and Path(image_path).exists():
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc

        message = raw["choices"][0]["message"]
        content_text = message.get("content") or ""
        if isinstance(content_text, list):
            content_text = "".join(part.get("text", "") for part in content_text if isinstance(part, dict))
        try:
            return parse_json_object(str(content_text))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return JSON: {content_text[:500]}") from exc
