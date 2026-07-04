"""Thin OpenAI-compatible chat wrapper for DashScope.

Responsibilities:
- inject ``enable_thinking=False`` via ``extra_body`` (DashScope Qwen3 switch),
- retry transient failures with exponential back-off,
- accept text-or-image content and return plain text,
- parse a JSON object out of a (possibly fenced) model reply.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import config

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                         timeout=config.REQUEST_TIMEOUT)
    return _client


def chat(messages: List[Dict[str, Any]], *, model: Optional[str] = None,
         temperature: float = 0.2, max_tokens: int = 1200) -> str:
    """Run one chat completion and return the assistant text."""
    model = model or config.CONTROLLER_MODEL
    extra_body = {"enable_thinking": config.ENABLE_THINKING}
    last_err: Optional[Exception] = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - back off on any transient error
            last_err = e
            msg = str(e).lower()
            # A model that rejects the non-standard switch is retried without it.
            if "enable_thinking" in msg or "extra_body" in msg:
                extra_body = {}
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"chat failed after {config.MAX_RETRIES} retries: {last_err}")


def user_text(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def user_multimodal(text: str, image_urls: List[str]) -> Dict[str, Any]:
    """Build a user message mixing one text prompt with several images."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def ask_text(prompt: str, *, model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 1200,
             system: Optional[str] = None) -> str:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user_text(prompt))
    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)


def ask_vision(prompt: str, image_urls: List[str], *, model: Optional[str] = None,
               temperature: float = 0.2, max_tokens: int = 1000,
               system: Optional[str] = None) -> str:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user_multimodal(prompt, image_urls))
    return chat(messages, model=model or config.VISION_MODEL,
                temperature=temperature, max_tokens=max_tokens)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from a model reply."""
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from model reply: {text[:200]}")


def ask_json(prompt: str, *, model: Optional[str] = None, vision_images: Optional[List[str]] = None,
             temperature: float = 0.1, max_tokens: int = 1600,
             system: Optional[str] = None) -> Any:
    """Ask for JSON; retry once with a stricter nudge if parsing fails."""
    reply = (ask_vision(prompt, vision_images, model=model, temperature=temperature,
                        max_tokens=max_tokens, system=system)
             if vision_images else
             ask_text(prompt, model=model, temperature=temperature,
                      max_tokens=max_tokens, system=system))
    try:
        return parse_json(reply)
    except ValueError:
        strict = prompt + "\n\n只输出合法 JSON，不要任何解释或 markdown 代码块。"
        reply = (ask_vision(strict, vision_images, model=model, temperature=0.0,
                            max_tokens=max_tokens, system=system)
                 if vision_images else
                 ask_text(strict, model=model, temperature=0.0,
                          max_tokens=max_tokens, system=system))
        return parse_json(reply)
