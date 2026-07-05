"""Thin OpenAI-compatible chat wrappers for LabARM-HV.

Responsibilities:
- route pure text calls to the configured LLM provider (DeepSeek by default),
- route multimodal image calls to the configured MLLM provider (Qwen by default),
- inject ``enable_thinking=False`` via ``extra_body`` when a provider supports it,
- retry transient failures with exponential back-off,
- accept text-or-image content and return plain text,
- parse a JSON object out of a (possibly fenced) model reply.

For whole-video understanding (Omni pipeline), the native dashscope SDK is used
with ``file:///absolute/path`` URIs so local videos are never uploaded.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dashscope
from dashscope import MultiModalConversation
from openai import OpenAI

from . import config

_llm_client: Optional[OpenAI] = None
_mllm_client: Optional[OpenAI] = None


def llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                             timeout=config.REQUEST_TIMEOUT)
    return _llm_client


def mllm_client() -> OpenAI:
    global _mllm_client
    if _mllm_client is None:
        _mllm_client = OpenAI(base_url=config.MLLM_BASE_URL, api_key=config.MLLM_API_KEY,
                              timeout=config.REQUEST_TIMEOUT)
    return _mllm_client


def client() -> OpenAI:
    """Backward-compatible default client for older imports."""
    return mllm_client()


def chat(messages: List[Dict[str, Any]], *, model: Optional[str] = None,
         temperature: float = 0.2, max_tokens: int = 1200,
         provider: str = "llm") -> str:
    """Run one chat completion and return the assistant text."""
    if provider == "mllm":
        openai_client = mllm_client()
        model = model or config.VISION_MODEL
    else:
        openai_client = llm_client()
        model = model or config.CONTROLLER_MODEL
    extra_body = {"enable_thinking": config.ENABLE_THINKING}
    last_err: Optional[Exception] = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = openai_client.chat.completions.create(
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
    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                provider="llm")


def ask_vision(prompt: str, image_urls: List[str], *, model: Optional[str] = None,
               temperature: float = 0.2, max_tokens: int = 1000,
               system: Optional[str] = None) -> str:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(user_multimodal(prompt, image_urls))
    return chat(messages, model=model or config.VISION_MODEL,
                temperature=temperature, max_tokens=max_tokens, provider="mllm")


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


def _dashscope_video_text(content: Any) -> str:
    """Extract plain text from a dashscope MultiModalConversation response content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()


def ask_video(prompt: str, video_path: Path, *, model: Optional[str] = None,
              fps: float = 1.0, max_tokens: int = 4000,
              system: Optional[str] = None) -> str:
    """Send a whole-video understanding request via the native dashscope SDK.

    Uses ``file:///absolute/path`` so no upload is required for local files.
    ``fps`` controls how densely the model samples frames from the video.
    """
    model = model or config.OMNI_MODEL
    dashscope.api_key = config.MLLM_API_KEY
    video_uri = f"file://{video_path.resolve()}"

    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"video": video_uri, "fps": fps},
            {"text": prompt},
        ],
    })

    last_err: Optional[Exception] = None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = MultiModalConversation.call(
                model=model,
                messages=messages,
                result_format="message",
                max_tokens=max_tokens,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"dashscope error {resp.status_code}: {resp.message} [{resp.code}]")
            return _dashscope_video_text(resp.output.choices[0].message.content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"ask_video failed after {config.MAX_RETRIES} retries: {last_err}")


def ask_video_json(prompt: str, video_path: Path, *, model: Optional[str] = None,
                   fps: float = 1.0, max_tokens: int = 4000,
                   system: Optional[str] = None) -> Any:
    """Ask an Omni model for JSON output given a whole video."""
    reply = ask_video(prompt, video_path, model=model, fps=fps,
                      max_tokens=max_tokens, system=system)
    try:
        return parse_json(reply)
    except ValueError:
        strict = prompt + "\n\n只输出合法 JSON，不要任何解释或 markdown 代码块。"
        reply = ask_video(strict, video_path, model=model, fps=fps,
                          max_tokens=max_tokens, system=system)
        return parse_json(reply)


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
