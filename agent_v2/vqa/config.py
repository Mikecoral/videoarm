from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_py_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location(f"vqa_config_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise RuntimeError(f"Config file must define CONFIG dict: {path}")
    return copy.deepcopy(config)


def load_config(config_path: str | Path, local_config_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = read_py_config(config_path)
    if local_config_path is None:
        local_config_path = config_path.with_name("config.local.py")
    config = deep_merge(config, read_py_config(local_config_path))

    api = config.setdefault("api", {})
    if os.environ.get("OPENAI_API_KEY"):
        api["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        api["base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.environ.get("OPENAI_MODEL"):
        api["model"] = os.environ["OPENAI_MODEL"]
    if os.environ.get("VQA_USE_MLLM"):
        config["use_mllm"] = os.environ["VQA_USE_MLLM"].lower() in {"1", "true", "yes", "on"}
    return config

