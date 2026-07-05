"""LabARM-HV configuration.

The hxa branch separates model routing by capability:

- MLLM / vision calls use DashScope + ``qwen3.5-27b`` by default.
- Pure text / structured reasoning calls use DeepSeek by default.

All values can be overridden with environment variables (``LABARM_*``), which
keeps checked-in source free of credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a gitignored ``agent_v2/.env`` for local runs.

    Secrets (the DashScope API key) live only in this file, never in source.
    Existing environment variables win, so CI/graders can override freely.
    """
    env_file = PACKAGE_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

DATA_ROOT = Path(_env("LABARM_DATA_ROOT", str(REPO_ROOT / "hackathon_release"))).expanduser()
OUTPUT_DIR = Path(_env("LABARM_OUTPUT_DIR", str(PACKAGE_ROOT / "outputs"))).expanduser()

# Frame cache lives outside the agent package so it is never shipped.
# Each video gets a subdirectory named by video_id under this path.
FRAME_CACHE_DIR = Path(_env("LABARM_FRAME_CACHE_DIR", str(REPO_ROOT / "frame_cache"))).expanduser()

# ---------------------------------------------------------------- endpoint --
DEFAULT_MLLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"

# Backward-compatible aliases: old LABARM_BASE_URL/API_KEY/MODEL map to the
# multimodal route.  New code should prefer LABARM_MLLM_* and LABARM_LLM_*.
MLLM_BASE_URL = _env("LABARM_MLLM_BASE_URL", _env("LABARM_BASE_URL", DEFAULT_MLLM_BASE_URL))
MLLM_API_KEY = _env(
    "LABARM_MLLM_API_KEY",
    _env("LABARM_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")),
)
MLLM_MODEL = _env("LABARM_MLLM_MODEL", _env("LABARM_MODEL", "qwen3.5-27b"))

LLM_BASE_URL = _env("LABARM_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
LLM_API_KEY = _env("LABARM_LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
LLM_MODEL = _env("LABARM_LLM_MODEL", "deepseek-chat")

# Compatibility names used by existing pipeline code.
DEFAULT_BASE_URL = MLLM_BASE_URL
BASE_URL = MLLM_BASE_URL
API_KEY = MLLM_API_KEY
DEFAULT_MODEL = MLLM_MODEL
VISION_MODEL = _env("LABARM_VISION_MODEL", MLLM_MODEL)
CONTROLLER_MODEL = _env("LABARM_CONTROLLER_MODEL", LLM_MODEL)
STRUCTURED_MODEL = _env("LABARM_STRUCTURED_MODEL", LLM_MODEL)

# Omni model for direct whole-video understanding (no frame extraction).
OMNI_MODEL = _env("LABARM_OMNI_MODEL", "qwen3.5-omni-plus")
# Frame sampling rate passed to the Omni model (frames per second).
OMNI_FPS = float(_env("LABARM_OMNI_FPS", "1"))

# DashScope Qwen3 exposes a `thinking` switch.  This task wants short, stable,
# parseable output, so thinking is disabled by default (see api_client).
ENABLE_THINKING = _env("LABARM_ENABLE_THINKING", "0") == "1"

# --------------------------------------------------------------- sampling ---
# Frames are sampled at this rate then down-scaled before being base64-encoded.
FRAME_FPS = float(_env("LABARM_FRAME_FPS", "1"))
FRAME_MAX_SIDE = int(_env("LABARM_FRAME_MAX_SIDE", "512"))
FRAME_JPEG_QUALITY = int(_env("LABARM_FRAME_QUALITY", "75"))

# Memory windows: the timeline is split into ~this many scene windows, each
# summarised by one vision call.  Bounded so long videos stay affordable.
MIN_WINDOWS = int(_env("LABARM_MIN_WINDOWS", "6"))
MAX_WINDOWS = int(_env("LABARM_MAX_WINDOWS", "16"))
SECONDS_PER_WINDOW = float(_env("LABARM_SECONDS_PER_WINDOW", "15"))
FRAMES_PER_WINDOW = int(_env("LABARM_FRAMES_PER_WINDOW", "3"))

# Per-segment verification (bounded vision cost).
VERIFY_ENABLED = _env("LABARM_VERIFY", "1") == "1"
VERIFY_FRAMES = int(_env("LABARM_VERIFY_FRAMES", "3"))

# One bounded repair pass, VideoHV-style: failed/low-confidence hypotheses are
# regenerated locally and then verified again before the final writer gate.
REPAIR_ENABLED = _env("LABARM_REPAIR", "1") == "1"
REPAIR_CONFIDENCE_THRESHOLD = float(_env("LABARM_REPAIR_CONFIDENCE", "0.55"))
REPAIR_FRAMES = int(_env("LABARM_REPAIR_FRAMES", "5"))
REPAIR_CONTEXT_SECONDS = float(_env("LABARM_REPAIR_CONTEXT_SECONDS", "2"))

# Output-only audit: flag suspiciously low temporal coverage or long gaps for
# human review without forcing the model to fill waiting/static intervals.
COVERAGE_AUDIT_ENABLED = _env("LABARM_COVERAGE_AUDIT", "1") == "1"
COVERAGE_MIN_RATIO = float(_env("LABARM_COVERAGE_MIN_RATIO", "0.4"))
COVERAGE_MAX_GAP_SECONDS = float(_env("LABARM_COVERAGE_MAX_GAP_SECONDS", "30"))
COVERAGE_BACKTRACK_ENABLED = _env("LABARM_COVERAGE_BACKTRACK", "1") == "1"

# If phase confidence is low, action segmentation receives broader ontology
# context so one mistaken phase does not force all actions into the wrong menu.
PHASE_FALLBACK_CONFIDENCE = float(_env("LABARM_PHASE_FALLBACK_CONFIDENCE", "0.65"))
MAX_PHASE_CANDIDATES = int(_env("LABARM_MAX_PHASE_CANDIDATES", "3"))
# When phase confidence is this high, use only the primary phase's actions to
# prevent alternative phase actions from leaking into the catalog.
PHASE_HIGH_CONF_THRESHOLD = float(_env("LABARM_PHASE_HIGH_CONF", "0.85"))

# Dense boundary resampling: long proposed segments are broken into smaller
# sub-windows and locally re-segmented to detect repeated short actions.
DENSE_ENABLED = _env("LABARM_DENSE", "1") == "1"
DENSE_MIN_DURATION = float(_env("LABARM_DENSE_MIN_DURATION", "14.0"))
DENSE_SUBWINDOW = float(_env("LABARM_DENSE_SUBWINDOW", "5.0"))
DENSE_FRAMES = int(_env("LABARM_DENSE_FRAMES", "4"))

# Hybrid hxa pipeline: MLLM local evidence agents + DeepSeek text audit.
HYBRID_OBJECT_ENABLED = _env("LABARM_HYBRID_OBJECT", "1") == "1"
HYBRID_OBJECT_FRAMES = int(_env("LABARM_HYBRID_OBJECT_FRAMES", "4"))
HYBRID_BOUNDARY_ENABLED = _env("LABARM_HYBRID_BOUNDARY", "1") == "1"
HYBRID_BOUNDARY_CONTEXT_SECONDS = float(_env("LABARM_HYBRID_BOUNDARY_CONTEXT", "3.0"))
HYBRID_BOUNDARY_FRAMES = int(_env("LABARM_HYBRID_BOUNDARY_FRAMES", "7"))
HYBRID_TEXT_AUDIT_ENABLED = _env("LABARM_HYBRID_TEXT_AUDIT", "1") == "1"

REQUEST_TIMEOUT = float(_env("LABARM_TIMEOUT", "120"))
MAX_RETRIES = int(_env("LABARM_MAX_RETRIES", "4"))
