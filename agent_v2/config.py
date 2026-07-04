"""LabARM-HV configuration.

Backend is the official Aliyun DashScope OpenAI-compatible endpoint, following
`VideoHV-Agent/video_hv/config.py`.  The model `qwen3.5-27b` is multimodal:
it accepts both text and images, so a single model id serves the controller,
the vision tools, and the structured-output calls.

All values can be overridden with environment variables (``LABARM_*``), which
keeps the checked-in defaults out of the way of graders who supply their own
credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = REPO_ROOT / "hackathon_release"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"


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

# ---------------------------------------------------------------- endpoint --
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

BASE_URL = _env("LABARM_BASE_URL", DEFAULT_BASE_URL)
# Provide the key via env var LABARM_API_KEY / DASHSCOPE_API_KEY, or agent_v2/.env
# (see .env.example). Never commit the real key.
API_KEY = _env("LABARM_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))

# One multimodal model for every role (per project decision).  The three names
# are kept separate so a grader can point a role at a different model later.
DEFAULT_MODEL = _env("LABARM_MODEL", "qwen3.5-27b")
CONTROLLER_MODEL = _env("LABARM_CONTROLLER_MODEL", DEFAULT_MODEL)
VISION_MODEL = _env("LABARM_VISION_MODEL", DEFAULT_MODEL)
STRUCTURED_MODEL = _env("LABARM_STRUCTURED_MODEL", DEFAULT_MODEL)

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

REQUEST_TIMEOUT = float(_env("LABARM_TIMEOUT", "120"))
MAX_RETRIES = int(_env("LABARM_MAX_RETRIES", "4"))
