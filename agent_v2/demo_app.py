"""Gradio demo UI for LabARM-HV.

The app is intentionally lightweight: it reads ``outputs/predictions.json``
when available, falls back to a built-in sample, and lets a reviewer inspect
segments, edit one segment, and run a minimal feedback update.
"""

from __future__ import annotations

import base64
import functools
import html
import http.server
import json
import mimetypes
import os
import re
import socket
import sys
import tempfile
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import gradio as gr

try:
    from . import config
    from . import memory as agent_memory
    from . import qa_runtime
except ImportError:  # Allows: python agent_v2/demo_app.py
    import config  # type: ignore
    import memory as agent_memory  # type: ignore
    import qa_runtime  # type: ignore


# Gradio 6 performs a local startup health check.  Some Windows/proxy setups
# route even localhost through HTTP(S)_PROXY, which makes that check fail with
# a 502.  Keep loopback traffic local.
_NO_PROXY_HOSTS = "127.0.0.1,localhost,::1"
for _key in ("NO_PROXY", "no_proxy"):
    current = os.environ.get(_key, "")
    if _NO_PROXY_HOSTS not in current:
        os.environ[_key] = f"{current},{_NO_PROXY_HOSTS}".strip(",")


APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"
DEFAULT_PREDICTIONS = OUTPUTS_DIR / "predictions.json"
GLOBAL_MEMORY = OUTPUTS_DIR / "global_memory.json"
LOCAL_HACKATHON_RELEASE = APP_DIR / "hackathon_release"
FULL_VQA_PACKAGE_DIR = APP_DIR / "test_15vids_vqa_package"
LEGACY_VQA_PACKAGE_DIR = APP_DIR / "vqa_demo_package"
VQA_PACKAGE_DIR = FULL_VQA_PACKAGE_DIR if FULL_VQA_PACKAGE_DIR.exists() else LEGACY_VQA_PACKAGE_DIR
VQA_DATA_PATHS = [
    VQA_PACKAGE_DIR / "vqa" / "vqa_output.json",
    VQA_PACKAGE_DIR / "vqa" / "vqa_output_dev_mllm_filtered.json",
    VQA_PACKAGE_DIR / "vqa" / "vqa_output_dev_mllm.json",
    VQA_PACKAGE_DIR / "vqa" / "vqa_output_dev_template.json",
]
RUNTIME_OUTPUT_DIR = Path(tempfile.gettempdir()) / "labarm_hv_demo"
UPDATED_PREDICTIONS = RUNTIME_OUTPUT_DIR / "predictions_reviewed.json"
FEEDBACK_LOG = RUNTIME_OUTPUT_DIR / "feedback_log.json"
VQA_STATIC_URL = ""
VQA_STATIC_SERVER: http.server.ThreadingHTTPServer | None = None
RELEASE_STATIC_URL = ""
RELEASE_STATIC_SERVER: http.server.ThreadingHTTPServer | None = None

TABLE_HEADERS = [
    "segment_id",
    "start",
    "end",
    "action_zh",
    "objects",
    "caption",
    "confidence",
    "status",
    "review",
]

SAMPLE_VQA_IMAGE = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCA2NDAgMzYwJz48cmVjdCB3aWR0aD0nNjQwJyBoZWlnaHQ9JzM2MCcgZmlsbD0nI2Y4ZmFmYycvPjxyZWN0IHg9Jzc0JyB5PScxMDgnIHdpZHRoPSczMDAnIGhlaWdodD0nMTIwJyByeD0nMTAnIGZpbGw9JyNmZmZmZmYnIHN0cm9rZT0nIzk0YTNiOCcgc3Ryb2tlLXdpZHRoPSc0Jy8+PGxpbmUgeDE9JzExMCcgeTE9JzE4MCcgeDI9JzMzNScgeTI9JzE4MCcgc3Ryb2tlPScjMjU2M2ViJyBzdHJva2Utd2lkdGg9JzYnLz48bGluZSB4MT0nMTEyJyB5MT0nMTUwJyB4Mj0nMzQwJyB5Mj0nMTUwJyBzdHJva2U9JyMwZjE3MmEnIHN0cm9rZS13aWR0aD0nMycvPjxyZWN0IHg9JzQyNCcgeT0nOTYnIHdpZHRoPSczNCcgaGVpZ2h0PScxNzAnIHJ4PSc4JyBmaWxsPScjZGJlYWZlJyBzdHJva2U9JyMyNTYzZWInIHN0cm9rZS13aWR0aD0nNCcvPjxyZWN0IHg9JzQ4OCcgeT0nMTI1JyB3aWR0aD0nNzAnIGhlaWdodD0nMTE2JyByeD0nOCcgZmlsbD0nI2VjZmVmZicgc3Ryb2tlPScjMGY3NjZlJyBzdHJva2Utd2lkdGg9JzQnLz48dGV4dCB4PSc3NCcgeT0nMzEwJyBmb250LWZhbWlseT0nQXJpYWwnIGZvbnQtc2l6ZT0nMjgnIGZpbGw9JyMwZjE3MmEnPlRMQyBwbGF0ZSBiYXNlbGluZSBtYXJraW5nPC90ZXh0Pjwvc3ZnPg=="
)


SAMPLE_PREDICTIONS: List[Dict[str, Any]] = [
    {
        "video_id": "demo_0061",
        "video_path": "videos/0061.mp4",
        "phase": "TLC_analysis",
        "phase_zh": "薄层色谱分析",
        "phase_confidence": 0.88,
        "segments": [
            {
                "segment_id": "demo_0061_s001",
                "start": 0.0,
                "end": 32.9,
                "action": "prepare_TLC_plate",
                "action_zh": "准备TLC板",
                "objects": ["TLC板", "铅笔", "尺子"],
                "caption": "实验人员取出TLC板，并用铅笔和尺子标记基线。",
                "evidence_timestamps": [1, 8, 25],
                "confidence": 0.86,
                "verification_status": "verified",
                "uncertainty_reason": "",
                "needs_human_review": False,
                "object_evidence": [{"object": "TLC板", "present": True}],
            },
            {
                "segment_id": "demo_0061_s002",
                "start": 41.5,
                "end": 64.5,
                "action": "spot_TLC_plate",
                "action_zh": "点样TLC板",
                "objects": ["毛细管", "样品瓶", "TLC板"],
                "caption": "实验人员使用毛细管蘸取样品，并在TLC板基线上点样。",
                "evidence_timestamps": [42, 51, 63],
                "confidence": 0.62,
                "verification_status": "partial",
                "uncertainty_reason": "关键接触动作只在少数帧中可见，建议人工复核。",
                "needs_human_review": True,
                "object_evidence": [{"object": "毛细管", "present": True}],
            },
            {
                "segment_id": "demo_0061_s003",
                "start": 88.0,
                "end": 112.0,
                "action": "develop_TLC_plate",
                "action_zh": "展开TLC板",
                "objects": ["展开缸", "TLC板"],
                "caption": "实验人员将TLC板放入展开容器中，使溶剂沿板面上升。",
                "evidence_timestamps": [91, 99, 108],
                "confidence": 0.41,
                "verification_status": "rejected",
                "uncertainty_reason": "画面遮挡较重，无法确认TLC板是否已经进入展开缸。",
                "needs_human_review": True,
                "object_evidence": [{"object": "展开缸", "present": True}],
            },
        ],
        "vqa": [
            {
                "vqa_id": "demo_0061_q001",
                "level": "perception",
                "question": "视频开头使用了什么工具标记TLC板？",
                "answer": "使用铅笔和尺子标记TLC板基线。",
                "evidence_timestamps": [1, 8],
                "question_image": SAMPLE_VQA_IMAGE,
                "segment_id": "demo_0061_s001",
            }
        ],
        "clip_memory": [
            {"start": 0.0, "end": 32.9, "summary": "桌面上出现TLC板、铅笔和尺子，正在标记基线。"},
            {"start": 41.5, "end": 64.5, "summary": "手部拿起毛细管靠近样品容器和TLC板。"},
        ],
        "processing_note": "Built-in sample for UI demonstration.",
    }
]


CSS = """
:root {
  --lab-bg: #f5f7fb;
  --lab-card: #ffffff;
  --lab-border: #dfe5ef;
  --lab-border-soft: #edf1f7;
  --lab-text: #152033;
  --lab-muted: #64748b;
  --lab-blue: #2563eb;
  --lab-blue-soft: #eff6ff;
  --lab-amber: #b7791f;
  --lab-amber-soft: #fff7ed;
  --lab-green: #0f766e;
}

body,
.gradio-container {
  background: var(--lab-bg) !important;
  color: var(--lab-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 22px 28px 28px !important;
}

footer {display: none !important;}

.lab-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 24px;
  margin-bottom: 18px;
}

.lab-title {
  margin: 0;
  font-size: 30px;
  line-height: 1.12;
  letter-spacing: 0;
  color: #0f172a;
}

.lab-subtitle {
  margin: 8px 0 0;
  color: var(--lab-muted);
  font-size: 14px;
}

.lab-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 8px 12px;
  background: var(--lab-blue-soft);
  color: #1d4ed8;
  font-size: 13px;
  white-space: nowrap;
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 0 0 10px;
  color: #0f172a;
  font-size: 15px;
  font-weight: 700;
}

.section-title span {
  color: var(--lab-muted);
  font-size: 12px;
  font-weight: 500;
}

.card {
  border: 1px solid var(--lab-border) !important;
  border-radius: 8px !important;
  background: var(--lab-card) !important;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04) !important;
  padding: 14px !important;
}

.gradio-container .top-card {
  height: 790px !important;
  min-height: 790px !important;
  max-height: 790px !important;
  overflow: hidden !important;
}

.gradio-container #top_left.top-card {
  height: auto !important;
  min-height: 640px !important;
  max-height: none !important;
  overflow: visible !important;
}

.gradio-container .bottom-card {
  height: auto !important;
  min-height: 560px !important;
  max-height: none !important;
  overflow: visible !important;
}

.top-card > div {
  height: 100% !important;
}

#top_left,
#top_right,
#bottom_left,
#bottom_right {
  height: auto !important;
}

#top_right .vqa-panel {
  height: 320px !important;
  overflow: auto !important;
}

#top_right .qa-panel {
  height: 105px !important;
  overflow: auto !important;
}

#top_right,
#bottom_right {
  overflow-y: auto !important;
}

#segment_table,
#segment_table > div,
#segment_table .wrap {
  max-height: 240px !important;
}

#segment_table .table-wrap,
#segment_table [data-testid="dataframe"],
#segment_table [data-testid="dataframe"] > div {
  max-height: 240px !important;
  overflow-y: auto !important;
  overflow-x: auto !important;
}

#segment_table table {
  min-width: 980px !important;
}

#bottom_left textarea {
  min-height: 74px !important;
}

#bottom_left .caption-field textarea,
#bottom_left .uncertainty-field textarea {
  min-height: 74px !important;
}

#bottom_left .block,
#bottom_left .form {
  margin-bottom: 10px !important;
}

#bottom_right textarea {
  min-height: 72px !important;
}

#bottom_right .export-path textarea {
  min-height: 44px !important;
}

.tight-card {
  border: 1px solid var(--lab-border-soft) !important;
  border-radius: 8px !important;
  background: var(--lab-card) !important;
  padding: 12px !important;
}

.video-card video,
.video-card .wrap {
  border-radius: 6px !important;
  background: #f8fafc !important;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.metric-card {
  border: 1px solid var(--lab-border-soft);
  border-radius: 8px;
  padding: 12px 14px;
  background: #fbfdff;
}

.metric-card strong {
  display: block;
  font-size: 24px;
  line-height: 1;
  color: #0f172a;
}

.metric-card span {
  display: block;
  margin-top: 6px;
  color: var(--lab-muted);
  font-size: 12px;
}

.status-strip {
  border: 1px solid #dbeafe;
  border-left: 4px solid var(--lab-blue);
  padding: 12px 14px;
  background: #f8fbff;
  border-radius: 8px;
}

.status-strip b {
  color: #0f172a;
}

.review-box {
  border: 1px solid #fed7aa;
  background: var(--lab-amber-soft);
  color: #7c2d12;
  border-radius: 8px;
  padding: 12px 14px;
}

.small-note {
  color: var(--lab-muted);
  font-size: 12px;
}

.gradio-dataframe {
  border: 1px solid var(--lab-border) !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

.gradio-dataframe table {
  font-size: 13px !important;
}

.gradio-dataframe th {
  background: #f8fafc !important;
  color: #334155 !important;
  font-weight: 700 !important;
}

.gradio-dataframe td {
  vertical-align: middle !important;
}

textarea,
input,
.input-container,
.wrap {
  border-radius: 6px !important;
}

button.primary {
  border-radius: 6px !important;
  background: var(--lab-blue) !important;
}

.secondary-button button {
  border-radius: 6px !important;
  border: 1px solid var(--lab-border) !important;
  background: #ffffff !important;
  color: #0f172a !important;
}

.vqa-panel {
  min-height: 150px;
}

.vqa-bank {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.vqa-card,
.qa-answer {
  border: 1px solid var(--lab-border-soft);
  border-radius: 8px;
  background: #ffffff;
  padding: 10px;
}

.vqa-question {
  color: #0f172a;
  font-weight: 700;
  line-height: 1.45;
}

.vqa-answer {
  margin-top: 6px;
  color: #334155;
  line-height: 1.5;
}

.vqa-images {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-bottom: 8px;
}

.vqa-images img {
  width: 100%;
  height: 90px;
  object-fit: contain;
  border: 1px solid var(--lab-border-soft);
  border-radius: 6px;
  background: #f8fafc;
}

.vqa-media {
  margin-bottom: 8px;
}

.vqa-media img,
.vqa-media video {
  width: 100%;
  max-height: 220px;
  object-fit: contain;
  border: 1px solid var(--lab-border-soft);
  border-radius: 6px;
  background: #0f172a;
}

.vqa-media img {
  background: #f8fafc;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.info-chip {
  display: inline-flex;
  align-items: center;
  border: 1px solid #dbeafe;
  border-radius: 999px;
  padding: 3px 8px;
  background: var(--lab-blue-soft);
  color: #1e40af;
  font-size: 12px;
}

.qa-evidence {
  border-left: 3px solid var(--lab-blue);
  margin-top: 8px;
  padding: 8px 10px;
  background: #f8fbff;
  border-radius: 6px;
}
}

.export-path textarea {
  min-height: 48px !important;
}

.feedback-card textarea {
  min-height: 92px !important;
}

.form-card label span {
  color: #475569 !important;
  font-size: 12px !important;
  font-weight: 650 !important;
}

.subsection-divider {
  border-top: 1px solid var(--lab-border-soft);
  margin: 14px 0 12px;
  padding-top: 12px;
}

.qa-answer p {
  margin: 0 0 8px;
}

.qa-hint {
  border: 1px solid #dbeafe;
  border-left: 4px solid var(--lab-blue);
  border-radius: 8px;
  background: #f8fbff;
  color: #334155;
  line-height: 1.5;
  padding: 10px 12px;
  margin-bottom: 10px;
}

.qa-output-title {
  color: #0f172a;
  font-size: 14px;
  font-weight: 700;
  margin: 10px 0 8px;
}

#qa_full {
  margin-top: 14px !important;
}

#qa_full textarea {
  min-height: 84px !important;
}

#qa_full .qa-panel {
  min-height: 180px !important;
  overflow: auto !important;
}
"""


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def latest_predictions_path(default_path: Path = DEFAULT_PREDICTIONS) -> Path | None:
    if default_path.exists():
        return default_path
    if not OUTPUTS_DIR.exists():
        return None
    candidates = [
        item / "predictions.json"
        for item in OUTPUTS_DIR.iterdir()
        if item.is_dir() and (item / "predictions.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def attach_prediction_source(predictions: List[Dict[str, Any]], path: Path) -> List[Dict[str, Any]]:
    run_dir = path.parent
    tagged = deepcopy(predictions)
    for item in tagged:
        item["_prediction_path"] = str(path)
        item["_run_dir"] = str(run_dir)
    return tagged


def load_external_vqa_items() -> List[Dict[str, Any]]:
    for path in VQA_DATA_PATHS:
        data = read_json(path, None)
        if isinstance(data, dict) and isinstance(data.get("vqa"), list):
            items = deepcopy(data["vqa"])
            for item in items:
                item["_vqa_source"] = str(path)
            return items
    return []


def vqa_index_by_video() -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for item in load_external_vqa_items():
        video_id = str(item.get("video_id", "")).strip()
        if video_id:
            index.setdefault(video_id, []).append(item)
    return index


def segment_from_vqa(item: Dict[str, Any]) -> Dict[str, Any] | None:
    background = item.get("background") or {}
    if not isinstance(background, dict):
        return None
    segment_id = item.get("segment_id") or background.get("segment_id")
    if not segment_id:
        return None
    time_range = item.get("time_range") or {}
    return {
        "segment_id": segment_id,
        "start": background.get("start", time_range.get("start", 0)),
        "end": background.get("end", time_range.get("end", 0)),
        "action": background.get("action", ""),
        "action_zh": background.get("action_zh", ""),
        "objects": [],
        "caption": background.get("segment_caption", ""),
        "evidence_timestamps": item.get("evidence_timestamps") or [],
        "confidence": 1.0,
        "verification_status": "vqa_source",
        "uncertainty_type": "none",
        "uncertainty_reason": "",
        "needs_human_review": False,
    }


def build_vqa_prediction(video_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = items[0] if items else {}
    background = first.get("background") or {}
    video_path = background.get("video_path") or (first.get("visual") or {}).get("video_path") or f"videos/{video_id}.mp4"
    segments_by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        segment = segment_from_vqa(item)
        if segment and str(segment["segment_id"]) not in segments_by_id:
            segments_by_id[str(segment["segment_id"])] = segment
    segments = sorted(
        segments_by_id.values(),
        key=lambda segment: float(segment.get("start", 0) or 0),
    )
    return {
        "video_id": video_id,
        "video_path": video_path,
        "phase": background.get("phase", "vqa_demo"),
        "phase_zh": background.get("phase_zh", "VQA 数据演示"),
        "phase_confidence": 1.0,
        "segments": segments,
        "vqa": deepcopy(items),
        "clip_memory": [
            {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "summary": segment.get("caption", ""),
            }
            for segment in segments
        ],
        "processing_note": "Loaded from vqa_demo_package.",
    }


def attach_external_vqa(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = vqa_index_by_video()
    if not index:
        return predictions

    merged = deepcopy(predictions)
    existing_ids = {str(item.get("video_id")) for item in merged}
    for prediction in merged:
        video_id = str(prediction.get("video_id", "")).strip()
        external_items = index.get(video_id, [])
        if not external_items:
            continue
        existing = prediction.get("vqa") or []
        existing_keys = {str(item.get("id") or item.get("vqa_id")) for item in existing if isinstance(item, dict)}
        additions = [
            item for item in external_items
            if str(item.get("id") or item.get("vqa_id")) not in existing_keys
        ]
        prediction["vqa"] = deepcopy(existing) + deepcopy(additions)

    for video_id, items in index.items():
        if video_id not in existing_ids:
            merged.append(build_vqa_prediction(video_id, items))
    return merged


def load_predictions(path: Path = DEFAULT_PREDICTIONS) -> List[Dict[str, Any]]:
    pred_path = latest_predictions_path(path)
    if pred_path and pred_path.exists():
        try:
            data = json.loads(pred_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return attach_external_vqa(attach_prediction_source(data, pred_path))
        except json.JSONDecodeError:
            pass
    return attach_external_vqa(deepcopy(SAMPLE_PREDICTIONS))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def video_options(predictions: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("video_id", f"video_{i}")) for i, item in enumerate(predictions)]


def find_prediction(predictions: List[Dict[str, Any]], video_id: str) -> Dict[str, Any]:
    for item in predictions:
        if str(item.get("video_id")) == str(video_id):
            return item
    return predictions[0] if predictions else deepcopy(SAMPLE_PREDICTIONS[0])


def prediction_run_dir(prediction: Dict[str, Any]) -> Path | None:
    raw = prediction.get("_run_dir")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def load_video_memory(prediction: Dict[str, Any]) -> Dict[str, Any] | None:
    video_id = str(prediction.get("video_id", ""))
    run_dir = prediction_run_dir(prediction)
    candidates: List[Path] = []
    if run_dir:
        candidates.append(run_dir / "memory" / f"{video_id}.json")
    candidates.extend(
        [
            OUTPUTS_DIR / "memory" / f"{video_id}.json",
            RUNTIME_OUTPUT_DIR / "memory" / f"{video_id}.json",
        ]
    )
    for candidate in candidates:
        data = read_json(candidate, None)
        if isinstance(data, dict) and data.get("video_id"):
            return data

    if prediction.get("segments") or prediction.get("clip_memory"):
        try:
            return agent_memory.build_video_memory(prediction, None, OUTPUTS_DIR / "frames_cache")
        except Exception:
            return None
    return None


def load_global_memory(prediction: Dict[str, Any]) -> Dict[str, Any] | None:
    run_dir = prediction_run_dir(prediction)
    candidates: List[Path] = []
    if run_dir:
        candidates.extend([run_dir / "global_memory_snapshot.json", run_dir.parent / "global_memory.json"])
    candidates.append(GLOBAL_MEMORY)
    for candidate in candidates:
        data = read_json(candidate, None)
        if isinstance(data, dict) and data:
            return data
    return None


def resolve_video_path(prediction: Dict[str, Any]) -> str | None:
    raw = prediction.get("video_path")
    if not raw:
        return None
    video_id = str(prediction.get("video_id", "")).strip()
    candidates = [
        Path(raw),
        config.DATA_ROOT / raw,
        LOCAL_HACKATHON_RELEASE / raw,
        APP_DIR.parent / raw,
        APP_DIR / raw,
    ]
    if video_id:
        candidates.extend(
            [
                LOCAL_HACKATHON_RELEASE / "videos" / f"{video_id}.mp4",
                config.DATA_ROOT / "videos" / f"{video_id}.mp4",
                APP_DIR.parent / "hackathon_release" / "videos" / f"{video_id}.mp4",
                APP_DIR / "videos" / f"{video_id}.mp4",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def resolve_asset_path(raw: Any, prediction: Dict[str, Any] | None = None) -> Path | None:
    if not raw or not isinstance(raw, str):
        return None
    if raw.startswith(("http://", "https://", "data:image/")):
        return None
    normalized = raw.replace("\\", "/")
    candidates = [
        Path(raw),
        config.DATA_ROOT / raw,
        APP_DIR.parent / raw,
        APP_DIR / raw,
    ]
    for marker, base in [
        ("outputs/frames_cache/", LOCAL_HACKATHON_RELEASE / "frames"),
        ("frames_cache/", LOCAL_HACKATHON_RELEASE / "frames"),
        ("hackathon_release/frames/", LOCAL_HACKATHON_RELEASE / "frames"),
        ("hackathon_release/frames/", VQA_PACKAGE_DIR / "frames"),
        ("hackathon_release/vqa/clips/", VQA_PACKAGE_DIR / "vqa" / "clips"),
        ("frames/", LOCAL_HACKATHON_RELEASE / "frames"),
        ("frames/", VQA_PACKAGE_DIR / "frames"),
        ("vqa/clips/", VQA_PACKAGE_DIR / "vqa" / "clips"),
    ]:
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            candidates.append(base / Path(suffix))
    if prediction and prediction.get("video_path"):
        video_dir = Path(str(prediction["video_path"])).parent
        candidates.extend(
            [
                video_dir / raw,
                config.DATA_ROOT / video_dir / raw,
                APP_DIR.parent / video_dir / raw,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def image_src(raw: Any, prediction: Dict[str, Any] | None = None) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    if raw.startswith(("http://", "https://", "data:image/")):
        return raw
    path = resolve_asset_path(raw, prediction)
    if not path:
        return ""
    try:
        if path.resolve().is_relative_to(VQA_PACKAGE_DIR.resolve()):
            start_vqa_static_server()
            static_url = vqa_static_file_url(path)
            if static_url:
                return static_url
        if path.resolve().is_relative_to(LOCAL_HACKATHON_RELEASE.resolve()):
            static_url = release_static_file_url(path)
            if static_url:
                return static_url
    except ValueError:
        pass
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start_vqa_static_server(preferred_port: int = 7872) -> str:
    global VQA_STATIC_URL, VQA_STATIC_SERVER
    if VQA_STATIC_URL:
        return VQA_STATIC_URL
    if not VQA_PACKAGE_DIR.exists():
        return ""

    port = preferred_port
    while port < preferred_port + 20 and not is_port_available(port):
        port += 1
    if port >= preferred_port + 20:
        return ""

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(VQA_PACKAGE_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    VQA_STATIC_SERVER = server
    VQA_STATIC_URL = f"http://127.0.0.1:{port}"
    return VQA_STATIC_URL


def start_release_static_server(preferred_port: int = 7892) -> str:
    global RELEASE_STATIC_URL, RELEASE_STATIC_SERVER
    if RELEASE_STATIC_URL:
        return RELEASE_STATIC_URL
    if not LOCAL_HACKATHON_RELEASE.exists():
        return ""

    port = preferred_port
    while port < preferred_port + 20 and not is_port_available(port):
        port += 1
    if port >= preferred_port + 20:
        return ""

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(LOCAL_HACKATHON_RELEASE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    RELEASE_STATIC_SERVER = server
    RELEASE_STATIC_URL = f"http://127.0.0.1:{port}"
    return RELEASE_STATIC_URL


def vqa_static_file_url(path: Path | None) -> str:
    if not path or not VQA_STATIC_URL:
        return ""
    try:
        relative = path.resolve().relative_to(VQA_PACKAGE_DIR.resolve()).as_posix()
    except ValueError:
        return ""
    return f"{VQA_STATIC_URL}/{quote(relative, safe='/')}"


def release_static_file_url(path: Path | None) -> str:
    if not path:
        return ""
    static_url = start_release_static_server()
    if not static_url:
        return ""
    try:
        relative = path.resolve().relative_to(LOCAL_HACKATHON_RELEASE.resolve()).as_posix()
    except ValueError:
        return ""
    return f"{static_url}/{quote(relative, safe='/')}"


def file_url(path: Path | None) -> str:
    if not path:
        return ""
    release_url = release_static_file_url(path)
    if release_url:
        return release_url
    static_url = vqa_static_file_url(path)
    if static_url:
        return static_url
    return "/file=" + path.resolve().as_posix()


def media_file_url(raw: Any, prediction: Dict[str, Any] | None = None) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    if raw.startswith(("http://", "https://", "data:")):
        return raw
    return file_url(resolve_asset_path(raw, prediction))


def usable_media_path(raw: Any, prediction: Dict[str, Any] | None = None,
                      min_bytes: int = 1024) -> Path | None:
    path = resolve_asset_path(raw, prediction)
    if not path:
        return None
    try:
        if path.stat().st_size < min_bytes:
            return None
    except OSError:
        return None
    return path


def vqa_viewer_link() -> str:
    viewer = VQA_PACKAGE_DIR / "vqa" / "viewer.html"
    if not viewer.exists():
        return ""
    static_url = start_vqa_static_server()
    if static_url:
        cache_key = int(viewer.stat().st_mtime)
        package_name = quote(VQA_PACKAGE_DIR.name, safe="")
        return f"{static_url}/vqa/viewer.html?package={package_name}&v={cache_key}"
    return viewer.as_uri()


def first_present(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value:
            return value
    return None


def collect_vqa_images(item: Dict[str, Any], prediction: Dict[str, Any]) -> List[str]:
    single_keys = [
        "image_path",
        "question_image",
        "question_image_path",
        "image",
        "frame_path",
        "question_frame",
        "evidence_image",
        "thumbnail",
    ]
    list_keys = [
        "image_paths",
        "question_images",
        "question_frames",
        "evidence_images",
        "evidence_frames",
        "frames",
    ]
    values: List[Any] = []
    values.extend(item.get(key) for key in single_keys if item.get(key))
    for key in list_keys:
        value = item.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    visual = item.get("visual")
    if isinstance(visual, dict):
        values.extend(
            visual.get(key)
            for key in ["image_path", "frame_path", "question_image", "thumbnail"]
            if visual.get(key)
        )
    background = item.get("background")
    if isinstance(background, dict):
        keyframes = background.get("keyframes") or []
        if isinstance(keyframes, list):
            values.extend(keyframes)

    srcs: List[str] = []
    for value in values:
        if isinstance(value, dict):
            value = first_present(value, ["path", "image_path", "frame_path", "url"])
        src = image_src(value, prediction)
        if src and src not in srcs:
            srcs.append(src)
    return srcs[:3]


def fmt_objects(objects: Any) -> str:
    if isinstance(objects, list):
        return "、".join(str(o) for o in objects)
    return str(objects or "")


def parse_objects(text: str) -> List[str]:
    normalized = text.replace(",", "、").replace("，", "、").replace("/", "、")
    return [part.strip() for part in normalized.split("、") if part.strip()]


def segment_rows(prediction: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for segment in prediction.get("segments", []) or []:
        confidence = float(segment.get("confidence", 0) or 0)
        needs_review = bool(segment.get("needs_human_review"))
        uncertainty_type = segment.get("uncertainty_type") or "none"
        rows.append(
            [
                segment.get("segment_id", ""),
                segment.get("start", 0),
                segment.get("end", 0),
                segment.get("action_zh") or segment.get("action", ""),
                fmt_objects(segment.get("objects")),
                segment.get("caption", ""),
                round(confidence, 2),
                segment.get("verification_status", "unverified"),
                f"Needs review · {uncertainty_type}" if needs_review else f"OK · {uncertainty_type}",
            ]
        )
    return rows


def summary_markdown(prediction: Dict[str, Any]) -> str:
    segments = prediction.get("segments", []) or []
    review_count = sum(1 for s in segments if s.get("needs_human_review"))
    avg_conf = 0.0
    if segments:
        avg_conf = sum(float(s.get("confidence", 0) or 0) for s in segments) / len(segments)
    return f"""
<div class="status-strip">
  <b>{prediction.get("video_id", "-")}</b> 路 {prediction.get("phase_zh") or prediction.get("phase") or "Unknown phase"}
  <br><span class="small-note">Phase confidence: {float(prediction.get("phase_confidence", 0) or 0):.2f}</span>
</div>
<div class="metric-grid" style="margin-top:10px;">
  <div class="metric-card"><strong>{len(segments)}</strong><br><span>Segments</span></div>
  <div class="metric-card"><strong>{review_count}</strong><br><span>Need review</span></div>
  <div class="metric-card"><strong>{avg_conf:.2f}</strong><br><span>Avg confidence</span></div>
</div>
"""


def review_markdown(prediction: Dict[str, Any]) -> str:
    items = review_segments(prediction, strict=True)
    if not items:
        return "<div class='review-box'>No urgent review items. All visible segments are reasonably confident.</div>"
    lines = ["<div class='review-box'><b>Priority review queue</b><br>"]
    for s in items[:6]:
        suggestion = s.get("review_suggestion") or s.get("uncertainty_reason") or ""
        lines.append(
            f"{s.get('segment_id')} 路 {s.get('start')}s-{s.get('end')}s 路 "
            f"{s.get('action_zh') or s.get('action')} 路 conf {float(s.get('confidence', 0) or 0):.2f}"
            f" 路 {html.escape(str(s.get('uncertainty_type', 'none')))}"
            f"<br><span class='small-note'>{html.escape(str(suggestion))}</span><br>"
        )
    lines.append("</div>")
    return "".join(lines)


def segment_review_flags(segment: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    confidence = float(segment.get("confidence", 0) or 0)
    status = str(segment.get("verification_status", "") or "").lower()
    uncertainty_type = str(segment.get("uncertainty_type", "") or "").lower()
    if segment.get("needs_human_review"):
        flags.append("human review")
    if confidence < 0.65:
        flags.append(f"low confidence {confidence:.2f}")
    if status in {"partial", "rejected", "unverified"}:
        flags.append(f"status {status}")
    if uncertainty_type and uncertainty_type != "none":
        flags.append(f"{uncertainty_type} uncertainty")
    return flags


def review_segments(prediction: Dict[str, Any], strict: bool = False) -> List[Dict[str, Any]]:
    items = []
    for segment in prediction.get("segments", []) or []:
        flags = segment_review_flags(segment)
        if not flags:
            continue
        if strict and not (segment.get("needs_human_review") or float(segment.get("confidence", 0) or 0) < 0.65):
            continue
        enriched = deepcopy(segment)
        enriched["_review_flags"] = flags
        items.append(enriched)
    return sorted(
        items,
        key=lambda s: (
            not bool(s.get("needs_human_review")),
            float(s.get("confidence", 0) or 0),
            float(s.get("start", 0) or 0),
        ),
    )


def diagnosis_note(segment: Dict[str, Any]) -> str:
    parts: List[str] = []
    uncertainty = str(segment.get("uncertainty_reason", "") or "").strip()
    if uncertainty:
        parts.append(uncertainty)
    utype = segment.get("uncertainty_type")
    if utype:
        parts.append(f"Uncertainty type: {utype}")
    suggestion = str(segment.get("review_suggestion", "") or "").strip()
    if suggestion:
        parts.append(f"Review suggestion: {suggestion}")
    diagnosis = segment.get("evidence_diagnosis") or {}
    if isinstance(diagnosis, dict) and diagnosis:
        checks = []
        for key, label in [
            ("perception_ok", "perception"),
            ("temporal_ok", "temporal"),
            ("action_ok", "action"),
            ("caption_ok", "caption"),
        ]:
            if key in diagnosis:
                checks.append(f"{label}={diagnosis.get(key)}")
        missing = diagnosis.get("missing_evidence") or []
        if checks:
            parts.append("Evidence diagnosis: " + ", ".join(checks))
        if missing:
            parts.append("Missing evidence: " + "；".join(str(x) for x in missing))
    return "\n".join(parts)


def segment_detail(prediction: Dict[str, Any], index: int) -> Tuple[Any, ...]:
    segments = prediction.get("segments", []) or []
    if not segments:
        return "", 0, 0, "", "", "", "", "", False
    index = max(0, min(index, len(segments) - 1))
    s = segments[index]
    return (
        s.get("segment_id", ""),
        float(s.get("start", 0) or 0),
        float(s.get("end", 0) or 0),
        s.get("action", ""),
        s.get("action_zh", ""),
        fmt_objects(s.get("objects")),
        s.get("caption", ""),
        diagnosis_note(s),
        bool(s.get("needs_human_review")),
    )


def vqa_markdown(prediction: Dict[str, Any]) -> str:
    vqa = prediction.get("vqa") or []
    if not vqa:
        return "No VQA items yet. You can add this after the ExpVid-style generator is connected."
    lines = []
    for item in vqa:
        ts = ", ".join(str(t) for t in item.get("evidence_timestamps", []))
        lines.append(
            f"**{item.get('level', 'vqa')}** 路 {item.get('question', '')}\n\n"
            f"{item.get('answer', '')}\n\n"
            f"`evidence: {ts}`"
        )
    return "\n\n---\n\n".join(lines)


def chip(label: str, value: Any) -> str:
    if value in (None, "", []):
        return ""
    return f"<span class='info-chip'>{html.escape(label)}: {html.escape(str(value))}</span>"


def format_time_range(value: Any) -> str:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is not None and end is not None:
            return f"{start}s-{end}s"
    return ""


def vqa_options_html(item: Dict[str, Any]) -> str:
    multiple_choice = item.get("multiple_choice") or {}
    options = multiple_choice.get("options") if isinstance(multiple_choice, dict) else None
    answer = multiple_choice.get("answer") if isinstance(multiple_choice, dict) else None
    if not isinstance(options, dict) or not options:
        return ""
    rows = []
    for key, value in options.items():
        marker = " ✓" if answer and str(key) == str(answer) else ""
        rows.append(
            "<span class='info-chip'>"
            f"{html.escape(str(key))}. {html.escape(str(value))}{marker}"
            "</span>"
        )
    return "<div class='chip-row'>" + "".join(rows) + "</div>"


def vqa_selected_frame(item: Dict[str, Any]) -> Dict[str, Any] | None:
    visual = item.get("visual") or {}
    background = item.get("background") or {}
    frames = background.get("keyframes") if isinstance(background, dict) else []
    if not isinstance(visual, dict) or not isinstance(frames, list) or not frames:
        return None
    keyframe_index = visual.get("keyframe_index")
    if isinstance(keyframe_index, int) and 0 <= keyframe_index < len(frames):
        frame = frames[keyframe_index]
        return frame if isinstance(frame, dict) else None
    return None


def vqa_primary_media_html(item: Dict[str, Any], prediction: Dict[str, Any], index: int) -> str:
    visual = item.get("visual") or {}
    background = item.get("background") or {}
    modality = item.get("visual_modality") or (visual.get("modality") if isinstance(visual, dict) else "") or "video"
    selected_frame = vqa_selected_frame(item)

    if modality == "image":
        raw_image = ""
        if isinstance(visual, dict):
            raw_image = visual.get("image_path") or ""
        if not raw_image and selected_frame:
            raw_image = str(selected_frame.get("frame_path") or "")
        src = image_src(raw_image, prediction)
        if not src:
            return "<div class='qa-answer'>No image available for this VQA item.</div>"
        return (
            "<div class='vqa-media'>"
            f"<img src='{src}' alt='VQA question image {index}' />"
            "</div>"
        )

    clip_path = visual.get("clip_path") if isinstance(visual, dict) else ""
    clip_file = usable_media_path(str(clip_path), prediction) if clip_path else None
    video_src = file_url(clip_file) if clip_file else ""
    if not video_src and isinstance(background, dict) and background.get("video_path"):
        source_path = resolve_video_path({"video_path": background.get("video_path"), "video_id": item.get("video_id")})
        if source_path:
            video_src = file_url(Path(source_path))
            time_range = item.get("time_range") or {}
            start = time_range.get("start")
            end = time_range.get("end")
            if start is not None and end is not None:
                video_src = f"{video_src}#t={start},{end}"
    if not video_src:
        return "<div class='qa-answer'>No video clip available for this VQA item.</div>"
    return (
        "<div class='vqa-media'>"
        f"<video controls preload='metadata' src='{video_src}'></video>"
        "</div>"
    )


def vqa_markdown(prediction: Dict[str, Any]) -> str:
    vqa = prediction.get("vqa") or []
    viewer = vqa_viewer_link()
    viewer_label = (
        "Open full 15-video VQA viewer"
        if VQA_PACKAGE_DIR.name == "test_15vids_vqa_package"
        else "Open full VQA viewer"
    )
    viewer_html = (
        "<div class='chip-row' style='margin-bottom:8px;'>"
        f"<a class='info-chip' href='{html.escape(viewer)}' target='_blank' rel='noreferrer'>{html.escape(viewer_label)}</a>"
        f"{chip('source', VQA_PACKAGE_DIR.name)}"
        "</div>"
        if viewer else ""
    )
    if not vqa:
        return (
            viewer_html
            + "<div class='qa-answer'>No VQA items for this video yet. Generated VQA will appear here when a matching video_id exists in vqa_demo_package.</div>"
        )
    cards = []
    for index, item in enumerate(vqa[:24], start=1):
        question = item.get("question") or item.get("q") or ""
        answer = item.get("answer") or item.get("a") or ""
        ts = ", ".join(str(t) for t in item.get("evidence_timestamps", []) or [])
        level = item.get("level") or item.get("type") or "vqa"
        category = item.get("category") or item.get("question_type")
        modality = item.get("visual_modality") or (item.get("visual") or {}).get("modality")
        time_range = format_time_range(item.get("time_range"))
        segment_id = item.get("segment_id") or item.get("related_segment") or item.get("related_segment_id")
        media_html = vqa_primary_media_html(item, prediction, index)
        meta = "".join(
            [
                chip("level", level),
                chip("category", category),
                chip("modality", modality),
                chip("time", time_range),
                chip("evidence", ts),
                chip("segment", segment_id),
                chip("confidence", item.get("confidence")),
            ]
        )
        options_html = vqa_options_html(item)
        cards.append(
            "<div class='vqa-card'>"
            f"{media_html}"
            f"<div class='vqa-question'>Q{index}. {html.escape(str(question))}</div>"
            f"<div class='vqa-answer'>{html.escape(str(answer))}</div>"
            f"{options_html}"
            f"<div class='chip-row'>{meta}</div>"
            "</div>"
        )
    if len(vqa) > 24:
        cards.append(
            "<div class='qa-answer'>"
            f"Showing 24 of {len(vqa)} VQA items for this video."
            "</div>"
        )
    return viewer_html + "<div class='vqa-bank'>" + "".join(cards) + "</div>"


def segment_line(segment: Dict[str, Any]) -> str:
    return (
        f"<b>{html.escape(str(segment.get('start', 0)))}s-{html.escape(str(segment.get('end', 0)))}s</b> "
        f"{html.escape(str(segment.get('action_zh') or segment.get('action') or 'unknown'))}"
        f"<br><span class='small-note'>{html.escape(str(segment.get('caption', '')))}</span>"
    )


def matching_segments(question: str, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = question.lower()
    matches = []
    for segment in prediction.get("segments", []) or []:
        text = " ".join(
            str(field).lower()
            for field in [
                segment.get("segment_id", ""),
                segment.get("action", ""),
                segment.get("action_zh", ""),
                segment.get("caption", ""),
                fmt_objects(segment.get("objects")),
            ]
        )
        if any(token and token in text for token in re.split(r"[\s,，。？?、/]+", q)):
            matches.append(segment)
    return matches


def matching_vqa_items(question: str, prediction: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    q_tokens = [
        token for token in re.split(r"[\s,，。？?、/]+", question.lower())
        if token
    ]
    matches: List[Tuple[int, Dict[str, Any]]] = []
    for item in prediction.get("vqa", []) or []:
        text = " ".join(
            str(field).lower()
            for field in [
                item.get("question", ""),
                item.get("answer", ""),
                item.get("category", ""),
                item.get("question_type", ""),
                item.get("evidence_description", ""),
            ]
        )
        score = sum(1 for token in q_tokens if token in text)
        if score:
            matches.append((score, item))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in matches[:limit]]


def is_review_question(question: str) -> bool:
    text = question.lower()
    cues = [
        "人工复审",
        "人工复核",
        "复审",
        "复核",
        "审核",
        "待确认",
        "需要确认",
        "需要人工",
        "哪些结果需要",
        "不确定",
        "低置信",
        "置信度低",
        "置信",
        "review",
        "human review",
        "uncertain",
        "uncertainty",
        "low confidence",
        "confidence",
    ]
    return any(cue in text for cue in cues)


def render_review_answer(prediction: Dict[str, Any]) -> str:
    items = review_segments(prediction)
    video_id = html.escape(str(prediction.get("video_id", "-")))
    parts = ["<div class='qa-answer'>"]
    if not items:
        parts.append(
            f"<p><b>Answer</b><br>当前视频 {video_id} 没有发现需要人工复审的片段。</p>"
        )
    else:
        parts.append(
            f"<p><b>Answer</b><br>当前视频 {video_id} 建议人工复审或重点关注的片段有 {len(items)} 个。</p>"
        )
        for index, segment in enumerate(items, start=1):
            confidence = float(segment.get("confidence", 0) or 0)
            flags = "；".join(str(flag) for flag in segment.get("_review_flags", []))
            reason = segment.get("uncertainty_reason") or "No uncertainty note."
            suggestion = segment.get("review_suggestion") or ""
            parts.append(
                "<div class='qa-evidence'>"
                f"<b>{index}. {html.escape(str(segment.get('segment_id', '')))}</b> "
                f"{html.escape(str(segment.get('start', 0)))}s-{html.escape(str(segment.get('end', 0)))}s "
                f"{html.escape(str(segment.get('action_zh') or segment.get('action') or 'unknown'))}"
                f"<br><span class='small-note'>confidence={confidence:.2f}; "
                f"status={html.escape(str(segment.get('verification_status', '')))}; "
                f"type={html.escape(str(segment.get('uncertainty_type', 'none')))}"
                f"</span>"
                f"<br><span class='small-note'>flags: {html.escape(flags)}</span>"
                f"<br>{html.escape(str(segment.get('caption', '')))}"
                f"<br><span class='small-note'>reason: {html.escape(str(reason))}</span>"
                + (
                    f"<br><span class='small-note'>suggestion: {html.escape(str(suggestion))}</span>"
                    if suggestion else ""
                )
                + "</div>"
            )
    parts.append(
        "<div class='qa-evidence'>"
        "<b>Evidence status</b><br>Structured review fields"
        "<br><span class='small-note'>mode: prediction_json_review_query</span>"
        "<br><span class='small-note'>标准：needs_human_review=True，或 confidence&lt;0.65，或 status 为 partial/rejected/unverified，或 uncertainty_type 非 none。</span>"
        "</div>"
    )
    parts.append("</div>")
    return "".join(parts)


def render_vqa_evidence(items: List[Dict[str, Any]], prediction: Dict[str, Any]) -> str:
    if not items:
        return ""
    cards = ["<div class='qa-evidence'><b>Generated VQA evidence</b>"]
    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        time_range = format_time_range(item.get("time_range"))
        images = "".join(
            f"<img src='{src}' alt='matched VQA evidence' />"
            for src in collect_vqa_images(item, prediction)[:2]
        )
        image_html = f"<div class='vqa-images'>{images}</div>" if images else ""
        cards.append(
            "<div class='vqa-card'>"
            f"{image_html}"
            f"<div class='vqa-question'>{html.escape(str(question))}</div>"
            f"<div class='vqa-answer'>{html.escape(str(answer))}</div>"
            "<div class='chip-row'>"
            f"{chip('level', item.get('level'))}"
            f"{chip('category', item.get('category'))}"
            f"{chip('time', time_range)}"
            f"{chip('segment', item.get('segment_id'))}"
            "</div>"
            "</div>"
        )
    cards.append("</div>")
    return "".join(cards)


def collect_memory_images(item: Dict[str, Any], prediction: Dict[str, Any], limit: int = 3) -> List[str]:
    values: List[Any] = []
    for key in ("evidence_frame_paths", "frame_paths", "evidence_images", "frames"):
        value = item.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    urls: List[str] = []
    for value in values:
        src = image_src(str(value), prediction)
        if src and src not in urls:
            urls.append(src)
        if len(urls) >= limit:
            break
    return urls


def evidence_video_html(item: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    video_path = resolve_video_path(prediction)
    if not video_path:
        return ""
    try:
        start = max(0.0, float(item.get("start", 0) or 0))
    except (TypeError, ValueError):
        start = 0.0
    try:
        end = float(item.get("end", 0) or 0)
    except (TypeError, ValueError):
        end = 0.0
    if end <= start:
        end = start + 5.0
    src = file_url(Path(video_path)) + f"#t={start:.2f},{end:.2f}"
    return (
        "<div class='vqa-media'>"
        f"<video src='{html.escape(src)}' controls preload='metadata'></video>"
        f"<span class='small-note'>video evidence: {start:.1f}s-{end:.1f}s</span>"
        "</div>"
    )


def evidence_visual_html(item: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    images = collect_memory_images(item, prediction)
    if images:
        return (
            "<div class='vqa-images'>"
            + "".join(f"<img src='{src}' alt='retrieved evidence frame' />" for src in images)
            + "</div>"
        )
    return evidence_video_html(item, prediction)


def render_runtime_qa(result: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    sufficiency = result.get("sufficiency") or {}
    sufficient = bool(sufficiency.get("sufficient"))
    mode = result.get("mode", "memory_only")
    status_text = "Sufficient memory evidence" if sufficient else "Needs visual lookup"
    reasons = sufficiency.get("reasons") or []
    selected = result.get("selected_memory") or []
    frames = result.get("visual_frames") or []
    cited = result.get("cited_timestamps") or []

    parts = [
        "<div class='qa-answer'>",
        f"<p><b>Answer</b><br>{html.escape(str(result.get('answer', '')))}</p>",
        "<div class='qa-evidence'>"
        f"<b>Evidence status</b><br>{html.escape(status_text)}"
        f"<br><span class='small-note'>mode: {html.escape(str(mode))}</span>",
    ]
    if reasons:
        parts.append("<br><span class='small-note'>" + html.escape("；".join(str(r) for r in reasons)) + "</span>")
    parts.append("</div>")

    if selected:
        parts.append("<div class='qa-evidence'><b>Retrieved memory</b>")
        for item in selected[:5]:
            start = item.get("start", "?")
            end = item.get("end", "?")
            if item.get("kind") == "segment":
                label = item.get("action_zh") or item.get("action") or "segment"
                body = item.get("caption", "")
                meta = (
                    f"confidence={item.get('confidence', '')}; "
                    f"status={item.get('verification_status', '')}; "
                    f"type={item.get('uncertainty_type', 'none')}"
                )
            else:
                label = "scene memory"
                body = item.get("summary", "")
                meta = f"source={item.get('memory_id', '')}"
            parts.append(
                "<div class='qa-evidence'>"
                f"<b>{html.escape(str(start))}s-{html.escape(str(end))}s</b> "
                f"{html.escape(str(label))}"
                f"<br><span class='small-note'>{html.escape(str(body))}</span>"
                f"<br><span class='small-note'>{html.escape(str(meta))}</span>"
                + "</div>"
            )
        parts.append("</div>")

    if frames:
        parts.append("<div class='qa-evidence'><b>Visual lookup candidates</b><br>")
        image_cards = []
        for frame in frames:
            timestamp = frame.get("timestamp", "?")
            path = frame.get("path", "")
            src = image_src(str(path), prediction) if path else ""
            label = (
                f"{html.escape(str(timestamp))}s"
                f" · {html.escape(str(frame.get('source', 'memory')))}"
            )
            if src:
                image_cards.append(
                    "<div class='vqa-card'>"
                    f"<div class='vqa-images'><img src='{src}' alt='visual lookup frame {html.escape(str(timestamp))}' /></div>"
                    f"<span class='small-note'>{label}</span>"
                    "</div>"
                )
            else:
                image_cards.append(f"<span class='info-chip'>{label}</span>")
        parts.append("<div class='vqa-bank'>" + "".join(image_cards) + "</div>")
        parts.append("</div>")

    if cited:
        parts.append(
            "<p><span class='small-note'>Cited timestamps: "
            + html.escape("、".join(str(t) for t in cited))
            + "</span></p>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_llm_qa(result: Dict[str, Any], context: Dict[str, Any],
                  prediction: Dict[str, Any], error: Exception | None = None) -> str:
    selected = context.get("selected_memory") or []
    cited = result.get("cited_timestamps") or []
    mode = "retrieve_context + answer_question"
    status = "LLM answer from retrieved memory"
    if error:
        status = "LLM API unavailable; showing retrieved evidence"
        mode = "retrieve_context fallback"

    parts = [
        "<div class='qa-answer'>",
        f"<p><b>Answer</b><br>{html.escape(str(result.get('answer', '')))}</p>",
        "<div class='qa-evidence'>"
        f"<b>Evidence status</b><br>{html.escape(status)}"
        f"<br><span class='small-note'>mode: {html.escape(mode)}</span>",
    ]
    if error:
        parts.append(
            "<br><span class='small-note'>"
            + html.escape(f"{type(error).__name__}: {str(error)[:160]}")
            + "</span>"
        )
    parts.append("</div>")

    if selected:
        parts.append("<div class='qa-evidence'><b>Retrieved memory</b>")
        for item in selected[:6]:
            start = item.get("start", "?")
            end = item.get("end", "?")
            if item.get("kind") == "segment":
                label = item.get("action_zh") or item.get("action") or "segment"
                body = item.get("caption", "")
                meta = (
                    f"confidence={item.get('confidence', '')}; "
                    f"status={item.get('verification_status', '')}; "
                    f"type={item.get('uncertainty_type', 'none')}"
                )
            else:
                label = "scene memory"
                body = item.get("summary", "")
                meta = f"source={item.get('memory_id', '')}"
            parts.append(
                "<div class='qa-evidence'>"
                f"<b>{html.escape(str(start))}s-{html.escape(str(end))}s</b> "
                f"{html.escape(str(label))}"
                f"<br><span class='small-note'>{html.escape(str(body))}</span>"
                f"<br><span class='small-note'>{html.escape(str(meta))}</span>"
                + "</div>"
            )
        parts.append("</div>")

    if cited:
        parts.append(
            "<p><span class='small-note'>Cited timestamps: "
            + html.escape("、".join(str(t) for t in cited))
            + "</span></p>"
        )
    parts.append("</div>")
    return "".join(parts)


def fallback_answer_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    selected = context.get("selected_memory") or []
    if not selected:
        return {
            "answer": "当前没有检索到足够的结构化证据，无法可靠回答这个问题。",
            "cited_timestamps": [],
        }

    first = selected[0]
    start = first.get("start", "?")
    end = first.get("end", "?")
    if first.get("kind") == "segment":
        label = first.get("action_zh") or first.get("action") or "相关操作"
        body = first.get("caption", "")
    else:
        label = "相关场景"
        body = first.get("summary", "")
    return {
        "answer": f"检索到最相关证据位于 {start}s-{end}s：{label}。{body}。由于 LLM API 暂时不可用，这里先展示检索证据，未生成最终自然语言答案。",
        "cited_timestamps": [start] if isinstance(start, (int, float)) else [],
    }


def answer_interactive_question(question: str, prediction: Dict[str, Any]) -> str:
    question = (question or "").strip()
    if not question:
        video_memory = load_video_memory(prediction)
        source = "memory bundle" if video_memory else "prediction JSON"
        vqa_count = len(prediction.get("vqa", []) or [])
        vqa_note = f"；当前视频已接入 {vqa_count} 条 Generated VQA" if vqa_count else ""
        return (
            "<div class='qa-answer'>输入一个关于当前视频的问题，系统会基于 "
            f"{html.escape(source)}、片段结果、Generated VQA 和跨视频记忆进行回答{html.escape(vqa_note)}。</div>"
        )

    if is_review_question(question):
        return render_review_answer(prediction)

    video_memory = load_video_memory(prediction)
    global_memory = load_global_memory(prediction)
    vqa_matches = matching_vqa_items(question, prediction)
    if video_memory:
        context = agent_memory.retrieve_context(
            question,
            video_memory,
            global_memory,
            max_items=8,
        )
        try:
            result = agent_memory.answer_question(
                question,
                context,
                video_memory=video_memory,
            )
            return render_llm_qa(result, context, prediction) + render_vqa_evidence(vqa_matches, prediction)
        except Exception as exc:  # noqa: BLE001 - demo should still show retrieved evidence
            result = fallback_answer_from_context(context)
            return render_llm_qa(result, context, prediction, error=exc) + render_vqa_evidence(vqa_matches, prediction)

    q = question.lower()
    segments = prediction.get("segments", []) or []
    lines: List[str] = []
    if any(word in q for word in ["步骤", "流程", "操作", "process", "step", "action"]):
        lines.append("当前视频的关键步骤如下：")
        for segment in segments:
            lines.append(f"<div class='qa-evidence'>{segment_line(segment)}</div>")
    elif any(word in q for word in ["物体", "器具", "工具", "材料", "object", "tool"]):
        seen = []
        for segment in segments:
            for obj in segment.get("objects", []) or []:
                if obj not in seen:
                    seen.append(obj)
        lines.append("当前视频中解析到的关键物体包括：" + html.escape("、".join(str(x) for x in seen) or "暂无记录"))
    else:
        if vqa_matches:
            lines.append("Generated VQA 中找到以下相关证据：")
            for item in vqa_matches:
                lines.append(
                    "<div class='qa-evidence'>"
                    f"<b>{html.escape(str(item.get('question', '')))}</b><br>"
                    f"{html.escape(str(item.get('answer', '')))}"
                    "</div>"
                )
            return (
                "<div class='qa-answer'>"
                + "".join(f"<p>{line}</p>" for line in lines)
                + render_vqa_evidence(vqa_matches, prediction)
                + "</div>"
            )
        matched = matching_segments(question, prediction)
        if matched:
            lines.append("根据当前 JSON 记忆，相关片段是：")
            for segment in matched[:3]:
                lines.append(f"<div class='qa-evidence'>{segment_line(segment)}</div>")
        else:
            memory = prediction.get("clip_memory", []) or []
            lines.append("没有找到精确匹配片段，下面是当前视频记忆摘要：")
            for item in memory[:2]:
                lines.append(
                    f"<div class='qa-evidence'><b>{html.escape(str(item.get('start', 0)))}s-"
                    f"{html.escape(str(item.get('end', 0)))}s</b><br>"
                    f"{html.escape(str(item.get('summary', '')))}</div>"
                )
    return "<div class='qa-answer'>" + "".join(f"<p>{line}</p>" for line in lines) + "</div>"


def load_video_view(video_id: str, predictions: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    prediction = find_prediction(predictions, video_id)
    rows = segment_rows(prediction)
    detail = segment_detail(prediction, 0)
    return (
        resolve_video_path(prediction),
        summary_markdown(prediction),
        review_markdown(prediction),
        rows,
        vqa_markdown(prediction),
        answer_interactive_question("", prediction),
        0,
        prediction,
        *detail,
    )


def on_table_select(evt: gr.SelectData, prediction: Dict[str, Any]) -> Tuple[Any, ...]:
    index = 0
    if evt and isinstance(evt.index, (list, tuple)) and evt.index:
        index = int(evt.index[0])
    elif evt and isinstance(evt.index, int):
        index = int(evt.index)
    return (index, *segment_detail(prediction, index))


def apply_feedback(
    predictions: List[Dict[str, Any]],
    prediction: Dict[str, Any],
    selected_index: int,
    segment_id: str,
    start: float,
    end: float,
    action: str,
    action_zh: str,
    objects_text: str,
    caption: str,
    uncertainty: str,
    needs_review: bool,
) -> Tuple[Any, ...]:
    updated = deepcopy(prediction)
    segments = updated.get("segments", []) or []
    if not segments:
        return predictions, updated, [], "No segment to update.", summary_markdown(updated), review_markdown(updated)
    selected_index = max(0, min(int(selected_index), len(segments) - 1))
    segment = segments[selected_index]
    before = deepcopy(segment)

    segment["segment_id"] = segment_id or segment.get("segment_id")
    segment["start"] = round(float(start), 2)
    segment["end"] = round(float(end), 2)
    segment["action"] = action.strip() or segment.get("action", "unknown")
    segment["action_zh"] = action_zh.strip()
    segment["objects"] = parse_objects(objects_text)
    segment["caption"] = caption.strip()
    segment["uncertainty_reason"] = uncertainty.strip()
    segment["needs_human_review"] = bool(needs_review)
    segment["object_evidence"] = [{"object": obj, "present": True} for obj in segment["objects"]]

    # Minimal closed-loop update: treat human edits as verified constraints.
    segment["verification_status"] = "human_verified"
    segment["confidence"] = max(float(segment.get("confidence", 0) or 0), 0.88)
    if not segment["uncertainty_reason"]:
        segment["uncertainty_reason"] = "已根据人工反馈完成局部更新。"
    segment["needs_human_review"] = False

    all_predictions = deepcopy(predictions)
    for i, item in enumerate(all_predictions):
        if item.get("video_id") == updated.get("video_id"):
            all_predictions[i] = updated
            break
    else:
        all_predictions.append(updated)

    feedback_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_id": updated.get("video_id"),
        "segment_index": selected_index,
        "before": before,
        "after": segment,
        "mode": "human_feedback_local_update",
    }
    old_log = []
    if FEEDBACK_LOG.exists():
        try:
            old_log = json.loads(FEEDBACK_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_log = []
    old_log.append(feedback_entry)
    save_json(FEEDBACK_LOG, old_log)
    save_json(UPDATED_PREDICTIONS, all_predictions)

    message = (
        f"Updated {segment.get('segment_id')} with human feedback. "
        f"Saved reviewed predictions to {UPDATED_PREDICTIONS.name}."
    )
    return (
        all_predictions,
        updated,
        segment_rows(updated),
        message,
        summary_markdown(updated),
        review_markdown(updated),
        *segment_detail(updated, selected_index),
    )


def export_predictions(predictions: List[Dict[str, Any]]) -> str:
    save_json(UPDATED_PREDICTIONS, predictions)
    return str(UPDATED_PREDICTIONS)


def build_app() -> gr.Blocks:
    initial_predictions = load_predictions()
    options = video_options(initial_predictions)
    initial_video = options[0] if options else "demo_0061"
    initial_prediction = find_prediction(initial_predictions, initial_video)

    with gr.Blocks(title="LabARM-HV Demo") as demo:
        predictions_state = gr.State(initial_predictions)
        prediction_state = gr.State(initial_prediction)
        selected_index_state = gr.State(0)

        gr.HTML(
            """
            <div class="lab-header">
              <div>
                <h1 class="lab-title">LabARM-HV Video Parsing Console</h1>
                <p class="lab-subtitle">Scientific experiment video parsing 路 structured segments 路 confidence review 路 human feedback loop</p>
              </div>
              <div class="lab-badge">Training-free Agent Demo</div>
            </div>
            """
        )

        first_detail = segment_detail(initial_prediction, 0)

        with gr.Row():
            with gr.Column(scale=5, elem_id="top_left", elem_classes=["card", "top-card", "video-card"]):
                gr.HTML('<div class="section-title">Video Review <span>source + overview</span></div>')
                video_selector = gr.Dropdown(
                    label="",
                    show_label=False,
                    choices=options,
                    value=initial_video,
                    interactive=True,
                )
                video_player = gr.Video(
                    label="",
                    show_label=False,
                    value=resolve_video_path(initial_prediction),
                    height=360,
                )
                summary = gr.HTML(value=summary_markdown(initial_prediction))
                review_queue = gr.HTML(value=review_markdown(initial_prediction))
            with gr.Column(scale=7, elem_id="top_right", elem_classes=["card", "top-card"]):
                gr.HTML('<div class="section-title">Agent Segments <span>click a row to inspect</span></div>')
                segment_table = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    value=segment_rows(initial_prediction),
                    datatype=["str", "number", "number", "str", "str", "str", "number", "str", "str"],
                    label="",
                    show_label=False,
                    interactive=False,
                    wrap=True,
                    max_height=240,
                    elem_id="segment_table",
                    column_widths=["145px", "70px", "70px", "110px", "170px", "360px", "85px", "95px", "105px"],
                )
                gr.HTML('<div class="section-title" style="margin-top:14px;">VQA Preview <span>ExpVid-style evidence QA</span></div>')
                vqa_panel = gr.HTML(
                    value=vqa_markdown(initial_prediction),
                    label="",
                    show_label=False,
                    elem_classes=["tight-card", "vqa-panel"],
                )

        with gr.Row():
            with gr.Column(scale=6, elem_id="bottom_left", elem_classes=["card", "bottom-card", "form-card"]):
                gr.HTML('<div class="section-title">Segment Detail <span>human-editable fields</span></div>')
                segment_id = gr.Textbox(label="Segment ID", value=first_detail[0], container=True)
                with gr.Row():
                    start = gr.Number(label="Start (s)", precision=2, value=first_detail[1])
                    end = gr.Number(label="End (s)", precision=2, value=first_detail[2])
                with gr.Row():
                    action = gr.Textbox(label="Action ID", value=first_detail[3])
                    action_zh = gr.Textbox(label="Action", value=first_detail[4])
                objects = gr.Textbox(
                    label="Objects",
                    value=first_detail[5],
                    placeholder="Use 、 or comma to separate objects",
                )
                caption = gr.Textbox(label="Caption", lines=3, value=first_detail[6], elem_classes=["caption-field"])
            with gr.Column(scale=6, elem_id="bottom_right", elem_classes=["card", "bottom-card", "form-card", "feedback-card"]):
                gr.HTML('<div class="section-title">Feedback Loop <span>local update + export</span></div>')
                uncertainty = gr.Textbox(
                    label="Uncertainty / review note",
                    lines=4,
                    value=first_detail[7],
                    elem_classes=["uncertainty-field"],
                )
                needs_review = gr.Checkbox(label="Needs human review", value=first_detail[8])
                update_btn = gr.Button("Apply feedback and update segment", variant="primary")
                save_btn = gr.Button("Export reviewed predictions", elem_classes=["secondary-button"])
                status = gr.Textbox(label="Status", interactive=False)
                export_file = gr.Textbox(
                    label="Reviewed JSON path",
                    interactive=False,
                    elem_classes=["export-path"],
                )

        with gr.Row():
            with gr.Column(elem_id="qa_full", elem_classes=["card", "form-card", "qa-card"]):
                gr.HTML('<div class="section-title">Interactive QA <span>memory-based judge Q&amp;A</span></div>')
                gr.HTML(
                    "<div class='qa-hint'>输入一个关于当前视频的问题，系统会基于已解析片段、Generated VQA 和记忆 JSON 进行回答。</div>"
                )
                qa_question = gr.Textbox(
                    label="Judge question",
                    lines=2,
                    placeholder="例如：这段视频有哪些关键步骤？哪些结果需要复核？",
                    elem_classes=["qa-input"],
                )
                ask_btn = gr.Button("Submit question", variant="primary")
                gr.HTML("<div class='qa-output-title'>Answer</div>")
                qa_answer = gr.HTML(
                    value=answer_interactive_question("", initial_prediction),
                    label="",
                    show_label=False,
                    elem_classes=["tight-card", "qa-panel"],
                )

        video_selector.change(
            fn=load_video_view,
            inputs=[video_selector, predictions_state],
            outputs=[
                video_player,
                summary,
                review_queue,
                segment_table,
                vqa_panel,
                qa_answer,
                selected_index_state,
                prediction_state,
                segment_id,
                start,
                end,
                action,
                action_zh,
                objects,
                caption,
                uncertainty,
                needs_review,
            ],
        )

        ask_btn.click(
            fn=answer_interactive_question,
            inputs=[qa_question, prediction_state],
            outputs=[qa_answer],
        )

        segment_table.select(
            fn=on_table_select,
            inputs=[prediction_state],
            outputs=[
                selected_index_state,
                segment_id,
                start,
                end,
                action,
                action_zh,
                objects,
                caption,
                uncertainty,
                needs_review,
            ],
        )

        update_btn.click(
            fn=apply_feedback,
            inputs=[
                predictions_state,
                prediction_state,
                selected_index_state,
                segment_id,
                start,
                end,
                action,
                action_zh,
                objects,
                caption,
                uncertainty,
                needs_review,
            ],
            outputs=[
                predictions_state,
                prediction_state,
                segment_table,
                status,
                summary,
                review_queue,
                segment_id,
                start,
                end,
                action,
                action_zh,
                objects,
                caption,
                uncertainty,
                needs_review,
            ],
        )

        save_btn.click(fn=export_predictions, inputs=[predictions_state], outputs=[export_file])

    return demo


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=True,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=CSS,
        allowed_paths=[str(VQA_PACKAGE_DIR), str(LOCAL_HACKATHON_RELEASE)],
    )
    while True:
        time.sleep(3600)


