"""Whole-video pipeline using qwen3.5-omni-plus (no frame extraction).

Flow:
    Phase hypothesis  -> one video call via native dashscope SDK (file://)
    Action segments   -> one video call via native dashscope SDK (file://)
    Normalize         -> same _normalize_segments as the frame pipeline
    Write             -> same predictions.json schema

No verify/repair steps since no frames are extracted.  All segments get
verification_status="partial" and needs_human_review=True by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import api_client, config, ontology, prompts
from .media import VideoInfo
from .ontology import Phase
from .pipeline import (
    _action_catalog_for_repair,
    _clip_final_overlaps,
    _combined_action_catalog,
    _coverage_audit,
    _normalize_segments,
    _phase_candidates,
    _verify_and_maybe_repair,
    _writer_gate,
)
from .runlog import RunLog


def _default_unverified(seg: Dict[str, Any]) -> Dict[str, Any]:
    seg.setdefault("object_evidence", [{"object": o, "present": True} for o in seg["objects"]])
    seg.setdefault("confidence", 0.5)
    # "partial" passes the writer gate; signals model-predicted but not frame-verified.
    seg.setdefault("verification_status", "partial")
    seg.setdefault("uncertainty_reason", "整视频直接理解，未做逐帧视觉核验")
    seg.setdefault("needs_human_review", True)
    seg.setdefault("uncertainty_type", "mixed")
    seg.setdefault("review_suggestion", "人工复核该片段的动作类别、物体和边界是否与画面一致")
    seg.setdefault("evidence_diagnosis", {
        "perception_ok": False,
        "temporal_ok": False,
        "action_ok": False,
        "caption_ok": False,
        "objects_present": [],
        "objects_missing": seg.get("objects", []),
        "missing_evidence": ["整视频直接理解，未做逐帧核验"],
        "note": "omni pipeline，无帧级验证",
    })
    return seg


def hypothesize_phase_omni(video: VideoInfo, phases: List[Phase],
                            log: RunLog) -> Dict[str, Any]:
    prompt = prompts.OMNI_PHASE_HYPOTHESIS.format(
        phase_catalog=ontology.phase_catalog_text(phases),
        duration=video.duration,
    )
    with log.timed("omni_phase_hypothesis") as h:
        result = api_client.ask_video_json(prompt, video.path, model=config.OMNI_MODEL,
                                           fps=config.OMNI_FPS)
        h["output"] = result
    if not isinstance(result, dict) or "phase" not in result:
        result = {"phase": "other_visible_lab_operation", "phase_zh": "其他可见实验操作",
                  "confidence": 0.3, "alternative_phases": [],
                  "evidence_timestamps": [], "reason": "fallback"}
    return result


def segment_actions_omni(video: VideoInfo, phases: List[Phase],
                         phase_res: Dict[str, Any], log: RunLog) -> List[Dict[str, Any]]:
    candidates = _phase_candidates(phases, phase_res)
    catalog, catalog_phase_ids = _combined_action_catalog(phases, candidates, phase_res)
    prompt = prompts.OMNI_ACTION_SEGMENTATION.format(
        phase_id=phase_res.get("phase", "unknown"),
        phase_zh=phase_res.get("phase_zh", ""),
        duration=video.duration,
        action_catalog=catalog,
    )
    with log.timed("omni_action_segmentation",
                   inputs={"phase": phase_res.get("phase"),
                           "phase_confidence": phase_res.get("confidence"),
                           "catalog_phases": catalog_phase_ids}) as h:
        result = api_client.ask_video_json(prompt, video.path, model=config.OMNI_MODEL,
                                           fps=config.OMNI_FPS, max_tokens=4000)
        h["output"] = result
    segments = (result if isinstance(result, list)
                else result.get("segments", []) if isinstance(result, dict) else [])
    return _normalize_segments(segments, video.duration)


def process_video_omni(video_id: str, video: VideoInfo, phases: List[Phase],
                       log: RunLog) -> Dict[str, Any]:
    log.start_video(video_id, {"duration": round(video.duration, 1),
                               "fps": round(video.fps, 1),
                               "resolution": f"{video.width}x{video.height}",
                               "pipeline": "omni",
                               "omni_fps": config.OMNI_FPS})

    phase_res = hypothesize_phase_omni(video, phases, log)
    segments = segment_actions_omni(video, phases, phase_res, log)

    # Real verification when frames are available, otherwise unverified defaults
    if video.frames:
        repair_catalog = _action_catalog_for_repair(phases, phase_res)
        processed: List[Dict[str, Any]] = []
        for seg in segments:
            result = _verify_and_maybe_repair(video, seg, repair_catalog, log)
            if result is not None:
                processed.append(result)
        segments = processed
    else:
        segments = [_default_unverified(s) for s in segments]

    final_segments = _writer_gate(segments, log, keep_rejected=True)
    final_segments = _clip_final_overlaps(final_segments)
    coverage = _coverage_audit(final_segments, video.duration, [], log)

    for i, seg in enumerate(final_segments, 1):
        seg["segment_id"] = f"{video_id}_s{i:03d}"

    prediction = {
        "video_id": video_id,
        "video_path": f"videos/{video_id}.mp4",
        "phase": phase_res.get("phase"),
        "phase_zh": phase_res.get("phase_zh"),
        "phase_confidence": phase_res.get("confidence"),
        "alternative_phases": phase_res.get("alternative_phases", []),
        "segments": [
            {"segment_id": s["segment_id"], "start": s["start"], "end": s["end"],
             "action": s["action"], "action_zh": s["action_zh"], "objects": s["objects"],
             "caption": s["caption"], "evidence_timestamps": s["evidence_timestamps"],
             "confidence": s["confidence"], "verification_status": s["verification_status"],
             "uncertainty_reason": s["uncertainty_reason"],
             "needs_human_review": s["needs_human_review"],
             "uncertainty_type": s.get("uncertainty_type", "mixed"),
             "review_suggestion": s.get("review_suggestion", ""),
             "evidence_diagnosis": s.get("evidence_diagnosis", {}),
             "object_evidence": s["object_evidence"]}
            for s in final_segments
        ],
        "clip_memory": [],
        "processing_note": ("coverage_audit: long gaps or low coverage, review recommended"
                            if coverage.get("needs_review") else "omni pipeline: no frame extraction"),
    }
    log.add("write_prediction", output={"num_segments": len(final_segments),
                                        "phase": phase_res.get("phase"),
                                        "pipeline": "omni"})
    return prediction
