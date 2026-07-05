"""Hybrid hxa pipeline for experimental video parsing.

The pipeline keeps the reliable evidence loop from the frame-based agent while
routing models by capability:

    Qwen MLLM  -> frame/window caption, object grounding, boundary refinement,
                  visual verification and local repair
    DeepSeek   -> phase/action hypotheses and final text-only audit

It intentionally does not replace ``pipeline.py`` or ``pipeline_omni.py`` so
all three variants can be compared on the same split.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from . import api_client, config, media, prompts
from .media import Frame, VideoInfo
from .ontology import Phase
from .pipeline import (
    _action_catalog_for_repair,
    _clip_final_overlaps,
    _coverage_audit,
    _normalize_segments,
    _summaries_text,
    _urls,
    _verify_and_maybe_repair,
    _writer_gate,
    build_clip_memory,
    hypothesize_phase,
    segment_actions,
)
from .runlog import RunLog


def _frame_ts(frames: List[Frame]) -> List[float]:
    return [round(float(f.timestamp), 1) for f in frames]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_objects(*groups: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in groups:
        for item in group:
            name = str(item).strip()
            if name and name not in seen:
                out.append(name)
                seen.add(name)
    return out


def object_ground_segment(video: VideoInfo, seg: Dict[str, Any],
                          log: RunLog) -> Dict[str, Any]:
    """Use Qwen MLLM to confirm apparatus/material names for one segment."""
    if not config.HYBRID_OBJECT_ENABLED:
        return seg
    frames = media.frames_in_window(
        video.frames, seg["start"], seg["end"], config.HYBRID_OBJECT_FRAMES
    )
    if not frames:
        return seg

    prompt = prompts.HYBRID_OBJECT_GROUNDING.format(
        start=seg["start"],
        end=seg["end"],
        action=seg["action"],
        action_zh=seg["action_zh"],
        objects="，".join(seg.get("objects", [])) or "(无)",
        caption=seg.get("caption", ""),
        frame_ts=_frame_ts(frames),
    )
    with log.timed(
        "hybrid_object_grounding",
        inputs={"segment": [seg["start"], seg["end"]], "action": seg["action"]},
    ) as h:
        try:
            result = api_client.ask_json(
                prompt,
                vision_images=_urls(frames),
                model=config.VISION_MODEL,
                max_tokens=1200,
            )
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)[:160]}
        h["output"] = result

    if not isinstance(result, dict) or "error" in result:
        return seg

    confirmed = [str(x).strip() for x in result.get("confirmed_objects", []) if str(x).strip()]
    additional = [str(x).strip() for x in result.get("additional_objects", []) if str(x).strip()]
    missing = [str(x).strip() for x in result.get("missing_objects", []) if str(x).strip()]
    merged = _merge_objects(confirmed, additional)
    if merged:
        seg["objects"] = merged
    corrected = str(result.get("corrected_caption", "")).strip()
    if corrected:
        seg["caption"] = corrected
    seg["object_grounding"] = {
        "confirmed_objects": confirmed,
        "additional_objects": additional,
        "missing_objects": missing,
        "confidence": _safe_float(result.get("confidence"), 0.5),
        "note": str(result.get("note", "")).strip(),
    }
    return seg


def refine_boundary(video: VideoInfo, seg: Dict[str, Any], log: RunLog) -> Dict[str, Any]:
    """Use Qwen MLLM to tighten start/end around the visible core action."""
    if not config.HYBRID_BOUNDARY_ENABLED:
        return seg
    pad = config.HYBRID_BOUNDARY_CONTEXT_SECONDS
    context_start = max(0.0, float(seg["start"]) - pad)
    context_end = min(video.duration, float(seg["end"]) + pad)
    frames = media.frames_in_window(
        video.frames, context_start, context_end, config.HYBRID_BOUNDARY_FRAMES
    )
    if not frames:
        return seg

    prompt = prompts.HYBRID_BOUNDARY_REFINEMENT.format(
        start=seg["start"],
        end=seg["end"],
        context_start=context_start,
        context_end=context_end,
        action=seg["action"],
        action_zh=seg["action_zh"],
        caption=seg.get("caption", ""),
        frame_ts=_frame_ts(frames),
    )
    with log.timed(
        "hybrid_boundary_refinement",
        inputs={"segment": [seg["start"], seg["end"]], "context": [context_start, context_end]},
    ) as h:
        try:
            result = api_client.ask_json(
                prompt,
                vision_images=_urls(frames),
                model=config.VISION_MODEL,
                max_tokens=1200,
            )
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)[:160]}
        h["output"] = result

    if not isinstance(result, dict) or "error" in result:
        return seg

    refined_start = _safe_float(result.get("refined_start"), seg["start"])
    refined_end = _safe_float(result.get("refined_end"), seg["end"])
    refined_start = max(context_start, min(context_end, refined_start))
    refined_end = max(context_start, min(context_end, refined_end))
    if refined_end - refined_start >= 0.5:
        seg["start"] = round(refined_start, 1)
        seg["end"] = round(refined_end, 1)
        evidence = [
            int(t) for t in (result.get("evidence_timestamps") or [])
            if isinstance(t, (int, float)) and refined_start <= float(t) <= refined_end
        ]
        if evidence:
            seg["evidence_timestamps"] = evidence
    seg["boundary_refinement"] = {
        "confidence": _safe_float(result.get("boundary_confidence"), 0.5),
        "reason": str(result.get("reason", "")).strip(),
    }
    return seg


def text_audit(video: VideoInfo, summaries: List[Dict[str, Any]],
               segments: List[Dict[str, Any]], log: RunLog) -> Dict[str, Any]:
    """Use DeepSeek for a text-only consistency and coverage audit."""
    if not config.HYBRID_TEXT_AUDIT_ENABLED:
        return {}
    serializable = []
    for idx, seg in enumerate(segments):
        serializable.append({
            "segment_index": idx,
            "start": seg.get("start"),
            "end": seg.get("end"),
            "action": seg.get("action"),
            "action_zh": seg.get("action_zh"),
            "objects": seg.get("objects", []),
            "caption": seg.get("caption", ""),
            "confidence": seg.get("confidence"),
            "verification_status": seg.get("verification_status"),
            "uncertainty_reason": seg.get("uncertainty_reason", ""),
        })
    prompt = prompts.HYBRID_TEXT_AUDIT.format(
        duration=video.duration,
        clip_summaries=_summaries_text(summaries),
        segments_json=json.dumps(serializable, ensure_ascii=False, indent=2),
    )
    with log.timed("hybrid_text_audit", inputs={"segments": len(segments)}) as h:
        try:
            result = api_client.ask_json(
                prompt, model=config.STRUCTURED_MODEL, max_tokens=2200
            )
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)[:160]}
        h["output"] = result
    return result if isinstance(result, dict) else {}


def _apply_text_audit(segments: List[Dict[str, Any]],
                      audit: Dict[str, Any]) -> None:
    reviews = audit.get("segment_reviews") if isinstance(audit, dict) else None
    if not isinstance(reviews, list):
        return
    for review in reviews:
        if not isinstance(review, dict):
            continue
        try:
            idx = int(review.get("segment_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(segments):
            continue
        severity = str(review.get("severity", "")).lower()
        if severity not in {"warning", "error"}:
            continue
        seg = segments[idx]
        suggestion = str(review.get("suggestion", "")).strip()
        issue_type = str(review.get("issue_type", "consistency")).strip()
        seg["needs_human_review"] = True
        seg["uncertainty_type"] = issue_type or seg.get("uncertainty_type", "mixed")
        if suggestion:
            old = str(seg.get("review_suggestion", "")).strip()
            seg["review_suggestion"] = f"{old}；{suggestion}" if old else suggestion
            old_reason = str(seg.get("uncertainty_reason", "")).strip()
            seg["uncertainty_reason"] = f"{old_reason}；文本审计：{suggestion}" if old_reason else f"文本审计：{suggestion}"


def process_video_hybrid(video_id: str, video: VideoInfo, phases: List[Phase],
                         log: RunLog) -> Dict[str, Any]:
    log.start_video(video_id, {
        "duration": round(video.duration, 1),
        "fps": round(video.fps, 1),
        "resolution": f"{video.width}x{video.height}",
        "sampled_frames": len(video.frames),
        "pipeline": "hybrid_hxa",
        "mllm_model": config.VISION_MODEL,
        "llm_model": config.STRUCTURED_MODEL,
    })

    summaries = build_clip_memory(video, log)
    phase_res = hypothesize_phase(video, phases, summaries, log)
    segments = segment_actions(video, phases, phase_res, summaries, log)

    refined_segments = []
    for seg in segments:
        refined = refine_boundary(video, seg, log)
        grounded = object_ground_segment(video, refined, log)
        refined_segments.append(grounded)
    segments = _normalize_segments(refined_segments, video.duration)

    repair_catalog = _action_catalog_for_repair(phases, phase_res)
    processed_segments: List[Dict[str, Any]] = []
    for seg in segments:
        repaired_or_verified = _verify_and_maybe_repair(video, seg, repair_catalog, log)
        if repaired_or_verified is not None:
            processed_segments.append(repaired_or_verified)

    final_segments = _writer_gate(processed_segments, log)
    final_segments = _clip_final_overlaps(final_segments)
    coverage = _coverage_audit(final_segments, video.duration, summaries, log)
    audit = text_audit(video, summaries, final_segments, log)
    _apply_text_audit(final_segments, audit)

    for i, seg in enumerate(final_segments, 1):
        seg["segment_id"] = f"{video_id}_s{i:03d}"

    prediction = {
        "video_id": video_id,
        "video_path": f"videos/{video_id}.mp4",
        "pipeline": "hybrid_hxa",
        "model_routing": {
            "mllm": config.VISION_MODEL,
            "llm": config.STRUCTURED_MODEL,
        },
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
             "uncertainty_type": s.get("uncertainty_type", "none"),
             "review_suggestion": s.get("review_suggestion", ""),
             "evidence_diagnosis": s.get("evidence_diagnosis", {}),
             "object_evidence": s["object_evidence"],
             "object_grounding": s.get("object_grounding", {}),
             "boundary_refinement": s.get("boundary_refinement", {})}
            for s in final_segments
        ],
        "clip_memory": summaries,
        "hybrid_audit": audit,
        "processing_note": (
            "coverage_audit: long gaps or low coverage, review recommended"
            if coverage.get("needs_review")
            else str(audit.get("summary", "")) if audit else ""
        ),
    }
    log.add("write_prediction", output={
        "num_segments": len(final_segments),
        "raw_segments": len(segments),
        "processed_segments": len(processed_segments),
        "phase": phase_res.get("phase"),
        "pipeline": "hybrid_hxa",
    })
    return prediction
