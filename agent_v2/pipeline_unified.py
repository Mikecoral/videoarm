"""Unified hxa pipeline: Omni → multi-agent → VQA.

Combines the strengths of all three pipelines:

    pipeline_omni   — whole-video global understanding, atomic segmentation
    pipeline_hybrid — boundary refinement, object grounding, text audit
    pipeline        — verify/repair loop

Stage flow
----------
    1. Omni                   → phase hypothesis + atomic segments (2 API calls)
    2. Multi-agent refinement  → boundary refinement + object grounding
                               → visual verification + conditional repair
                               → text audit (consistency / coverage)
    3. VQA generation          → template-based VQA per segment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import config
from .media import VideoInfo
from .ontology import Phase
from .pipeline import (
    _action_catalog_for_repair,
    _clip_final_overlaps,
    _normalize_segments,
    _verify_and_maybe_repair,
    _writer_gate,
)
from .pipeline_hybrid import (
    _apply_text_audit,
    object_ground_segment,
    refine_boundary,
    text_audit,
)
from .pipeline_omni import (
    hypothesize_phase_omni,
    segment_actions_omni,
)
from .runlog import RunLog


# ------------------------------------------------------------------ #
# VQA generation                                                      #
# ------------------------------------------------------------------ #

def _generate_vqa_for_segments(
    video_id: str,
    video: VideoInfo,
    segments: List[Dict[str, Any]],
    phase_res: Dict[str, Any],
    questions_per_segment: int = 4,
) -> List[Dict[str, Any]]:
    """Extract one keyframe per segment, run MLLM hand recognition, and generate VQA."""
    try:
        from agent_v2.vqa.generator import TemplateVQAGenerator
        from agent_v2.vqa.keyframes import extract_video_frame
        from agent_v2.vqa.types import KeyframeContext, SegmentContext
    except ImportError:
        print("  [unified] vqa not available, skipping VQA")
        return []

    generator = TemplateVQAGenerator()
    all_vqa: List[Dict[str, Any]] = []
    keyframe_dir = config.FRAME_CACHE_DIR / video_id / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        # ── Pick one keyframe timestamp (midpoint of evidence, or segment midpoint) ──
        evidence_ts = seg.get("evidence_timestamps") or []
        if evidence_ts:
            kf_ts = sorted(evidence_ts)[len(evidence_ts) // 2]  # median
        else:
            kf_ts = (float(seg["start"]) + float(seg["end"])) / 2.0
        kf_ts = round(float(kf_ts), 1)

        # ── Extract actual frame from video ──
        out_path = keyframe_dir / f"seg_{i:03d}_{int(kf_ts * 1000):09d}.jpg"
        success = extract_video_frame(video.path, kf_ts, out_path)
        frame_path = str(out_path) if success else None

        # ── MLLM hand recognition on the keyframe ──
        hands: list[dict] = []
        if frame_path:
            try:
                from . import api_client, prompts
                import base64
                data_url = "data:image/jpeg;base64," + base64.b64encode(
                    Path(frame_path).read_bytes()
                ).decode()
                hand_prompt = prompts.HAND_RECOGNITION.format(timestamp=kf_ts)
                hand_result = api_client.ask_json(
                    hand_prompt,
                    vision_images=[data_url],
                    model=config.VISION_MODEL,
                    max_tokens=400,
                )
                if isinstance(hand_result, dict):
                    raw = hand_result.get("hands") or []
                    for h in raw:
                        if isinstance(h, dict):
                            hands.append({
                                "side": str(h.get("side", "")).strip(),
                                "visible": bool(h.get("visible", True)),
                                "holding": str(h.get("holding", "")).strip() or None,
                            })
            except Exception as exc:
                print(f"  [unified] hand recognition failed for seg {i}: {exc}")

        # ── Build single keyframe with per-frame hands ──
        kf = KeyframeContext(
            frame_index=int(round(kf_ts)),
            time=kf_ts,
            frame_path=frame_path,
            objects=seg.get("objects", []),
            hands=hands,
            caption=seg.get("caption", ""),
        )

        prev_cap = segments[i - 1].get("caption") if i > 0 else None
        next_cap = segments[i + 1].get("caption") if i + 1 < len(segments) else None

        ctx = SegmentContext(
            video_id=video_id,
            video_path=str(video.path),
            phase=phase_res.get("phase", ""),
            phase_zh=phase_res.get("phase_zh", ""),
            segment_id=seg.get("segment_id", f"{video_id}_s{i:03d}"),
            start=float(seg["start"]),
            end=float(seg["end"]),
            action=seg.get("action", ""),
            action_zh=seg.get("action_zh", ""),
            segment_caption=seg.get("caption", ""),
            keyframes=[kf],
            previous_caption=prev_cap,
            next_caption=next_cap,
        )

        try:
            items = generator.generate(ctx, questions_per_segment)
        except Exception:
            items = []

        for item in items:
            item["segment_id"] = seg.get("segment_id", "")
            item["video_id"] = video_id
        all_vqa.extend(items)

    return all_vqa


# ------------------------------------------------------------------ #
# Main orchestrator                                                    #
# ------------------------------------------------------------------ #

def process_video_unified(
    video_id: str,
    video: VideoInfo,
    phases: List[Phase],
    log: RunLog,
    *,
    skip_vqa: bool = False,
) -> Dict[str, Any]:
    """Run the full unified pipeline on one video."""

    log.start_video(
        video_id,
        {
            "duration": round(video.duration, 1),
            "fps": round(video.fps, 1),
            "resolution": f"{video.width}x{video.height}",
            "sampled_frames": len(video.frames),
            "pipeline": "unified_hxa",
            "omni_model": config.OMNI_MODEL,
            "mllm_model": config.VISION_MODEL,
            "llm_model": config.STRUCTURED_MODEL,
        },
    )

    # ═══════════════════════════════════════════════════════════════ #
    # Stage 1: Omni atomic segmentation                                #
    # ═══════════════════════════════════════════════════════════════ #
    phase_res = hypothesize_phase_omni(video, phases, log)
    segments = segment_actions_omni(video, phases, phase_res, log)
    print(
        f"  [unified] Stage 1 Omni → {len(segments)} atomic segments "
        f"(phase={phase_res.get('phase')} conf={phase_res.get('confidence')})"
    )
    for i, seg in enumerate(segments):
        print(
            f"    [{seg['start']:.0f}-{seg['end']:.0f}s] {seg.get('action_zh', seg.get('action', '?'))}"
            f"  {seg.get('caption', '')[:80]}"
        )

    # ═══════════════════════════════════════════════════════════════ #
    # Stage 2: Multi-agent refinement                                  #
    # ═══════════════════════════════════════════════════════════════ #

    # 2a. Boundary refinement (MLLM)
    if config.HYBRID_BOUNDARY_ENABLED:
        for seg in segments:
            refine_boundary(video, seg, log)

    # 2b. Object grounding (MLLM)
    if config.HYBRID_OBJECT_ENABLED:
        for seg in segments:
            object_ground_segment(video, seg, log)

    segments = _normalize_segments(segments, video.duration)

    # 2c. Visual verification + repair (MLLM)
    repair_catalog = _action_catalog_for_repair(phases, phase_res)
    processed_segments: List[Dict[str, Any]] = []
    for seg in segments:
        result = _verify_and_maybe_repair(video, seg, repair_catalog, log)
        if result is not None:
            processed_segments.append(result)

    # 2d. Writer gate + final overlap clipping
    final_segments = _writer_gate(processed_segments, log)
    final_segments = _clip_final_overlaps(final_segments)

    # 2e. Text audit (DeepSeek) — build clip_memory for audit context
    audit: Dict[str, Any] = {}
    if config.HYBRID_TEXT_AUDIT_ENABLED:
        from .pipeline import build_clip_memory
        summaries = build_clip_memory(video, log)
        audit = text_audit(video, summaries, final_segments, log)
        _apply_text_audit(final_segments, audit)

    # Assign stable segment IDs
    for i, seg in enumerate(final_segments, 1):
        seg["segment_id"] = f"{video_id}_s{i:03d}"

    verified = sum(
        1 for s in final_segments if s.get("verification_status") == "verified"
    )
    partial = sum(
        1 for s in final_segments if s.get("verification_status") == "partial"
    )
    print(
        f"  [unified] Stage 2 → {len(final_segments)} final segments "
        f"(verified={verified} partial={partial})"
    )

    # ═══════════════════════════════════════════════════════════════ #
    # Stage 3: VQA generation                                          #
    # ═══════════════════════════════════════════════════════════════ #
    vqa_items: List[Dict[str, Any]] = []
    if not skip_vqa:
        try:
            vqa_items = _generate_vqa_for_segments(
                video_id, video, final_segments, phase_res,
            )
            if vqa_items:
                print(f"  [unified] Stage 3 → {len(vqa_items)} VQA items")
        except Exception as exc:
            print(f"  [unified] Stage 3 VQA failed: {exc}")

    # ═══════════════════════════════════════════════════════════════ #
    # Assemble prediction                                              #
    # ═══════════════════════════════════════════════════════════════ #
    prediction: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": f"videos/{video_id}.mp4",
        "pipeline": "unified_hxa",
        "model_routing": {
            "omni": config.OMNI_MODEL,
            "mllm": config.VISION_MODEL,
            "llm": config.STRUCTURED_MODEL,
        },
        "phase": phase_res.get("phase"),
        "phase_zh": phase_res.get("phase_zh"),
        "phase_confidence": phase_res.get("confidence"),
        "alternative_phases": phase_res.get("alternative_phases", []),
        "segments": [
            {
                "segment_id": s["segment_id"],
                "start": s["start"],
                "end": s["end"],
                "action": s["action"],
                "action_zh": s["action_zh"],
                "objects": s["objects"],
                "caption": s["caption"],
                "evidence_timestamps": s["evidence_timestamps"],
                "confidence": s["confidence"],
                "verification_status": s["verification_status"],
                "uncertainty_reason": s["uncertainty_reason"],
                "needs_human_review": s["needs_human_review"],
                "uncertainty_type": s.get("uncertainty_type", "none"),
                "review_suggestion": s.get("review_suggestion", ""),
                "evidence_diagnosis": s.get("evidence_diagnosis", {}),
                "object_evidence": s["object_evidence"],
                "object_grounding": s.get("object_grounding", {}),
                "boundary_refinement": s.get("boundary_refinement", {}),
            }
            for s in final_segments
        ],
        "vqa": vqa_items,
        "hybrid_audit": audit,
        "processing_note": (
            "unified pipeline: omni atomic → multi-agent → vqa. "
            + (str(audit.get("summary", "")) if audit else "")
        ),
    }

    log.add(
        "write_prediction",
        output={
            "num_segments": len(final_segments),
            "num_vqa": len(vqa_items),
            "phase": phase_res.get("phase"),
            "pipeline": "unified_hxa",
        },
    )
    return prediction
