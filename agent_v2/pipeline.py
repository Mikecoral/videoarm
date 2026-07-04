"""Per-video processing pipeline (base requirements).

Flow, following PLAN.md:
    Observe   -> sample frames, split timeline into windows
    Memorize  -> scene_snapper vision call per window (hierarchical clip memory)
    Think     -> phase hypothesis over the clip memory
    Act       -> action segmentation over the clip memory + ontology
    Verify    -> bounded per-segment vision check -> confidence / uncertainty
    Write     -> assemble structured prediction
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import api_client, config, media, ontology, prompts
from .media import Frame, VideoInfo
from .ontology import Phase
from .runlog import RunLog


# --------------------------------------------------------------- windows ----
def _plan_windows(duration: float) -> List[tuple[float, float]]:
    n = round(duration / config.SECONDS_PER_WINDOW) if duration else config.MIN_WINDOWS
    n = max(config.MIN_WINDOWS, min(config.MAX_WINDOWS, int(n) or config.MIN_WINDOWS))
    step = duration / n if duration else config.SECONDS_PER_WINDOW
    return [(round(i * step, 1), round((i + 1) * step, 1)) for i in range(n)]


def _urls(frames: List[Frame]) -> List[str]:
    return [media.to_data_url(f) for f in frames]


# ----------------------------------------------------------- clip memory ----
def build_clip_memory(video: VideoInfo, log: RunLog) -> List[Dict[str, Any]]:
    windows = _plan_windows(video.duration)
    summaries: List[Dict[str, Any]] = []
    for (ws, we) in windows:
        frames = media.frames_in_window(video.frames, ws, we, config.FRAMES_PER_WINDOW)
        if not frames:
            continue
        prompt = prompts.SCENE_SNAPPER.format(start=ws, end=we)
        with log.timed("scene_snapper", inputs={"window": [ws, we],
                                                "frame_ts": [f.timestamp for f in frames]}) as h:
            text = api_client.ask_vision(prompt, _urls(frames))
            h["output"] = {"summary": text}
        summaries.append({"start": ws, "end": we,
                          "frame_ts": [f.timestamp for f in frames], "summary": text})
    return summaries


def _summaries_text(summaries: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{s['start']:.0f}-{s['end']:.0f}s] {s['summary']}" for s in summaries)


# --------------------------------------------------------- phase / actions --
def hypothesize_phase(video: VideoInfo, phases: List[Phase],
                      summaries: List[Dict[str, Any]], log: RunLog) -> Dict[str, Any]:
    prompt = prompts.PHASE_HYPOTHESIS.format(
        phase_catalog=ontology.phase_catalog_text(phases),
        duration=video.duration,
        clip_summaries=_summaries_text(summaries))
    with log.timed("phase_hypothesis") as h:
        result = api_client.ask_json(prompt, model=config.STRUCTURED_MODEL)
        h["output"] = result
    if not isinstance(result, dict) or "phase" not in result:
        result = {"phase": "other_visible_lab_operation", "phase_zh": "其他可见实验操作",
                  "confidence": 0.3, "evidence_timestamps": [], "reason": "fallback"}
    return result


def segment_actions(video: VideoInfo, phase: Phase, phase_zh: str,
                    summaries: List[Dict[str, Any]], log: RunLog) -> List[Dict[str, Any]]:
    catalog = ontology.action_catalog_text(phase) if phase else "(无对应本体动作，可自行判断)"
    prompt = prompts.ACTION_SEGMENTATION.format(
        phase_id=phase.phase_id if phase else "unknown",
        phase_zh=phase_zh,
        duration=video.duration,
        action_catalog=catalog,
        clip_summaries=_summaries_text(summaries))
    with log.timed("action_segmentation") as h:
        result = api_client.ask_json(prompt, model=config.STRUCTURED_MODEL, max_tokens=2400)
        h["output"] = result
    segments = result if isinstance(result, list) else result.get("segments", []) if isinstance(result, dict) else []
    return _normalize_segments(segments, video.duration)


def _normalize_segments(raw: List[Dict[str, Any]], duration: float) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for s in raw:
        try:
            start = max(0.0, float(s.get("start", 0)))
            end = min(duration, float(s.get("end", 0)))
        except (TypeError, ValueError):
            continue
        if end - start < 0.5:
            continue
        clean.append({
            "start": round(start, 1), "end": round(end, 1),
            "action": str(s.get("action", "")).strip() or "unknown",
            "action_zh": str(s.get("action_zh", "")).strip(),
            "objects": [str(o).strip() for o in (s.get("objects") or []) if str(o).strip()],
            "caption": str(s.get("caption", "")).strip(),
            "evidence_timestamps": [int(t) for t in (s.get("evidence_timestamps") or [])
                                    if isinstance(t, (int, float))],
        })
    clean.sort(key=lambda x: x["start"])
    # clip overlaps so segments stay ordered and non-overlapping
    for i in range(1, len(clean)):
        if clean[i]["start"] < clean[i - 1]["end"]:
            clean[i]["start"] = clean[i - 1]["end"]
    return [s for s in clean if s["end"] - s["start"] >= 0.5]


# ------------------------------------------------------------- verify -------
def verify_segment(video: VideoInfo, seg: Dict[str, Any], log: RunLog) -> Dict[str, Any]:
    frames = media.frames_in_window(video.frames, seg["start"], seg["end"], config.VERIFY_FRAMES)
    if not frames:
        return {}
    prompt = prompts.VERIFY_SEGMENT.format(
        start=seg["start"], end=seg["end"], action=seg["action"],
        action_zh=seg["action_zh"], objects="，".join(seg["objects"]) or "(无)",
        caption=seg["caption"])
    with log.timed("verify_segment", inputs={"segment": [seg["start"], seg["end"]],
                                            "action": seg["action"]}) as h:
        try:
            result = api_client.ask_json(prompt, vision_images=_urls(frames),
                                         model=config.VISION_MODEL)
        except Exception as e:  # noqa: BLE001
            result = {"error": str(e)[:120]}
        h["output"] = result
    return result if isinstance(result, dict) else {}


def _apply_verification(seg: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
    conf = v.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else 0.5
    action_ok = bool(v.get("action_ok", True))
    caption_ok = bool(v.get("caption_ok", True))
    # Keep confidence consistent with the check: a mismatched action cannot be
    # "high confidence correct", regardless of the number the model volunteered.
    if not action_ok:
        conf = min(conf, 0.4)
    elif not caption_ok:
        conf = min(conf, 0.6)

    present = [o for o in (v.get("objects_present") or []) if isinstance(o, str)]
    if present:  # keep only visually confirmed objects when the check ran
        seg["objects"] = present
    if caption_ok is False and v.get("corrected_caption"):
        seg["caption"] = str(v["corrected_caption"]).strip()

    if action_ok and caption_ok and conf >= 0.6:
        status = "verified"
    elif action_ok or caption_ok:
        status = "partial"
    else:
        status = "rejected"

    note = str(v.get("note", "")).strip()
    if conf >= 0.80 and status == "verified":
        uncertainty, human = "", False
    elif conf >= 0.60 and status != "rejected":
        uncertainty = note or "视觉证据部分可见，建议抽查"
        human = False
    else:
        uncertainty = note or "关键动作/物体证据不足"
        human = True

    seg["object_evidence"] = [{"object": o, "present": True} for o in seg["objects"]]
    seg["confidence"] = round(conf, 2)
    seg["verification_status"] = status
    seg["uncertainty_reason"] = uncertainty
    seg["needs_human_review"] = human
    return seg


def _default_fields(seg: Dict[str, Any]) -> Dict[str, Any]:
    seg.setdefault("object_evidence", [{"object": o, "present": True} for o in seg["objects"]])
    seg.setdefault("confidence", 0.5)
    seg.setdefault("verification_status", "unverified")
    seg.setdefault("uncertainty_reason", "未做视觉复核")
    seg.setdefault("needs_human_review", True)
    return seg


# ------------------------------------------------------------- driver -------
def process_video(video_id: str, video: VideoInfo, phases: List[Phase],
                  log: RunLog) -> Dict[str, Any]:
    log.start_video(video_id, {"duration": round(video.duration, 1), "fps": round(video.fps, 1),
                               "resolution": f"{video.width}x{video.height}",
                               "sampled_frames": len(video.frames)})

    summaries = build_clip_memory(video, log)
    phase_res = hypothesize_phase(video, phases, summaries, log)
    phase_obj = ontology.find_phase(phases, phase_res.get("phase", ""))
    segments = segment_actions(video, phase_obj, phase_res.get("phase_zh", ""), summaries, log)

    for seg in segments:
        if config.VERIFY_ENABLED:
            v = verify_segment(video, seg, log)
            _apply_verification(seg, v) if v and "error" not in v else _default_fields(seg)
        else:
            _default_fields(seg)
        # segment_id assigned after ordering
    for i, seg in enumerate(segments, 1):
        seg["segment_id"] = f"{video_id}_s{i:03d}"

    prediction = {
        "video_id": video_id,
        "video_path": f"videos/{video_id}.mp4",
        "phase": phase_res.get("phase"),
        "phase_zh": phase_res.get("phase_zh"),
        "phase_confidence": phase_res.get("confidence"),
        "segments": [
            {"segment_id": s["segment_id"], "start": s["start"], "end": s["end"],
             "action": s["action"], "action_zh": s["action_zh"], "objects": s["objects"],
             "caption": s["caption"], "evidence_timestamps": s["evidence_timestamps"],
             "confidence": s["confidence"], "verification_status": s["verification_status"],
             "uncertainty_reason": s["uncertainty_reason"],
             "needs_human_review": s["needs_human_review"],
             "object_evidence": s["object_evidence"]}
            for s in segments
        ],
        "clip_memory": summaries,
        "processing_note": "",
    }
    log.add("write_prediction", output={"num_segments": len(segments),
                                        "phase": phase_res.get("phase")})
    return prediction
