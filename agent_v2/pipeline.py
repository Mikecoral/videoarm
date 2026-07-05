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
                  "confidence": 0.3, "alternative_phases": [],
                  "evidence_timestamps": [], "reason": "fallback"}
    return result


def _phase_confidence(phase_res: Dict[str, Any]) -> float:
    conf = phase_res.get("confidence")
    try:
        return float(conf)
    except (TypeError, ValueError):
        return 0.0


def _phase_candidates(phases: List[Phase], phase_res: Dict[str, Any]) -> List[Phase]:
    """Return primary + alternative phase objects, preserving order."""
    by_id = {p.phase_id: p for p in phases}
    ids: List[str] = []
    primary = str(phase_res.get("phase", "")).strip()
    if primary:
        ids.append(primary)

    alternatives = phase_res.get("alternative_phases") or []
    if isinstance(alternatives, list):
        for item in alternatives:
            pid = ""
            if isinstance(item, dict):
                pid = str(item.get("phase", "")).strip()
            elif isinstance(item, str):
                pid = item.strip()
            if pid:
                ids.append(pid)

    out: List[Phase] = []
    seen = set()
    for pid in ids:
        phase = by_id.get(pid)
        if phase and phase.phase_id not in seen:
            out.append(phase)
            seen.add(phase.phase_id)
        if len(out) >= config.MAX_PHASE_CANDIDATES:
            break
    return out


def _combined_action_catalog(phases: List[Phase], candidates: List[Phase],
                             phase_res: Dict[str, Any]) -> tuple[str, List[str]]:
    selected = candidates[:]
    conf = _phase_confidence(phase_res)
    if conf >= config.PHASE_HIGH_CONF_THRESHOLD:
        # High confidence: use only primary phase to prevent alternative actions leaking in.
        selected = candidates[:1]
    elif conf < config.PHASE_FALLBACK_CONFIDENCE and len(selected) < 2:
        selected = phases

    blocks = []
    used_ids = []
    for phase in selected:
        catalog = ontology.action_catalog_text(phase).strip()
        if not catalog:
            continue
        used_ids.append(phase.phase_id)
        blocks.append(f"## {phase.phase_id} ({phase.zh})\n{catalog}")

    if not blocks:
        return "(无对应本体动作，可自行判断)", used_ids
    return "\n\n".join(blocks), used_ids


def _action_catalog_for_repair(phases: List[Phase], phase_res: Dict[str, Any]) -> str:
    candidates = _phase_candidates(phases, phase_res)
    catalog, _ = _combined_action_catalog(phases, candidates, phase_res)
    return catalog


def segment_actions(video: VideoInfo, phases: List[Phase], phase_res: Dict[str, Any],
                    summaries: List[Dict[str, Any]], log: RunLog) -> List[Dict[str, Any]]:
    candidates = _phase_candidates(phases, phase_res)
    catalog, catalog_phase_ids = _combined_action_catalog(phases, candidates, phase_res)
    prompt = prompts.ACTION_SEGMENTATION.format(
        phase_id=phase_res.get("phase", "unknown"),
        phase_zh=phase_res.get("phase_zh", ""),
        duration=video.duration,
        action_catalog=catalog,
        clip_summaries=_summaries_text(summaries))
    with log.timed("action_segmentation",
                   inputs={"phase": phase_res.get("phase"),
                           "phase_confidence": phase_res.get("confidence"),
                           "catalog_phases": catalog_phase_ids}) as h:
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


# ------------------------------------------------- dense boundary sampling ---
def _dense_resample_and_split(video: VideoInfo, seg: Dict[str, Any],
                              phases: List[Phase], phase_res: Dict[str, Any],
                              log: RunLog) -> List[Dict[str, Any]]:
    """Re-sample a long segment with dense sub-windows and locally re-segment.

    Replaces one coarse segment with multiple finer ones when the dense
    scene_snapper reveals distinct sub-operations (e.g. repeated injections).
    Returns the original segment unchanged if no improvement is found.
    """
    start, end = float(seg["start"]), float(seg["end"])
    duration = end - start
    n = max(3, int(round(duration / config.DENSE_SUBWINDOW)))
    step = duration / n

    sub_summaries: List[Dict[str, Any]] = []
    for i in range(n):
        ws = round(start + i * step, 1)
        we = round(start + (i + 1) * step, 1)
        frames = media.frames_in_window(video.frames, ws, we, config.DENSE_FRAMES)
        if not frames:
            continue
        prompt = prompts.SCENE_SNAPPER.format(start=ws, end=we)
        with log.timed("dense_snapper",
                       inputs={"window": [ws, we],
                               "frame_ts": [f.timestamp for f in frames]}) as h:
            text = api_client.ask_vision(prompt, _urls(frames))
            h["output"] = {"summary": text}
        sub_summaries.append({"start": ws, "end": we,
                               "frame_ts": [f.timestamp for f in frames],
                               "summary": text})

    if len(sub_summaries) < 3:
        return [seg]

    local_segs = segment_actions(video, phases, phase_res, sub_summaries, log)
    if len(local_segs) <= 1:
        return [seg]

    log.add("dense_split", inputs={"original": [start, end]},
            output={"sub_segments": len(local_segs)})
    return local_segs


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


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "是", "对", "正确"}:
            return True
        if lowered in {"false", "no", "0", "否", "不", "错误"}:
            return False
    return default


def _str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _uncertainty_type(raw: Any, reasons: List[str]) -> str:
    text = str(raw or "").strip().lower()
    if reasons and text in {"", "none", "no"}:
        return "mixed" if len(set(reasons)) > 1 else reasons[0]
    aliases = {
        "none": "none",
        "no": "none",
        "perception": "perception",
        "perceptive": "perception",
        "visual": "perception",
        "temporal": "temporal",
        "time": "temporal",
        "cognitive": "cognitive",
        "caption": "cognitive",
        "reasoning": "cognitive",
        "mixed": "mixed",
        "multiple": "mixed",
    }
    if text in aliases:
        return aliases[text]
    if len(set(reasons)) > 1:
        return "mixed"
    return reasons[0] if reasons else "none"


def _apply_verification(seg: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
    conf = v.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    objects_missing = _str_list(v.get("objects_missing"))
    present = _str_list(v.get("objects_present"))
    perception_ok = _as_bool(v.get("perception_ok"), default=not objects_missing)
    temporal_ok = _as_bool(v.get("temporal_ok"), default=_as_bool(v.get("action_ok"), True))
    action_ok = _as_bool(v.get("action_ok"), temporal_ok)
    caption_ok = _as_bool(v.get("caption_ok"), True)
    missing_evidence = _str_list(v.get("missing_evidence"))
    uncertainty_reasons: List[str] = []

    # Layered Dr.V-style confidence guards.  The model may volunteer a high
    # score, but unsupported perception/temporal/cognitive evidence caps it.
    #
    # action_ok=False means the wrong operation type was identified — hard cap.
    # temporal_ok=False covers boundary/ordering uncertainty which is common
    # when only 3 frames are sampled; use a softer cap so segments survive as
    # "partial" rather than being mass-rejected.
    if objects_missing or not perception_ok:
        conf = min(conf, 0.7)
        uncertainty_reasons.append("perception")
    if not action_ok:
        conf = min(conf, 0.4)
        uncertainty_reasons.append("temporal")
    elif not temporal_ok:
        conf = min(conf, 0.6)
        uncertainty_reasons.append("temporal")
    elif missing_evidence and any(
        cue in " ".join(missing_evidence)
        for cue in ("边界", "顺序", "先后", "起止", "持续", "方向")
    ):
        conf = min(conf, 0.65)
        uncertainty_reasons.append("temporal")
    if not caption_ok:
        conf = min(conf, 0.6)
        uncertainty_reasons.append("cognitive")

    uncertainty_type = _uncertainty_type(v.get("uncertainty_level"), uncertainty_reasons)
    if present:  # keep only visually confirmed objects when the check ran
        seg["objects"] = present
    if caption_ok is False and v.get("corrected_caption"):
        seg["caption"] = str(v["corrected_caption"]).strip()

    if perception_ok and temporal_ok and action_ok and caption_ok and conf >= 0.6:
        status = "verified"
    elif not action_ok and conf <= 0.4:
        status = "rejected"
    elif perception_ok or temporal_ok or action_ok or caption_ok or conf >= 0.45:
        # conf >= 0.45 with some evidence: treat as partial rather than hard reject
        status = "partial"
    else:
        status = "rejected"

    note = str(v.get("note", "")).strip()
    review_suggestion = str(v.get("review_suggestion", "")).strip()
    evidence_diagnosis = {
        "perception_ok": perception_ok,
        "temporal_ok": temporal_ok,
        "action_ok": action_ok,
        "caption_ok": caption_ok,
        "objects_present": present,
        "objects_missing": objects_missing,
        "missing_evidence": missing_evidence,
        "note": note,
    }
    reason_parts = []
    if note:
        reason_parts.append(note)
    if missing_evidence:
        reason_parts.append("缺失证据：" + "；".join(missing_evidence))

    if conf >= 0.80 and status == "verified":
        uncertainty, human = "", False
    elif conf >= 0.55 and status != "rejected":
        uncertainty = "；".join(reason_parts) or "视觉证据部分可见，建议抽查"
        human = uncertainty_type in {"temporal", "mixed"} and conf < 0.70
    else:
        uncertainty = "；".join(reason_parts) or "关键动作/物体证据不足"
        human = True
    if uncertainty_type != "none" and not review_suggestion:
        if uncertainty_type == "perception":
            review_suggestion = "人工复核关键物体是否可见及命名是否正确"
        elif uncertainty_type == "temporal":
            review_suggestion = "人工复核动作类别、方向和起止边界"
        elif uncertainty_type == "cognitive":
            review_suggestion = "人工复核 caption 是否包含画面不支持的推理"
        else:
            review_suggestion = "人工复核该片段的物体、动作边界和 caption"

    object_names = list(dict.fromkeys(seg.get("objects", []) + objects_missing))
    seg["object_evidence"] = [
        {"object": o, "present": o not in objects_missing}
        for o in object_names
    ]
    seg["confidence"] = round(conf, 2)
    seg["verification_status"] = status
    seg["uncertainty_reason"] = uncertainty
    seg["needs_human_review"] = human
    seg["uncertainty_type"] = uncertainty_type
    seg["review_suggestion"] = review_suggestion
    seg["evidence_diagnosis"] = evidence_diagnosis
    return seg


def _default_fields(seg: Dict[str, Any]) -> Dict[str, Any]:
    seg.setdefault("object_evidence", [{"object": o, "present": True} for o in seg["objects"]])
    seg.setdefault("confidence", 0.5)
    seg.setdefault("verification_status", "unverified")
    seg.setdefault("uncertainty_reason", "未做视觉复核")
    seg.setdefault("needs_human_review", True)
    seg.setdefault("uncertainty_type", "mixed")
    seg.setdefault("review_suggestion", "人工复核该片段的动作、物体和边界")
    seg.setdefault("evidence_diagnosis", {
        "perception_ok": False,
        "temporal_ok": False,
        "action_ok": False,
        "caption_ok": False,
        "objects_present": [],
        "objects_missing": seg.get("objects", []),
        "missing_evidence": ["未做视觉复核"],
        "note": "未做视觉复核",
    })
    return seg


def _needs_repair(seg: Dict[str, Any]) -> bool:
    if not config.REPAIR_ENABLED:
        return False
    status = seg.get("verification_status")
    conf = seg.get("confidence")
    try:
        conf_value = float(conf)
    except (TypeError, ValueError):
        conf_value = 0.0
    return status == "rejected" or conf_value < config.REPAIR_CONFIDENCE_THRESHOLD


def _repair_window(seg: Dict[str, Any], duration: float) -> tuple[float, float]:
    pad = config.REPAIR_CONTEXT_SECONDS
    start = max(0.0, float(seg["start"]) - pad)
    end = min(duration, float(seg["end"]) + pad)
    return round(start, 1), round(end, 1)


def repair_segment(video: VideoInfo, seg: Dict[str, Any], action_catalog: str,
                   log: RunLog) -> Dict[str, Any]:
    context_start, context_end = _repair_window(seg, video.duration)
    frames = media.frames_in_window(video.frames, context_start, context_end,
                                    config.REPAIR_FRAMES)
    if not frames:
        return {"delete": True, "repair_reason": "no frames available"}
    prompt = prompts.REPAIR_SEGMENT.format(
        start=seg["start"],
        end=seg["end"],
        action=seg["action"],
        action_zh=seg["action_zh"],
        objects="，".join(seg["objects"]) or "(无)",
        caption=seg["caption"],
        status=seg.get("verification_status", "unknown"),
        confidence=seg.get("confidence", ""),
        uncertainty_reason=seg.get("uncertainty_reason", ""),
        action_catalog=action_catalog,
        context_start=context_start,
        context_end=context_end,
    )
    with log.timed("repair_segment",
                   inputs={"segment": [seg["start"], seg["end"]],
                           "action": seg["action"],
                           "status": seg.get("verification_status"),
                           "confidence": seg.get("confidence"),
                           "context": [context_start, context_end]}) as h:
        try:
            result = api_client.ask_json(prompt, vision_images=_urls(frames),
                                         model=config.VISION_MODEL, max_tokens=1600)
        except Exception as e:  # noqa: BLE001
            result = {"delete": True, "error": str(e)[:120],
                      "repair_reason": "repair call failed"}
        h["output"] = result
    return result if isinstance(result, dict) else {"delete": True, "repair_reason": "invalid repair result"}


def _apply_repair(seg: Dict[str, Any], repair: Dict[str, Any],
                  duration: float) -> Dict[str, Any] | None:
    if repair.get("delete") is True:
        seg["verification_status"] = "rejected"
        try:
            conf = float(seg.get("confidence", 0.4) or 0.4)
        except (TypeError, ValueError):
            conf = 0.4
        seg["confidence"] = min(conf, 0.4)
        seg["needs_human_review"] = True
        seg["uncertainty_reason"] = str(repair.get("repair_reason", "repair deleted segment")).strip()
        seg["repair_action"] = "delete"
        return None

    normalized = _normalize_segments([repair], duration)
    if not normalized:
        seg["repair_action"] = "invalid"
        return None

    repaired = normalized[0]
    repaired["repair_action"] = "rewrite"
    repaired["repair_reason"] = str(repair.get("repair_reason", "")).strip()
    repaired["original_segment"] = {
        "start": seg["start"],
        "end": seg["end"],
        "action": seg["action"],
        "action_zh": seg["action_zh"],
        "caption": seg["caption"],
        "verification_status": seg.get("verification_status"),
        "confidence": seg.get("confidence"),
        "uncertainty_reason": seg.get("uncertainty_reason", ""),
    }
    return repaired


def _verify_and_maybe_repair(video: VideoInfo, seg: Dict[str, Any],
                             action_catalog: str, log: RunLog) -> Dict[str, Any] | None:
    if config.VERIFY_ENABLED:
        v = verify_segment(video, seg, log)
        _apply_verification(seg, v) if v and "error" not in v else _default_fields(seg)
    else:
        _default_fields(seg)

    if not _needs_repair(seg):
        return seg

    repair = repair_segment(video, seg, action_catalog, log)
    repaired = _apply_repair(seg, repair, video.duration)
    if repaired is None:
        return None

    if config.VERIFY_ENABLED:
        v2 = verify_segment(video, repaired, log)
        _apply_verification(repaired, v2) if v2 and "error" not in v2 else _default_fields(repaired)
    else:
        _default_fields(repaired)
    return repaired


def _writer_gate(segments: List[Dict[str, Any]], log: RunLog,
                 keep_rejected: bool = False) -> List[Dict[str, Any]]:
    def conf_value(seg: Dict[str, Any]) -> float:
        try:
            return float(seg.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    allowed = {"verified", "partial"}
    if not config.VERIFY_ENABLED:
        allowed.add("unverified")
    if keep_rejected:
        allowed.add("rejected")

    accepted = [s for s in segments
                if s.get("verification_status") in allowed]
    rejected = [s for s in segments
                if s.get("verification_status") not in allowed]
    fallback_used = False
    if not accepted and rejected:
        fallback = max(rejected, key=conf_value)
        fallback["needs_human_review"] = True
        fallback["uncertainty_reason"] = fallback.get("uncertainty_reason") or "全部片段被过滤，保留最高置信片段供人工复核"
        accepted = [fallback]
        fallback_used = True
    log.add("writer_gate",
            inputs={"total_segments": len(segments)},
            output={"accepted": len(accepted), "filtered": len(rejected),
                    "fallback_used": fallback_used})
    return accepted


def _clip_final_overlaps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments.sort(key=lambda x: x["start"])
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["end"]:
            segments[i]["start"] = segments[i - 1]["end"]
    for seg in segments:
        seg["evidence_timestamps"] = [
            t for t in seg.get("evidence_timestamps", [])
            if seg["start"] <= t <= seg["end"]
        ]
    return [s for s in segments if s["end"] - s["start"] >= 0.5]


_ACTION_CUES = (
    "倒", "倾倒", "滴加", "加入", "添加", "移液", "吸取", "转移", "放入", "取出",
    "拿起", "放置", "插入", "连接", "固定", "调节", "调整", "点击", "启动",
    "搅拌", "摇晃", "过滤", "抽滤", "放液", "排放", "称量", "读数", "旋紧",
    "拧开", "盖上", "标记", "书写", "点样", "洗涤", "润洗",
)

_NO_ACTION_CUES = (
    "未见明显", "没有发生", "无操作", "未观察到", "静止", "处于静置",
    "未见液体倾倒", "未见明显的液体倾倒", "未出现实验人员",
)


def _summary_has_action(summary: str) -> bool:
    if not summary:
        return False
    if any(cue in summary for cue in _ACTION_CUES):
        # A summary can say "未见明显倒液" while still mentioning no action.
        # Keep it suspicious only if it also names a concrete hand/tool action.
        if any(cue in summary for cue in _NO_ACTION_CUES):
            concrete = ("正在" in summary or "双手" in summary or "右手" in summary or
                        "左手" in summary or "随后" in summary)
            return concrete and not summary.strip().startswith(_NO_ACTION_CUES)
        return True
    return False


def _backtrack_coverage_gaps(gaps: List[Dict[str, Any]],
                             summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not config.COVERAGE_BACKTRACK_ENABLED or not gaps:
        return []

    findings = []
    for gap in gaps:
        relevant = []
        for summary in summaries:
            if summary["end"] <= gap["start"] or summary["start"] >= gap["end"]:
                continue
            text = str(summary.get("summary", ""))
            if _summary_has_action(text):
                relevant.append({
                    "window": [summary["start"], summary["end"]],
                    "frame_ts": summary.get("frame_ts", []),
                    "summary": text,
                })
        if relevant:
            findings.append({
                "gap": gap,
                "suspicious": True,
                "reason": "clip_memory in this gap describes possible substantive operations",
                "evidence": relevant,
            })
    return findings


def _coverage_audit(segments: List[Dict[str, Any]], duration: float,
                    summaries: List[Dict[str, Any]], log: RunLog) -> Dict[str, Any]:
    if not config.COVERAGE_AUDIT_ENABLED:
        return {}

    ordered = sorted(segments, key=lambda x: x["start"])
    covered = round(sum(max(0.0, s["end"] - s["start"]) for s in ordered), 1)
    ratio = round(covered / duration, 3) if duration > 0 else 0.0
    gaps = []
    cursor = 0.0
    for seg in ordered:
        gap = seg["start"] - cursor
        if gap >= config.COVERAGE_MAX_GAP_SECONDS:
            gaps.append({"start": round(cursor, 1), "end": round(seg["start"], 1),
                         "duration": round(gap, 1)})
        cursor = max(cursor, seg["end"])
    tail_gap = duration - cursor
    if tail_gap >= config.COVERAGE_MAX_GAP_SECONDS:
        gaps.append({"start": round(cursor, 1), "end": round(duration, 1),
                     "duration": round(tail_gap, 1)})

    findings = _backtrack_coverage_gaps(gaps, summaries)
    needs_review = bool(gaps) or ratio < config.COVERAGE_MIN_RATIO or bool(findings)
    audit = {
        "video_duration": round(duration, 1),
        "covered_seconds": covered,
        "coverage_ratio": ratio,
        "large_gaps": gaps,
        "backtrack_findings": findings,
        "suspicious_gap_count": len(findings),
        "needs_review": needs_review,
    }
    log.add("coverage_audit", output=audit)
    return audit


# ------------------------------------------------------------- driver -------
def process_video(video_id: str, video: VideoInfo, phases: List[Phase],
                  log: RunLog) -> Dict[str, Any]:
    log.start_video(video_id, {"duration": round(video.duration, 1), "fps": round(video.fps, 1),
                               "resolution": f"{video.width}x{video.height}",
                               "sampled_frames": len(video.frames)})

    summaries = build_clip_memory(video, log)
    phase_res = hypothesize_phase(video, phases, summaries, log)
    segments = segment_actions(video, phases, phase_res, summaries, log)

    if config.DENSE_ENABLED:
        expanded: List[Dict[str, Any]] = []
        for seg in segments:
            if seg["end"] - seg["start"] >= config.DENSE_MIN_DURATION:
                expanded.extend(_dense_resample_and_split(
                    video, seg, phases, phase_res, log))
            else:
                expanded.append(seg)
        segments = _normalize_segments(expanded, video.duration)

    repair_catalog = _action_catalog_for_repair(phases, phase_res)

    processed_segments: List[Dict[str, Any]] = []
    for seg in segments:
        repaired_or_verified = _verify_and_maybe_repair(video, seg, repair_catalog, log)
        if repaired_or_verified is not None:
            processed_segments.append(repaired_or_verified)

    final_segments = _writer_gate(processed_segments, log)
    final_segments = _clip_final_overlaps(final_segments)
    coverage = _coverage_audit(final_segments, video.duration, summaries, log)
    # segment_id assigned after final ordering/gating
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
             "uncertainty_type": s.get("uncertainty_type", "none"),
             "review_suggestion": s.get("review_suggestion", ""),
             "evidence_diagnosis": s.get("evidence_diagnosis", {}),
             "object_evidence": s["object_evidence"]}
            for s in final_segments
        ],
        "clip_memory": summaries,
        "processing_note": ("coverage_audit: long gaps or low coverage, review recommended"
                            if coverage.get("needs_review") else ""),
    }
    log.add("write_prediction", output={"num_segments": len(final_segments),
                                        "raw_segments": len(segments),
                                        "processed_segments": len(processed_segments),
                                        "phase": phase_res.get("phase")})
    return prediction
