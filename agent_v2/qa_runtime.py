"""Runtime helpers for memory-grounded interactive QA.

The first version is deliberately deterministic: it retrieves context from the
per-video memory bundle, checks whether that evidence is enough to answer, and
selects visual frames to revisit when memory evidence is weak.  A later version
can call the vision model on those frames without changing the demo contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from . import memory
except ImportError:  # Allows direct script-style imports during local demos.
    import memory  # type: ignore


LOW_CONFIDENCE = 0.65
VISUAL_CUES = (
    "有没有看到",
    "是否可见",
    "能否确认",
    "确认",
    "是不是",
    "哪个器具",
    "什么器具",
    "有没有倒",
    "是否倒",
    "有没有加入",
    "是否加入",
    "有没有点样",
    "是否点样",
    "是否进入",
    "看得清",
    "visible",
    "confirm",
    "whether",
    "which object",
    "which tool",
)
PROCESS_CUES = ("步骤", "流程", "操作", "过程", "step", "process", "action")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _question_needs_visual(question: str) -> bool:
    text = question.lower()
    return any(cue.lower() in text for cue in VISUAL_CUES)


def _evidence_timestamps(item: Dict[str, Any]) -> List[Any]:
    if item.get("kind") == "segment":
        return _as_list(item.get("evidence_timestamps"))
    return _as_list(item.get("frame_ts"))


def assess_context_sufficiency(question: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether retrieved memory is enough or visual lookup is needed."""
    selected = _as_list(context.get("selected_memory"))
    reasons: List[str] = []
    target_windows: List[Dict[str, Any]] = []
    visual_question = _question_needs_visual(question)

    if not selected:
        reasons.append("No memory item matched the question.")

    for item in selected:
        kind = item.get("kind", "memory")
        start = item.get("start")
        end = item.get("end")
        item_reasons: List[str] = []

        if kind == "segment":
            confidence = _as_float(item.get("confidence"), 0.0)
            uncertainty_type = str(item.get("uncertainty_type", "none") or "none")
            if confidence and confidence < LOW_CONFIDENCE:
                item_reasons.append(f"low confidence ({confidence:.2f})")
            if item.get("needs_human_review"):
                item_reasons.append("marked for human review")
            if uncertainty_type in {"perception", "temporal", "mixed"}:
                item_reasons.append(f"{uncertainty_type} uncertainty")
            if not _evidence_timestamps(item) and visual_question:
                item_reasons.append("no evidence timestamps for a visual question")
        elif visual_question and not _as_list(item.get("frame_ts")):
            item_reasons.append("clip memory has no frame timestamps")

        if item_reasons:
            reasons.append(f"{item.get('memory_id', kind)}: " + "; ".join(item_reasons))
            target_windows.append({
                "memory_id": item.get("memory_id", ""),
                "kind": kind,
                "start": start,
                "end": end,
                "reason": "; ".join(item_reasons),
            })

    if visual_question and not target_windows and selected:
        target_windows.append({
            "memory_id": selected[0].get("memory_id", ""),
            "kind": selected[0].get("kind", "memory"),
            "start": selected[0].get("start"),
            "end": selected[0].get("end"),
            "reason": "visual confirmation requested by the question",
        })
        reasons.append("The question asks for visual confirmation.")

    sufficient = bool(selected) and not reasons
    if any(cue.lower() in question.lower() for cue in PROCESS_CUES) and selected:
        # Process-summary questions are usually answerable from structured memory.
        sufficient = not any("No memory item" in r for r in reasons)

    return {
        "sufficient": sufficient,
        "need_visual_lookup": not sufficient and bool(target_windows),
        "reasons": reasons,
        "target_windows": target_windows,
    }


def select_visual_frames(context: Dict[str, Any], max_frames: int = 5) -> List[Dict[str, Any]]:
    """Pick frame candidates from retrieved clips/segments for visual revisit."""
    frames: List[Dict[str, Any]] = []
    selected = _as_list(context.get("selected_memory"))

    for item in selected:
        if item.get("kind") == "segment":
            timestamps = _as_list(item.get("evidence_timestamps"))
            paths = _as_list(item.get("evidence_frame_paths"))
            for idx, ts in enumerate(timestamps):
                frames.append({
                    "timestamp": ts,
                    "path": paths[idx] if idx < len(paths) else "",
                    "source": item.get("memory_id", ""),
                    "window": [item.get("start"), item.get("end")],
                })
        else:
            timestamps = _as_list(item.get("frame_ts"))
            paths = _as_list(item.get("frame_paths"))
            for idx, ts in enumerate(timestamps):
                frames.append({
                    "timestamp": ts,
                    "path": paths[idx] if idx < len(paths) else "",
                    "source": item.get("memory_id", ""),
                    "window": [item.get("start"), item.get("end")],
                })

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for frame in frames:
        key = (str(frame.get("timestamp")), str(frame.get("path")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)
        if len(deduped) >= max_frames:
            break
    return deduped


def _memory_answer(question: str, selected: List[Dict[str, Any]]) -> str:
    if not selected:
        return "未检索到足够相关的记忆项，建议先回看候选帧或换一个更具体的问题。"

    process_like = any(cue.lower() in question.lower() for cue in PROCESS_CUES)
    if process_like:
        parts = []
        for item in selected:
            if item.get("kind") == "segment":
                parts.append(
                    f"{item.get('start')}s-{item.get('end')}s："
                    f"{item.get('action_zh') or item.get('action')}，{item.get('caption', '')}"
                )
        if parts:
            return "根据结构化片段，当前视频相关步骤包括：" + "；".join(parts[:6]) + "。"

    item = selected[0]
    if item.get("kind") == "segment":
        return (
            f"根据已解析片段，相关证据位于 {item.get('start')}s-{item.get('end')}s："
            f"{item.get('caption', '')}"
        )
    return (
        f"根据场景记忆，相关时间窗为 {item.get('start')}s-{item.get('end')}s："
        f"{item.get('summary', '')}"
    )


def answer_with_optional_visual_lookup(
    question: str,
    video_memory: Dict[str, Any],
    global_memory: Dict[str, Any] | None = None,
    *,
    max_items: int = 8,
    max_frames: int = 5,
) -> Dict[str, Any]:
    """Return QA result plus evidence sufficiency and visual lookup candidates."""
    context = memory.retrieve_context(question, video_memory, global_memory, max_items=max_items)
    selected = _as_list(context.get("selected_memory"))
    sufficiency = assess_context_sufficiency(question, context)
    visual_frames = select_visual_frames(context, max_frames=max_frames) if sufficiency["need_visual_lookup"] else []
    cited: List[Any] = []
    for item in selected:
        cited.extend(_evidence_timestamps(item))

    return {
        "question": question,
        "mode": "memory_plus_visual_candidates" if visual_frames else "memory_only",
        "answer": _memory_answer(question, selected),
        "sufficiency": sufficiency,
        "selected_memory": selected,
        "visual_frames": visual_frames,
        "cross_video_hints": context.get("cross_video_hints", {}),
        "cited_timestamps": list(dict.fromkeys(str(t) for t in cited if t not in (None, ""))),
    }
