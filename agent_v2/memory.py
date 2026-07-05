"""Persistent video memory bundles and bounded cross-video memory.

The project does not need a full VideoARM HM3 store.  For the hackathon demo we
persist a compact per-video memory bundle for retrieval-grounded QA, plus a
bounded cross-video memory that accumulates reusable action/object/failure
patterns across processed videos.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MEMORY_VERSION = 1
MAX_EXAMPLES_PER_KEY = 8
MAX_VIDEOS_PER_KEY = 20

REVIEW_CUES = ("复核", "不确定", "低置信", "人工", "uncertain", "review", "需要复核")

SYNONYM_GROUPS = (
    ("倒液", "倾倒", "倒入", "倒", "pour", "倾倒液体", "转移液体"),
    ("转移", "移入", "移出", "转运", "transfer"),
    ("移液", "吸取", "加液", "滴加", "取样", "加样", "pipette", "移取"),
    ("过滤", "抽滤", "滤过", "滤液", "滤饼", "filter", "filtration"),
    ("点样", "TLC", "薄层", "薄层色谱", "展开", "显色"),
    ("称量", "称重", "天平", "weigh"),
    ("搅拌", "混合", "摇晃", "振荡", "mix", "stir"),
    ("洗涤", "润洗", "冲洗", "rinse", "wash"),
    ("加热", "升温", "热板", "水浴", "heat"),
    ("冷却", "降温", "冰浴", "cool"),
    ("连接", "组装", "固定", "安装", "调节", "装置"),
    ("读数", "观察", "记录", "标记", "书写"),
)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame_path(frame_cache_root: Path, video_id: str, timestamp: float) -> str:
    frame_idx = int(round(float(timestamp)))
    return str(frame_cache_root / video_id / f"{frame_idx:06d}.jpg")


def _keywords_for_segment(seg: Dict[str, Any]) -> List[str]:
    words = [
        seg.get("action", ""),
        seg.get("action_zh", ""),
        seg.get("uncertainty_type", ""),
    ]
    words.extend(seg.get("objects", []))
    caption = str(seg.get("caption", ""))
    for cue in ("倒液", "倾倒", "移液", "吸取", "转移", "过滤", "抽滤", "搅拌",
                "洗涤", "润洗", "点样", "称量", "读数", "连接", "加热", "冷却"):
        if cue in caption:
            words.append(cue)
    if seg.get("needs_human_review"):
        words.extend(["复核", "需要复核", "不确定", "人工"])
    try:
        if float(seg.get("confidence") or 1.0) < 0.65:
            words.append("低置信")
    except (TypeError, ValueError):
        pass
    return _dedupe_keep_order(words)


def _expand_query_terms(question: str) -> List[str]:
    terms = [question]
    for group in SYNONYM_GROUPS:
        if any(term and term in question for term in group):
            terms.extend(group)
    return _dedupe_keep_order(terms)


def _video_duration(video_memory: Dict[str, Any]) -> float:
    info = video_memory.get("video_info", {})
    duration = _as_float(info.get("duration"), 0.0)
    if duration > 0:
        return duration
    ends = []
    for key in ("clip_memory", "segment_memory"):
        for item in video_memory.get(key, []):
            ends.append(_as_float(item.get("end"), 0.0))
    return max(ends) if ends else 0.0


def _parse_time_ranges(question: str, video_memory: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Extract coarse time constraints such as 前30秒 / 1分钟左右 / 最后阶段."""
    text = str(question)
    duration = _video_duration(video_memory)
    ranges: List[Tuple[float, float]] = []

    def add(start: float, end: float) -> None:
        if duration > 0:
            start = max(0.0, min(duration, start))
            end = max(0.0, min(duration, end))
        if end > start:
            ranges.append((round(start, 1), round(end, 1)))

    for match in re.finditer(r"前\s*(\d+(?:\.\d+)?)\s*(秒|s|分钟|分|min)?", text, re.I):
        value = float(match.group(1))
        unit = match.group(2) or "秒"
        seconds = value * 60 if unit in {"分钟", "分", "min"} else value
        add(0.0, seconds)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(秒|s|分钟|分|min)\s*(?:左右|附近|前后|上下|的时候|时)?", text, re.I):
        if match.start() > 0 and text[match.start() - 1] == "前":
            continue
        value = float(match.group(1))
        unit = match.group(2)
        seconds = value * 60 if unit in {"分钟", "分", "min"} else value
        add(seconds - 15.0, seconds + 15.0)

    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(秒|s|分钟|分|min)\s*(?:-|到|至|~)\s*(\d+(?:\.\d+)?)\s*(秒|s|分钟|分|min)?",
        text,
        re.I,
    ):
        start = float(match.group(1))
        start_unit = match.group(2)
        end = float(match.group(3))
        end_unit = match.group(4) or start_unit
        if start_unit in {"分钟", "分", "min"}:
            start *= 60
        if end_unit in {"分钟", "分", "min"}:
            end *= 60
        add(start, end)

    if any(cue in text for cue in ("开头", "开始", "最开始", "一开始", "前半段")):
        add(0.0, duration * 0.5 if "前半段" in text and duration > 0 else min(30.0, duration or 30.0))
    if duration > 0 and any(cue in text for cue in ("最后", "末尾", "结尾", "后半段", "后面")):
        start = duration * 0.5 if "后半段" in text else max(0.0, duration - 30.0)
        add(start, duration)
    if duration > 0 and any(cue in text for cue in ("中间", "中段", "中部")):
        mid = duration / 2.0
        add(mid - 20.0, mid + 20.0)

    return ranges


def _overlaps(item: Dict[str, Any], ranges: List[Tuple[float, float]]) -> bool:
    if not ranges:
        return False
    start = _as_float(item.get("start"), 0.0)
    end = _as_float(item.get("end"), start)
    return any(start < r_end and end > r_start for r_start, r_end in ranges)


def _timeline_sample(video_memory: Dict[str, Any], max_items: int = 6) -> List[Dict[str, Any]]:
    clips = list(video_memory.get("clip_memory", []))
    if not clips:
        return []
    if len(clips) <= max_items:
        return clips
    # Evenly sample the full clip timeline so fallback still covers beginning,
    # middle, and end rather than only high-confidence action segments.
    indexes = []
    for i in range(max_items):
        pos = round(i * (len(clips) - 1) / max(1, max_items - 1))
        indexes.append(pos)
    return [clips[i] for i in sorted(set(indexes))]


def _index_add(index: Dict[str, Dict[str, List[str]]], group: str,
               key: str, memory_id: str) -> None:
    if not key:
        return
    index.setdefault(group, {}).setdefault(key, [])
    if memory_id not in index[group][key]:
        index[group][key].append(memory_id)


def build_video_memory(prediction: Dict[str, Any], video_log: Dict[str, Any] | None,
                       frame_cache_root: Path) -> Dict[str, Any]:
    """Build one persistent retrieval bundle for a processed video."""
    video_id = str(prediction.get("video_id", ""))
    clip_memory = []
    retrieval_index: Dict[str, Dict[str, List[str]]] = {
        "objects": {},
        "actions": {},
        "keywords": {},
        "time_bins": {},
        "review": {},
    }

    for i, clip in enumerate(prediction.get("clip_memory", []), 1):
        memory_id = f"{video_id}_c{i:03d}"
        item = {
            "memory_id": memory_id,
            "kind": "clip",
            "start": clip.get("start"),
            "end": clip.get("end"),
            "frame_ts": clip.get("frame_ts", []),
            "summary": clip.get("summary", ""),
            "frame_paths": [
                _frame_path(frame_cache_root, video_id, ts)
                for ts in clip.get("frame_ts", [])
            ],
        }
        clip_memory.append(item)
        start = float(clip.get("start", 0) or 0)
        _index_add(retrieval_index, "time_bins", f"{int(start // 30 * 30)}-{int(start // 30 * 30 + 30)}", memory_id)

    segment_memory = []
    evidence_memory = []
    for seg in prediction.get("segments", []):
        memory_id = str(seg.get("segment_id", ""))
        item = {
            "memory_id": memory_id,
            "kind": "segment",
            "start": seg.get("start"),
            "end": seg.get("end"),
            "action": seg.get("action", ""),
            "action_zh": seg.get("action_zh", ""),
            "objects": seg.get("objects", []),
            "caption": seg.get("caption", ""),
            "evidence_timestamps": seg.get("evidence_timestamps", []),
            "evidence_frame_paths": [
                _frame_path(frame_cache_root, video_id, ts)
                for ts in seg.get("evidence_timestamps", [])
            ],
            "confidence": seg.get("confidence"),
            "verification_status": seg.get("verification_status"),
            "uncertainty_type": seg.get("uncertainty_type", "none"),
            "uncertainty_reason": seg.get("uncertainty_reason", ""),
            "needs_human_review": seg.get("needs_human_review", False),
            "review_suggestion": seg.get("review_suggestion", ""),
        }
        segment_memory.append(item)
        evidence_memory.append({
            "target_id": memory_id,
            **seg.get("evidence_diagnosis", {}),
            "object_evidence": seg.get("object_evidence", []),
            "review_suggestion": seg.get("review_suggestion", ""),
        })
        _index_add(retrieval_index, "actions", str(seg.get("action", "")), memory_id)
        _index_add(retrieval_index, "actions", str(seg.get("action_zh", "")), memory_id)
        for obj in seg.get("objects", []):
            _index_add(retrieval_index, "objects", str(obj), memory_id)
        for keyword in _keywords_for_segment(seg):
            _index_add(retrieval_index, "keywords", keyword, memory_id)
        start = float(seg.get("start", 0) or 0)
        _index_add(retrieval_index, "time_bins", f"{int(start // 30 * 30)}-{int(start // 30 * 30 + 30)}", memory_id)
        if seg.get("needs_human_review"):
            _index_add(retrieval_index, "review", "needs_review", memory_id)
        utype = str(seg.get("uncertainty_type", "none") or "none")
        if utype != "none":
            _index_add(retrieval_index, "review", utype, memory_id)

    return {
        "version": MEMORY_VERSION,
        "video_id": video_id,
        "created_at": datetime.now().isoformat(),
        "video_info": (video_log or {}).get("video_info", {}),
        "phase": prediction.get("phase"),
        "phase_zh": prediction.get("phase_zh"),
        "phase_confidence": prediction.get("phase_confidence"),
        "clip_memory": clip_memory,
        "segment_memory": segment_memory,
        "evidence_memory": evidence_memory,
        "retrieval_index": retrieval_index,
        "run_trace_summary": {
            "step_count": len((video_log or {}).get("steps", [])),
            "steps": [s.get("step") for s in (video_log or {}).get("steps", [])],
        },
    }


def save_video_memory(run_dir: Path, memory: Dict[str, Any]) -> Path:
    video_id = str(memory.get("video_id", "unknown"))
    path = run_dir / "memory" / f"{video_id}.json"
    _write_json(path, memory)
    return path


def _empty_global_memory() -> Dict[str, Any]:
    return {
        "version": MEMORY_VERSION,
        "updated_at": None,
        "videos": {},
        "actions": {},
        "objects": {},
        "uncertainty_patterns": {},
    }


def load_global_memory(path: Path) -> Dict[str, Any]:
    memory = _read_json(path, _empty_global_memory())
    if not isinstance(memory, dict) or memory.get("version") != MEMORY_VERSION:
        return _empty_global_memory()
    for key in ("videos", "actions", "objects", "uncertainty_patterns"):
        memory.setdefault(key, {})
    return memory


def _bump_video_list(bucket: Dict[str, Any], video_id: str) -> None:
    videos = _dedupe_keep_order([video_id] + bucket.get("videos", []))
    bucket["videos"] = videos[:MAX_VIDEOS_PER_KEY]


def _add_example(bucket: Dict[str, Any], example: Dict[str, Any]) -> None:
    examples = bucket.setdefault("examples", [])
    dedupe_key = (example.get("video_id"), example.get("segment_id"))
    if not any((e.get("video_id"), e.get("segment_id")) == dedupe_key for e in examples):
        examples.insert(0, example)
    bucket["examples"] = examples[:MAX_EXAMPLES_PER_KEY]


def update_global_memory(global_memory: Dict[str, Any], video_memory: Dict[str, Any],
                         memory_path: Path | None = None) -> Dict[str, Any]:
    """Accumulate bounded cross-video action/object/uncertainty patterns."""
    video_id = str(video_memory.get("video_id", ""))
    global_memory["updated_at"] = datetime.now().isoformat()
    global_memory.setdefault("videos", {})[video_id] = {
        "phase": video_memory.get("phase"),
        "phase_zh": video_memory.get("phase_zh"),
        "phase_confidence": video_memory.get("phase_confidence"),
        "segment_count": len(video_memory.get("segment_memory", [])),
        "memory_path": str(memory_path) if memory_path else "",
    }

    for seg in video_memory.get("segment_memory", []):
        action = str(seg.get("action", "") or "unknown")
        action_bucket = global_memory.setdefault("actions", {}).setdefault(action, {
            "action_zh": seg.get("action_zh", ""),
            "count": 0,
            "videos": [],
            "objects": {},
            "examples": [],
        })
        action_bucket["count"] = int(action_bucket.get("count", 0)) + 1
        action_bucket["action_zh"] = action_bucket.get("action_zh") or seg.get("action_zh", "")
        _bump_video_list(action_bucket, video_id)
        for obj in seg.get("objects", []):
            action_bucket.setdefault("objects", {})
            action_bucket["objects"][obj] = int(action_bucket["objects"].get(obj, 0)) + 1
        _add_example(action_bucket, {
            "video_id": video_id,
            "segment_id": seg.get("memory_id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "caption": seg.get("caption"),
            "confidence": seg.get("confidence"),
        })

        for obj in seg.get("objects", []):
            obj_bucket = global_memory.setdefault("objects", {}).setdefault(obj, {
                "count": 0,
                "videos": [],
                "actions": {},
                "examples": [],
            })
            obj_bucket["count"] = int(obj_bucket.get("count", 0)) + 1
            _bump_video_list(obj_bucket, video_id)
            obj_bucket.setdefault("actions", {})
            obj_bucket["actions"][action] = int(obj_bucket["actions"].get(action, 0)) + 1
            _add_example(obj_bucket, {
                "video_id": video_id,
                "segment_id": seg.get("memory_id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "action": action,
                "caption": seg.get("caption"),
            })

        uncertainty = str(seg.get("uncertainty_type", "none") or "none")
        if uncertainty != "none":
            pattern_bucket = global_memory.setdefault("uncertainty_patterns", {}).setdefault(uncertainty, {
                "count": 0,
                "videos": [],
                "examples": [],
            })
            pattern_bucket["count"] = int(pattern_bucket.get("count", 0)) + 1
            _bump_video_list(pattern_bucket, video_id)
            _add_example(pattern_bucket, {
                "video_id": video_id,
                "segment_id": seg.get("memory_id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "action": action,
                "reason": seg.get("uncertainty_reason", ""),
                "review_suggestion": seg.get("review_suggestion", ""),
            })

    return global_memory


def save_global_memory(path: Path, global_memory: Dict[str, Any]) -> None:
    _write_json(path, global_memory)


def build_global_summary(global_memory: Dict[str, Any], top_k: int = 8) -> Dict[str, Any]:
    """Return a compact summary useful for reports or a demo sidebar."""
    action_counts = Counter({
        k: int(v.get("count", 0))
        for k, v in global_memory.get("actions", {}).items()
    })
    object_counts = Counter({
        k: int(v.get("count", 0))
        for k, v in global_memory.get("objects", {}).items()
    })
    uncertainty_counts = Counter({
        k: int(v.get("count", 0))
        for k, v in global_memory.get("uncertainty_patterns", {}).items()
    })
    return {
        "video_count": len(global_memory.get("videos", {})),
        "top_actions": action_counts.most_common(top_k),
        "top_objects": object_counts.most_common(top_k),
        "top_uncertainty_patterns": uncertainty_counts.most_common(top_k),
    }


def _memory_by_id(video_memory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key in ("clip_memory", "segment_memory"):
        for item in video_memory.get(key, []):
            out[str(item.get("memory_id", ""))] = item
    return out


def retrieve_context(question: str, video_memory: Dict[str, Any],
                     global_memory: Dict[str, Any] | None = None,
                     max_items: int = 8) -> Dict[str, Any]:
    """Retrieve compact context for an interactive video-QA prompt.

    This is intentionally lexical and deterministic.  The Demo can pass the
    returned context to an LLM/VLM, and optionally load listed frame paths.
    """
    text = str(question)
    index = video_memory.get("retrieval_index", {})
    hits: List[str] = []
    query_terms = _expand_query_terms(text)
    time_ranges = _parse_time_ranges(text, video_memory)

    for item in video_memory.get("segment_memory", []) + video_memory.get("clip_memory", []):
        if _overlaps(item, time_ranges):
            hits.append(str(item.get("memory_id", "")))

    for group in ("actions", "objects", "keywords", "time_bins"):
        for key, memory_ids in index.get(group, {}).items():
            if key and any(key in term or term in key for term in query_terms):
                hits.extend(memory_ids)

    if any(cue in text for cue in REVIEW_CUES):
        for memory_ids in index.get("review", {}).values():
            hits.extend(memory_ids)

    if not hits:
        # Fallback: surface reliable structured segments plus a sampled global
        # timeline, so broad questions still get whole-video context.
        segments = sorted(
            video_memory.get("segment_memory", []),
            key=lambda x: _as_float(x.get("confidence"), 0.0),
            reverse=True,
        )
        hits.extend(str(s.get("memory_id", "")) for s in segments[: max(1, max_items // 2)])
        hits.extend(str(c.get("memory_id", "")) for c in _timeline_sample(
            video_memory, max_items=max(1, max_items - len(hits))
        ))

    by_id = _memory_by_id(video_memory)
    selected = []
    for memory_id in _dedupe_keep_order(hits):
        if memory_id in by_id:
            selected.append(by_id[memory_id])
        if len(selected) >= max_items:
            break

    cross_video_hints = {
        "actions": {},
        "objects": {},
        "uncertainty_patterns": {},
    }
    if global_memory:
        for item in selected:
            action = str(item.get("action", ""))
            if action and action in global_memory.get("actions", {}):
                cross_video_hints["actions"][action] = global_memory["actions"][action]
            for obj in item.get("objects", []):
                if obj in global_memory.get("objects", {}):
                    cross_video_hints["objects"][obj] = global_memory["objects"][obj]
            uncertainty = str(item.get("uncertainty_type", "none"))
            if uncertainty != "none" and uncertainty in global_memory.get("uncertainty_patterns", {}):
                cross_video_hints["uncertainty_patterns"][uncertainty] = (
                    global_memory["uncertainty_patterns"][uncertainty]
                )

    needs_review_query = any(cue in text for cue in REVIEW_CUES)
    return {
        "question": question,
        "video_id": video_memory.get("video_id"),
        "selected_memory": selected,
        "cross_video_hints": cross_video_hints,
        "review_suggestions": (
            get_review_suggestions(video_memory) if needs_review_query else None
        ),
        "retrieval_debug": {
            "query_terms": query_terms,
            "time_ranges": time_ranges,
            "hit_count": len(_dedupe_keep_order(hits)),
        },
    }


def get_review_suggestions(video_memory: Dict[str, Any],
                           include_all_uncertain: bool = True) -> Dict[str, Any]:
    """Return a deterministic human-review checklist for one video."""
    items = []
    for seg in video_memory.get("segment_memory", []):
        conf = _as_float(seg.get("confidence"), 0.0)
        needs_review = bool(seg.get("needs_human_review"))
        uncertain = str(seg.get("uncertainty_type", "none") or "none") != "none"
        if not needs_review and not (include_all_uncertain and uncertain):
            continue
        suggestion = str(seg.get("review_suggestion", "")).strip()
        if not suggestion:
            if conf < 0.55:
                suggestion = "人工复核动作类别、关键物体和起止边界"
            elif uncertain:
                suggestion = "人工抽查该片段的不确定性来源是否成立"
            else:
                suggestion = "人工抽查该片段标注是否与画面一致"
        items.append({
            "segment_id": seg.get("memory_id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "action": seg.get("action"),
            "action_zh": seg.get("action_zh"),
            "confidence": seg.get("confidence"),
            "verification_status": seg.get("verification_status"),
            "uncertainty_type": seg.get("uncertainty_type", "none"),
            "uncertainty_reason": seg.get("uncertainty_reason", ""),
            "review_suggestion": suggestion,
        })

    items.sort(key=lambda x: (_as_float(x.get("confidence"), 0.0), _as_float(x.get("start"), 0.0)))
    return {
        "video_id": video_memory.get("video_id"),
        "needs_review_count": len(items),
        "items": items,
    }


def answer_question(question: str, context: Dict[str, Any], *,
                    video_memory: Dict[str, Any] | None = None,
                    model: str | None = None) -> Dict[str, Any]:
    """Generate a natural-language answer from retrieve_context() output.

    Pass ``video_memory`` (the full bundle from build_video_memory) to include
    the complete clip_memory scene timeline in the prompt — analogous to how
    VideoARM injects the full HM³ so the LLM can answer semantic questions
    without relying solely on keyword-retrieved segments.

    Returns {"answer": str, "cited_timestamps": list[float]}.
    """
    from . import api_client, config  # deferred to avoid circular import at module load

    video_id = context.get("video_id", "")
    selected = context.get("selected_memory", [])

    if not selected and not video_memory:
        return {"answer": "未找到相关片段，无法回答该问题。", "cited_timestamps": []}

    # ── 1. Full clip timeline (videoarm HM³ style) ──────────────────────────
    clip_lines: List[str] = []
    clips = (video_memory or {}).get("clip_memory", [])
    for clip in clips:
        start = clip.get("start", "?")
        end = clip.get("end", "?")
        summary = str(clip.get("summary", ""))[:150]
        clip_lines.append(f"[{start}-{end}s] {summary}")

    # ── 2. Retrieved structured segments ────────────────────────────────────
    seg_lines: List[str] = []
    for item in selected:
        start = item.get("start", "?")
        end = item.get("end", "?")
        if item.get("kind") == "segment":
            action_zh = item.get("action_zh") or item.get("action", "")
            caption = item.get("caption", "")
            conf = item.get("confidence", "")
            flag = " ⚠需复核" if item.get("needs_human_review") else ""
            review = str(item.get("review_suggestion", "")).strip()
            reason = str(item.get("uncertainty_reason", "")).strip()
            extra = ""
            if flag and review:
                extra += f" 复核建议: {review}"
            if reason:
                extra += f" 不确定性: {reason[:120]}"
            seg_lines.append(f"[{start}-{end}s] {action_zh}(conf={conf}){flag}: {caption}{extra}")
        else:
            summary = str(item.get("summary", ""))[:150]
            seg_lines.append(f"[{start}-{end}s] 场景: {summary}")

    # ── 3. Cross-video hints ─────────────────────────────────────────────────
    cross_lines: List[str] = []
    for action, bucket in context.get("cross_video_hints", {}).get("actions", {}).items():
        zh = bucket.get("action_zh") or action
        cross_lines.append(f"  {zh}({action}) 跨视频共出现 {bucket.get('count', 0)} 次")

    clip_block = "\n".join(clip_lines) if clip_lines else "（无场景快照）"
    seg_block = "\n".join(seg_lines) if seg_lines else "（无精确匹配片段）"
    cross_block = "\n".join(cross_lines) if cross_lines else "（无）"

    prompt = f"""你是实验视频问答助手。以下是视频 {video_id} 的完整场景时间线（粗粒度）：

{clip_block}

以下是与问题相关的精确动作片段（结构化标注）：

{seg_block}

跨视频参考信息：
{cross_block}

请综合场景时间线和精确片段，用中文简洁回答问题，回答中必须引用具体时间戳（格式如 12s 或 12-34s）。

问题：{question}

只输出合法 JSON，字段：
{{
  "answer": "你的回答（含时间戳引用）",
  "cited_timestamps": [引用到的时间点列表，浮点数]
}}"""

    result = api_client.ask_json(prompt, model=model or config.STRUCTURED_MODEL, max_tokens=800)
    if not isinstance(result, dict):
        return {"answer": str(result), "cited_timestamps": []}
    timestamps: List[float] = []
    for t in (result.get("cited_timestamps") or []):
        try:
            timestamps.append(float(t))
        except (TypeError, ValueError):
            pass
    return {"answer": str(result.get("answer", "")), "cited_timestamps": timestamps}
