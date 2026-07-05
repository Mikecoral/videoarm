from __future__ import annotations

import hashlib
import random
from typing import Any

from agent_v2.vqa.types import SegmentContext


LETTERS = ["A", "B", "C", "D", "E", "F"]

PHASE_DISTRACTORS = [
    "称量",
    "开反应",
    "过滤与抽滤",
    "萃取",
    "薄层色谱分析",
    "分析样品制备",
    "柱层析准备",
    "标记与暂存",
    "旋蒸",
]

ACTION_DISTRACTORS = [
    "称取固体样品。",
    "向反应体系加入液体。",
    "摇晃或混合容器。",
    "将液体转移至分液漏斗。",
    "振摇分液漏斗并放气。",
    "静置等待液相分层。",
    "点样 TLC 板。",
    "配置 TLC 展开剂。",
    "安装装样柱。",
    "设置过柱机分离参数。",
]

OBJECT_DISTRACTORS = [
    "移液枪、EP管",
    "分液漏斗、烧杯",
    "TLC板、铅笔",
    "装样柱、分离柱",
    "圆底烧瓶、注射器",
    "布氏漏斗、滤纸",
]


def _stable_rng(seed: str) -> random.Random:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:8], 16))


def _normalize(text: Any) -> str:
    return str(text or "").strip()


def _canonical_object_set(text: str) -> str:
    normalized = text.replace("，", "、").replace(",", "、").replace("；", "、").replace(";", "、")
    items = sorted({item.strip() for item in normalized.split("、") if item.strip()})
    return "、".join(items)


def _dedupe(values: list[str], answer: str, *, object_set: bool = False) -> list[str]:
    canonical = _canonical_object_set if object_set else _normalize
    seen = {canonical(_normalize(answer))}
    out = []
    for value in values:
        text = _normalize(value)
        key = canonical(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _objects_answer(ctx: SegmentContext) -> str:
    objects = []
    for keyframe in ctx.keyframes:
        for obj in keyframe.objects:
            if obj and obj not in objects:
                objects.append(obj)
    return "、".join(objects) if objects else "关键帧中未标注明确物体"


def _time_answer(ctx: SegmentContext) -> str:
    return f"{ctx.start:.2f}s 到 {ctx.end:.2f}s"


def _same_video(contexts: list[SegmentContext], ctx: SegmentContext) -> list[SegmentContext]:
    return [item for item in contexts if item.video_id == ctx.video_id and item.segment_id != ctx.segment_id]


def _same_phase(contexts: list[SegmentContext], ctx: SegmentContext) -> list[SegmentContext]:
    return [
        item
        for item in contexts
        if item.phase == ctx.phase and item.segment_id != ctx.segment_id
    ]


def _candidate_distractors(question_type: str, answer: str, ctx: SegmentContext, contexts: list[SegmentContext]) -> list[str]:
    same_video = _same_video(contexts, ctx)
    same_phase = _same_phase(contexts, ctx)

    if question_type == "action_recognition":
        candidates = [c.segment_caption for c in same_video]
        candidates += [c.segment_caption for c in same_phase]
        candidates += [c.segment_caption for c in contexts if c.segment_id != ctx.segment_id]
        candidates += ACTION_DISTRACTORS
        return _dedupe(candidates, answer)

    if question_type == "object_grounding":
        candidates = [_objects_answer(c) for c in same_video]
        candidates += [_objects_answer(c) for c in same_phase]
        candidates += [_objects_answer(c) for c in contexts if c.segment_id != ctx.segment_id]
        candidates += OBJECT_DISTRACTORS
        return _dedupe(candidates, answer, object_set=True)

    if question_type == "temporal_grounding":
        candidates = [_time_answer(c) for c in same_video]
        candidates += [_time_answer(c) for c in same_phase]
        candidates += [_time_answer(c) for c in contexts if c.segment_id != ctx.segment_id]
        duration = max(1.0, ctx.end - ctx.start)
        candidates += [
            f"{max(0.0, ctx.start - duration):.2f}s 到 {max(0.0, ctx.end - duration):.2f}s",
            f"{ctx.start + duration:.2f}s 到 {ctx.end + duration:.2f}s",
            f"{max(0.0, ctx.start - 10.0):.2f}s 到 {max(0.0, ctx.end - 10.0):.2f}s",
        ]
        return _dedupe(candidates, answer)

    if question_type == "phase_understanding":
        candidates = [c.phase_zh or c.phase for c in contexts if c.segment_id != ctx.segment_id]
        candidates += PHASE_DISTRACTORS
        return _dedupe(candidates, answer)

    if question_type == "procedure_order":
        candidates = [c.segment_caption for c in same_video]
        candidates += [c.segment_caption for c in same_phase]
        candidates += [c.segment_caption for c in contexts if c.segment_id != ctx.segment_id]
        candidates += ACTION_DISTRACTORS
        return _dedupe(candidates, answer)

    candidates = [c.segment_caption for c in same_video + same_phase + contexts]
    candidates += ACTION_DISTRACTORS
    return _dedupe(candidates, answer)


def build_multiple_choice(
    qa: dict[str, Any],
    ctx: SegmentContext,
    contexts: list[SegmentContext],
    item_id: str,
    num_options: int = 4,
) -> dict[str, Any]:
    answer = _normalize(qa.get("answer")) or "未提供明确答案"
    question_type = _normalize(qa.get("question_type")) or "unknown"
    mllm_distractors = qa.get("distractors") or []
    if isinstance(mllm_distractors, list):
        distractors = _dedupe([_normalize(item) for item in mllm_distractors], answer)
    else:
        distractors = []
    if len(distractors) < num_options - 1:
        distractors += _candidate_distractors(question_type, answer, ctx, contexts)
        distractors = _dedupe(distractors, answer)

    options = [answer] + distractors[: max(0, num_options - 1)]
    fallback_pool = ACTION_DISTRACTORS + OBJECT_DISTRACTORS + PHASE_DISTRACTORS
    for value in fallback_pool:
        if len(options) >= num_options:
            break
        text = _normalize(value)
        if text and text not in options:
            options.append(text)

    rng = _stable_rng(item_id)
    rng.shuffle(options)
    options = options[:num_options]
    if answer not in options:
        options[-1] = answer
        rng.shuffle(options)

    answer_index = options.index(answer)
    labeled = {LETTERS[i]: option for i, option in enumerate(options)}
    return {
        "question": qa.get("question", ""),
        "options": labeled,
        "ground_truth": {
            "answer": LETTERS[answer_index],
            "answer_text": answer,
        },
        "distractor_source": "mllm_then_context",
    }
