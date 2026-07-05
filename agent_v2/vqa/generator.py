from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_v2.vqa.io_utils import read_text
from agent_v2.vqa.openai_compatible import OpenAICompatibleClient
from agent_v2.vqa.types import SegmentContext


def _objects(ctx: SegmentContext) -> list[str]:
    seen = []
    for kf in ctx.keyframes:
        for obj in kf.objects:
            if obj and obj not in seen:
                seen.append(obj)
    return seen


def _evidence(ctx: SegmentContext) -> list[float]:
    return [kf.time for kf in ctx.keyframes] or [round((ctx.start + ctx.end) / 2.0, 3)]


class TemplateVQAGenerator:
    def generate(self, ctx: SegmentContext, questions_per_segment: int) -> list[dict[str, Any]]:
        objects = _objects(ctx)
        obj_answer = "、".join(objects) if objects else "关键帧中未标注明确物体"
        evidence = _evidence(ctx)
        candidates = [
            {
                "level": "L1",
                "category": "operation",
                "visual_modality": "video",
                "keyframe_index": None,
                "question_type": "action_recognition",
                "question": "这段视频主要展示了什么实验操作？",
                "answer": ctx.segment_caption or ctx.action_zh or ctx.action,
                "evidence_timestamps": evidence,
                "answer_basis": "segment_caption",
            },
            {
                "level": "L1",
                "category": "entity",
                "visual_modality": "image",
                "keyframe_index": 0,
                "question_type": "object_grounding",
                "question": "图中主要涉及哪些实验器具或物体？",
                "answer": obj_answer,
                "evidence_timestamps": evidence,
                "answer_basis": "keyframe_objects",
            },
            {
                "level": "L1",
                "category": "operation",
                "visual_modality": "video",
                "keyframe_index": None,
                "question_type": "phase_understanding",
                "question": "这段操作属于哪个实验阶段？",
                "answer": ctx.phase_zh or ctx.phase or "未知阶段",
                "evidence_timestamps": evidence,
                "answer_basis": "phase_label",
            },
        ]
        if ctx.previous_caption:
            candidates.append(
                {
                    "level": "L1",
                    "category": "procedure",
                    "visual_modality": "video",
                    "keyframe_index": None,
                    "question_type": "procedure_order",
                    "question": "在这段操作之前，实验人员刚进行了什么操作？",
                    "answer": ctx.previous_caption,
                    "evidence_timestamps": [ctx.start],
                    "answer_basis": "previous_segment_caption",
                }
            )
        if ctx.next_caption:
            candidates.append(
                {
                    "level": "L1",
                    "category": "procedure",
                    "visual_modality": "video",
                    "keyframe_index": None,
                    "question_type": "procedure_order",
                    "question": "在这段操作之后，实验人员接下来进行了什么操作？",
                    "answer": ctx.next_caption,
                    "evidence_timestamps": [ctx.end],
                    "answer_basis": "next_segment_caption",
                }
            )
        return candidates[:questions_per_segment]


class MLLMVQAGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.client = OpenAICompatibleClient(config)
        self.prompt = read_text(Path(__file__).parent / "prompts" / "agent_v2.vqa.md")

    def generate(self, ctx: SegmentContext, questions_per_segment: int) -> list[dict[str, Any]]:
        send_images = bool(self.client.config.get("send_images", False)) if hasattr(self.client, "config") else False
        image_paths = [kf.frame_path for kf in ctx.keyframes if kf.frame_path] if send_images else []
        keyframes = [
            {
                "index": index,
                "objects": kf.objects,
                "hands": kf.hands,
                "caption": kf.caption,
            }
            for index, kf in enumerate(ctx.keyframes)
        ]
        payload = {
            "questions_per_segment": questions_per_segment,
            "background": {
                "video_id": ctx.video_id,
                "phase": ctx.phase_zh or ctx.phase,
                "segment_caption": ctx.segment_caption,
                "action": ctx.action_zh or ctx.action,
                "previous_segment_caption": ctx.previous_caption,
                "next_segment_caption": ctx.next_caption,
                "keyframes": keyframes,
            },
            "allowed_schema": {
                "vqa": [
                    {
                        "level": "L1 | L2",
                        "category": "entity | operation | procedure | state | attribute | function | safety",
                        "visual_modality": "video | image",
                        "keyframe_index": "integer index from background.keyframes when visual_modality=image, otherwise null",
                        "question_type": "action_recognition | object_grounding | state_change | procedure_order | visual_detail | function_reasoning | safety_reasoning",
                        "question": "string",
                        "answer": "string",
                        "evidence_description": "string",
                        "answer_basis": "segment_caption | keyframe_caption | temporal_context | lab_common_knowledge | keyframe_caption_and_common_lab_knowledge",
                        "distractors": ["string", "string", "string"],
                    }
                ]
            },
        }
        result = self.client.chat_json(
            self.prompt,
            user_text=json.dumps(payload, ensure_ascii=False),
            image_paths=image_paths,
        )
        items = result.get("vqa", [])
        if not isinstance(items, list):
            raise RuntimeError("MLLM response must contain list field: vqa")
        return items[:questions_per_segment]


def build_generator(config: dict[str, Any]):
    if config.get("use_mllm"):
        return MLLMVQAGenerator(config)
    return TemplateVQAGenerator()
