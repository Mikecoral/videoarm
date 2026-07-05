from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KeyframeContext:
    frame_index: int
    time: float
    frame_path: str | None
    objects: list[str] = field(default_factory=list)
    hands: list[str] = field(default_factory=list)
    caption: str = ""


@dataclass
class SegmentContext:
    video_id: str
    video_path: str
    phase: str
    phase_zh: str
    segment_id: str
    start: float
    end: float
    action: str
    action_zh: str
    segment_caption: str
    keyframes: list[KeyframeContext]
    previous_caption: str | None = None
    next_caption: str | None = None

    def background(self) -> str:
        frame_lines = [
            f"- {kf.time:.2f}s: {kf.caption}" for kf in self.keyframes
        ]
        return "\n".join(
            [
                f"视频: {self.video_id}",
                f"阶段: {self.phase_zh or self.phase}",
                f"片段: {self.start:.2f}s-{self.end:.2f}s",
                f"动作标签: {self.action_zh or self.action}",
                f"片段 caption: {self.segment_caption}",
                f"上一片段: {self.previous_caption or '无'}",
                f"下一片段: {self.next_caption or '无'}",
                "关键帧 caption:",
                *frame_lines,
            ]
        )

    def to_background_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "phase": self.phase,
            "phase_zh": self.phase_zh,
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "action": self.action,
            "action_zh": self.action_zh,
            "segment_caption": self.segment_caption,
            "previous_caption": self.previous_caption,
            "next_caption": self.next_caption,
            "keyframes": [
                {
                    "frame_index": kf.frame_index,
                    "time": kf.time,
                    "frame_path": kf.frame_path,
                    "objects": kf.objects,
                    "hands": kf.hands,
                    "caption": kf.caption,
                }
                for kf in self.keyframes
            ],
        }

