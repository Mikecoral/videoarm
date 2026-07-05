from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_v2.vqa.types import KeyframeContext


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def frame_time(frame: dict[str, Any], sample_fps: float) -> float:
    return float(frame.get("frame_index", 0)) / sample_fps


def make_frame_caption(objects: list[str], hands: list) -> str:
    obj_text = "、".join(objects) if objects else "未标注关键物体"

    if not hands:
        hand_text = "未标注手部"
    else:
        parts = []
        for h in hands:
            if isinstance(h, dict):
                side = h.get("side", "?")
                visible = h.get("visible", True)
                holding = h.get("holding")
                if not visible:
                    parts.append(f"{side}手不可见")
                elif holding:
                    parts.append(f"{side}手持{holding}")
                else:
                    parts.append(f"{side}手可见")
            elif isinstance(h, str):
                parts.append(f"{h}手")
            else:
                parts.append(str(h))
        hand_text = "；".join(parts) if parts else "未标注手部"

    return f"画面中可见物体：{obj_text}；手部：{hand_text}。"


def extract_video_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    try:
        import cv2
    except Exception:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = max(0, int(round(float(timestamp) * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output_path), frame))
    finally:
        cap.release()


def keyframes_from_prediction_segment(
    *,
    seg: dict[str, Any],
    dataset_root: Path,
    video_path: str,
    video_id: str,
    max_keyframes: int,
) -> list[KeyframeContext]:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    evidence_timestamps = seg.get("evidence_timestamps") or []
    timestamps = []
    for value in evidence_timestamps:
        try:
            ts = float(value)
        except (TypeError, ValueError):
            continue
        if start <= ts <= end and ts not in timestamps:
            timestamps.append(ts)
    if not timestamps:
        if max_keyframes <= 1:
            timestamps = [(start + end) / 2.0]
        else:
            timestamps = [start, (start + end) / 2.0, end]
    timestamps = timestamps[:max_keyframes]

    source = Path(video_path)
    if not source.is_absolute():
        source = dataset_root / source
    objects = _unique_preserve_order([str(obj) for obj in seg.get("objects", []) if obj])
    contexts: list[KeyframeContext] = []
    for ts in timestamps:
        frame_index = int(round(ts))
        rel_path = Path("frames") / str(video_id) / f"{int(round(ts * 1000)):09d}.jpg"
        output_path = dataset_root / rel_path
        frame_path = str(output_path) if extract_video_frame(source, ts, output_path) else None
        contexts.append(
            KeyframeContext(
                frame_index=frame_index,
                time=round(ts, 3),
                frame_path=frame_path,
                objects=objects,
                hands=[],
                caption=make_frame_caption(objects, []),
            )
        )
    return contexts


def select_keyframes(
    spatial: dict[str, Any] | None,
    start: float,
    end: float,
    dataset_root: Path,
    max_keyframes: int,
) -> list[KeyframeContext]:
    if not spatial or not spatial.get("frames"):
        mid = round((start + end) / 2.0, 3)
        return [
            KeyframeContext(
                frame_index=int(round(mid)),
                time=mid,
                frame_path=None,
                caption="该片段没有可用空间标注帧；使用片段中点作为证据时间。",
            )
        ]

    sample_fps = float(spatial.get("sample_fps", 1.0))
    frames = spatial.get("frames", [])
    in_segment = [
        frame for frame in frames if start <= frame_time(frame, sample_fps) <= end
    ]
    if not in_segment:
        targets = [(start + end) / 2.0]
    elif max_keyframes <= 1:
        targets = [(start + end) / 2.0]
    else:
        targets = [start, (start + end) / 2.0, end]
        if max_keyframes > 3:
            step = (end - start) / max(1, max_keyframes - 1)
            targets = [start + i * step for i in range(max_keyframes)]

    selected: list[dict[str, Any]] = []
    candidate_pool = in_segment or frames
    for target in targets:
        nearest = min(candidate_pool, key=lambda f: abs(frame_time(f, sample_fps) - target))
        if nearest not in selected:
            selected.append(nearest)
        if len(selected) >= max_keyframes:
            break

    contexts = []
    for frame in selected:
        objects = _unique_preserve_order(
            [obj.get("category", "") for obj in frame.get("objects", [])]
        )
        hands = _unique_preserve_order(
            [hand.get("side", "") for hand in frame.get("hands", [])]
        )
        frame_path = frame.get("frame_path")
        full_path = str(dataset_root / frame_path) if frame_path else None
        contexts.append(
            KeyframeContext(
                frame_index=int(frame.get("frame_index", 0)),
                time=round(frame_time(frame, sample_fps), 3),
                frame_path=full_path,
                objects=objects,
                hands=hands,
                caption=make_frame_caption(objects, hands),
            )
        )
    return contexts
