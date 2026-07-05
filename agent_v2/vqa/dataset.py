from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_v2.vqa.io_utils import read_json
from agent_v2.vqa.keyframes import keyframes_from_prediction_segment, select_keyframes
from agent_v2.vqa.types import SegmentContext


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def contexts_from_split(dataset_root: str | Path, split: str, keyframes_per_segment: int) -> list[SegmentContext]:
    root = Path(dataset_root)
    split_data = read_json(root / "splits" / f"{split}.json")
    contexts: list[SegmentContext] = []
    for item in split_data["videos"]:
        temporal_path = item.get("temporal_annotation")
        if temporal_path:
            temporal = read_json(root / temporal_path)
        else:
            predicted_path = root / "predictions.json"
            temporal = None
            if predicted_path.exists():
                temporal = read_json(predicted_path)
        if not temporal or "segments" not in temporal:
            continue

        spatial = None
        if item.get("spatial_annotation"):
            spatial = _load_optional_json(root / item["spatial_annotation"])

        segments = temporal.get("segments", [])
        for i, seg in enumerate(segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            contexts.append(
                SegmentContext(
                    video_id=temporal.get("video_id", item["video_id"]),
                    video_path=temporal.get("video_path", item.get("video_path", "")),
                    phase=temporal.get("phase", ""),
                    phase_zh=temporal.get("phase_zh", ""),
                    segment_id=f"{item['video_id']}_seg_{i:04d}",
                    start=start,
                    end=end,
                    action=seg.get("action", ""),
                    action_zh=seg.get("action_zh", ""),
                    segment_caption=seg.get("caption", ""),
                    keyframes=select_keyframes(spatial, start, end, root, keyframes_per_segment),
                    previous_caption=segments[i - 1].get("caption") if i > 0 else None,
                    next_caption=segments[i + 1].get("caption") if i + 1 < len(segments) else None,
                )
            )
    return contexts


def contexts_from_predictions(
    predictions_path: str | Path,
    dataset_root: str | Path,
    keyframes_per_segment: int,
) -> list[SegmentContext]:
    root = Path(dataset_root)
    data = read_json(predictions_path)
    if isinstance(data, list):
        videos = data
    elif isinstance(data, dict):
        videos = data.get("videos", [])
    else:
        videos = []
    contexts: list[SegmentContext] = []
    for video in videos:
        segments = video.get("segments", [])
        spatial = _load_optional_json(root / "annotations" / "spatial" / f"{video['video_id']}.mp4.json")
        for i, seg in enumerate(segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if spatial:
                keyframes = select_keyframes(spatial, start, end, root, keyframes_per_segment)
            else:
                keyframes = keyframes_from_prediction_segment(
                    seg=seg,
                    dataset_root=root,
                    video_path=video.get("video_path", ""),
                    video_id=str(video.get("video_id", "")),
                    max_keyframes=keyframes_per_segment,
                )
            contexts.append(
                SegmentContext(
                    video_id=video.get("video_id", ""),
                    video_path=video.get("video_path", ""),
                    phase=video.get("phase", ""),
                    phase_zh=video.get("phase_zh", ""),
                    segment_id=f"{video.get('video_id', 'video')}_seg_{i:04d}",
                    start=start,
                    end=end,
                    action=seg.get("action", ""),
                    action_zh=seg.get("action_zh", ""),
                    segment_caption=seg.get("caption", ""),
                    keyframes=keyframes,
                    previous_caption=segments[i - 1].get("caption") if i > 0 else None,
                    next_caption=segments[i + 1].get("caption") if i + 1 < len(segments) else None,
                )
            )
    return contexts
