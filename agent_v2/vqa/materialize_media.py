from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_v2.vqa.clips import make_clip
from agent_v2.vqa.io_utils import read_json, write_json


CATEGORY_BY_TYPE = {
    "action_recognition": "operation",
    "object_grounding": "entity",
    "visual_detail": "attribute",
    "procedure_order": "procedure",
    "temporal_grounding": "procedure",
    "state_change": "state",
    "phase_understanding": "operation",
    "function_reasoning": "function",
    "safety_reasoning": "safety",
}


def nearest_keyframe_index(item: dict[str, Any]) -> int:
    frames = item.get("background", {}).get("keyframes") or []
    if not frames:
        return 0
    timestamps = item.get("evidence_timestamps") or []
    if not timestamps:
        return 0
    target = float(timestamps[0])
    return min(
        range(len(frames)),
        key=lambda index: abs(float(frames[index].get("time", 0.0)) - target),
    )


def infer_modality(item: dict[str, Any]) -> tuple[str, int | None]:
    if item.get("visual_modality") in {"image", "video"}:
        modality = item["visual_modality"]
    elif item.get("visual_reference") == "keyframe":
        modality = "image"
    else:
        modality = "video"
    if modality == "image":
        return "image", nearest_keyframe_index(item)
    return "video", None


def enrich_item(item: dict[str, Any], dataset_root: Path, clips_dir: Path, generation_log: list[dict[str, Any]]) -> None:
    bg = item.get("background", {})
    frames = bg.get("keyframes") or []
    modality, keyframe_index = infer_modality(item)
    item.setdefault("level", "L1")
    item.setdefault("category", CATEGORY_BY_TYPE.get(item.get("question_type"), "operation"))
    item["visual_modality"] = modality

    visual = {
        "modality": modality,
        "keyframe_index": keyframe_index,
        "image_path": None,
        "video_path": bg.get("video_path"),
        "clip_path": None,
    }
    if modality == "image":
        if frames:
            index = max(0, min(int(keyframe_index or 0), len(frames) - 1))
            visual["keyframe_index"] = index
            visual["image_path"] = frames[index].get("frame_path")
    else:
        try:
            visual["clip_path"] = make_clip(
                dataset_root=dataset_root,
                video_path=bg.get("video_path", ""),
                video_id=item.get("video_id", ""),
                start=float(item.get("time_range", {}).get("start", 0.0)),
                end=float(item.get("time_range", {}).get("end", 0.0)),
                clips_dir=clips_dir,
            )
        except Exception as exc:
            generation_log.append(
                {
                    "segment_id": item.get("segment_id"),
                    "warning": f"clip generation failed: {exc}",
                }
            )
    item["visual"] = visual
    item["visual_reference"] = modality


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize VQA image/video assets and clip files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dataset-root", default="视频解析智能体数据包/hackathon_release")
    parser.add_argument("--clips-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    dataset_root = Path(args.dataset_root)
    clips_dir = Path(args.clips_dir) if args.clips_dir else dataset_root / "vqa" / "clips"
    data = read_json(input_path)
    generation_log = data.setdefault("generation_log", [])
    for item in data.get("vqa", []):
        enrich_item(item, dataset_root, clips_dir, generation_log)

    output_path = Path(args.output) if args.output else input_path
    write_json(output_path, data)
    print(f"wrote {output_path}")
    print(f"clips_dir: {clips_dir}")


if __name__ == "__main__":
    main()
