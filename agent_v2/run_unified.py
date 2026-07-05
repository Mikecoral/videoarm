"""Unified pipeline batch entry point.

    python -m agent_v2.run_unified --split test
    python -m agent_v2.run_unified --split dev
    python -m agent_v2.run_unified --video_id video1
    python -m agent_v2.run_unified --video_id 0061 --skip-vqa

Each run auto-creates a timestamped subdirectory under outputs/ tagged with
``_unified`` so unified runs are easy to distinguish.  Output schema matches
the other pipelines (predictions.json + run_log.json + run_info.json) with an
additional ``vqa`` array per prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Make vqa importable from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import config, media, memory, ontology
from . import pipeline_unified as pipeline
from .runlog import RunLog


def _load_split(split: str) -> List[Dict[str, Any]]:
    path = config.DATA_ROOT / "splits" / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["videos"]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run_dir(base: Path, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="LabARM-HV unified pipeline (omni → frame sub-seg → multi-agent → vqa)",
    )
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument(
        "--video_id", help="process a single video id (overrides --split)",
    )
    ap.add_argument(
        "--out_dir", default=str(config.OUTPUT_DIR),
        help="parent output directory; a timestamped subdir is created inside",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="skip videos already in predictions.json",
    )
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    ap.add_argument(
        "--skip-vqa", action="store_true",
        help="skip Stage 4 VQA generation",
    )
    args = ap.parse_args(argv)

    # Resolve the video work list
    if args.video_id:
        videos = [
            {"video_id": args.video_id, "video_path": f"videos/{args.video_id}.mp4"}
        ]
        split_tag = f"single_{args.video_id}"
    elif args.split == "all":
        videos = _load_split("dev") + _load_split("test")
        split_tag = "all"
    else:
        videos = _load_split(args.split)
        split_tag = args.split
    if args.limit:
        videos = videos[: args.limit]

    video_ids = [str(v["video_id"]) for v in videos]
    tag = f"unified_{split_tag}_{len(video_ids)}vids"
    run_dir = _make_run_dir(Path(args.out_dir), tag)

    pred_path = run_dir / "predictions.json"
    log_path = run_dir / "run_log.json"
    info_path = run_dir / "run_info.json"
    cache_root = config.FRAME_CACHE_DIR
    global_memory_path = Path(args.out_dir) / "global_memory.json"
    global_snapshot_path = run_dir / "global_memory_snapshot.json"

    run_start = datetime.now().isoformat()
    run_info: Dict[str, Any] = {
        "started_at": run_start,
        "split": split_tag,
        "pipeline": "unified_hxa",
        "omni_model": config.OMNI_MODEL,
        "mllm_model": config.VISION_MODEL,
        "llm_model": config.STRUCTURED_MODEL,
        "video_count": len(video_ids),
        "video_ids": video_ids,
        "finished_at": None,
        "total_seconds": None,
        "skip_vqa": args.skip_vqa,
    }
    _write_json(info_path, run_info)

    predictions: List[Dict[str, Any]] = []
    done_ids: set = set()
    if args.resume and pred_path.exists():
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        done_ids = {p["video_id"] for p in predictions}

    run_log = RunLog()
    if args.resume and log_path.exists():
        try:
            run_log.videos = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    global_memory_obj = memory.load_global_memory(global_memory_path)

    phases = ontology.load_ontology()
    print(f"[unified] run dir : {run_dir}")
    print(f"[unified] split   : {split_tag}  videos: {len(video_ids)}  ids: {video_ids}")
    print(f"[unified] omni    : {config.OMNI_MODEL}")
    print(f"[unified] mllm    : {config.VISION_MODEL}")
    print(f"[unified] llm     : {config.STRUCTURED_MODEL}")
    print(f"[unified] ontology: {len(phases)} phases")
    print(f"[unified] skip-vqa: {args.skip_vqa}")

    wall_start = time.time()
    for i, entry in enumerate(videos, 1):
        vid = str(entry["video_id"])
        if vid in done_ids:
            print(f"[{i}/{len(videos)}] {vid} already done, skip")
            continue

        video_path = config.DATA_ROOT / entry["video_path"]
        print(f"[{i}/{len(videos)}] {vid} loading frames ...", flush=True)
        t0 = time.time()
        try:
            video = media.load_video(vid, video_path, cache_root=cache_root)
            print(
                f"    {len(video.frames)} frames, {video.duration:.0f}s → processing",
                flush=True,
            )
            pred = pipeline.process_video_unified(
                vid, video, phases, run_log, skip_vqa=args.skip_vqa,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR on {vid}: {e}", file=sys.stderr)
            pred = {
                "video_id": vid,
                "video_path": entry["video_path"],
                "segments": [],
                "vqa": [],
                "processing_note": f"error: {e}",
            }

        predictions.append(pred)
        video_log = run_log.videos[-1] if run_log.videos else {}
        if pred.get("segments") or pred.get("clip_memory"):
            video_memory = memory.build_video_memory(pred, video_log, cache_root)
            video_memory_path = memory.save_video_memory(run_dir, video_memory)
            global_memory_obj = memory.update_global_memory(
                global_memory_obj, video_memory, memory_path=video_memory_path,
            )
            memory.save_global_memory(global_memory_path, global_memory_obj)
            memory.save_global_memory(global_snapshot_path, global_memory_obj)
            run_info["global_memory_summary"] = memory.build_global_summary(
                global_memory_obj,
            )

        _write_json(pred_path, predictions)
        _write_json(log_path, run_log.as_list())
        _write_json(info_path, run_info)
        elapsed = time.time() - t0
        segs = len(pred.get("segments", []))
        vqa_n = len(pred.get("vqa", []))
        print(f"    done in {elapsed:.1f}s, {segs} segments, {vqa_n} vqa")

    total = time.time() - wall_start
    run_info["finished_at"] = datetime.now().isoformat()
    run_info["total_seconds"] = round(total, 1)
    _write_json(info_path, run_info)

    print(f"[unified] finished in {total:.0f}s")
    print(f"[unified] results  → {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
