"""Hybrid hxa batch entry point.

Examples:
    python -m agent_v2.run_hybrid --split test
    python -m agent_v2.run_hybrid --split dev
    python -m agent_v2.run_hybrid --video_id 0061

The output schema is compatible with ``run.py`` and stored in timestamped
``outputs/*_hybrid_*`` directories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from . import config, media, memory, ontology
from . import pipeline_hybrid as pipeline
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
    ap = argparse.ArgumentParser(description="LabARM-HV hxa hybrid video parser")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--video_id", help="process a single video id (overrides --split)")
    ap.add_argument("--out_dir", default=str(config.OUTPUT_DIR),
                    help="parent output directory; a timestamped subdir is created inside")
    ap.add_argument("--resume", action="store_true", help="skip videos already in predictions.json")
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    args = ap.parse_args(argv)

    if args.video_id:
        videos = [{"video_id": args.video_id, "video_path": f"videos/{args.video_id}.mp4"}]
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
    run_dir = _make_run_dir(Path(args.out_dir), f"hybrid_{split_tag}_{len(video_ids)}vids")
    pred_path = run_dir / "predictions.json"
    log_path = run_dir / "run_log.json"
    info_path = run_dir / "run_info.json"
    cache_root = Path(args.out_dir) / "frames_cache"
    global_memory_path = Path(args.out_dir) / "global_memory.json"
    global_snapshot_path = run_dir / "global_memory_snapshot.json"

    run_info: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "split": split_tag,
        "pipeline": "hybrid_hxa",
        "mllm_model": config.VISION_MODEL,
        "llm_model": config.STRUCTURED_MODEL,
        "video_count": len(video_ids),
        "video_ids": video_ids,
        "finished_at": None,
        "total_seconds": None,
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
    global_memory = memory.load_global_memory(global_memory_path)
    phases = ontology.load_ontology()

    print(f"[hybrid] run dir : {run_dir}")
    print(f"[hybrid] split   : {split_tag}  videos: {len(video_ids)}  ids: {video_ids}")
    print(f"[hybrid] mllm    : {config.VISION_MODEL}")
    print(f"[hybrid] llm     : {config.STRUCTURED_MODEL}")
    print(f"[hybrid] ontology: {len(phases)} phases")

    wall_start = time.time()
    for i, entry in enumerate(videos, 1):
        vid = str(entry["video_id"])
        if vid in done_ids:
            print(f"[{i}/{len(videos)}] {vid} already done, skip")
            continue
        video_path = config.DATA_ROOT / entry["video_path"]
        frame_dir = config.DATA_ROOT / "frames" / vid
        print(f"[{i}/{len(videos)}] {vid} loading frames ...", flush=True)
        t0 = time.time()
        try:
            video = media.load_video(
                vid, video_path, cache_root=cache_root, release_frame_dir=frame_dir
            )
            print(f"    {len(video.frames)} frames, {video.duration:.0f}s -> hybrid processing", flush=True)
            pred = pipeline.process_video_hybrid(vid, video, phases, run_log)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR on {vid}: {e}", file=sys.stderr)
            pred = {"video_id": vid, "video_path": entry["video_path"],
                    "segments": [], "processing_note": f"error: {e}"}
        predictions.append(pred)
        video_log = run_log.videos[-1] if run_log.videos else {}
        if pred.get("segments") or pred.get("clip_memory"):
            video_memory = memory.build_video_memory(pred, video_log, cache_root)
            video_memory_path = memory.save_video_memory(run_dir, video_memory)
            global_memory = memory.update_global_memory(
                global_memory, video_memory, memory_path=video_memory_path
            )
            memory.save_global_memory(global_memory_path, global_memory)
            memory.save_global_memory(global_snapshot_path, global_memory)
            run_info["global_memory_summary"] = memory.build_global_summary(global_memory)
        _write_json(pred_path, predictions)
        _write_json(log_path, run_log.as_list())
        _write_json(info_path, run_info)
        print(f"    done in {time.time() - t0:.1f}s, {len(pred.get('segments', []))} segments")

    total = time.time() - wall_start
    run_info["finished_at"] = datetime.now().isoformat()
    run_info["total_seconds"] = round(total, 1)
    _write_json(info_path, run_info)

    print(f"[hybrid] finished in {total:.0f}s")
    print(f"[hybrid] results  -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
