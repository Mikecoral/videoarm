"""Omni batch entry point (qwen3.5-omni-plus, no frame extraction).

    python -m agent_v2.run_omni --split test
    python -m agent_v2.run_omni --split dev
    python -m agent_v2.run_omni --video_id 0061
    python -m agent_v2.run_omni --video_id 1 --limit 3

Each run auto-creates a timestamped subdirectory under outputs/ tagged with
``_omni`` so omni runs are easy to distinguish from frame-based runs.
Output schema is identical to run.py (predictions.json + run_log.json +
run_info.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from . import config, media, ontology
from . import pipeline_omni as pipeline
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
    ap = argparse.ArgumentParser(description="LabARM-HV omni video parser (no frame extraction)")
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
    tag = f"omni_{split_tag}_{len(video_ids)}vids"
    run_dir = _make_run_dir(Path(args.out_dir), tag)

    pred_path = run_dir / "predictions.json"
    log_path = run_dir / "run_log.json"
    info_path = run_dir / "run_info.json"

    run_start = datetime.now().isoformat()
    run_info: Dict[str, Any] = {
        "started_at": run_start,
        "split": split_tag,
        "pipeline": "omni",
        "model": config.OMNI_MODEL,
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

    phases = ontology.load_ontology()
    print(f"[omni] run dir : {run_dir}")
    print(f"[omni] split   : {split_tag}  videos: {len(video_ids)}  ids: {video_ids}")
    print(f"[omni] model   : {config.OMNI_MODEL}")
    print(f"[omni] ontology: {len(phases)} phases")

    wall_start = time.time()
    for i, entry in enumerate(videos, 1):
        vid = str(entry["video_id"])
        if vid in done_ids:
            print(f"[{i}/{len(videos)}] {vid} already done, skip")
            continue
        video_path = config.DATA_ROOT / entry["video_path"]
        print(f"[{i}/{len(videos)}] {vid} probing video ...", flush=True)
        t0 = time.time()
        try:
            # Only probe metadata; no frame extraction.
            duration, fps, w, h = media.probe(video_path)
            video = media.VideoInfo(
                video_id=vid, path=video_path,
                duration=duration, fps=fps, width=w, height=h,
                frames=[],
            )
            print(f"    duration={duration:.0f}s  {w}x{h} -> sending to omni", flush=True)
            pred = pipeline.process_video_omni(vid, video, phases, run_log)
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR on {vid}: {e}", file=sys.stderr)
            pred = {"video_id": vid, "video_path": entry["video_path"],
                    "segments": [], "processing_note": f"error: {e}"}
        predictions.append(pred)
        _write_json(pred_path, predictions)
        _write_json(log_path, run_log.as_list())
        _write_json(info_path, run_info)
        elapsed = time.time() - t0
        segs = len(pred.get("segments", []))
        print(f"    done in {elapsed:.1f}s, {segs} segments")

    total = time.time() - wall_start
    run_info["finished_at"] = datetime.now().isoformat()
    run_info["total_seconds"] = round(total, 1)
    _write_json(info_path, run_info)

    print(f"[omni] finished in {total:.0f}s")
    print(f"[omni] results  -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
