"""Batch entry point.

    python -m agent_v2.run --split test
    python -m agent_v2.run --split dev
    python -m agent_v2.run --video_id 0061

Writes predictions.json and run_log.json incrementally so a crash mid-run
still leaves valid partial output, and finished videos are skipped on resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from . import config, media, ontology, pipeline
from .runlog import RunLog


def _load_split(split: str) -> List[Dict[str, Any]]:
    path = config.DATA_ROOT / "splits" / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["videos"]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LabARM-HV batch video parser")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--video_id", help="process a single video id (overrides --split)")
    ap.add_argument("--out_dir", default=str(config.OUTPUT_DIR))
    ap.add_argument("--resume", action="store_true", help="skip videos already in predictions.json")
    ap.add_argument("--limit", type=int, default=0, help="max videos (0 = all)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    cache_root = out_dir / "frames_cache"
    pred_path = out_dir / "predictions.json"
    log_path = out_dir / "run_log.json"

    # resolve the video work list
    if args.video_id:
        videos = [{"video_id": args.video_id, "video_path": f"videos/{args.video_id}.mp4"}]
    elif args.split == "all":
        videos = _load_split("dev") + _load_split("test")
    else:
        videos = _load_split(args.split)
    if args.limit:
        videos = videos[: args.limit]

    predictions: List[Dict[str, Any]] = []
    done_ids = set()
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
    print(f"[labarm] loaded ontology: {len(phases)} phases")

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
            video = media.load_video(vid, video_path, cache_root=cache_root,
                                     release_frame_dir=frame_dir)
            print(f"    {len(video.frames)} frames, {video.duration:.0f}s -> processing", flush=True)
            pred = pipeline.process_video(vid, video, phases, run_log)
        except Exception as e:  # noqa: BLE001 - never let one video kill the batch
            print(f"    ERROR on {vid}: {e}", file=sys.stderr)
            pred = {"video_id": vid, "video_path": entry["video_path"],
                    "segments": [], "processing_note": f"error: {e}"}
        predictions.append(pred)
        # incremental persist
        _write_json(pred_path, predictions)
        _write_json(log_path, run_log.as_list())
        print(f"    done in {time.time() - t0:.1f}s, {len(pred.get('segments', []))} segments")

    print(f"[labarm] wrote {pred_path} and {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
