"""Batch Omni pipeline for all dataset videos (test + dev).

    python -m agent_v2.run_batch [--split test|dev|all] [--limit N] [--resume]
"""

from __future__ import annotations

import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from . import api_client, config, media, memory, ontology
from .pipeline_omni import process_video_omni
from .runlog import RunLog


def _load_split(split: str) -> List[Dict[str, Any]]:
    path = config.DATA_ROOT / "splits" / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))["videos"]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Batch Omni video parser")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out_dir", default=str(config.OUTPUT_DIR))
    args = ap.parse_args(argv)

    if args.split == "all":
        videos = _load_split("dev") + _load_split("test")
    else:
        videos = _load_split(args.split)
    if args.limit:
        videos = videos[: args.limit]

    video_ids = [str(v["video_id"]) for v in videos]
    tag = f"batch_{args.split}_{len(video_ids)}vids"
    run_dir = Path(args.out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / "predictions.json"
    global_memory_path = Path(args.out_dir) / "global_memory.json"

    predictions: List[Dict[str, Any]] = []
    done_ids: set = set()
    if args.resume and pred_path.exists():
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        done_ids = {p["video_id"] for p in predictions}

    global_mem = memory.load_global_memory(global_memory_path)
    phases = ontology.load_ontology()
    run_log = RunLog()
    cache_root = config.FRAME_CACHE_DIR

    print(f"[batch] split={args.split}  videos={len(video_ids)}  resume={args.resume}")
    print(f"[batch] omni={config.OMNI_MODEL}  mllm={config.VISION_MODEL}")
    print(f"[batch] out_dir={run_dir}")

    wall_start = time.time()
    for i, entry in enumerate(videos, 1):
        vid = str(entry["video_id"])
        if vid in done_ids:
            print(f"[{i}/{len(videos)}] {vid} already done, skip")
            continue

        video_path = config.DATA_ROOT / entry["video_path"]
        print(f"[{i}/{len(videos)}] {vid} ...", flush=True)
        t0 = time.time()

        try:
            video = media.load_video(vid, video_path, cache_root=cache_root)
            print(f"  {len(video.frames)} frames, {video.duration:.0f}s", flush=True)

            prediction = process_video_omni(vid, video, phases, run_log)

            # Video summary
            segs = prediction.get("segments", [])
            timeline = "\n".join(
                f"[{s['start']:.0f}s–{s['end']:.0f}s] {s.get('caption', '')}"
                for s in segs
            )
            summary = api_client.ask_text(
                f"你是化学实验视频分析专家。以下是视频 {vid}（总时长 {video.duration:.0f}s，"
                f"实验阶段：{prediction.get('phase_zh', '')}）的全部原子操作分段 caption：\n\n"
                f"{timeline}\n\n"
                "请用一段中文（3-5 句）概括整个视频的核心实验流程（目的、步骤、器具、结果）。"
                "只输出概括文本。",
                model=config.STRUCTURED_MODEL, max_tokens=400,
            )
            prediction["video_summary"] = summary

            # Memory
            video_mem = memory.build_video_memory(
                prediction, run_log.videos[-1] if run_log.videos else {}, cache_root,
            )
            memory.save_video_memory(run_dir, video_mem)
            global_mem = memory.update_global_memory(
                global_mem, video_mem, memory_path=str(run_dir / "memory" / f"{vid}.json"),
            )
            memory.save_global_memory(global_memory_path, global_mem)

            predictions.append(prediction)
            _write_json(pred_path, predictions)

            dt = time.time() - t0
            verified = sum(1 for s in segs if s.get("verification_status") == "verified")
            print(f"  {len(segs)} segments (verified={verified}) in {dt:.0f}s", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            predictions.append({
                "video_id": vid, "video_path": entry["video_path"],
                "segments": [], "processing_note": f"error: {e}",
            })
            _write_json(pred_path, predictions)

    total = time.time() - wall_start
    # Write run info
    _write_json(run_dir / "run_info.json", {
        "split": args.split, "pipeline": "omni_batch",
        "omni_model": config.OMNI_MODEL, "mllm_model": config.VISION_MODEL,
        "video_count": len(video_ids), "total_seconds": round(total, 1),
        "global_memory_summary": memory.build_global_summary(global_mem),
    })
    _write_json(run_dir / "run_log.json", run_log.as_list())

    print(f"\n[batch] {len(predictions)} videos in {total:.0f}s → {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
