"""Batch Omni-only: segmentation + caption, no verify/repair. Fast.

    python -m agent_v2.run_batch_omni_only --split all
"""

from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from . import config, media, ontology
from .pipeline_omni import hypothesize_phase_omni, segment_actions_omni
from .pipeline import _normalize_segments, _clip_final_overlaps, _writer_gate
from .runlog import RunLog

CONCURRENCY = 1


def _load_split(split: str) -> List[Dict[str, Any]]:
    path = config.DATA_ROOT / "splits" / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))["videos"]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Omni-only batch segmentation")
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
    tag = f"omni_{args.split}_{len(video_ids)}vids"
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / "predictions.json"
    predictions: List[Dict[str, Any]] = []
    done_ids: set = set()
    if args.resume and pred_path.exists():
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        done_ids = {p["video_id"] for p in predictions}

    phases = ontology.load_ontology()

    print(f"[omni] split={args.split}  videos={len(video_ids)}  resume={args.resume}")
    print(f"[omni] model={config.OMNI_MODEL}  concurrency={CONCURRENCY}  out_dir={run_dir}")

    # Filter out already-done videos
    pending = [(i, entry) for i, entry in enumerate(videos, 1)
               if str(entry["video_id"]) not in done_ids]
    total = len(videos)

    def process_one(idx: int, entry: Dict[str, Any]) -> Dict[str, Any]:
        vid = str(entry["video_id"])
        video_path = config.DATA_ROOT / entry["video_path"]
        t0 = time.time()

        dur, fps, w, h = media.probe(video_path)
        video = media.VideoInfo(video_id=vid, path=video_path,
                                duration=dur, fps=fps, width=w, height=h, frames=[])

        log = RunLog()
        log.start_video(vid, {"duration": dur, "fps": fps, "pipeline": "omni_only"})
        phase_res = hypothesize_phase_omni(video, phases, log)
        segments = segment_actions_omni(video, phases, phase_res, log)

        for seg in segments:
            seg.setdefault("confidence", 0.5)
            seg.setdefault("verification_status", "partial")
            seg.setdefault("needs_human_review", True)
            seg.setdefault("uncertainty_reason", "Omni-only, 未做逐帧视觉核验")
            seg.setdefault("uncertainty_type", "mixed")
            seg.setdefault("review_suggestion", "人工复核该片段的动作、物体和边界")
            seg.setdefault("evidence_diagnosis", {})
            seg.setdefault("object_evidence",
                [{"object": o, "present": True} for o in seg.get("objects", [])])

        final = _writer_gate(segments, log, keep_rejected=True)
        final = _clip_final_overlaps(final)

        for j, seg in enumerate(final, 1):
            seg["segment_id"] = f"{vid}_s{j:03d}"

        prediction = {
            "video_id": vid,
            "video_path": f"videos/{vid}.mp4",
            "phase": phase_res.get("phase"),
            "phase_zh": phase_res.get("phase_zh"),
            "phase_confidence": phase_res.get("confidence"),
                "alternative_phases": phase_res.get("alternative_phases", []),
                "segments": [
                    {"segment_id": s["segment_id"], "start": s["start"], "end": s["end"],
                     "action": s["action"], "action_zh": s["action_zh"],
                     "objects": s["objects"], "caption": s["caption"],
                     "evidence_timestamps": s["evidence_timestamps"],
                     "confidence": s["confidence"],
                     "verification_status": s["verification_status"],
                     "uncertainty_reason": s["uncertainty_reason"],
                     "needs_human_review": s["needs_human_review"],
                     "uncertainty_type": s.get("uncertainty_type", "mixed"),
                     "review_suggestion": s.get("review_suggestion", ""),
                     "evidence_diagnosis": s.get("evidence_diagnosis", {}),
                     "object_evidence": s["object_evidence"]}
                    for s in final
                ],
                "clip_memory": [],
                "processing_note": "omni-only pipeline: no frame extraction, no verification",
            }
        dt = time.time() - t0
        return {"idx": idx, "prediction": prediction, "elapsed": dt,
                "n_segs": len(final), "phase_zh": phase_res.get("phase_zh", "")}

    wall_start = time.time()
    completed = len(done_ids)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_vid = {executor.submit(process_one, idx, entry): str(entry["video_id"])
                         for idx, entry in pending}
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                result = future.result()
                completed += 1
                predictions.append(result["prediction"])
                predictions.sort(key=lambda p: video_ids.index(p["video_id"])
                                 if p["video_id"] in video_ids else 999)
                _write_json(pred_path, predictions)
                print(f"[{completed}/{total}] {vid}: "
                      f"{result['n_segs']} segs ({result['phase_zh']}) "
                      f"in {result['elapsed']:.0f}s", flush=True)
            except Exception as e:
                completed += 1
                print(f"[{completed}/{total}] {vid}: ERROR {e}", file=sys.stderr, flush=True)
                predictions.append({
                    "video_id": vid, "video_path": f"videos/{vid}.mp4",
                    "segments": [], "processing_note": f"error: {e}",
                })
                _write_json(pred_path, predictions)

    total_elapsed = time.time() - wall_start
    _write_json(run_dir / "run_info.json", {
        "split": args.split, "pipeline": "omni_only",
        "model": config.OMNI_MODEL, "video_count": len(video_ids),
        "concurrency": CONCURRENCY, "total_seconds": round(total_elapsed, 1),
    })
    print(f"\n[omni] {len(predictions)} videos in {total_elapsed:.0f}s → {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
