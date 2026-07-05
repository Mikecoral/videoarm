"""Full pipeline: Omni → verify → JSON → VQA → viewer.

    python -m agent_v2.run_full --video_id 1
    python -m agent_v2.run_full --split test
"""

from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path
from typing import Any, Dict, List

from . import api_client, config, media, memory, ontology
from .pipeline import (
    _action_catalog_for_repair, _clip_final_overlaps,
    _verify_and_maybe_repair, _writer_gate,
)
from .pipeline_omni import hypothesize_phase_omni, segment_actions_omni
from .runlog import RunLog


def _load_split(split: str) -> List[Dict[str, Any]]:
    return json.loads((config.DATA_ROOT / "splits" / f"{split}.json").read_text())["videos"]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_vqa(predictions: List[Dict], concurrency: int = 100) -> List[Dict]:
    """Generate template VQA items for all segments."""
    from .vqa.types import SegmentContext, KeyframeContext
    from .vqa.generator import TemplateVQAGenerator
    from concurrent.futures import ThreadPoolExecutor, as_completed

    generator = TemplateVQAGenerator()
    all_items: List[Dict] = []

    # Build contexts
    tasks = []
    for video in predictions:
        vid = video["video_id"]
        segs = video.get("segments", [])
        for i, seg in enumerate(segs):
            ts = (seg.get("evidence_timestamps") or [None])[0]
            if ts is None:
                ts = (seg["start"] + seg["end"]) / 2
            kf = KeyframeContext(
                frame_index=int(ts), time=float(ts),
                frame_path=None, objects=seg.get("objects", []),
                hands=[], caption=seg.get("caption", ""),
            )
            prev = segs[i-1].get("caption") if i > 0 else None
            next_cap = segs[i+1].get("caption") if i+1 < len(segs) else None
            ctx = SegmentContext(
                video_id=vid, video_path=f"videos/{vid}.mp4",
                phase=video.get("phase", ""), phase_zh=video.get("phase_zh", ""),
                segment_id=seg.get("segment_id", f"{vid}_s{i:03d}"),
                start=float(seg["start"]), end=float(seg["end"]),
                action=seg.get("action", ""), action_zh=seg.get("action_zh", ""),
                segment_caption=seg.get("caption", ""),
                keyframes=[kf], previous_caption=prev, next_caption=next_cap,
            )
            tasks.append(ctx)

    def _gen(ctx):
        items = generator.generate(ctx, 4)
        for it in items:
            it["segment_id"] = ctx.segment_id
            it["video_id"] = ctx.video_id
        return items

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_gen, ctx) for ctx in tasks]
        for f in as_completed(futures):
            all_items.extend(f.result())

    return all_items


def generate_viewer(pred: Dict, video_path: str) -> Path:
    """Generate a standalone viewer HTML for one video."""
    segs = pred.get("segments", [])
    dur = max(s["end"] for s in segs) if segs else 0

    viewer_data = {
        "video_id": pred["video_id"], "video_path": video_path,
        "duration": dur, "fps": 0, "width": 0, "height": 0,
        "omni_model": config.OMNI_MODEL,
        "phase": pred.get("phase"), "phase_zh": pred.get("phase_zh"),
        "phase_confidence": pred.get("phase_confidence"),
        "video_summary": pred.get("video_summary", ""),
        "segments": segs,
    }

    template_src = (Path(__file__).parent / "run_omni_seg.py").read_text()
    m = re.search(r'_VIEWER_TEMPLATE = r"""(.+?)"""', template_src, re.DOTALL)
    template = m.group(1)

    html = template.format(
        video_id=pred["video_id"],
        phase_zh=pred.get("phase_zh", ""),
        phase_confidence=pred.get("phase_confidence", ""),
        n_segs=len(segs), omni_model=config.OMNI_MODEL,
        duration=dur, video_path=video_path,
        video_summary=pred.get("video_summary", ""),
        data_json=json.dumps(viewer_data, ensure_ascii=False),
    )
    out = Path(pred.get("_viewer_path", f"outputs/viewers/{pred['video_id']}.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def process_one(video_id: str, video_path: Path, phases,
                extract_frames: bool = False, skip_verify: bool = False,
                skip_vqa: bool = False, skip_viewer: bool = False):
    """Run full pipeline on one video."""
    t0 = time.time()

    if extract_frames:
        video = media.load_video(video_id, video_path, cache_root=config.FRAME_CACHE_DIR)
        print(f"  Frames: {len(video.frames)}, {video.duration:.0f}s")
    else:
        dur, fps, w, h = media.probe(video_path)
        video = media.VideoInfo(video_id=video_id, path=video_path,
                                duration=dur, fps=fps, width=w, height=h, frames=[])

    log = RunLog()
    log.start_video(video_id, {"duration": video.duration, "pipeline": "full"})

    # Stage 1: Omni
    phase_res = hypothesize_phase_omni(video, phases, log)
    segments = segment_actions_omni(video, phases, phase_res, log)
    print(f"  Omni: {len(segments)} segments ({phase_res.get('phase_zh', '')})")

    # Stage 2: Verify (if frames available)
    if not skip_verify and video.frames:
        repair_catalog = _action_catalog_for_repair(phases, phase_res)
        verified = []
        for seg in segments:
            result = _verify_and_maybe_repair(video, seg, repair_catalog, log)
            if result is not None:
                verified.append(result)
        segments = verified
        print(f"  Verify: {len(segments)} after verify/repair")
    else:
        for seg in segments:
            seg.setdefault("confidence", 0.5)
            seg.setdefault("verification_status", "partial")
            seg.setdefault("needs_human_review", True)
            seg.setdefault("uncertainty_reason", "未做逐帧视觉核验")
            seg.setdefault("uncertainty_type", "mixed")
            seg.setdefault("review_suggestion", "人工复核")
            seg.setdefault("evidence_diagnosis", {})
            seg.setdefault("object_evidence",
                [{"object": o, "present": True} for o in seg.get("objects", [])])

    # Stage 3: Writer gate
    final_segments = _writer_gate(segments, log, keep_rejected=True)
    final_segments = _clip_final_overlaps(final_segments)
    for i, seg in enumerate(final_segments, 1):
        seg["segment_id"] = f"{video_id}_s{i:03d}"

    # Video summary
    timeline = "\n".join(
        f"[{s['start']:.0f}s–{s['end']:.0f}s] {s.get('caption', '')}"
        for s in final_segments
    )
    video_summary = api_client.ask_text(
        f"你是化学实验视频分析专家。以下是视频 {video_id}（实验阶段：{phase_res.get('phase_zh', '')}）"
        f"的全部原子操作分段 caption：\n\n{timeline}\n\n"
        "请用一段中文（3-5 句）概括整个视频的核心实验流程。只输出概括文本。",
        model=config.STRUCTURED_MODEL, max_tokens=400,
    )

    prediction = {
        "video_id": video_id, "video_path": f"videos/{video_id}.mp4",
        "pipeline": "full", "video_summary": video_summary,
        "phase": phase_res.get("phase"), "phase_zh": phase_res.get("phase_zh"),
        "phase_confidence": phase_res.get("confidence"),
        "alternative_phases": phase_res.get("alternative_phases", []),
        "segments": [{
            "segment_id": s["segment_id"], "start": s["start"], "end": s["end"],
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
            "object_evidence": s["object_evidence"],
        } for s in final_segments],
        "processing_note": "",
    }
    prediction["run_log"] = log.as_list()

    view_path = None
    if not skip_viewer:
        view_path = generate_viewer(prediction, str(video_path))

    vqa_items = []
    if not skip_vqa:
        vqa_items = generate_vqa([prediction])
        prediction["vqa"] = vqa_items

    dt = time.time() - t0
    print(f"  Done: {len(final_segments)} segs, {len(vqa_items)} VQA in {dt:.0f}s")
    return prediction, view_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LabARM-HV Full Pipeline")
    ap.add_argument("--video_id", help="Single video ID")
    ap.add_argument("--split", choices=["dev", "test", "all"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default="agent_v2/outputs/full_pipeline")
    ap.add_argument("--verify", action="store_true", help="Extract frames and run MLLM verification")
    ap.add_argument("--skip-vqa", action="store_true")
    args = ap.parse_args(argv)

    if args.video_id:
        videos = [{"video_id": args.video_id, "video_path": f"videos/{args.video_id}.mp4"}]
    elif args.split == "all":
        videos = _load_split("dev") + _load_split("test")
    elif args.split:
        videos = _load_split(args.split)
    else:
        ap.print_help()
        return 1
    if args.limit:
        videos = videos[:args.limit]

    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    phases = ontology.load_ontology()

    print(f"[full] {len(videos)} videos, verify={args.verify}, out={run_dir}")

    predictions = []
    for i, entry in enumerate(videos, 1):
        vid = str(entry["video_id"])
        vpath = config.DATA_ROOT / entry["video_path"]
        print(f"\n[{i}/{len(videos)}] {vid}")
        try:
            pred, _ = process_one(vid, vpath, phases,
                                  extract_frames=args.verify,
                                  skip_verify=not args.verify,
                                  skip_vqa=args.skip_vqa,
                                  skip_viewer=False)
            predictions.append(pred)
        except Exception as e:
            print(f"  ERROR: {e}")
            predictions.append({"video_id": vid, "segments": [], "processing_note": f"error: {e}"})
        _write_json(run_dir / "predictions.json", predictions)

    # Generate combined VQA
    if not args.skip_vqa:
        print(f"\n[full] Generating VQA for {sum(len(p['segments']) for p in predictions)} segments...")
        all_vqa = generate_vqa(predictions)
        vqa_out = {
            "source": str(run_dir / "predictions.json"),
            "generation_method": "template",
            "num_segments": sum(len(p["segments"]) for p in predictions),
            "num_vqa": len(all_vqa),
            "vqa": all_vqa,
        }
        _write_json(run_dir / "vqa_output.json", vqa_out)
        print(f"  {len(all_vqa)} VQA items")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
