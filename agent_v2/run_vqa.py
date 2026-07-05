"""VQA generation from omni_batch predictions.json. Concurrency 100.

    python -m agent_v2.run_vqa
"""

from __future__ import annotations

import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_AI_LAB = _HERE.parent.parent
sys.path.insert(0, str(_AI_LAB))

from agent_v2 import config

PRED_PATH = _HERE / "outputs" / "omni_batch" / "predictions.json"
OUT_PATH = _HERE / "outputs" / "omni_batch" / "vqa_output.json"
CONCURRENCY = 100

# Build SegmentContext per segment
from vqa_generation.types import SegmentContext, KeyframeContext
from vqa_generation.generator import TemplateVQAGenerator, MLLMVQAGenerator
from vqa_generation.io_utils import write_json

predictions = json.loads(PRED_PATH.read_text())
generator = TemplateVQAGenerator()  # template-based
all_items: List[Dict] = []
contexts: List[SegmentContext] = []

# Build all contexts first
for video in predictions:
    vid = video["video_id"]
    segs = video.get("segments", [])
    for i, seg in enumerate(segs):
        evidence_ts = seg.get("evidence_timestamps") or []
        if not evidence_ts:
            evidence_ts = [(seg["start"] + seg["end"]) / 2]
        kf_ts = evidence_ts[len(evidence_ts) // 2]  # median
        kf = KeyframeContext(
            frame_index=int(kf_ts), time=float(kf_ts),
            frame_path=None, objects=seg.get("objects", []),
            hands=[], caption=seg.get("caption", ""),
        )
        prev_cap = segs[i-1].get("caption") if i > 0 else None
        next_cap = segs[i+1].get("caption") if i+1 < len(segs) else None
        ctx = SegmentContext(
            video_id=vid, video_path=f"videos/{vid}.mp4",
            phase=video.get("phase", ""), phase_zh=video.get("phase_zh", ""),
            segment_id=seg.get("segment_id", f"{vid}_s{i:03d}"),
            start=float(seg["start"]), end=float(seg["end"]),
            action=seg.get("action", ""), action_zh=seg.get("action_zh", ""),
            segment_caption=seg.get("caption", ""),
            keyframes=[kf], previous_caption=prev_cap, next_caption=next_cap,
        )
        contexts.append(ctx)

print(f"Segments: {len(contexts)}  Concurrency: {CONCURRENCY}")

def process_one(ctx: SegmentContext) -> tuple[int, list]:
    items = generator.generate(ctx, questions_per_segment=4)
    for item in items:
        item["segment_id"] = ctx.segment_id
        item["video_id"] = ctx.video_id
    return ctx.segment_id, items

t0 = time.time()
with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = {executor.submit(process_one, ctx): ctx.segment_id for ctx in contexts}
    for future in as_completed(futures):
        seg_id, items = future.result()
        all_items.extend(items)

dt = time.time() - t0
output = {
    "source": str(PRED_PATH),
    "generation_method": "template",
    "num_segments": len(contexts),
    "num_vqa": len(all_items),
    "vqa": all_items,
    "generation_log": [],
}
write_json(str(OUT_PATH), output)
print(f"{len(all_items)} VQA items from {len(contexts)} segments in {dt:.1f}s")
print(f"Saved: {OUT_PATH}")
