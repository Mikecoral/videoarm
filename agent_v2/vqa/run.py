from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import re
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_v2.vqa.config import load_config
from agent_v2.vqa.clips import make_clip
from agent_v2.vqa.dataset import contexts_from_predictions, contexts_from_split
from agent_v2.vqa.generator import TemplateVQAGenerator, build_generator
from agent_v2.vqa.io_utils import write_json
from agent_v2.vqa.multiple_choice import build_multiple_choice


BAD_ANSWERS = {
    "",
    "其它操作",
    "其他操作",
    "其他可见实验操作",
    "other_visible_lab_operation",
    "未知",
    "无法判断",
    "不确定",
}

QUESTION_LEAK_TERMS = {
    "caption",
    "描述文本",
    "标注",
    "关键帧",
    "时间戳",
    "frame",
    "第几帧",
    "答案",
}
LOW_VALUE_QUESTION_TERMS = {
    "哪一部分",
    "什么时候出现",
    "何时出现",
    "哪个阶段出现",
}
GENERIC_QUESTION_PATTERNS = {
    "图中有什么物体？",
    "图中有哪些实验器具？",
    "图中主要涉及哪些实验器具或物体？",
}
L2_VISUAL_LEAK_PATTERNS = {
    "图中可见",
    "图中出现",
    "图片中可见",
    "图片中出现",
    "视频中可见",
    "视频里可见",
}
L2_GENERIC_COMMONSENSE_PATTERNS = {
    "这类",
    "通常用于",
    "一般用于",
    "主要用于",
    "常用于",
}
QUESTION_TIME_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*s\b|\d+(?:\.\d+)?\s*秒|\d{1,2}:\d{2})",
    re.IGNORECASE,
)
ALLOWED_LEVELS = {"L1", "L2"}
ALLOWED_CATEGORIES = {"entity", "operation", "procedure", "state", "attribute", "function", "safety"}
ALLOWED_MODALITIES = {"video", "image"}


class GenerationTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def generation_deadline(seconds: float | int | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise GenerationTimeout(f"generation timed out after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def valid_qa(qa: dict) -> bool:
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    if not question or not answer:
        return False
    question_lower = question.lower()
    if any(term.lower() in question_lower for term in QUESTION_LEAK_TERMS):
        return False
    if any(term in question for term in LOW_VALUE_QUESTION_TERMS):
        return False
    if question in GENERIC_QUESTION_PATTERNS:
        return False
    if str(qa.get("level", "")).strip().upper() == "L2":
        if any(pattern in question for pattern in L2_VISUAL_LEAK_PATTERNS) and any(
            pattern in question for pattern in L2_GENERIC_COMMONSENSE_PATTERNS
        ):
            return False
    if QUESTION_TIME_PATTERN.search(question):
        return False
    if answer in BAD_ANSWERS:
        return False
    if "其他可见实验操作" in answer or "other_visible_lab_operation" in answer:
        return False
    return True


def normalized_level(qa: dict) -> str:
    level = str(qa.get("level", "")).strip().upper()
    return level if level in ALLOWED_LEVELS else "L1"


def normalized_category(qa: dict) -> str:
    category = str(qa.get("category", "")).strip()
    return category if category in ALLOWED_CATEGORIES else "operation"


def normalized_modality(qa: dict, ctx) -> tuple[str, int | None]:
    modality = str(qa.get("visual_modality", "")).strip().lower()
    if modality not in ALLOWED_MODALITIES:
        modality = "image" if ctx.keyframes else "video"
    keyframe_index = qa.get("keyframe_index")
    if modality == "image":
        try:
            index = int(keyframe_index)
        except (TypeError, ValueError):
            index = 0
        if not ctx.keyframes:
            return "video", None
        index = max(0, min(index, len(ctx.keyframes) - 1))
        return "image", index
    return "video", None


def build_visual_payload(qa: dict, ctx, dataset_root: str, clips_dir: str, generation_log: list[dict]) -> dict:
    modality, keyframe_index = normalized_modality(qa, ctx)
    payload = {
        "modality": modality,
        "keyframe_index": keyframe_index,
        "image_path": None,
        "video_path": ctx.video_path,
        "clip_path": None,
    }
    if modality == "image":
        frame = ctx.keyframes[keyframe_index] if keyframe_index is not None and ctx.keyframes else None
        payload["image_path"] = frame.frame_path if frame else None
        return payload

    try:
        payload["clip_path"] = make_clip(
            dataset_root=dataset_root,
            video_path=ctx.video_path,
            video_id=ctx.video_id,
            start=ctx.start,
            end=ctx.end,
            clips_dir=clips_dir,
        )
    except Exception as exc:
        generation_log.append(
            {
                "segment_id": ctx.segment_id,
                "warning": f"clip generation failed: {exc}",
            }
        )
    return payload


def process_context(
    *,
    ctx_index: int,
    ctx,
    generator,
    fallback,
    config: dict,
    questions_per_segment: int,
    num_options: int,
    dataset_root: str,
    clips_dir: str,
    contexts: list,
) -> tuple[int, list[dict], list[dict]]:
    logs: list[dict] = []
    max_retries = int(config.get("generation_retries", 2 if config.get("use_mllm") else 0))
    generated = []
    method = "mllm" if config.get("use_mllm") else "template"
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            generated = generator.generate(ctx, questions_per_segment)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logs.append(
                    {
                        "segment_id": ctx.segment_id,
                        "warning": f"VQA generation retry {attempt + 1}/{max_retries}: {exc}",
                    }
                )
    if last_exc is not None:
        if config.get("use_mllm") and not config.get("template_fallback_on_mllm_error", False):
            generated = []
            method = "mllm_failed"
        else:
            generated = fallback.generate(ctx, questions_per_segment)
            method = "template_fallback"
        logs.append(
            {
                "segment_id": ctx.segment_id,
                "warning": f"VQA generation failed: {last_exc}",
            }
        )

    before_filter = len(generated)
    generated = [qa for qa in generated if valid_qa(qa)]
    if before_filter != len(generated):
        logs.append(
            {
                "segment_id": ctx.segment_id,
                "warning": f"filtered {before_filter - len(generated)} invalid QA items",
            }
        )

    items = []
    for i, qa in enumerate(generated):
        item_id = f"{ctx.segment_id}_qa_{i:03d}"
        answer_text = qa.get("answer", "")
        multiple_choice = build_multiple_choice(qa, ctx, contexts, item_id, num_options)
        visual = build_visual_payload(qa, ctx, dataset_root, clips_dir, logs)
        items.append(
            {
                "id": item_id,
                "video_id": ctx.video_id,
                "segment_id": ctx.segment_id,
                "time_range": {"start": ctx.start, "end": ctx.end},
                "background": ctx.to_background_dict(),
                "level": normalized_level(qa),
                "category": normalized_category(qa),
                "visual_modality": visual["modality"],
                "visual": visual,
                "question_type": qa.get("question_type", "unknown"),
                "question_format": "open_ended_and_multiple_choice",
                "question": qa.get("question", ""),
                "answer": answer_text,
                "ground_truth": {
                    "open_ended": {
                        "answer_text": answer_text,
                    },
                    "multiple_choice": multiple_choice["ground_truth"],
                },
                "open_ended": {
                    "question": qa.get("question", ""),
                    "ground_truth": {
                        "answer_text": answer_text,
                    },
                },
                "multiple_choice": multiple_choice,
                "evidence_timestamps": qa.get("evidence_timestamps", []),
                "answer_basis": qa.get("answer_basis", ""),
                "visual_reference": visual["modality"],
                "evidence_description": qa.get("evidence_description", ""),
                "generation_method": method,
            }
        )
    return ctx_index, items, logs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VQA from existing video parsing JSON.")
    parser.add_argument("--config", default="agent_v2/vqa/config.example.py")
    parser.add_argument("--local-config", default="agent_v2/vqa/config.local.py")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--questions-per-segment", type=int, default=None)
    parser.add_argument("--keyframes-per-segment", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--use-mllm", action="store_true")
    parser.add_argument("--no-mllm", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.local_config)
    if args.dataset_root:
        config["dataset_root"] = args.dataset_root
    if args.split:
        config["split"] = args.split
    if args.output:
        config["output"] = args.output
    if args.questions_per_segment is not None:
        config["questions_per_segment"] = args.questions_per_segment
    if args.keyframes_per_segment is not None:
        config["keyframes_per_segment"] = args.keyframes_per_segment
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    if args.use_mllm:
        config["use_mllm"] = True
    if args.no_mllm:
        config["use_mllm"] = False

    dataset_root = config["dataset_root"]
    keyframes_per_segment = int(config.get("keyframes_per_segment", 3))
    if args.predictions:
        contexts = contexts_from_predictions(args.predictions, dataset_root, keyframes_per_segment)
        source = args.predictions
    else:
        contexts = contexts_from_split(dataset_root, config.get("split", "dev"), keyframes_per_segment)
        source = f"split:{config.get('split', 'dev')}"

    generator = build_generator(config)
    fallback = TemplateVQAGenerator()
    questions_per_segment = int(config.get("questions_per_segment", 4))
    num_options = int(config.get("multiple_choice_num_options", 4))
    clips_dir = str(Path(config.get("clips_dir", Path(dataset_root) / "vqa" / "clips")))
    all_items = []
    generation_log = []
    configured_concurrency = int(config.get("concurrency", 16 if config.get("use_mllm") else 1))
    concurrency = max(16, configured_concurrency) if config.get("use_mllm") else max(1, configured_concurrency)
    print(f"using concurrency={concurrency}", flush=True)

    results: dict[int, tuple[list[dict], list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for ctx_index, ctx in enumerate(contexts, start=1):
            print(f"[{ctx_index}/{len(contexts)}] queued {ctx.segment_id}", flush=True)
            future = executor.submit(
                process_context,
                ctx_index=ctx_index,
                ctx=ctx,
                generator=generator,
                fallback=fallback,
                config=config,
                questions_per_segment=questions_per_segment,
                num_options=num_options,
                dataset_root=dataset_root,
                clips_dir=clips_dir,
                contexts=contexts,
            )
            futures[future] = ctx.segment_id
        for completed, future in enumerate(as_completed(futures), start=1):
            segment_id = futures[future]
            try:
                ctx_index, items, logs = future.result()
            except Exception as exc:
                ctx_index = len(results) + 1
                items = []
                logs = [{"segment_id": segment_id, "warning": f"worker failed: {exc}"}]
            results[ctx_index] = (items, logs)
            print(f"[{completed}/{len(contexts)}] finished {segment_id}: {len(items)} items", flush=True)

    for ctx_index in sorted(results):
        items, logs = results[ctx_index]
        all_items.extend(items)
        generation_log.extend(logs)

    output = {
        "source": source,
        "dataset_root": dataset_root,
        "generation_method": "mllm" if config.get("use_mllm") else "template",
        "num_segments": len(contexts),
        "num_vqa": len(all_items),
        "vqa": all_items,
        "generation_log": generation_log,
    }
    write_json(config.get("output", "vqa_output.json"), output)
    print(f"loaded {len(contexts)} segment contexts from {source}")
    print(f"wrote {len(all_items)} VQA items to {config.get('output', 'vqa_output.json')}")


if __name__ == "__main__":
    main()
