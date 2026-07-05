from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_v2.vqa.config import load_config
from agent_v2.vqa.io_utils import read_text, write_json
from agent_v2.vqa.openai_compatible import OpenAICompatibleClient


LEAK_TERMS = [
    "caption",
    "描述文本",
    "标注",
    "关键帧",
    "时间戳",
    "frame",
    "第几帧",
    "答案",
]
LOW_VALUE_TERMS = [
    "哪一部分",
    "什么时候出现",
    "何时出现",
    "哪个阶段出现",
    "开始阶段",
    "中间阶段",
    "结尾阶段",
]
TIME_PATTERN = re.compile(r"(\d+(?:\.\d+)?\s*s\b|\d+(?:\.\d+)?\s*秒|\d{1,2}:\d{2})", re.I)
EXPERIMENT_TERMS = [
    "实验",
    "试剂",
    "烧杯",
    "烧瓶",
    "锥形瓶",
    "试剂瓶",
    "分液漏斗",
    "漏斗",
    "注射器",
    "移液枪",
    "吸取",
    "加入",
    "滴加",
    "量取",
    "转移",
    "振摇",
    "混合",
    "过滤",
    "抽滤",
    "TLC",
    "薄层",
    "色谱",
    "层析",
    "装样柱",
    "分离柱",
    "溶液",
    "液体",
    "固体",
    "容器",
    "器具",
    "避光",
    "安全",
    "反应",
    "样品",
    "紫外灯",
    "直尺",
    "尺子",
    "铅笔",
    "盖子",
    "塞子",
]
GENERIC_QUESTION_PATTERNS = [
    "图中主要涉及哪些实验器具或物体？",
    "图中有什么物体？",
    "图中有哪些实验器具？",
]
ALLOWED_REPEATED_QUESTIONS = {
    "这段视频中实验人员正在进行什么操作？",
    "视频中实验人员正在进行什么操作？",
}
L2_VISUAL_LEAK_PATTERNS = [
    "图中可见",
    "图中出现",
    "图片中可见",
    "图片中出现",
    "视频中可见",
    "视频里可见",
]
L2_GENERIC_COMMONSENSE_PATTERNS = [
    "这类",
    "通常用于",
    "一般用于",
    "主要用于",
    "常用于",
]
AWKWARD_CHINESE_PATTERNS = [
    "从外观上暗示",
    "基于可见信息",
    "根据图像内容可推断",
    "根据视觉信息可推断",
]
ALLOWED_LEVELS = {"L1", "L2"}
ALLOWED_CATEGORIES = {"entity", "operation", "procedure", "state", "attribute", "function", "safety"}
ALLOWED_MODALITIES = {"video", "image"}


def text_of(item: dict[str, Any]) -> str:
    options = item.get("multiple_choice", {}).get("options", {})
    return " ".join(
        [
            str(item.get("question", "")),
            str(item.get("answer", "")),
            " ".join(str(value) for value in options.values()),
        ]
    )


def evaluate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("vqa", [])
    rule_issues_by_id, issue_counter = collect_rule_issues(items)
    hard_fail_issue_names = {
        "question_leaks_hidden_metadata",
        "low_value_temporal_question",
        "invalid_level",
        "invalid_category",
        "invalid_visual_modality",
        "not_experiment_related",
        "missing_video_clip",
        "missing_image",
        "l2_visual_anchor_leaked_as_text",
        "awkward_chinese_question",
    }
    hard_fail_count = sum(issue_counter[name] for name in hard_fail_issue_names)
    soft_issue_count = issue_counter["overly_generic_question"] + issue_counter["repeated_question_template"]
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        for issue in rule_issues_by_id.get(item.get("id", ""), []):
            if len(examples[issue["type"]]) < 8:
                examples[issue["type"]].append(
                    {
                        "id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "detail": issue.get("detail", ""),
                    }
                )
    summary = {
        "file": str(path),
        "num_vqa": len(items),
        "levels": dict(Counter(item.get("level") for item in items)),
        "categories": dict(Counter(item.get("category") for item in items)),
        "modalities": dict(Counter(item.get("visual_modality") for item in items)),
        "generation_warnings": len(data.get("generation_log", [])),
        "issue_counts": {name: count for name, count in sorted(issue_counter.items()) if count},
        "hard_fail_count": hard_fail_count,
        "soft_issue_count": soft_issue_count,
        "pass": hard_fail_count == 0 and soft_issue_count <= max(8, len(items) // 10) and len(items) >= 100,
        "examples": dict(examples),
    }
    return summary


def collect_rule_issues(items: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, str]]], Counter]:
    issues: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    question_counts = Counter(str(item.get("question", "")).strip() for item in items)

    def add_issue(item: dict[str, Any], issue_type: str, detail: str = "") -> None:
        entry = {"type": issue_type, "detail": detail}
        by_id[str(item.get("id", ""))].append(entry)
        issues[issue_type].append(
            {"id": item.get("id", ""), "question": item.get("question", ""), "detail": detail}
        )

    for item in items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        question_lower = question.lower()
        combined = text_of(item)
        combined_lower = combined.lower()

        leaks = [term for term in LEAK_TERMS if term.lower() in question_lower]
        if leaks or TIME_PATTERN.search(question):
            add_issue(item, "question_leaks_hidden_metadata", ", ".join(leaks))
        if any(term in question for term in LOW_VALUE_TERMS) or any(term in answer for term in ["开始阶段", "中间阶段", "结尾阶段"]):
            add_issue(item, "low_value_temporal_question", answer)
        if item.get("level") not in ALLOWED_LEVELS:
            add_issue(item, "invalid_level", str(item.get("level")))
        if item.get("category") not in ALLOWED_CATEGORIES:
            add_issue(item, "invalid_category", str(item.get("category")))
        if item.get("visual_modality") not in ALLOWED_MODALITIES:
            add_issue(item, "invalid_visual_modality", str(item.get("visual_modality")))
        if not any(term.lower() in combined_lower for term in EXPERIMENT_TERMS):
            add_issue(item, "not_experiment_related", answer)
        if question in GENERIC_QUESTION_PATTERNS:
            add_issue(item, "overly_generic_question", answer)
        if any(pattern in question for pattern in AWKWARD_CHINESE_PATTERNS):
            add_issue(item, "awkward_chinese_question", answer)
        if item.get("level") == "L2":
            has_visual_leak = any(pattern in question for pattern in L2_VISUAL_LEAK_PATTERNS)
            has_generic_common_sense = any(pattern in question for pattern in L2_GENERIC_COMMONSENSE_PATTERNS)
            if has_visual_leak and has_generic_common_sense:
                add_issue(item, "l2_visual_anchor_leaked_as_text", answer)
        visual = item.get("visual", {})
        if item.get("visual_modality") == "video" and not visual.get("clip_path"):
            add_issue(item, "missing_video_clip")
        if item.get("visual_modality") == "image" and not visual.get("image_path"):
            add_issue(item, "missing_image")

    for question, count in question_counts.items():
        if count > 5:
            if question in ALLOWED_REPEATED_QUESTIONS:
                continue
            matching = [item for item in items if str(item.get("question", "")).strip() == question]
            for item in matching:
                add_issue(item, "repeated_question_template", f"count={count}")
    return by_id, Counter({name: len(values) for name, values in issues.items()})


def judge_payload(item: dict[str, Any], rule_issues: list[dict[str, str]]) -> dict[str, Any]:
    mc = item.get("multiple_choice", {})
    bg = item.get("background", {})
    frames = bg.get("keyframes", [])
    selected_frame = None
    visual = item.get("visual", {})
    if item.get("visual_modality") == "image" and frames:
        index = visual.get("keyframe_index")
        if isinstance(index, int) and 0 <= index < len(frames):
            selected_frame = {
                "index": index,
                "objects": frames[index].get("objects"),
                "hands": frames[index].get("hands"),
                "caption": frames[index].get("caption"),
            }
    return {
        "item": {
            "id": item.get("id"),
            "level": item.get("level"),
            "category": item.get("category"),
            "visual_modality": item.get("visual_modality"),
            "question_type": item.get("question_type"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "options": mc.get("options"),
            "correct_option": mc.get("ground_truth"),
            "answer_basis": item.get("answer_basis"),
            "evidence_description": item.get("evidence_description"),
        },
        "visual_context": {
            "segment_caption": bg.get("segment_caption"),
            "action": bg.get("action_zh") or bg.get("action"),
            "previous_segment_caption": bg.get("previous_caption"),
            "next_segment_caption": bg.get("next_caption"),
            "selected_image_text": selected_frame,
        },
        "rule_issues": rule_issues,
    }


def run_llm_filter(
    *,
    data: dict[str, Any],
    config: dict[str, Any],
    concurrency: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    items = data.get("vqa", [])
    rule_issues, _ = collect_rule_issues(items)
    prompt = read_text(Path(__file__).parent / "prompts" / "vqa_judge.md")

    def judge_one(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        client = OpenAICompatibleClient(config)
        payload = judge_payload(item, rule_issues.get(item.get("id", ""), []))
        try:
            result = client.chat_json(prompt, json.dumps(payload, ensure_ascii=False))
            passed = bool(result.get("pass"))
            return str(item.get("id", "")), {
                "pass": passed,
                "reason": str(result.get("reason", "")),
                "issue_type": str(result.get("issue_type", "none" if passed else "unknown")),
            }
        except Exception as exc:
            return str(item.get("id", "")), {
                "pass": False,
                "reason": f"judge failed: {exc}",
                "issue_type": "judge_error",
            }

    decisions: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(judge_one, item) for item in items]
        for future in as_completed(futures):
            item_id, decision = future.result()
            decisions[item_id] = decision

    kept = [item for item in items if decisions.get(str(item.get("id", "")), {}).get("pass")]
    rejected = [
        {
            "id": item.get("id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            **decisions.get(str(item.get("id", "")), {"pass": False, "reason": "missing decision", "issue_type": "judge_error"}),
        }
        for item in items
        if not decisions.get(str(item.get("id", "")), {}).get("pass")
    ]
    filtered = copy.deepcopy(data)
    filtered["vqa"] = kept
    filtered["num_vqa"] = len(kept)
    filtered["quality_filter"] = {
        "method": "rule_prescreen_plus_llm_judge",
        "input_count": len(items),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
    }
    report = {
        **filtered["quality_filter"],
        "rejected_by_issue_type": dict(Counter(item["issue_type"] for item in rejected)),
        "rejected": rejected,
    }
    return filtered, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated VQA quality.")
    parser.add_argument("path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="agent_v2/vqa/config.example.py")
    parser.add_argument("--local-config", default="agent_v2/vqa/config.local.py")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--filtered-output", default=None)
    parser.add_argument("--judge-report", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.path)
    result = evaluate(input_path)
    if args.llm_judge:
        config = load_config(args.config, args.local_config)
        concurrency = args.concurrency or int(config.get("judge_concurrency", config.get("concurrency", 16)))
        data = json.loads(input_path.read_text(encoding="utf-8"))
        filtered, judge_report = run_llm_filter(data=data, config=config, concurrency=max(1, concurrency))
        if args.filtered_output:
            write_json(args.filtered_output, filtered)
        if args.judge_report:
            write_json(args.judge_report, judge_report)
        result["llm_judge"] = {
            "input_count": judge_report["input_count"],
            "kept_count": judge_report["kept_count"],
            "rejected_count": judge_report["rejected_count"],
            "rejected_by_issue_type": judge_report["rejected_by_issue_type"],
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
