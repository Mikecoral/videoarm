# 基于分片视频 JSON 的 VQA 生成与筛选流程

本文档说明如何从已经完成视频分片的 JSON 信息生成实验视频 VQA，并在生成后用 LLM judge 筛掉不合格题目。

## 1. 输入

输入不是原始未处理视频，而是已经完成视频解析/分片后的数据目录：

```text
视频解析智能体数据包/hackathon_release/
  splits/dev.json
  annotations/temporal/*.json
  annotations/spatial/*.json
  videos/*.mp4
  frames/<video_id>/*.jpg
```

核心输入是 `temporal` JSON，每个视频包含若干已经切好的片段：

```json
{
  "video_id": "3",
  "video_path": "videos/3.mp4",
  "phase": "reaction_setup",
  "phase_zh": "开反应",
  "segments": [
    {
      "start": 5.5,
      "end": 20.71,
      "action": "measure_liquid_for_addition",
      "action_zh": "量取液体",
      "caption": "加入反应溶剂二氯甲烷 20 mL。"
    }
  ]
}
```

`spatial` JSON 提供候选图片帧、物体和手部标注。当前 pipeline 不重新检测物体，只使用已有空间标注生成关键帧文本。

也支持 `new_test` 这类自标注/预测结果格式：

```text
new_test/<run_id>/predictions.json
```

这类文件的顶层可以直接是视频列表：

```json
[
  {
    "video_id": "1",
    "video_path": "videos/1.mp4",
    "phase": "filtration",
    "phase_zh": "过滤与抽滤",
    "segments": [
      {
        "segment_id": "1_s001",
        "start": 14.0,
        "end": 19.0,
        "action": "wash_filter_cake",
        "action_zh": "洗涤滤饼",
        "objects": ["布氏漏斗", "吸滤瓶", "白色固体"],
        "caption": "实验人员手持棕色试剂瓶，使用滴管吸取液体...",
        "evidence_timestamps": [14, 16, 17]
      }
    ]
  }
]
```

如果 `new_test` 没有 `annotations/spatial/<video_id>.mp4.json`，pipeline 会使用 segment 中的 `evidence_timestamps` 从视频里自动抽帧，并用 segment 的 `objects` 构造关键帧文本。

## 2. 构造 SegmentContext

入口代码：

```text
vqa_generation/dataset.py
```

`contexts_from_split()` 会读取 `splits/dev.json`，再按其中的路径加载：

- `temporal_annotation`
- `spatial_annotation`
- `video_path`
- `frame_dir`

每个 temporal segment 被转成一个 `SegmentContext`：

```text
video_id
video_path
phase / phase_zh
segment_id
start / end
action / action_zh
segment_caption
keyframes
previous_caption
next_caption
```

其中 `previous_caption` 和 `next_caption` 用于生成流程顺序类问题。

## 3. 关键帧选择

入口代码：

```text
vqa_generation/keyframes.py
```

关键帧不从视频重新抽取，而是在 `spatial` JSON 的已标注帧中选择。

步骤：

1. 使用 `sample_fps` 将空间标注中的 `frame_index` 转成时间：

```python
frame_time = frame_index / sample_fps
```

2. 找出落在当前 segment 内的候选帧：

```text
start <= frame_time <= end
```

3. 默认每个 segment 选择 3 个目标时间点：

```text
start
(start + end) / 2
end
```

4. 对每个目标时间点，选择最近的空间标注帧。

5. 去重。如果开始、中点、结束选中了同一帧，只保留一次。

6. 为每帧构造文本 caption：

```text
画面中可见物体：...；手部：...
```

这些关键帧会进入 LLM 输入，但生成模型默认不接收真实图片。

## 4. Caption-only VQA 生成

入口代码：

```text
vqa_generation/generator.py
vqa_generation/prompts/vqa_generation.md
```

当前生成模式是 `caption-only`：

```python
"send_images": False
```

也就是说，DeepSeek 只接收文本化背景：

```json
{
  "video_id": "...",
  "phase": "...",
  "segment_caption": "...",
  "action": "...",
  "previous_segment_caption": "...",
  "next_segment_caption": "...",
  "keyframes": [
    {
      "index": 0,
      "objects": ["..."],
      "hands": ["..."],
      "caption": "画面中可见物体：..."
    }
  ]
}
```

生成模型需要一次性输出：

- `level`: `L1` 或 `L2`
- `category`: 固定类别 label
- `visual_modality`: `video` 或 `image`
- `keyframe_index`: 图片题对应的候选图片编号
- `question`
- `answer`
- `distractors`
- `answer_basis`
- `evidence_description`

L1 是直接视觉问答。L2 必须是多跳题：答题者需要先看图片/视频识别视觉事实，再结合实验知识、器具功能、流程关系或安全规则作答。

## 5. 开放式问答与选择题的生成顺序

生成顺序是：

```text
开放式 QA + distractors -> 选择题
```

LLM 先生成开放式问题和标准答案：

```json
{
  "question": "这段视频中实验人员正在进行什么操作？",
  "answer": "用注射器量取液体。",
  "distractors": [
    "称取固体样品。",
    "振摇分液漏斗。",
    "点样 TLC 板。"
  ]
}
```

随后 `vqa_generation/multiple_choice.py` 将 `answer + distractors` 洗牌为 A/B/C/D。

因此选择题不是独立生成的，而是基于开放式 QA 的标准答案构造。

## 6. 题目后处理与媒体落盘

入口代码：

```text
vqa_generation/run.py
vqa_generation/clips.py
```

每条 VQA 会被补充 `visual` 字段：

```json
{
  "visual_modality": "video",
  "visual": {
    "modality": "video",
    "keyframe_index": null,
    "image_path": null,
    "video_path": "videos/3.mp4",
    "clip_path": "视频解析智能体数据包/hackathon_release/vqa/clips/3_5p50_20p71.mp4"
  }
}
```

如果是 `video` 题，系统会用 ffmpeg 将原视频按 segment 的 `start/end` 裁成独立 mp4：

```text
vqa/clips/*.mp4
```

如果是 `image` 题，系统会绑定对应 `keyframe_index` 的图片路径：

```json
{
  "visual_modality": "image",
  "visual": {
    "modality": "image",
    "keyframe_index": 0,
    "image_path": "视频解析智能体数据包/hackathon_release/frames/3/000006.jpg"
  }
}
```

网页展示时直接使用这些 `clip_path` 或 `image_path`。

## 7. 生成命令

基于 dev split 生成未筛选 VQA：

```bash
python vqa_generation/run.py \
  --config vqa_generation/config.example.py \
  --local-config vqa_generation/config.local.py \
  --dataset-root "视频解析智能体数据包/hackathon_release" \
  --split dev \
  --use-mllm \
  --concurrency 16 \
  --output "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_raw.json"
```

基于 `new_test` 格式 predictions 生成未筛选 VQA：

```bash
python vqa_generation/run.py \
  --config vqa_generation/config.example.py \
  --local-config vqa_generation/config.local.py \
  --dataset-root "视频解析智能体数据包/hackathon_release" \
  --predictions "new_test/20260705_100926_test_3vids/predictions.json" \
  --use-mllm \
  --concurrency 16 \
  --output "视频解析智能体数据包/hackathon_release/vqa/vqa_output_new_test_raw.json"
```

如果 `new_test` 没有 spatial annotation，运行时会自动将证据帧抽到：

```text
视频解析智能体数据包/hackathon_release/frames/<video_id>/
```

关键配置：

```python
CONFIG = {
    "use_mllm": True,
    "send_images": False,
    "concurrency": 16,
    "api": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
        "max_tokens": 8192
    }
}
```

`config.local.py` 存放本地 API key，已被 `.gitignore` 忽略。

## 8. LLM Judge 筛选

入口代码：

```text
vqa_generation/evaluate_quality.py
vqa_generation/prompts/vqa_judge.md
```

生成完成后，不再回到生成模型做迭代。筛选阶段会逐题判断是否保留：

```text
规则硬筛辅助信息 + LLM judge -> pass/fail
```

不合格题目直接丢弃，不补生成。

LLM judge 会检查：

- 中文题面是否自然
- 是否实验相关
- L1 是否是直接视觉问题
- L2 是否真的是视觉锚点 + 实验知识的多跳题
- 是否变成纯常识题
- 是否泄漏 caption、关键帧、时间戳、frame、答案等不可见信息
- 干扰项是否合理
- 答案是否明确且可由视觉材料支撑

筛选命令：

```bash
python vqa_generation/evaluate_quality.py \
  "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_raw.json" \
  --config vqa_generation/config.example.py \
  --local-config vqa_generation/config.local.py \
  --llm-judge \
  --concurrency 16 \
  --filtered-output "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_mllm.json" \
  --judge-report "视频解析智能体数据包/hackathon_release/vqa/vqa_judge_report.json" \
  --output "视频解析智能体数据包/hackathon_release/vqa/vqa_quality_final.json"
```

输出报告中会包含：

```json
{
  "llm_judge": {
    "input_count": 148,
    "kept_count": 135,
    "rejected_count": 13,
    "rejected_by_issue_type": {
      "l2_visual_anchor_leaked": 9,
      "too_trivial_or_generic": 3,
      "unsupported_by_visual": 1
    }
  }
}
```

## 9. 输出 JSON

筛选后的正式 VQA 文件：

```text
视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_mllm.json
```

顶层字段：

```json
{
  "source": "split:dev",
  "dataset_root": "视频解析智能体数据包/hackathon_release",
  "generation_method": "mllm",
  "num_segments": 37,
  "num_vqa": 135,
  "quality_filter": {
    "method": "rule_prescreen_plus_llm_judge",
    "input_count": 148,
    "kept_count": 135,
    "rejected_count": 13
  },
  "vqa": []
}
```

单条 VQA 同时包含开放式问答和选择题：

```json
{
  "id": "3_seg_0000_qa_000",
  "video_id": "3",
  "segment_id": "3_seg_0000",
  "time_range": {"start": 5.5, "end": 20.71},
  "level": "L1",
  "category": "operation",
  "visual_modality": "video",
  "question": "这段视频中实验人员正在进行什么操作？",
  "answer": "量取液体。",
  "ground_truth": {
    "open_ended": {"answer_text": "量取液体。"},
    "multiple_choice": {"answer": "B", "answer_text": "量取液体。"}
  },
  "open_ended": {
    "question": "这段视频中实验人员正在进行什么操作？",
    "ground_truth": {"answer_text": "量取液体。"}
  },
  "multiple_choice": {
    "question": "这段视频中实验人员正在进行什么操作？",
    "options": {
      "A": "称取固体样品。",
      "B": "量取液体。",
      "C": "点样 TLC 板。",
      "D": "过滤反应混合物。"
    },
    "ground_truth": {"answer": "B", "answer_text": "量取液体。"}
  }
}
```

## 10. 网页展示

构建网页：

```bash
python vqa_generation/build_viewer.py \
  --vqa-dir "视频解析智能体数据包/hackathon_release/vqa"
```

启动服务时必须从 `hackathon_release` 根目录启动，否则图片和视频 clip 路径会 404：

```bash
cd "视频解析智能体数据包/hackathon_release"
python -m http.server 8765 --bind 0.0.0.0
```

访问：

```text
http://localhost:8765/vqa/viewer.html
```

答题视图展示：

- `video` 题：裁剪后的 mp4 片段
- `image` 题：指定关键帧图片
- 问题、开放式作答区、选择题选项

标注视图额外展示：

- ground truth
- background caption
- evidence description
- 关键帧元信息

## 11. 运行时流程总结

运行时 pipeline 是：

```text
分片 JSON
  -> SegmentContext
  -> 关键帧文本化
  -> caption-only LLM 生成 VQA
  -> 构造开放式问答和选择题
  -> 裁剪视频片段 / 绑定图片
  -> 规则硬筛辅助信息
  -> LLM judge 逐题筛选
  -> 丢弃不合格 VQA
  -> 输出正式 JSON
  -> 构建网页展示
```

注意：Codex 为了改 prompt 或修代码做的迭代不属于运行时 pipeline。运行时不会把不合格题目送回生成模型重写，只会直接丢弃。
