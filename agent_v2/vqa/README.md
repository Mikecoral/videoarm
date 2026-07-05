# VQA 生成模块

这是独立于 `agent/` 的 VQA 构造步骤。它假设已经存在视频解析结果 JSON，例如数据包中的：

- `annotations/temporal/*.json`
- `annotations/spatial/*.json`

或外部 pipeline 生成的 `predictions.json`。

完整流程说明见：[基于分片视频 JSON 的 VQA 生成与筛选流程](docs/vqa_generation_from_segment_json.md)。

## 方法

参考 LiveVQA / VideoQA benchmark 的构造方式：

1. 使用已有 temporal JSON 作为视频分段。
2. 对每个分段选择关键帧：默认选开始、中点、结束附近的空间标注帧。
3. 为片段构造 background：
   - 片段起止时间
   - 片段 caption
   - 动作标签
   - 关键帧 caption
   - 前后片段上下文
4. 使用 OpenAI-compatible 文本 LLM 生成开放式 VQA。默认只把片段/图片 caption 和结构化标签作为文本输入，不上传图片或视频。
5. 生成时要求模型同时给出：
   - `level`: `L1` 直接视觉问答，或 `L2` 轻量多跳/常识推理。
   - `category`: `entity`、`operation`、`procedure`、`state`、`attribute`、`function`、`safety`。
   - `visual_modality`: `video` 或 `image`。图片题需要指定 `keyframe_index`。
6. 基于模型给出的 ground truth 和 distractors 构造选择题。
7. 对 `video` 题裁剪对应片段并保存到 `vqa/clips/`，网页直接展示裁剪后的 mp4；对 `image` 题展示选中的关键帧图片。
8. 每条 VQA 同时保留开放式问答和多选题格式，并给出两种 ground truth。

## 运行

基于 dev 标注生成：

```bash
python vqa_generation/run.py \
  --config vqa_generation/config.example.py \
  --dataset-root "视频解析智能体数据包/hackathon_release" \
  --split dev \
  --output vqa_output_dev.json
```

基于 `predictions.json` 生成：

```bash
python vqa_generation/run.py \
  --config vqa_generation/config.example.py \
  --dataset-root "视频解析智能体数据包/hackathon_release" \
  --predictions predictions.json \
  --output vqa_output_predictions.json
```

## LLM 配置

默认使用模板生成，便于离线复现。若要使用 OpenAI-compatible API：

```bash
cp vqa_generation/config.example.py vqa_generation/config.local.py
```

在 `vqa_generation/config.local.py` 中设置：

```python
CONFIG = {
    "use_mllm": True,
    "send_images": False,
    "api": {
        "base_url": "https://your-api/v1",
        "model": "deepseek-v4-pro",
        "api_key": "sk-...",
        "use_response_format": True,
    },
}
```

`send_images=False` 表示只进行 caption-only 文本生成。此模式适合 DeepSeek 这类文本模型，也便于低成本批量生成。vLLM 示例：

```python
CONFIG = {
    "use_mllm": True,
    "api": {
        "base_url": "http://localhost:8000/v1",
        "model": "deepseek-v4-pro",
        "api_key": "",
        "use_response_format": False,
    },
}
```

`vqa_generation/config.local.py` 已被 `.gitignore` 忽略。

## 媒体补全与网页

如果已有 VQA JSON 还没有 `visual` 字段，或者需要重新生成裁剪片段：

```bash
python vqa_generation/materialize_media.py \
  --input "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_mllm.json" \
  --dataset-root "视频解析智能体数据包/hackathon_release"
```

生成后使用规则硬筛 + LLM judge 筛选问题。不合格 VQA 会直接丢弃，不补生成，也不会回到生成模型：

```bash
python vqa_generation/evaluate_quality.py \
  "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_mllm.json" \
  --config vqa_generation/config.example.py \
  --local-config vqa_generation/config.local.py \
  --llm-judge \
  --concurrency 16 \
  --filtered-output "视频解析智能体数据包/hackathon_release/vqa/vqa_output_dev_mllm_filtered.json" \
  --judge-report "视频解析智能体数据包/hackathon_release/vqa/vqa_judge_report.json" \
  --output "视频解析智能体数据包/hackathon_release/vqa/vqa_quality_final.json"
```

构建静态网页：

```bash
python vqa_generation/build_viewer.py \
  --vqa-dir "视频解析智能体数据包/hackathon_release/vqa"
```

## 输出格式

```json
{
  "source": "split:dev",
  "dataset_root": "视频解析智能体数据包/hackathon_release",
  "generation_method": "template",
  "num_segments": 37,
  "num_vqa": 148,
  "vqa": [
    {
      "id": "8_seg_0000_qa_000",
      "video_id": "8",
      "segment_id": "8_seg_0000",
      "time_range": {"start": 0.0, "end": 15.56},
      "background": {
        "segment_caption": "将反应液转移至分液漏斗中。",
        "keyframes": []
      },
      "level": "L1",
      "category": "operation",
      "visual_modality": "video",
      "visual": {
        "modality": "video",
        "keyframe_index": null,
        "image_path": null,
        "video_path": "videos/8.mp4",
        "clip_path": "视频解析智能体数据包/hackathon_release/vqa/clips/8_0p00_15p56.mp4"
      },
      "question_type": "action_recognition",
      "question_format": "open_ended_and_multiple_choice",
      "question": "这段视频主要展示了什么实验操作？",
      "answer": "将反应液转移至分液漏斗中。",
      "ground_truth": {
        "open_ended": {
          "answer_text": "将反应液转移至分液漏斗中。"
        },
        "multiple_choice": {
          "answer": "B",
          "answer_text": "将反应液转移至分液漏斗中。"
        }
      },
      "open_ended": {
        "question": "这段视频主要展示了什么实验操作？",
        "ground_truth": {
          "answer_text": "将反应液转移至分液漏斗中。"
        }
      },
      "multiple_choice": {
        "question": "这段视频主要展示了什么实验操作？",
        "options": {
          "A": "向分液漏斗中加入萃取溶剂。",
          "B": "将反应液转移至分液漏斗中。",
          "C": "振摇分液漏斗并放气。",
          "D": "静置等待液相分层。"
        },
        "ground_truth": {
          "answer": "B",
          "answer_text": "将反应液转移至分液漏斗中。"
        }
      },
      "evidence_timestamps": [0.0, 8.0, 15.0],
      "answer_basis": "segment_caption",
      "generation_method": "template"
    }
  ]
}
```
