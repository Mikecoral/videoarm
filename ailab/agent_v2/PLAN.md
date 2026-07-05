# LabARM-HV: 实验视频解析智能体新方案

> 目标：以 `videoarm/` 为主要工程模板和运行范式，补入 `VideoHV-Agent/VideoHV-Agent` 的假设验证机制，完成实验视频解析黑客松的全进阶交付。

## 0. 设计定位

本方案不沿用旧 `agent/` 的实现计划。新的 agent 底座以 `videoarm/` 为主：

- 采用 VideoARM 的 `observe -> think -> act -> memorize` 控制循环；
- 采用 VideoARM 的 hierarchical multimodal memory 思路；
- 采用 VideoARM 的工具注册、工具调用、结果记忆、run trace 保存方式；
- 将 VideoHV-Agent 的 hypothesis-verification 机制嵌入到 VideoARM 的 controller loop 中；
- 面向赛题输出结构化动作标注，而不是只回答单个视频问题。

系统名称建议：

```text
LabARM-HV: Hierarchical-Memory Video Agent with Hypothesis Verification
```

核心叙事：

```text
VideoARM 提供长视频分层记忆和按需回看机制；
VideoHV 提供假设生成、区分线索提取、工具验证和失败再生机制；
EXPVID 提供实验视频三层理解目标：细粒度感知、过程理解、科学/复盘推理。
```

## 1. 赛题交付目标

基础输出：

- 每个视频关键实验操作的起止时间；
- 每段操作的原子动作类别；
- 每段涉及的关键物体；
- 简洁准确 caption；
- 对应证据时间戳；
- 统一入口批量处理测试视频；
- `predictions.json`；
- `run_log.json`；
- 技术报告和 PPT。

进阶输出：

- 每条标注结果的置信度；
- 不确定性原因；
- 人工复核建议；
- 每视频若干条 VQA；
- 可交互视频问答 Demo；
- 可视化标注界面；
- 人工反馈后局部更新结果。

## 2. 主工程模板：VideoARM

优先参考 `videoarm/` 中这些设计：

```text
videoarm/main.py
videoarm/videoarm/core/agent.py
videoarm/videoarm/video/utils.py
videoarm/videoarm/api/client.py
videoarm/videoarm/config/model_config.py
videoarm/videoarm/config/settings.py
```

需要保留的 VideoARM 机制：

1. 统一 agent 类
   - 新建 `LabARMHVAgent`，类似 `VideoARMAgent`。
   - 每个视频一次 session。
   - 每个 session 保存 memory、tool log、controller history。

2. HM3 分层记忆
   - VideoARM 原始 memory：
     - `scene_snapshots`
     - `audio_transcripts`
     - `clip_analyses`
   - 本任务扩展为：
     - `frame_observations`
     - `clip_summaries`
     - `phase_hypotheses`
     - `action_hypotheses`
     - `segment_verifications`
     - `repairs`
     - `final_segments`
     - `vqa_items`

3. 工具化回看
   - 沿用 scene snapper / clip analyzer 的思想；
   - 但工具目标改成“标注验证”：
     - `scene_snapper`
     - `clip_analyzer`
     - `object_presence_checker`
     - `boundary_checker`
     - `ontology_lookup`
     - `hypothesis_verifier`

4. 消息重建
   - 每次工具调用后将结果写入 HM3；
   - controller 下一轮只读取压缩后的 HM3；
   - 避免长视频帧描述撑爆上下文。

5. trace 保存
   - 每个工具调用、参数、结果、耗时都写入 `run_log.json`；
   - 报告中展示“Agent 中间轨迹”。

## 3. 补充机制：VideoHV 假设验证

优先参考 `VideoHV-Agent/VideoHV-Agent` 中这些模块：

```text
video_hv/config.py
video_hv/verifier.py
video_hv/vision_tools.py
video_hv/pipelines/egoschema_openai/runner.py
video_hv/pipelines/egoschema_openai/openai_stages.py
```

需要迁移的 VideoHV 思想：

1. Hypothesis generation
   - 对每个视频先生成 phase 假设；
   - 再生成动作段假设；
   - 每个动作段本质上是一条待验证假设：

```json
{
  "claim": "12.0-18.5s 是 pipette_sample_solution",
  "expected_evidence": ["移液枪出现", "液体被转移", "目标容器出现"],
  "segment": {...}
}
```

2. Distinguishing clue
   - VideoHV 用 clue 区分多选题选项；
   - 本任务改成用 clue 区分相近 action：
     - `pipette_sample_solution` vs `add_solvent_to_sample_container`
     - `transfer_liquid_to_sep_funnel` vs `rinse_container`
     - `prepare_TLC_plate` vs `spot_TLC_plate`
     - `add_liquid_to_reaction` vs `measure_liquid_for_addition`

3. Tool-based verification
   - 每条动作假设必须调用视觉工具回看；
   - 验证器不能只根据文字 memory 下判断；
   - 工具返回 evidence 后，controller 再给结论。

4. Regeneration after failed verification
   - 如果验证失败，不是简单标低置信；
   - 要进入 repair loop：
     - 重新生成边界；
     - 重新生成 action；
     - 重新生成 objects；
     - 必要时拆分/合并片段。

5. Verification trace
   - 记录：
     - 初始假设；
     - 区分线索；
     - 调用的工具；
     - 工具观察；
     - 验证结论；
     - 修复结果。

## 4. 新系统总体架构

```text
agent_v2/run.py
  -> BatchVideoRunner
      -> LabARMHVAgent.process_video(video)
          -> [0] initialize_video
          -> [1] build_multiscale_memory
          -> [2] phase_hypothesis_loop
          -> [3] action_hypothesis_loop
          -> [4] hypothesis_verification_loop
          -> [5] repair_and_reflect_loop
          -> [6] final_writer
          -> [7] expvid_vqa_generator
          -> [8] interactive_memory_export
```

单视频内部流程：

```text
Observe:
  抽帧、读取 metadata、构建初始 HM3

Think:
  controller 读取 HM3，判断下一步需要 phase、动作、边界、物体还是 caption 验证

Act:
  调用工具：scene_snapper / clip_analyzer / object_presence / boundary_check / ontology_lookup

Memorize:
  工具结果写入 HM3

Verify:
  用 VideoHV 风格的 hypothesis verifier 判断假设是否成立

Repair:
  对失败或低置信假设局部重生

Write:
  输出 predictions、VQA 和 demo memory
```

## 5. Agent 模块设计

### 5.1 Batch Runner

职责：

- 读取 `hackathon_release/splits/test.json`；
- 支持 `--split dev/test/all`；
- 支持 `--video_id`；
- 支持断点续跑；
- 每跑完一个视频增量写 `predictions.json` 和 `run_log.json`。

入口：

```bash
python agent_v2/run.py --split test
python agent_v2/run.py --split dev --eval
python agent_v2/run.py --video_id 0061
```

### 5.2 Video Memory Builder

职责：

- 默认 1 fps 抽帧；
- 对候选边界附近加密抽帧；
- 构建多尺度 memory。

输出：

```json
{
  "video_info": {"duration": 141.0, "fps": 50.0, "frames": 7048},
  "frame_observations": [],
  "clip_summaries": [],
  "global_summary": ""
}
```

### 5.3 Phase Hypothesis Agent

输入：

- clip summaries；
- ontology phase 列表；
- dev 标注格式示例。

输出：

```json
{
  "phase": "TLC_analysis",
  "phase_zh": "薄层色谱分析",
  "confidence": 0.88,
  "evidence_timestamps": [0, 31, 68, 126],
  "alternatives": [
    {"phase": "analytical_sample_preparation", "reason": "..."}
  ]
}
```

### 5.4 Action Hypothesis Agent

输入：

- phase hypothesis；
- HM3；
- ontology 中该 phase 的 action 列表；
- clip summaries。

输出动作假设：

```json
{
  "hypothesis_id": "0061_h03",
  "start": 41.5,
  "end": 64.5,
  "action": "spot_TLC_plate",
  "action_zh": "点样 TLC 板",
  "objects": ["毛细管", "EP管", "TLC板"],
  "caption": "打开EP管并用毛细管蘸取样品进行点样准备。",
  "evidence_timestamps": [42, 51, 63],
  "distinguishing_clues": [
    "是否出现毛细管接触样品或TLC板",
    "是否不是单纯划线标记TLC板"
  ]
}
```

### 5.5 Hypothesis Verification Agent

这是 VideoHV 机制的核心迁移。

每条动作假设生成验证任务：

```text
Claim:
  在 start-end 时间内发生 action。

Need verify:
  - boundary
  - action
  - objects
  - caption
  - evidence timestamp
```

最小化实现规则：

- 直接复用 VideoARM `clip_analyzer` 返回的 `confidence` 作为主要置信度来源；
- 每条 segment 至少调用一次 `clip_analyzer`，问题格式固定为“该时间段是否支持 action/object/caption 这条标注？”；
- 不额外引入复杂校准模型，先用规则合成标注级字段。

验证输出：

```json
{
  "hypothesis_id": "...",
  "status": "verified | partial | rejected",
  "confidence": 0.74,
  "boundary_ok": true,
  "action_ok": true,
  "objects_ok": false,
  "caption_ok": true,
  "evidence_found": [
    {"timestamp": 43, "observation": "..."}
  ],
  "missing_evidence": ["TLC板被遮挡，无法确认接触位置"],
  "suggested_revision": {
    "start": null,
    "end": null,
    "action": null,
    "objects": ["毛细管", "EP管"]
  },
  "uncertainty_reason": "关键接触动作只在少数帧中可见",
  "needs_human_review": true
}
```

### 5.6 Repair Agent

触发条件：

- `status == rejected`
- `confidence < 0.65`
- `boundary_ok == false`
- 相邻 segment 重叠严重；
- caption 和视觉证据冲突。

修复动作：

- shift boundary；
- split segment；
- merge repeated action；
- relabel action；
- remove hallucinated object；
- regenerate caption。

最多两轮：

```text
K = 0: no repair baseline
K = 1: first repair
K = 2: second repair if still low confidence
```

报告里展示 dev 上 K=0/K=1 的收益。

### 5.7 Writer Agent

职责：

- 将 verified / partial 片段转成最终 `predictions.json`；
- rejected 片段默认不输出，除非是唯一关键动作且标记 `needs_human_review=true`；
- 做排序、去重、轻量 NMS；
- 生成 final caption；
- 写 confidence 和 uncertainty。

标注级置信度最小派生规则：

```text
segments[].confidence = verifier.clip_analyzer_confidence

if confidence >= 0.80 and verification_status == "verified":
  uncertainty_reason = ""
  needs_human_review = false
elif confidence >= 0.60:
  uncertainty_reason = verifier.missing_evidence 或 "视觉证据部分可见，建议抽查"
  needs_human_review = false
else:
  uncertainty_reason = verifier.missing_evidence 或 "关键动作/物体证据不足"
  needs_human_review = true
```

这样只是在 VideoARM 已有 confidence 上补充赛题要求字段，不增加额外模型调用。

### 5.8 EXPVID VQA Agent

按三层生成：

```text
Level 1 Fine-grained Perception:
  工具、材料、容器、动作。

Level 2 Procedural Understanding:
  步骤顺序、完整性、下一步预测。

Level 3 Scientific / Review Reasoning:
  操作目的、状态变化、异常风险、复盘建议。
```

每个视频输出 6 条：

- perception 2 条；
- procedural 2 条；
- reasoning 2 条。

每条 VQA 必须有 `evidence_timestamps`。

### 5.9 Demo Agent

功能：

- 选择视频；
- 展示视频播放器；
- 展示 segment 时间轴；
- 点击 segment 跳转；
- 展示 VQA；
- 支持开放式问答。

问答不每次重跑全流程，而是基于：

- final segments；
- HM3 memory；
- VQA；
- 必要时调用 clip analyzer 回看局部帧。

### 5.10 Feedback Agent

标注 UI：

- 表格编辑 start/end/action/objects/caption；
- 表格编辑和补充 VQA，包括 question/answer/level/evidence_timestamps；
- 保存 feedback；
- 对修改片段局部重跑 verifier/writer；
- 对修改或新增 VQA 局部重跑 VQA verifier，检查问题是否能被视频证据支撑；
- 输出 `feedback_log.json`。

## 6. 工具池设计

工具参考 VideoARM 的注册方式，补充 VideoHV 的 caption 工具风格。

### 6.1 scene_snapper

用途：

- 粗看一段较长区间；
- 生成 scene / stage 描述；
- 写入 `clip_summaries`。

### 6.2 clip_analyzer

用途：

- 对一个短区间回答具体子问题；
- 用于验证动作和 caption。

### 6.3 object_presence_checker

用途：

- 判断物体是否出现；
- 输出近似位置；
- 可选 bbox，但不强依赖。

### 6.4 boundary_checker

用途：

- 看动作开始/结束前后帧；
- 判断边界是否过早/过晚；
- 给建议调整。

### 6.5 ontology_lookup

用途：

- phase -> candidate actions；
- action -> candidate objects；
- action -> visually distinguishing clues。

### 6.6 hypothesis_verifier

用途：

- VideoHV 风格总控验证；
- 汇总多个工具观察；
- 输出验证 JSON。

## 7. API 配置

配置优先参考本仓库 `test.py`，VideoHV-Agent 的配置作为结构参考。

`test.py` 当前可用调用方式：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://34.13.73.248:3888/v1",
    api_key="<YOUR_API_KEY>",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",
    messages=[{"role": "user", "content": "自我介绍一下"}],
    temperature=1,
)
```

因此 `agent_v2/config.py` 的默认值应改为：

```python
DEFAULT_BASE_URL = "http://34.13.73.248:3888/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_API_KEY = "<YOUR_API_KEY>"
```

仍保留环境变量覆盖，避免把本地测试配置写死：

```text
LABARM_BASE_URL
LABARM_API_KEY
LABARM_CONTROLLER_MODEL
LABARM_VISION_MODEL
LABARM_STRUCTURED_MODEL
```

VideoHV-Agent 风格的配置结构：

```python
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_BASE_URL = VIDEOHV_LLM_BASE_URL or DEFAULT_BASE_URL
CAPTION_BASE_URL = VIDEOHV_CAPTION_BASE_URL or DEFAULT_BASE_URL
STRUCTURED_LLM_BASE_URL = VIDEOHV_STRUCTURED_LLM_BASE_URL or DEFAULT_BASE_URL
```

但需要区分模型能力：

```text
LABARM_CONTROLLER_MODEL:
  负责规划、假设生成、修复和写结果。

LABARM_VISION_MODEL:
  必须支持图像输入，负责 scene_snapper / clip_analyzer / object_presence。

LABARM_STRUCTURED_MODEL:
  负责严格 JSON 输出。
```

默认建议：

```text
LABARM_CONTROLLER_MODEL = Qwen/Qwen3.5-35B-A3B
LABARM_VISION_MODEL = Qwen/Qwen3.5-35B-A3B
LABARM_STRUCTURED_MODEL = Qwen/Qwen3.5-35B-A3B
```

如果该部署不支持图像输入，视觉工具再单独切到可用 VL 模型，例如 `qwen-vl-max` 或服务端暴露的视觉模型。

### 7.1 关闭 Qwen thinking

需要默认关闭 thinking，原因：

- 本任务需要短、稳定、可解析 JSON；
- thinking 会增加延迟和 token 成本；
- thinking 内容可能污染结构化输出；
- 工具调用与强制工具选择在非思考模式下更稳定。

联网查证结果：

- 阿里云百炼 OpenAI-compatible Chat 文档中，`enable_thinking` 是非 OpenAI 标准参数；
- 使用 Python OpenAI SDK 时需要放入 `extra_body`；
- 配置方式为：

```python
extra_body={"enable_thinking": False}
```

因此所有 `client.chat.completions.create(...)` 统一经 `api_client.py` 封装，默认注入：

```python
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    tools=tools,
    tool_choice=tool_choice,
    extra_body={"enable_thinking": False},
)
```

兼容兜底：

- 如果当前代理服务不接受 `extra_body.enable_thinking`，捕获 `400` 或参数错误后自动重试一次；
- 重试时在 system message 或 user message 末尾添加 `/no_think`；
- Qwen 官方文档也说明 `/no_think` 可以关闭 thinking；
- 若模型是仅 non-thinking 的 instruct 版本，则该参数可无害保留或由兼容重试去除。

## 8. 目录结构

```text
agent_v2/
  PLAN.md
  run.py
  config.py
  schemas.py
  api_client.py
  media.py
  ontology.py

  core/
    labarm_hv_agent.py
    memory.py
    controller.py
    trace.py

  tools/
    registry.py
    scene_snapper.py
    clip_analyzer.py
    object_presence.py
    boundary_checker.py
    ontology_lookup.py

  agents/
    phase_hypothesis.py
    action_hypothesis.py
    hypothesis_verifier.py
    repair.py
    writer.py
    vqa.py
    interactive_qa.py
    feedback.py

  prompts/
    controller.md
    phase_hypothesis.md
    action_hypothesis.md
    verify_hypothesis.md
    repair.md
    writer.md
    vqa.md
    interactive_qa.md

  demo/
    app.py

  outputs/
    predictions.json
    run_log.json
    eval_dev.json
    feedback_log.json
```

注意：`agent_v2/` 是开发目录。最终提交时必须按题目要求整理为 `agent/`：

```text
agent/
  run.py
  requirements.txt
  README.md
  ...
```

最终根目录交付：

```text
agent/
run_log.json
predictions.json
report.pdf
slides.pptx 或 presentation.pdf
demo/ 或 agent/demo/
```

## 9. 输出 JSON 建议

`predictions.json` 顶层建议为视频结果数组，便于评委直接遍历：

```json
[
  {"video_id": "0061", "segments": []},
  {"video_id": "0068", "segments": []}
]
```

单个视频对象如下：

```json
{
  "video_id": "0061",
  "video_path": "videos/0061.mp4",
  "phase": "TLC_analysis",
  "phase_zh": "薄层色谱分析",
  "phase_confidence": 0.9,
  "segments": [
    {
      "segment_id": "0061_s001",
      "start": 0.0,
      "end": 32.9,
      "action": "prepare_TLC_plate",
      "action_zh": "准备TLC板",
      "objects": ["TLC板", "铅笔", "尺子"],
      "caption": "取出TLC板并用铅笔和尺子标记基线。",
      "evidence_timestamps": [1, 8, 25],
      "confidence": 0.86,
      "verification_status": "verified",
      "uncertainty_reason": "",
      "needs_human_review": false,
      "object_evidence": [
        {
          "object": "TLC板",
          "timestamp": 3,
          "present": true,
          "location": "画面中央偏右"
        }
      ]
    }
  ],
  "vqa": [
    {
      "vqa_id": "0061_q001",
      "level": "perception",
      "question": "该视频开头使用了什么工具标记TLC板？",
      "answer": "使用铅笔和尺子标记TLC板基线。",
      "evidence_timestamps": [1, 3, 8]
    }
  ],
  "processing_note": ""
}
```

字段与题目要求映射：

| 题目要求 | JSON 字段 |
|---|---|
| 关键实验操作起止时间 | `segments[].start`, `segments[].end` |
| 原子操作类别 | `segments[].action`, `segments[].action_zh` |
| 关键物体 | `segments[].objects`, `segments[].object_evidence` |
| caption | `segments[].caption` |
| 证据时间戳 | `segments[].evidence_timestamps` |
| 置信度 | `segments[].confidence` |
| 不确定性原因 | `segments[].uncertainty_reason` |
| 人工复核建议 | `segments[].needs_human_review` |
| VQA 数据 | `vqa[]` |
| 其他关键内容 | `phase`, `phase_zh`, `phase_confidence`, `processing_note` |

`run_log.json` 需要独立保存完整运行轨迹，至少包括：

```json
{
  "video_id": "0061",
  "steps": [
    {
      "step": "scene_snapper",
      "input": {"frame_range": "0-30"},
      "output": {"caption": "..."},
      "elapsed": 1.23
    },
    {
      "step": "hypothesis_verifier",
      "hypothesis_id": "0061_s001",
      "tool_calls": [],
      "verification_result": {}
    }
  ]
}
```

## 10. Dev 自评

先实现轻量评估：

- phase accuracy；
- tIoU@0.5 precision / recall / F1；
- action accuracy；
- object recall；
- average confidence；
- human review ratio；
- repair gain：K=0 vs K=1。

不要追求复杂 mAP，黑客松时间内 F1 和案例展示更重要。

## 11. 24 小时实施顺序

### P0: 新底座骨架，2 小时

- 建 `agent_v2/`；
- 复制/改造 VideoARM 的 agent class 结构；
- 接入 VideoHV 风格 config；
- 跑通单视频 metadata + 抽帧 + trace 保存。

### P1: VideoARM 主循环，4 小时

- 实现 HM3 memory；
- 实现 scene_snapper；
- 实现 clip_analyzer；
- controller 可以根据 memory 调用工具；
- 单视频生成粗 summary。

### P2: 动作假设生成，4 小时

- phase hypothesis；
- action hypothesis；
- ontology lookup；
- 初版 segment list。

### P3: VideoHV 验证融合，5 小时

- hypothesis verifier；
- distinguishing clues；
- object presence；
- boundary checker；
- failed verification regeneration；
- 输出 confidence / uncertainty / human review。

### P4: Repair + Writer，3 小时

- 低置信修复；
- segment 去重和排序；
- 写 `predictions.json`；
- 写完整 `run_log.json`。

### P5: VQA + Demo，4 小时

- EXPVID 三层 VQA；
- Gradio 视频问答；
- segment 时间轴展示；
- 标注表格编辑。

### P6: 全量运行和报告，2 小时

- 跑 dev；
- 跑 test；
- 生成 eval；
- 写 report/PPT 素材；
- 整理启动说明。

## 12. 风险和兜底

### 风险 1：视觉模型成本或速度过高

兜底：

- 先 1 fps；
- verifier 只看 start/mid/end 和边界前后；
- 每个 segment 最多 2 次工具调用。

### 风险 2：动作边界不稳定

兜底：

- 输出较保守的连续片段；
- 低置信标记人工复核；
- 报告中强调 boundary repair 机制。

### 风险 3：物体识别幻觉

兜底：

- object 必须来自视觉 evidence；
- 看不到就写 uncertainty；
- 不把 ontology candidate object 直接当成 observed object。

### 风险 4：Demo 来不及

兜底：

- 先做 predictions viewer + QA；
- 标注反馈只需保存 feedback 并局部重跑 verifier；
- 不做复杂前端。

## 13. 最终技术报告结构

1. Task and Dataset
2. System Overview
3. VideoARM-based Hierarchical Memory
4. VideoHV-inspired Hypothesis Verification
5. Tool Pool and Calling Strategy
6. Reflection and Repair Loop
7. Structured Outputs and VQA
8. Demo and Human Feedback Interface
9. Dev Evaluation and Case Study
10. Limitations and Future Work

报告核心图：

```text
Video -> HM3 Memory -> Hypothesis Agents -> Tool Verification -> Repair -> Predictions/VQA/Demo
```

核心卖点：

- 不是单次 MLLM caption；
- 是分层记忆 + 假设验证 + 失败修复；
- 每条结果有证据时间戳和置信度；
- 能被人工反馈局部更新。
