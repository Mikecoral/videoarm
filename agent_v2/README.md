# LabARM-HV — 实验视频解析智能体（基础版）

第一视角化学实验视频的多模态解析智能体。以「分层记忆 + 假设 + 视觉校验」的方式，
对每个视频输出关键操作片段（起止时间、原子动作、关键物体、caption、证据时间戳），
并附带置信度 / 不确定性 / 人工复核建议。

## 运行环境

```bash
pip install -r requirements.txt
```

后端在 `hxa` 分支中按能力分路：

- MLLM / 视觉调用：DashScope OpenAI 兼容端点，默认 `qwen3.5-27b`；
- 纯文本 / 结构化推理：DeepSeek OpenAI 兼容端点，默认 `deepseek-chat`。

密钥通过环境变量或本地 `.env` 提供，不写入源码：

```bash
export LABARM_MLLM_API_KEY=sk-...       # DashScope key
export LABARM_MLLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LABARM_MLLM_MODEL=qwen3.5-27b

export LABARM_LLM_API_KEY=sk-...        # DeepSeek key
export LABARM_LLM_BASE_URL=https://api.deepseek.com
export LABARM_LLM_MODEL=deepseek-chat
```

## 统一入口

```bash
# 批量处理测试集（15 个视频）
python -m agent_v2.run --split test

# 处理带参考标注的 dev 集
python -m agent_v2.run --split dev

# 单个视频
python -m agent_v2.run --video_id 0061

# 断点续跑（跳过已完成视频）
python -m agent_v2.run --split test --resume
```

## hxa Hybrid 入口

`hxa` 分支新增一条 hybrid pipeline：先用帧级 clip memory 生成候选，再由独立的对象确认、
边界精修、视觉核验/修复和文本审计 agent 逐步收紧结果。

```bash
# 单个视频
python -m agent_v2.run_hybrid --video_id 0061

# dev / test 批处理
python -m agent_v2.run_hybrid --split dev
python -m agent_v2.run_hybrid --split test
```

当前三条 pipeline 的定位：

| 入口 | 模型输入 | 优点 | 不足 |
|---|---|---|---|
| `run.py` | 抽帧窗口 + 局部帧核验 | 有 verify/repair 闭环 | 全局理解依赖窗口摘要 |
| `run_omni.py` | 完整视频 | 全局理解强，不抽帧 | 无逐段核验，默认 partial |
| `run_hybrid.py` | 抽帧窗口 + 局部证据 + 文本审计 | 兼顾候选生成、证据核验、边界/对象收紧 | 调用次数更多 |

输出写到 `agent_v2/outputs/`：

- `predictions.json` — 每个视频的结构化解析结果；
- `run_log.json` — 每个视频的完整运行轨迹（工具调用、输入、输出、耗时）；
- `memory/{video_id}.json` — 单视频持久记忆包，供交互式 Demo 检索问答；
- `global_memory_snapshot.json` / `global_memory.json` — 跨视频经验库快照/累计库；
- `frames_cache/` — 抽帧缓存（1 fps、缩放后）。

批量处理不对任何单条视频做人工干预；结果每处理完一个视频即增量落盘。

## 处理流程（单视频）

1. **Observe** — OpenCV 按 1 fps 抽帧并缩放（dev 若已带 1fps 帧则复用）。
2. **Memorize** — 时间轴切成若干窗口，每窗口一次视觉调用生成场景摘要（分层 clip 记忆）。
3. **Think** — 基于 clip 记忆 + ontology 判定视频级实验阶段 phase。
4. **Act** — 基于 clip 记忆 + 该 phase 的候选动作，切分并标注关键操作片段。
5. **Verify** — 对每个片段回看少量帧做视觉核验，产出置信度、修正 objects/caption、
   不确定性原因与人工复核建议。
6. **Write** — 组装 `predictions.json` 条目并记录轨迹。
7. **Persist Memory** — 将 `clip_memory`、片段标注、证据帧、核验诊断写成 per-video
   memory bundle，并把高置信动作/物体/不确定性模式累计到 bounded cross-video memory。

## 记忆机制

`agent_v2` 使用轻量版 VideoARM-style 记忆，不做不可控的跨视频自动改写：

- **Per-video memory**：`memory/{video_id}.json` 保存当前视频的 frame index、clip summaries、
  segment memory、verification evidence 和 lexical retrieval index。交互式 Demo 可以先检索该文件，
  再把相关时间段、caption、证据帧路径交给模型回答。
- **Cross-video memory**：`global_memory.json` 累计已处理视频中的 action/object/uncertainty pattern
  统计和少量例子，用于 Demo 的“经验提示”和报告里的 self-evolving 叙事。它只提供参考上下文，
  不会覆盖当前视频的结构化标注。

## 关键配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LABARM_FRAME_FPS` | 1 | 抽帧频率 |
| `LABARM_FRAME_MAX_SIDE` | 512 | 帧最长边（控制 base64 体积） |
| `LABARM_MIN_WINDOWS`/`LABARM_MAX_WINDOWS` | 6/16 | 记忆窗口数量范围 |
| `LABARM_FRAMES_PER_WINDOW` | 3 | 每个窗口送入视觉的帧数 |
| `LABARM_VERIFY` | 1 | 是否开启逐片段视觉核验 |
| `LABARM_ENABLE_THINKING` | 0 | 是否开启 Qwen thinking |
| `LABARM_MLLM_MODEL` | qwen3.5-27b | 视觉/多模态模型 |
| `LABARM_LLM_MODEL` | deepseek-chat | 纯文本结构化模型 |
| `LABARM_HYBRID_OBJECT` | 1 | hybrid 是否开启对象确认 agent |
| `LABARM_HYBRID_BOUNDARY` | 1 | hybrid 是否开启边界精修 agent |
| `LABARM_HYBRID_TEXT_AUDIT` | 1 | hybrid 是否开启 DeepSeek 文本审计 |

## 输出字段

`predictions.json` 为视频结果数组，单个视频：

```json
{
  "video_id": "0061",
  "phase": "TLC_analysis",
  "phase_zh": "薄层色谱分析",
  "phase_confidence": 0.9,
  "segments": [{
    "segment_id": "0061_s001",
    "start": 0.0, "end": 32.9,
    "action": "prepare_TLC_plate", "action_zh": "准备TLC板",
    "objects": ["TLC板", "铅笔"],
    "caption": "...", "evidence_timestamps": [1, 8, 25],
    "confidence": 0.86, "verification_status": "verified",
    "uncertainty_reason": "", "needs_human_review": false,
    "object_evidence": [{"object": "TLC板", "present": true}]
  }],
  "clip_memory": [...]
}
```
