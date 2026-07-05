# LabARM-HV: 实验视频解析智能体

基于 VideoARM 框架，结合 Omni 全局视频理解与多 Agent 精修，对第一视角化学实验视频进行原子操作分段、物体识别和 VQA 生成。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 DashScope + DeepSeek API 密钥
```

## 全流程 Pipeline

```
Video → Omni 分段 → MLLM 核验 → JSON → VQA → Viewer/Demo
```

### 单视频

```bash
# 基础版：Omni 分段 + 概括 + Viewer + VQA（无需抽帧）
python -m agent_v2.run_full --video_id 1

# 完整版：含逐帧 MLLM 核验（需抽帧）
python -m agent_v2.run_full --video_id 1 --verify
```

输出 `outputs/full_pipeline/`:
- `predictions.json` — 分段结果（phase/segments/confidence/VQA）
- `vqa_output.json` — VQA 问答对
- `viewers/<video_id>.html` — 可交互查看器

### 批处理

```bash
# Omni 快速批处理（~50s/视频，5并发）
python -m agent_v2.run_batch_omni_only --split all

# 全流程批处理
python -m agent_v2.run_full --split all
```

### 从已有 predictions 生成 VQA

```bash
python -m agent_v2.run_vqa
```

## Demo 前端

```bash
python -m agent_v2.demo_server --port 8080
```

浏览器 `http://localhost:8080` — 浏览视频列表、分段时间轴（点击跳转）、键盘 j/k 切换、空格暂停/播放。

## 架构

```
agent_v2/
├── run_full.py            # 全流程入口
├── run_batch_omni_only.py # Omni 批处理
├── run_omni_seg.py        # 单视频 Omni + Viewer
├── run_vqa.py             # VQA 生成
├── demo_server.py         # Demo 前端（含 Range 支持）
├── range_server.py        # HTTP Range 服务器
│
├── pipeline_omni.py       # Omni 全局分段
├── pipeline.py            # 帧级核验/修复/记忆
├── pipeline_unified.py    # 统一 Pipeline
├── pipeline_hybrid.py     # 混合 Pipeline
│
├── prompts.py             # Prompt 模板
├── config.py              # 配置
├── api_client.py          # API 客户端
├── media.py               # 视频抽帧
├── memory.py              # 跨视频记忆
├── ontology.py            # 实验操作本体
├── runlog.py              # 运行日志
│
├── vqa/                   # VQA 模块（自包含）
│   ├── generator.py       # 模板/MLLM VQA 生成器
│   ├── types.py           # SegmentContext / KeyframeContext
│   └── keyframes.py       # 关键帧提取
│
└── outputs/               # 输出目录
```

## 输出格式

```json
[{
  "video_id": "1",
  "phase": "filtration", "phase_zh": "过滤与抽滤",
  "phase_confidence": 0.98,
  "video_summary": "该视频展示了过滤与抽滤实验的核心流程...",
  "segments": [{
    "segment_id": "1_s001",
    "start": 0, "end": 13,
    "action": "pour_mixture_to_filter",
    "action_zh": "将混合物倒入过滤装置",
    "objects": ["棕色试剂瓶", "布氏漏斗"],
    "caption": "操作者手持棕色试剂瓶，将瓶内的固液混合物倒入布氏漏斗中。",
    "evidence_timestamps": [5, 8, 11],
    "confidence": 0.5,
    "verification_status": "partial",
    "needs_human_review": true
  }],
  "vqa": [{
    "question": "这段视频主要展示了什么实验操作？",
    "answer": "...",
    "level": "L1", "category": "operation"
  }]
}]
```

## 关键配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LABARM_DATA_ROOT` | `../hackathon_release` | 数据集路径 |
| `LABARM_OMNI_MODEL` | `qwen3.5-omni-plus` | Omni 视频理解模型 |
| `LABARM_MLLM_MODEL` | `qwen3.5-27b` | 视觉/多模态模型 |
| `LABARM_LLM_MODEL` | `deepseek-chat` | 文本/结构化模型 |
| `LABARM_VERIFY` | `0` | 是否逐帧核验 |

## Pipeline 选择

| 入口 | 速度 | 置信度 | 适用 |
|---|---|---|---|
| `run_batch_omni_only` | ~50s/视频 | 无 | 快速探索 |
| `run_full` (默认) | ~50s/视频 | 无 | Viewer+VQA |
| `run_full --verify` | ~3min/视频 | 真实核验 | 最终交付 |
