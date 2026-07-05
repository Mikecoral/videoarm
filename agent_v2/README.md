# LabSleuth: 实验视频解析智能体

基于 VideoARM 框架，结合 Omni 全局视频理解与 MLLM 逐帧核验，对第一视角化学实验视频进行原子操作分段、物体识别和 VQA 生成。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 DashScope + DeepSeek API 密钥
```

## 全流程 Pipeline

```
Video → Omni 分段 → MLLM 逐帧核验 → JSON → VQA → Viewer/Demo
```

### 单视频

```bash
python -m agent_v2.run_full --video_id 1 --verify
```

### 批处理

```bash
python -m agent_v2.run_full --split all --verify
```

输出 `outputs/full_pipeline/`:
- `predictions.json` — 分段结果（phase/segments/confidence/VQA）
- `vqa_output.json` — VQA 问答对
- `viewers/<video_id>.html` — 可交互查看器

## Demo 前端

```bash
python -m agent_v2.demo_server --port 8080
```

浏览器 `http://localhost:8080` — 浏览视频列表、分段时间轴（点击跳转）、键盘 j/k 切换、空格暂停/播放。

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
    "confidence": 0.95,
    "verification_status": "verified",
    "needs_human_review": false
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
| `LABARM_MLLM_MODEL` | `qwen3.5-27b` | 视觉核验模型 |
| `LABARM_LLM_MODEL` | `deepseek-chat` | 文本推理模型 |
