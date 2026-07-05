# CLAUDE.md — agent_v2

## 开发日志工作流（每次必须遵守）

1. **开工前**：先读 `agent_v2/develop.md`，了解「现在在做什么 / 还没做什么」，再动手。
2. **收工后**：每完成一处改动，立即更新 `develop.md`。
3. **保持动态精简**：`develop.md` 不是流水账，只保留有效信息。
   - 旧的、已完成到不再需要参考的、过时的条目——删掉。
   - 最新状态写在最上面。
   - 目标是任何时候打开它都能一眼看清当前进度，而不是越写越长。

> 详细规划见 `PLAN.md`（长期不常变）；`develop.md` 只写动态进度。

## 项目概览

实验流程视频理解 Agent。后端：DashScope 官方端点 + `qwen3.5-27b`（多模态，`enable_thinking=False`）。

流程：抽帧 → clip 记忆(scene_snapper) → phase 假设 → 动作分段 → 逐段视觉核验 → 写结果。
输出：`predictions.json` + `run_log.json`。

### 模块
- `config.py` / `api_client.py` / `media.py`（抽帧）/ `ontology.py`
- `prompts.py` / `runlog.py` / `pipeline.py`（主流程）/ `run.py`（入口）

### 运行
入口 `run.py`。密钥在 `.env`（见 `.env.example`）。
