# 开发日志

> 精简进度记录。只写「现在在做什么」+「还没做什么」。最新在上。

## 当前状态（2026-07-04）

**已完成：基础要求端到端跑通。**

- 后端确定：DashScope 官方端点 + `qwen3.5-27b`（实测多模态，可读图），`enable_thinking=False`。
- `agent_v2/` 基础系统建成：config / api_client / media / ontology / prompts / runlog / pipeline / run。
- 流程：抽帧 → clip 记忆(scene_snapper) → phase 假设 → 动作分段 → 逐段视觉核验 → 写结果。
- 输出：`predictions.json` + `run_log.json`（字段齐全，含 confidence/uncertainty/human_review）。
- 已验证：dev/0061 端到端跑通；test/5.mp4 的 OpenCV 抽帧路径单独验证通过。
- 文档：README.md、ARCHITECTURE.md（架构+来源对照+审计）已写。

## 待办

**下一步（等确认）**
- [ ] 跑全量 test（15 个视频），产出正式 predictions/run_log。

**质量调优（基础范围内）**
- [ ] phase 判定偏差（0061 判成 analytical_sample_preparation，参考 TLC_analysis）——更密采样/更强 prompt。
- [ ] 动作边界精度有限（目前来自窗口摘要推断）。

**进阶要求（尚未开始）**
- [ ] repair 修复循环（K=1/K=2）+ dev 上收益对比。
- [ ] EXPVID 三层 VQA 生成（每视频 6 条，带证据时间戳）。
- [ ] 可交互视频问答 Demo（Gradio）。
- [ ] 可视化标注 + 人工反馈 UI，反馈后局部重跑。
- [ ] dev 自评脚本（phase acc / tIoU@0.5 P-R-F1 / action acc / object recall）。

**交付整理（最后做）**
- [ ] 按题目要求把 agent_v2/ 整理为提交用 agent/。
- [ ] 技术报告 report.pdf + PPT。
