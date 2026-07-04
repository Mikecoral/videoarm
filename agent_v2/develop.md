# 开发日志

> 精简进度记录。只写「现在在做什么」+「还没做什么」。最新在上。

## 当前状态（2026-07-04）

**已完成：基础系统 + 验证/修复/审计闭环。**

- 后端：DashScope 官方端点 + `qwen3.5-27b`（多模态），`enable_thinking=False`。
- `agent_v2/` 系统：config / api_client / media / ontology / prompts / runlog / pipeline / run。
- 流程：抽帧 → clip 记忆(scene_snapper) → phase 假设 → 动作分段 → 逐段视觉核验 → K=1 repair → writer gate → coverage audit → 写结果。
- 输出：`predictions.json` + `run_log.json`（字段齐全，含 confidence/uncertainty/human_review）。
- 已跑旧版全量 test：`outputs/20260704_223009_test_15vids/`，15/15 完整；该 run 是修复前基线。

**最近修复/补充：**
- Phase fallback：phase prompt 输出 `alternative_phases`，action segmentation 可使用主 phase + 候选 phase 的 action catalog；低置信时退回更宽本体，减少 phase 误判级联。
- K=1 repair：默认开启 `LABARM_REPAIR=1`。对 rejected / 低置信片段回看局部帧，可删除幻觉片段或重写 start/end/action/objects/caption，并二次核验。
- Writer gate：最终输出默认只保留 `verified` / `partial`；全空时保留最高置信 rejected 作为人工复核兜底。
- Coverage audit + backtrack：记录覆盖率、长空洞，并回溯 `clip_memory` 判断空洞里是否有疑似实质操作；只标记复核，不自动补 segment。
- 旧 run 审计结论：主要问题是 rejected 直接进入输出、部分 rejected 可修复、少量 action_id 脱离 ontology、caption 过度具体、部分视频覆盖率偏低。

## 待办

**下一步**
- [ ] 用新流程重跑小样本（优先 `5`, `22`, `0075`, `12`），检查 repair/writer gate/coverage backtrack 的实际输出。
- [ ] 重跑全量 test（15 个视频），产出新的正式 `predictions.json` / `run_log.json`。
- [ ] 对比修复前基线：rejected 数量、writer gate 过滤数、repair 成功数、coverage suspicious gap 数。

**质量调优（基础范围内）**
- [ ] 收紧 action ontology 映射：segmentation 阶段默认必须优先使用本体 action_id；自拟 action 需显式说明。
- [ ] 收紧 caption：只写画面直接支持的核心动作，避免加入未确认物体、方向、工具细节。
- [ ] dev 自评脚本：phase acc / tIoU@0.5 P-R-F1 / action acc / object recall。

**进阶要求（可选）**
- [ ] repair K=2 + dev 上收益对比。
- [ ] EXPVID 三层 VQA 生成（每视频 6 条，带证据时间戳）。
- [ ] 可交互视频问答 Demo（Gradio）。
- [ ] 可视化标注 + 人工反馈 UI，反馈后局部重跑。

**交付整理（最后做）**
- [ ] 按题目要求把 agent_v2/ 整理为提交用 agent/。
- [ ] 技术报告 report.pdf + PPT。
