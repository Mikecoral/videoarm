# 开发日志

> 精简进度记录。只写「现在在做什么」+「还没做什么」。最新在上。

## 当前状态（2026-07-05）

**最新：Omni pipeline（qwen3.5-omni-plus 整视频直接理解）接入完成。**

- 新增 `pipeline_omni.py` + `run_omni.py`：不抽帧，直接用 dashscope 原生 SDK + `file:///` 路径把本地视频发给模型。
- 技术路径：`dashscope.MultiModalConversation.call(model="qwen3.5-omni-plus", content=[{"video": "file://..."}])` — OpenAI 兼容端点不支持本地文件，fileid:// 也不适用，官方 dashscope SDK 的原生 API 才是正确姿势。
- 配置：`OMNI_MODEL`（默认 `qwen3.5-omni-plus`）、`OMNI_FPS`（默认 1fps）均可通过环境变量覆盖。
- 输出：与帧pipeline完全相同 schema（predictions.json + run_log.json + run_info.json），存到 `outputs/` 的带 `_omni_` 标签的 run 目录。
- 测试结果：
  - 视频 1（22MB 160s）：phase=filtration(0.98)，11段，84s完成。
  - 视频 5（14MB 67s）：phase=analytical_sample_preparation(0.95)，3段，37s完成。
  - 结果质量明显优于帧pipeline：caption 更详细，segment 粒度合理。
- 入口：`python -m agent_v2.run_omni --video_id 1`（单视频）或 `--split test`（批量）。


**最新：dev 评测 + 三轮优化完成。**

- 跑完 dev 5 个视频（3/7/8/0061/0068）基线评测：phase 3/5，tIoU@0.5 F1=0.18。
- 主要问题：phase 误判（TLC/柱色谱被归为 analytical_sample_preparation）、欠分割、相近动作混淆。

**Fix 1：Phase 识别（scene_snapper 改造）**
- `SCENE_SNAPPER` 新增第 4 条描述项，要求明确点名 TLC 板、展开缸、过柱机等专用仪器。
- `PHASE_HYPOTHESIS` 加视觉区分线索（TLC/柱层析/萃取/称量特征对比）。
- 结果：0061 phase ✓（TLC_analysis），0068 phase ✓（column_chromatography_setup）；总 phase 准确率 3/5→4/5。

**Fix 2：欠分割（action segmentation prompt）**
- 加"粒度规则"：同一动作重复出现时每次独立输出，不合并。
- Dense boundary resampling（pipeline 结构改动）：
  - `segment_actions` 初步分段后，对 ≥14s 的段用 5s 子窗口 + 4帧重跑 `scene_snapper`，再做局部再分割。
  - 配置项：`DENSE_ENABLED`, `DENSE_MIN_DURATION=14s`, `DENSE_SUBWINDOW=5s`, `DENSE_FRAMES=4`。
  - 效果：视频 3 从 4 段→10 段（ann=14），视频 8 从 6 段→12 段（ann=11），Recall 0.14→0.19。

**Fix 3：相近动作区分（action segmentation prompt）**
- 新增易混动作区分线索（measure vs add、rinse vs transfer、mix/vent vs add_extraction_solvent、prepare_TLC vs spot_TLC、transfer_to_sep_funnel vs add_liquid_to_reaction、allow_phase_separation）。
- Pipeline 结构修复：高置信度（≥0.85）时只用主 phase 的 action catalog，不混入 alternative phase 的动作，消除 `wet_and_equilibrate_column` 等跨 phase 幻觉。

**最终 dev 结果（dense 版）**：phase 4/5，TP=7，FP=35，FN=30，Prec=0.17，Rec=0.19，F1=0.18。
- 瓶颈已从「欠分割」转移到「action label 精度」：段数已接近 ann 水平，但 action 名称仍有混淆。

**最新：per-video memory bundle + bounded cross-video memory 已接入。**

- 新增 `memory.py`：每个视频处理后生成 `memory/{video_id}.json`，包含 frame index、clip memory、segment memory、evidence memory 和 retrieval index。
- `run.py` 现在会在每个视频完成后更新 `global_memory.json`，并在当前 run 目录写 `global_memory_snapshot.json`。
- 跨视频 memory 累计 action/object/uncertainty pattern 的统计和少量例子，用于交互式 Demo 的经验提示与 self-evolving 叙事。
- 设计边界：跨视频 memory 不反向覆盖当前视频标注，只作为检索问答上下文，避免历史错误污染新预测。
- 离线验证：用已跑完的 `5`, `22`, `0075` 输出构建 memory 成功；global summary 显示 3 个视频、top actions 包括 `pipette_sample_solution`/`add_solid_to_reaction`，top uncertainty 包括 `perception`/`temporal`；`retrieve_context("这个视频有没有过滤或者倒液操作？")` 能返回相关片段。

**最新：temporal_ok 惩罚软化，三视频验证完成。**

- `_apply_verification` 中 `temporal_ok` 与 `action_ok` 拆分：
  - `action_ok=False`（动作类型彻底不符）：`conf <= 0.40`，status=rejected。
  - `temporal_ok=False`（边界/顺序不确定，3帧采样下常见）：`conf <= 0.60`，status=partial，不再触发 rejected。
  - 旧代码把两者合并为 `conf <= 0.40` 导致几乎所有段被 rejected（0075 视频 10% coverage）。
- 实跑结果（`outputs_layered_eval/`）：
  - 视频 5：5→4 段，rejected 降为 0，有 perception 诊断和命名纠错建议。
  - 视频 22：10→6 段，avg_conf 提升（0.77→0.82），有 perception review suggestion。
  - 视频 0075（修复后）：3 段，fallback 不再被迫触发，coverage_audit 正确标记低覆盖供人工复核。

**已完成：基础系统 + 验证/修复/审计闭环 + 分层诊断。**

- 后端：DashScope 官方端点 + `qwen3.5-27b`（多模态），`enable_thinking=False`。
- `agent_v2/` 系统：config / api_client / media / ontology / prompts / runlog / pipeline / run。
- 流程：抽帧 → clip 记忆(scene_snapper) → phase 假设 → 动作分段 → 逐段视觉核验 → K=1 repair → writer gate → coverage audit → 写结果。
- 输出：`predictions.json` + `run_log.json`（字段齐全，含 confidence/uncertainty/human_review/uncertainty_type/review_suggestion/evidence_diagnosis）。
- 已跑旧版全量 test：`outputs/20260704_223009_test_15vids/`，15/15 完整；该 run 是修复前基线。

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
- [x] 小样本验证（`5`, `22`, `0075`）——分层诊断字段正确，temporal_ok 软化后 coverage 恢复正常。
- [x] 做交互式 Demo 后端：`retrieve_context()` + `answer_question()` 已实现，支持带 clip_memory 全量上下文的开放式 QA，测试通过。
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
