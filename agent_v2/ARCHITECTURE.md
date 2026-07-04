# LabARM-HV 系统架构说明（基础版审计文档）

> 本文档面向审计：说明当前**基础要求**版本的系统架构、数据流、输出格式，
> 以及每个部分**参考了什么来源**、哪些是原创/改造。
> 对应代码目录：`agent_v2/`。

---

## 0. 一句话定位

LabARM-HV 是一个针对第一视角化学实验视频的**多模态解析智能体**：
对每个视频输出「关键操作片段（起止时间 + 原子动作 + 关键物体 + caption + 证据时间戳）」，
并附带置信度 / 不确定性 / 人工复核建议。

系统骨架来自三份参考：

| 来源 | 提供的思想 | 在本系统中的落点 |
|---|---|---|
| **VideoARM**（Agentic Reasoning over Hierarchical Memory） | Observe→Think→Act→Memorize 控制循环；分层记忆；工具化回看 | `pipeline.py` 主循环、clip 记忆、`scene_snapper`/`verify` 视觉工具、`run_log.json` 轨迹 |
| **VideoHV**（Think, Then Verify：假设-验证多智能体） | 先生成假设，再用视觉工具校验；失败可修复 | 「动作分段=假设」→「逐片段视觉核验」→置信度/状态 |
| **EXPVID**（实验视频理解基准） | 实验视频三层理解目标（细粒度感知 / 过程理解 / 科学推理） | 输出粒度设计：动作+物体（感知）、有序分段（过程）；推理层留给进阶版 VQA |

以及赛题自带资源：`hackathon_release/ontology.md`（阶段/动作/器具参考本体）、
`splits/*.json`（官方划分）、DashScope `qwen3.5-27b`（多模态后端）。

---

## 1. 目录与模块职责

```
agent_v2/
  config.py       # 端点/模型/采样超参，全部可用 LABARM_* 环境变量覆盖
  api_client.py   # OpenAI 兼容封装：thinking off + retry + 多模态 + JSON 解析
  media.py        # 抽帧（OpenCV 1fps）→ 缩放 → base64；dev 帧可复用
  ontology.py     # 解析 ontology.md → 10 个 phase + 各自动作/候选器具
  prompts.py      # 各步骤提示词模板
  runlog.py       # 运行轨迹记录器 → run_log.json
  pipeline.py     # 单视频主流程（本系统核心）
  run.py          # 批处理入口 + 增量落盘 + 断点续跑
  outputs/        # predictions.json / run_log.json / frames_cache/
```

各模块参考来源：

- `api_client.py` — 结构对标 **VideoARM `api/client.py`**（统一 chat/工具封装 + 指数退避 retry），
  `enable_thinking=False` 走 `extra_body` 参考 **DashScope 官方文档**（Qwen3 thinking 开关）。
- `config.py` — 变量分层（controller/vision/structured 三角色 + 环境覆盖）参考 **VideoHV `config.py`**，
  端点/模型默认值即取自 VideoHV 的 DashScope 配置。
- `media.py` — 抽帧/缩放/base64 思路参考 **VideoARM `video/utils.py`**，改为纯 OpenCV（本机无 ffmpeg）。
- `pipeline.py` — Observe→Memorize→Think→Act→Verify→Write 的循环来自 **VideoARM**；
  「分段即假设、逐段视觉核验」来自 **VideoHV**。
- `ontology.py`、`prompts.py`、分段/置信度派生规则 — **本任务原创**。

---

## 2. 单视频数据流

```
run.py
 └─ pipeline.process_video(video)
     [Observe]  media.load_video → 1fps 抽帧 + 缩放(≤512px) + 缓存
     [Memorize] build_clip_memory
                  时间轴切 6–16 个窗口，每窗口取 3 帧
                  → scene_snapper(视觉) → clip_memory[] 场景摘要
     [Think]    hypothesize_phase
                  clip_memory + ontology.phase_catalog → 判定视频级 phase(JSON)
     [Act]      segment_actions
                  clip_memory + 该 phase 的候选动作/器具
                  → 有序、不重叠的动作片段假设(JSON)
                  → _normalize_segments 裁剪越界/重叠、丢弃过短
     [Verify]   verify_segment（每段一次视觉调用，回看 3 帧）
                  → action_ok / objects_present / caption_ok / confidence
                  → _apply_verification 派生 confidence/status/uncertainty/human_review
     [Write]    组装 prediction，全程步骤写入 run_log
```

**成本与稳定性设计**（对应 PLAN 风险 1）：视觉调用是最稀缺资源，因此
窗口数封顶 16、每次仅送 3 帧、每段最多 1 次核验、帧缩放到 ≤512px（实测单帧 base64 ~15KB、
DashScope 1–2s 稳定返回）。单视频（~140s）总视觉调用 ≈ 窗口数 + 片段数 ≈ 10–20 次。

---

## 3. 抽帧说明（审计重点：test 视频）

- **dev 视频**：release 已带 1fps 帧（`frames/<id>/`），`media.py` 直接复用（缩放后缓存），省解码。
- **test 视频**：**没有任何预抽帧**，`media.py` 走 `_extract_with_opencv`：
  按源 fps 计算步长，逐秒取帧、缩放、缓存。
- **已实测**：test `5.mp4`（67s, 1280×720, 30fps）→ 68 帧@1fps、缩放 512×288、~15KB/帧、3.0s 完成。
  所以「面对只有原始 mp4 的正常数据」这条路径是打通的，抽帧是流程内的必备第一步。

抽帧参数（环境变量可调）：`LABARM_FRAME_FPS=1`、`LABARM_FRAME_MAX_SIDE=512`、`LABARM_FRAME_QUALITY=75`。

---

## 4. 输出格式（实测样例）

### 4.1 `predictions.json`

顶层是**视频结果数组**，便于评委直接遍历。单个视频对象字段：
`video_id, video_path, phase, phase_zh, phase_confidence, segments[], clip_memory[], processing_note`。

单个 **segment**（真实输出，含进阶字段）：

```json
{
  "segment_id": "0061_s002",
  "start": 63.0,
  "end": 78.0,
  "action": "pipette_sample_solution",
  "action_zh": "移取样品溶液",
  "objects": ["移液枪", "透明小瓶"],
  "caption": "实验人员使用移液枪将液体从透明小瓶中吸取并转移至离心管中。",
  "evidence_timestamps": [65, 70, 75],
  "confidence": 0.4,
  "verification_status": "rejected",
  "uncertainty_reason": "画面显示操作是将液体从小瓶转移到离心管，而非标注所述……",
  "needs_human_review": true,
  "object_evidence": [
    {"object": "移液枪", "present": true},
    {"object": "透明小瓶", "present": true}
  ]
}
```

`clip_memory` 单元（分层记忆，随结果一并导出，供 Demo/复核复用）：

```json
{"start": 0.0, "end": 15.7, "frame_ts": [0.0, 8.0, 15.0], "summary": "画面中可见实验台、移液枪架……"}
```

### 4.2 字段 ↔ 赛题要求映射

| 赛题要求 | JSON 字段 | 归属 |
|---|---|---|
| 关键操作起止时间 | `segments[].start` / `end` | 基础 |
| 原子操作类别 | `segments[].action` / `action_zh` | 基础 |
| 关键物体 | `segments[].objects` / `object_evidence` | 基础 |
| caption | `segments[].caption` | 基础 |
| 证据时间戳 | `segments[].evidence_timestamps` | 基础 |
| 其他关键内容 | `phase` / `phase_zh` / `phase_confidence` / `clip_memory` | 基础 |
| 置信度 | `segments[].confidence` | 进阶（已顺带实现） |
| 不确定性原因 | `segments[].uncertainty_reason` | 进阶（已顺带实现） |
| 人工复核建议 | `segments[].needs_human_review` | 进阶（已顺带实现） |

### 4.3 `run_log.json`

顶层是**每视频一条轨迹**的数组。单条：`video_id, video_info, steps[]`。
每个 step 记录 `step 名 / input / output / elapsed`，工具调用可完整回放。
实测某 `verify_segment` step：

```json
{
  "step": "verify_segment",
  "input": {"segment": [0.0, 31.0], "action": "finish_sample_container"},
  "output": {"action_ok": true, "objects_present": ["白色小方块","绿色细笔"],
             "caption_ok": true, "confidence": 0.95, "note": "画面清晰显示……"},
  "elapsed": 1.893
}
```

step 类型：`scene_snapper`(建记忆) / `phase_hypothesis` / `action_segmentation` /
`verify_segment` / `write_prediction`。

---

## 5. 关键设计规则（原创部分）

1. **允许片段间空隙**：分段提示词明确「不要强行铺满整段视频」，因为参考标注本身有空隙
   （如 0061 的 32.9s→41.5s 无标注）。强行连续会拉低精度。
2. **objects 宁缺毋滥 + 视觉确认覆盖**：分段先给候选 objects，核验时用 `objects_present`
   仅保留画面可确认的物体，抑制幻觉（对应 PLAN 风险 3）。
3. **置信度一致性约束**：核验判定动作不符时，强制 `confidence ≤ 0.4`、`status=rejected`、
   `needs_human_review=true`——避免「高置信却被否」的自相矛盾。
4. **派生分级**（`_apply_verification`）：
   `conf≥0.8 且 verified` → 无需复核；`0.6≤conf<0.8` → 建议抽查；
   `<0.6 或 rejected` → 转人工复核并给出不确定原因。

---

## 6. 与 PLAN.md 的关系 / 当前边界

**已实现（基础要求全覆盖 + 部分进阶）**：分层记忆、phase/动作假设、视觉核验、
置信度/不确定性/人工复核、统一批处理入口、predictions/run_log、断点续跑。

**尚未实现（PLAN 中的进阶，本轮不做）**：
repair 修复循环（K=1/K=2）、EXPVID 三层 VQA 生成、Gradio 交互 Demo、可视化标注反馈 UI、
dev 自评脚本（tIoU/action acc 等）。

**已知质量局限**（诚实记录）：
- phase 判定在动作歧义大的视频上会偏（如 0061 被判成 `analytical_sample_preparation`
  而参考为 `TLC_analysis`，因该视频确含大量移液操作）。可通过更密采样 / 更强 phase prompt / 
  引入 repair 改善。
- 动作边界来自窗口级摘要推断，精度有限；进阶版的 `boundary_checker` 尚未接入。
- 视觉证据只做「是否可见」的定性判断，未做 bbox 级定位（spatial 标注方向留待进阶）。

---

## 7. 复现方式

```bash
pip install -r agent_v2/requirements.txt
python -m agent_v2.run --video_id 0061      # 单视频
python -m agent_v2.run --split test         # 全量测试集（15 个）
python -m agent_v2.run --split test --resume  # 断点续跑
```
输出在 `agent_v2/outputs/`。
```
