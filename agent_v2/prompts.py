"""Prompt templates for the LabARM-HV pipeline (kept in one place)."""

SCENE_SNAPPER = """你在分析一段第一视角化学实验视频的一个时间窗口（{start:.0f}s–{end:.0f}s）。
下面按时间顺序给出该窗口内的若干帧。请用中文简洁描述：
1) 画面中出现的实验器具/容器/材料；
2) 实验人员双手正在做的具体操作；
3) 是否发生了明显的状态变化（如倒液、盖盖、书写、点样等）。
只描述你真实看到的内容，看不清就说不确定。控制在 3 句以内。"""

PHASE_HYPOTHESIS = """你在判断一整段化学实验视频属于哪个「实验阶段(phase)」。

候选阶段（参考本体，非封闭集合）：
{phase_catalog}

视频总时长约 {duration:.0f} 秒。以下是按时间顺序的窗口观察摘要：
{clip_summaries}

请输出 JSON：
{{
  "phase": "<最匹配的 phase_id>",
  "phase_zh": "<该阶段中文名>",
  "confidence": <0-1>,
  "alternative_phases": [
    {{"phase": "<第二候选 phase_id>", "phase_zh": "<中文名>", "confidence": <0-1>, "reason": "<一句话>"}},
    {{"phase": "<第三候选 phase_id>", "phase_zh": "<中文名>", "confidence": <0-1>, "reason": "<一句话>"}}
  ],
  "evidence_timestamps": [<支持该判断的秒数, 3-5个>],
  "reason": "<一句话理由>"
}}
如果最匹配阶段并不确定，必须给出 1-2 个 alternative_phases，尤其注意区分 TLC_analysis、
analytical_sample_preparation、reaction_setup、extraction 这类画面上都可能出现移液/倒液的阶段。
只输出 JSON。"""

ACTION_SEGMENTATION = """你在把一段化学实验视频切分成若干「关键操作片段」，并为每段生成结构化标注。

该视频的实验阶段：{phase_id}（{phase_zh}）。
视频总时长约 {duration:.0f} 秒。

候选阶段常见的原子动作（参考本体，可超出此集合）。先优先使用主阶段动作；
如果观察摘要明显更符合其他候选阶段的动作，应选择更贴切的 action_id，不要被主阶段误判限制：
{action_catalog}

按时间顺序的窗口观察摘要（每行含大致时间）：
{clip_summaries}

要求：
- 依据观察摘要，输出时间上从前到后、不重叠的关键操作片段；
- 凡摘要中明确提到有实质性操作（如倒液、移液、搅拌、抽滤、转移、洗涤、点样等）的时间窗口，
  都必须产出对应的片段，不可遗漏；仅纯粹的静止等待或背景不变的段落可跳过；
- 每段选择最贴切的 action_id（尽量用本体中的；确实不在本体内可自拟并注明）；
- objects 写该片段中出现的关键实验器具和材料；
- caption 用一句中文客观描述该片段发生了什么；
- evidence_timestamps 给出该片段内最能佐证的 2-4 个秒数。

输出 JSON 数组，每个元素：
{{
  "start": <秒>,
  "end": <秒>,
  "action": "<action_id>",
  "action_zh": "<动作中文名>",
  "objects": ["..."],
  "caption": "...",
  "evidence_timestamps": [<秒>, ...]
}}
只输出 JSON 数组。"""

VERIFY_SEGMENT = """请核对一条实验操作标注是否与画面一致。

标注：
- 时间段：{start:.0f}s–{end:.0f}s
- 动作(action)：{action} ({action_zh})
- 关键物体(objects)：{objects}
- 描述(caption)：{caption}

下面是该时间段内按时间顺序抽取的若干帧。请只根据画面判断，输出 JSON：
{{
  "action_ok": <true/false，核心操作类型是否与画面相符；判断重点是"做了什么动作"，
               物体名称细节差异（如"试管"与"离心管"、"烧杯"与"容器"）不影响此判断>,
  "objects_present": ["<画面中确实能看到的、objects里的物体>"],
  "objects_missing": ["<objects里但画面看不到的>"],
  "caption_ok": <true/false，描述的核心操作是否与画面基本相符；
                容器/物体名称的细微差异不算错误，仅操作类型或方向明显有误时才为false>,
  "corrected_caption": "<若原caption核心操作描述有明显错误则给出修正，否则复述原caption>",
  "confidence": <0-1，你对本时间段整体标注正确性的信心>,
  "note": "<一句话说明证据或不确定原因>"
}}
只输出 JSON。"""

REPAIR_SEGMENT = """你在修复一条未通过核验或低置信的实验操作片段假设。

原始片段：
- 时间段：{start:.1f}s–{end:.1f}s
- 动作(action)：{action} ({action_zh})
- 关键物体(objects)：{objects}
- 描述(caption)：{caption}
- 上一轮核验状态：{status}
- 上一轮置信度：{confidence}
- 上一轮问题说明：{uncertainty_reason}

该视频候选动作本体：
{action_catalog}

下面是原片段附近按时间顺序抽取的若干帧。请只根据画面修复这条假设：
- 如果画面中没有可确认的关键实验操作，或原假设明显是幻觉，返回 delete=true；
- 如果有可确认操作，返回 delete=false，并给出修复后的 start/end/action/action_zh/objects/caption/evidence_timestamps；
- start/end 必须留在 {context_start:.1f}s–{context_end:.1f}s 范围内；
- action 尽量使用本体中的 action_id；确实没有合适本体动作才自拟；
- 不要为了保留原假设而编造看不见的物体或动作。

输出 JSON：
{{
  "delete": <true/false>,
  "start": <秒，delete=true时可为null>,
  "end": <秒，delete=true时可为null>,
  "action": "<action_id，delete=true时可为空>",
  "action_zh": "<中文动作名，delete=true时可为空>",
  "objects": ["<画面可确认的关键物体>"],
  "caption": "<一句中文客观描述，delete=true时说明删除原因>",
  "evidence_timestamps": [<秒>, ...],
  "repair_reason": "<一句话说明如何修复或为何删除>"
}}
只输出 JSON。"""
