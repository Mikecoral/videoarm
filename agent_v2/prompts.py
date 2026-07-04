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
  "evidence_timestamps": [<支持该判断的秒数, 3-5个>],
  "reason": "<一句话理由>"
}}
只输出 JSON。"""

ACTION_SEGMENTATION = """你在把一段化学实验视频切分成若干「关键操作片段」，并为每段生成结构化标注。

该视频的实验阶段：{phase_id}（{phase_zh}）。
视频总时长约 {duration:.0f} 秒。

该阶段常见的原子动作（参考本体，可超出此集合）：
{action_catalog}

按时间顺序的窗口观察摘要（每行含大致时间）：
{clip_summaries}

要求：
- 依据观察摘要，输出时间上从前到后、不重叠的关键操作片段；
- 片段之间允许存在没有关键操作的空隙，不要强行铺满整段视频；
- 每段选择最贴切的 action_id（尽量用本体中的；确实不在本体内可自拟并注明）；
- objects 只写你有证据认为出现的关键物体，宁缺毋滥；
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
  "action_ok": <true/false，动作是否与画面相符>,
  "objects_present": ["<画面中确实能看到的、objects里的物体>"],
  "objects_missing": ["<objects里但画面看不到的>"],
  "caption_ok": <true/false，描述是否与画面基本相符>,
  "corrected_caption": "<若原caption有明显错误则给出修正，否则复述原caption>",
  "confidence": <0-1，你对本时间段整体标注正确性的信心>,
  "note": "<一句话说明证据或不确定原因>"
}}
只输出 JSON。"""
