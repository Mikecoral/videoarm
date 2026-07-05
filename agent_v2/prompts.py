"""Prompt templates for the LabARM-HV pipeline (kept in one place)."""

SCENE_SNAPPER = """你在分析一段第一视角化学实验视频的一个时间窗口（{start:.0f}s–{end:.0f}s）。
下面按时间顺序给出该窗口内的若干帧。请用中文简洁描述：
1) 画面中出现的实验器具/容器/材料；
2) 实验人员双手正在做的具体操作（按"左手：<操作>；右手：<操作>"格式输出；
   若某只手不可见或未参与操作，注明"左手：不可见"或"左手：未参与"）；
3) 是否发生了明显的状态变化（如倒液、盖盖、书写、点样等）；
4) 是否出现任何专用分析/分离仪器或耗材，如：TLC 板、毛细管、展开缸、紫外灯、
   过柱机、装样柱、天平、旋蒸仪、分液漏斗、真空泵——若出现请明确点名。
只描述你真实看到的内容，看不清就说不确定。控制在 4 句以内。"""

PHASE_HYPOTHESIS = """你在判断一整段化学实验视频属于哪个「实验阶段(phase)」。

候选阶段（参考本体，非封闭集合）：
{phase_catalog}

视频总时长约 {duration:.0f} 秒。以下是按时间顺序的窗口观察摘要：
{clip_summaries}

【视觉区分关键线索 — 优先用以下特征区分，不要仅靠移液/倒液动作来判断】
- TLC_analysis：画面中出现 TLC 薄板（白色/铝基薄片）、毛细管点样、展开缸、铅笔划线、紫外灯；
  不以移液枪为主要器具；板上可见斑点。
- column_chromatography_setup：出现自动过柱机（大型仪器主机）、装样柱/样品筒（短粗柱）、
  管路连接；操作以安装、插入、启动仪器为主。
- analytical_sample_preparation：主要使用移液枪+EP管/离心管/核磁管，无 TLC 板和过柱机，
  核心操作是稀释/转移/封管/标记。
- extraction：出现分液漏斗（梨形/筒形玻璃漏斗+活塞），含振摇排气、放出下层、加萃取溶剂。
- reaction_setup：出现烧瓶/圆底瓶+磁力搅拌/加热台，或注射器加液进烧瓶体系。
- weighing：出现天平台面和称量纸/称量舟，动作以加减固体和读数为主。

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
  "reason": "<一句话理由，必须引用至少一个视觉区分线索>"
}}
如果最匹配阶段并不确定，必须给出 1-2 个 alternative_phases。只输出 JSON。"""

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
- 【粒度规则】同一类动作若在视频中重复出现多次（如多次用注射器逐次加液、多次移液），
  每次独立操作应单独输出一个片段，不要合并为一大段；
  判断依据：相邻窗口若描述"再次加入"、"继续量取"、"又一次"等，说明是独立重复操作；
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
【易混动作区分线索】
- measure_liquid_for_addition vs add_liquid_to_reaction：
  measure = 将液体从试剂瓶吸入注射器/量筒（吸取阶段）；
  add = 将注射器/量筒中的液体推入/倒入反应容器（注入阶段）。
- rinse_container vs transfer_liquid_to_sep_funnel：
  rinse = 用少量液体润洗容器内壁，随后将液体倒出丢弃；
  transfer = 将液体从一个容器直接倒入另一个容器（目的是转移，不是润洗）。
- transfer_liquid_to_sep_funnel vs add_liquid_to_reaction：
  transfer_to_sep_funnel = 将液体从烧杯/圆底烧瓶倒入分液漏斗（目标容器是分液漏斗）；
  add_to_reaction = 向圆底烧瓶/锥形瓶等反应容器中加液（目标是反应瓶，不是分液漏斗）。
- mix_and_vent_sep_funnel vs add_extraction_solvent：
  mix/vent = 分液漏斗盖好后倒置振摇并开活塞排气，容器本身在手中翻转；
  add = 从外部试剂瓶向分液漏斗内倒入液体，是倒入动作而非振摇。
- allow_phase_separation：
  分液漏斗静置在铁架台上等待液体自然分层，无明显手部操作；
  凡摘要中描述"静置"、"分层"、"固定在架子上"且持续较长时间，应标注此动作。
- prepare_TLC_plate vs spot_TLC_plate：
  prepare = 用铅笔在 TLC 板上划线、标记位置，板上尚无样品点；
  spot = 用毛细管蘸取样品并将液滴点到 TLC 板指定位置。
- transfer_and_add_small_volume_liquid vs add_extraction_solvent：
  transfer_and_add_small_volume_liquid = 用注射器/移液枪向反应瓶体系中少量多次注入液体，
    容器为圆底烧瓶或锥形瓶，常见于反应建立阶段；
  add_extraction_solvent = 向分液漏斗（必须有玻璃活塞）中大量倾倒有机萃取溶剂；
  若不能确认容器为真正带活塞的分液漏斗，不要使用 add_extraction_solvent。

只输出 JSON 数组。"""

VERIFY_SEGMENT = """请核对一条实验操作标注是否与画面一致。

标注：
- 时间段：{start:.0f}s–{end:.0f}s
- 动作(action)：{action} ({action_zh})
- 关键物体(objects)：{objects}
- 描述(caption)：{caption}

下面是该时间段内按时间顺序抽取的若干帧。请只根据画面判断，并按 Dr.V 风格拆成三层诊断：
- perception：物体/器具/材料是否真实可见，是否存在遮挡、模糊、命名不确定；
- temporal：动作是否真的发生，起止边界和动作顺序是否与画面一致；
- cognition：caption 是否只表达画面支持的事实，是否加入了过度推理的实验意图、方向或结果。

输出 JSON：
{{
  "perception_ok": <true/false，关键物体是否足以从画面确认>,
  "temporal_ok": <true/false，核心动作、动作方向、边界/顺序是否与画面一致>,
  "action_ok": <true/false，核心操作类型是否与画面相符；判断重点是"做了什么动作"，
               物体名称细节差异（如"试管"与"离心管"、"烧杯"与"容器"）不影响此判断>,
  "objects_present": ["<画面中确实能看到的、objects里的物体>"],
  "objects_missing": ["<objects里但画面看不到的>"],
  "caption_ok": <true/false，描述的核心操作是否与画面基本相符；
                容器/物体名称的细微差异不算错误，仅操作类型或方向明显有误时才为false>,
  "corrected_caption": "<若原caption核心操作描述有明显错误则给出修正，否则复述原caption>",
  "missing_evidence": ["<缺失或不足的证据，如'看不到液体是否转移'、'边界前后帧不足'>"],
  "uncertainty_level": "<none|perception|temporal|cognitive|mixed>",
  "review_suggestion": "<给人工复核员的短建议；若无须复核则为空字符串>",
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

# ====================================================================
# Omni whole-video prompts (qwen3.5-omni-plus, no frame extraction)
# ====================================================================

OMNI_PHASE_HYPOTHESIS = """你在观看一段第一视角化学实验视频（完整视频，总时长约 {duration:.0f} 秒）。

候选实验阶段（参考本体，非封闭集合）：
{phase_catalog}

【视觉区分关键线索 — 优先用以下特征区分，不要仅靠移液/倒液动作来判断】
- TLC_analysis：画面中出现 TLC 薄板（白色/铝基薄片）、毛细管点样、展开缸、铅笔划线、紫外灯；
  不以移液枪为主要器具；板上可见斑点。
- column_chromatography_setup：出现自动过柱机（大型仪器主机）、装样柱/样品筒（短粗柱）、
  管路连接；操作以安装、插入、启动仪器为主。
- analytical_sample_preparation：主要使用移液枪+EP管/离心管/核磁管，无 TLC 板和过柱机，
  核心操作是稀释/转移/封管/标记。
- extraction：出现分液漏斗（梨形/筒形玻璃漏斗+活塞），含振摇排气、放出下层、加萃取溶剂。
- reaction_setup：出现烧瓶/圆底瓶+磁力搅拌/加热台，或注射器加液进烧瓶体系。
- weighing：出现天平台面和称量纸/称量舟，动作以加减固体和读数为主。

请认真观看整段视频，输出 JSON：
{{
  "phase": "<最匹配的 phase_id>",
  "phase_zh": "<该阶段中文名>",
  "confidence": <0-1>,
  "alternative_phases": [
    {{"phase": "<第二候选 phase_id>", "phase_zh": "<中文名>", "confidence": <0-1>, "reason": "<一句话>"}},
    {{"phase": "<第三候选 phase_id>", "phase_zh": "<中文名>", "confidence": <0-1>, "reason": "<一句话>"}}
  ],
  "evidence_timestamps": [<支持该判断的秒数, 3-5个>],
  "reason": "<一句话理由，必须引用至少一个视觉区分线索>"
}}
只输出 JSON。"""

OMNI_ACTION_SEGMENTATION = """你在观看一段第一视角化学实验视频（完整视频，总时长约 {duration:.0f} 秒）。
该视频的实验阶段：{phase_id}（{phase_zh}）。

候选动作本体（先优先使用主阶段动作；若观察与其他阶段动作更符合，可选择更贴切的 action_id）：
{action_catalog}

要求：
- 观看完整视频，输出时间上从前到后、不重叠的关键操作片段；
- 【原子粒度规则 — 最重要的硬性约束】每个片段必须且只能包含一个原子实验操作。
  一个原子操作 = 单一实验意图 + 连续动作 + 不更换核心器具。

  禁止将以下情况合并为一个片段（必须拆分为多个片段）：
    ❌ "加入溶剂并用药匙搅拌"        → 拆为 "加入溶剂" + "用药匙搅拌"
    ❌ "倒入液体润洗，随后将润洗液倒出" → 拆为 "倒入液体润洗" + "倒出润洗液"
    ❌ "拿起试剂瓶，将其中的液体倒入漏斗" → 拆为 "拿起试剂瓶" + "倒入液体"
    ❌ "调整管路并进行抽滤"           → 拆为 "调整管路" + "抽滤"
    ❌ "打开容器，用滴管取样"         → 拆为 "打开容器" + "吸取样品"
    ❌ "将液体倒入漏斗进行转移和润洗"    → 拆为 "倒入液体" + "转移/润洗"
    ❌ "注入洗涤液，摇晃后倒入漏斗"       → 拆为 "注入洗涤液" + "摇晃容器" + "倒入漏斗"
    ❌ 同一类动作重复出现多次          → 每次独立操作单独输出一个片段

  判断标准：如果 caption 中出现了"并"、"随后"、"然后"、"同时"、"后"（如"摇晃后"）等连接词，
  说明该片段包含多个动作，必须拆分。特别关注：摇晃/摇匀/振摇是独立的原子操作（本体 action_id:
  swirl_or_mix_container），切勿将摇晃合并到加液或倒液动作中。

- 【全覆盖规则】整个视频时间轴从 0s 到最后一秒必须被连续覆盖，不允许有任何时间空隙。
  相邻片段的 end 必须等于下一个片段的 start（允许 ±0.5s 误差）。
  不重要的操作也必须单独分段，例如：
    "打开开关"、"摇匀试剂瓶"、"将烧杯放到桌面"、"拿起容器"、"放下器具"、
    "调整铁架台"、"连接管路"、"拧紧接口"、"等待液体流下"、"观察状态"等。
  即使画面中实验人员只是短暂离开或仅做准备工作，也要标注为片段（可自拟 action）；
  不得因为"不重要"而跳过任何时间段；
- 每段选择最贴切的 action_id（尽量用本体中的；确实不在本体内可自拟并注明）；
- objects 写该片段中出现的关键实验器具和材料；
- caption 用一句中文客观描述该片段发生了什么；
- evidence_timestamps 给出该片段内最能佐证的 2-4 个秒数。

【易混动作区分线索】
- measure_liquid_for_addition vs add_liquid_to_reaction：
  measure = 将液体从试剂瓶吸入注射器/量筒（吸取阶段）；
  add = 将注射器/量筒中的液体推入/倒入反应容器（注入阶段）。
- rinse_container vs transfer_liquid_to_sep_funnel：
  rinse = 用少量液体润洗容器内壁，随后将液体倒出丢弃；
  transfer = 将液体从一个容器直接倒入另一个容器（目的是转移，不是润洗）。
- transfer_liquid_to_sep_funnel vs add_liquid_to_reaction：
  transfer_to_sep_funnel = 将液体倒入分液漏斗（目标容器是分液漏斗）；
  add_to_reaction = 向反应容器中加液（目标是反应瓶，不是分液漏斗）。
- mix_and_vent_sep_funnel vs add_extraction_solvent：
  mix/vent = 分液漏斗盖好后倒置振摇并开活塞排气；
  add = 从外部试剂瓶向分液漏斗内倒入液体。
- prepare_TLC_plate vs spot_TLC_plate：
  prepare = 用铅笔在 TLC 板上划线、标记位置；
  spot = 用毛细管蘸取样品并将液滴点到 TLC 板指定位置。

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
【输出前必须逐段自检】
对每一段，检查其 caption 中是否包含"并"/"随后"/"然后"/"同时"等连接词。
如果有，该段不是原子操作，请立即拆分。反复检查直到每段都只描述一个动作。
只输出 JSON 数组。"""

# ====================================================================
# Hybrid agent prompts (hxa branch)
# ====================================================================

HYBRID_OBJECT_GROUNDING = """你是实验视频中的器具/材料确认 agent。

候选片段：
- 时间段：{start:.1f}s–{end:.1f}s
- 动作：{action}（{action_zh}）
- 原始 objects：{objects}
- 原始 caption：{caption}
- 抽帧时间戳：{frame_ts}

下面按时间顺序给出该片段内的若干帧。请只根据画面确认实验器具、容器、材料。
重点区分容易混淆的对象：TLC板/白纸/标签纸、毛细管/滴管/注射器针头、移液枪/注射器、
离心管/试管/核磁管、分液漏斗/普通漏斗/梨形瓶、称量纸/滤纸。

输出 JSON：
{{
  "confirmed_objects": ["画面中能确认存在的关键对象"],
  "missing_objects": ["原始 objects 中画面不能确认的对象"],
  "additional_objects": ["原始 objects 漏掉但画面明显可见且与操作相关的对象"],
  "corrected_caption": "若原 caption 中对象名称明显错误，用一句中文改写；否则复述原 caption",
  "confidence": <0-1>,
  "note": "一句话说明依据和不确定性"
}}
只输出 JSON。"""

HYBRID_BOUNDARY_REFINEMENT = """你是实验视频操作边界精修 agent。

候选片段：
- 原始时间段：{start:.1f}s–{end:.1f}s
- 可调整范围：{context_start:.1f}s–{context_end:.1f}s
- 动作：{action}（{action_zh}）
- caption：{caption}
- 抽帧时间戳：{frame_ts}

下面按时间顺序给出可调整范围内的若干帧。请判断核心动作真正开始和结束的大致时间。
要求：
- refined_start/refined_end 必须落在可调整范围内；
- 不要把纯等待、取放前准备、动作完成后的静止画面纳入核心动作；
- 如果这些帧不足以判断边界，可保持原始时间段，但要降低 confidence 并说明原因；
- evidence_timestamps 给出最能支持该边界判断的 2-5 个秒数。

输出 JSON：
{{
  "refined_start": <秒>,
  "refined_end": <秒>,
  "evidence_timestamps": [<秒>, ...],
  "boundary_confidence": <0-1>,
  "reason": "一句话说明边界依据"
}}
只输出 JSON。"""

HYBRID_TEXT_AUDIT = """你是实验视频结构化解析结果的文本审计 agent。

注意：你只能审计以下 clip 观察摘要和结构化片段，不要编造画面中没有写到的内容。

视频总时长约 {duration:.1f}s。

clip 观察摘要：
{clip_summaries}

结构化片段 JSON：
{segments_json}

审计目标：
1) 检查是否存在明显遗漏：clip 摘要中有实质性实验操作，但 segments 没覆盖；
2) 检查动作标签是否明显不贴切或过度具体；
3) 检查 caption 是否加入了摘要/片段不支持的实验意图、结果或物体名称；
4) 检查片段顺序、重叠、长空洞是否需要人工复核；
5) 不做视觉臆测，只根据文字证据给出审计意见。

输出 JSON：
{{
  "overall_quality": "<good|needs_review|poor>",
  "summary": "一句中文总体评价",
  "segment_reviews": [
    {{
      "segment_index": <从0开始的片段序号>,
      "severity": "<info|warning|error>",
      "issue_type": "<label|caption|boundary|object|coverage|consistency>",
      "suggestion": "给人工复核或后处理的短建议"
    }}
  ],
  "possible_missing_operations": [
    {{
      "start": <秒>,
      "end": <秒>,
      "reason": "为什么怀疑有遗漏"
    }}
  ]
}}
只输出 JSON。"""


HAND_RECOGNITION = """你正在分析一张第一视角化学实验画面的关键帧（来自时间点 {timestamp:.0f}s）。

请识别画面中实验人员的双手情况：
- 左手是否可见？如果可见，持握什么器具？
- 右手是否可见？如果可见，持握什么器具？
- 若某只手完全不在画面内或被遮挡到无法判断，标为不可见。
- holding 写该手持握的具体器具名称（如"移液枪""锥形瓶""药匙"）；
  若可见但未持握任何器具，holding 为 null。
- 只根据画面内容判断，不要猜测画面外的情况。

输出 JSON：
{{
  "hands": [
    {{"side": "left",  "visible": true,  "holding": "锥形瓶"}},
    {{"side": "right", "visible": false, "holding": null}}
  ]
}}
只输出 JSON。"""
