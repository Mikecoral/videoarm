"""Prompt templates for the LabARM-HV pipeline (kept in one place)."""

SCENE_SNAPPER = """你在分析一段第一视角化学实验视频的一个时间窗口（{start:.0f}s–{end:.0f}s）。
下面按时间顺序给出该窗口内的若干帧。请用中文简洁描述：
1) 画面中出现的实验器具/容器/材料；
2) 实验人员双手正在做的具体操作；
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
