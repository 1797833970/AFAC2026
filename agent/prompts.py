"""Prompt templates for the optimized 4-step agent pipeline."""


# ============================================================
# Step 1: Retrieval Planning — 智能文档理解 + 查询拆解 + 多轮策略
# ============================================================

def _retrieval_output_format():
    return """{
  "doc_analysis": [
    {
      "doc_id": "文档ID",
      "relevance": "high|medium|low",
      "reason": "简述该文档与题干的相关性，20字以内"
    }
  ],
  "retrieval_plan": [
    {
      "round": 1,
      "description": "检索目的，用于验证哪个选项/哪部分计算",
      "semantic_query": "自然语言查询，15~30字，保留关键约束",
      "keywords": "BM25关键词，空格分隔，5~8个词，使用原文术语",
      "answer_type": "数值|比例|金额|公式|条款|是/否|日期|列表",
      "doc_ids": ["目标文档ID列表"]
    }
  ],
  "needs_second_round": true|false,
  "second_round_reason": "为什么需要第二轮检索，简述"
}"""

def _retrieval_common_rules():
    return """通用规则：
- 先做文档理解，再做检索规划——从给定文档列表中判断每个文档的相关性
- high: 文档标题直接对应题干中的产品/机构/法规名
- medium: 文档主题相关但不确定具体位置，需要检索确认
- low: 文档主题与题干明显无关，可跳过
- 第一轮检索只查 high + medium 文档，第二轮再决定是否查 low
- 同一检索目标涉及多个文档时，doc_ids放列表，不要拆分
- semantic_query必须显式包含题干约束（产品名、日期、计划类型等），但**禁止把选项中的具体数值写入检索词**（如选项说"上限10亿元"，检索词只写"发行规模上限"，不写"10亿元"）
- keywords使用文档原文专业术语，优先条款编号/字段名，**禁止包含选项中的具体数值**（避免错误数值污染检索）
- 否定选项必须检索：选项声称"不包含/不存在/未给出/未提及"时，必须创建检索目标验证
- 检索为空的否定选项：压缩阶段会自动标记NEGATIVE_VERIFIED，回答阶段据此判断"""

RETRIEVAL_CALC = f"""你是一位保险精算与金融计算检索专家。为计算题设计检索策略，找到计算公式、赔付规则、参数数值。

## 工作流程
1. **文档理解**：先阅读所有可用文档标题，判断每个文档与本题的相关性
2. **查询拆解**：每个保险产品/计算对象独立一个检索目标，找其赔付公式、免赔额规则、赔付比例
3. **多轮策略**：第一轮查核心产品参数，第二轮补充验证边界条件

## 输出格式
{_retrieval_output_format()}

## 特有规则
- semantic_query必须包含：产品名、免赔额数值、计算场景（如"医保报销后""自费部分"）
- keywords除产品名外，补充：赔付计算公式、免赔额、赔付比例、保险金计算

{_retrieval_common_rules()}
"""

RETRIEVAL_MCQ = f"""你是一位金融文档检索策略专家。为单选题设计检索策略。

## 工作流程
1. **文档理解**：先阅读所有可用文档标题，判断每个文档与本题的相关性
2. **查询拆解**：每个选项都是独立的判断题，各建独立检索目标；同一知识点跨文档的合并doc_ids
3. **多轮策略**：第一轮查所有选项的核心证据，第二轮查缺补漏

## 输出格式
{_retrieval_output_format()}

## 特有规则
- semantic_query聚焦选项核心主张
- 四个选项都要覆盖，不能漏项
- 每个选项至少一个检索目标，不要因为是单选就只查你觉得对的那个

{_retrieval_common_rules()}"""

RETRIEVAL_MULTI = f"""你是一位金融文档检索策略专家。多选题的每个选项都是一个独立的子判断题，必须逐个验证选项。

## 工作流程
1. **文档理解**：先阅读所有可用文档标题，判断每个文档与本题的相关性
2. **查询拆解**：每个选项至少一个检索目标，把每个选项当作独立判断题
3. **多轮策略**：第一轮查所有选项的核心证据，第二轮查缺补漏

## 输出格式
{_retrieval_output_format()}

## 核心要求
- **每个选项至少一个检索目标，不能合并
- 否定声称也须检索：选项说"未给出公式""不包含某内容"时，必须创建检索目标
- 同一选项涉及多份文档时合并doc_ids

{_retrieval_common_rules()}
"""

RETRIEVAL_TF = f"""你是一位金融文档检索策略专家。为判断题设计检索策略，核心是验证题干中每个子命题是否被文档证实。

## 工作流程
1. **文档理解**：先阅读所有可用文档标题，判断每个文档与本题的相关性
2. **查询拆解**：题干包含多个子命题时，每个独立命题各建一个检索目标
3. **多轮策略**：第一轮查核心命题，第二轮补查边界命题

## 输出格式
{_retrieval_output_format()}

## 特有规则
- 无法确定命题对应哪个文档时，标记medium相关性的文档全查

{_retrieval_common_rules()}
"""


def get_retrieval_system(question_data: dict) -> str:
    """根据题目类型路由到对应的检索策略 system prompt。计算题优先级最高。"""
    qtype = question_data.get('type', '')
    af = question_data.get('answer_format', 'mcq')

    if '计算' in qtype:
        return RETRIEVAL_CALC
    elif af == 'mcq':
        return RETRIEVAL_MCQ
    elif af == 'multi':
        return RETRIEVAL_MULTI
    elif af == 'tf':
        return RETRIEVAL_TF
    return RETRIEVAL_MCQ


# ============================================================
# Step 2: Compress — extract relevant info based on description
# ============================================================

COMPRESS_SYSTEM = """你是一位文档压缩专家。从检索到的原文段落中，删除无关内容，保留相关信息。

## 输入
每条记录包含：description（检索目标）、semantic_query（查询语句）、chunk_ids、text（原文段落，多个用 --- 分隔）

## 压缩原则
- **原文优先**：compressed_text 保留原文措辞，保留核心原文内容数据等，不做意译、不做总结。只删除明显无关的段落，
- **三个不删**：不删数字和单位、不删限定词（除/不含/仅/非）、不删数值范围

## 输出格式（严格JSON数组）
[
  {"description":"获取本期发行金额上限","chunk_ids":["text01_17"],"semantic_query":"本期债券发行规模上限","compressed_text":"[广晟控股] 本次债券注册金额不超过180亿元（含180亿元）"}
]"""


# ============================================================
# Step 4: Answer — 按题型路由不同提示词
# ============================================================

def _answer_output_format():
    return """{
  "correct_options": ["A"],
  "evidence_retrieval": [
    {
      "chunk_ids": ["text01_17","text01_18"],
      "quoted_clause": "原文精确引用，保留条款编号和数据",
      "reasoning": "支持/反对选项X，逻辑说明"
    }
  ]
}"""

_COMMON_RULES = (
    "## 通用规则\n"
    "- 仅输出纯JSON，无额外文本、无markdown包裹。\n"
    "- 仅基于所给文档内容，不编造。\n"
    "- chunk_ids 照抄原文段落编号。\n"
    "- quoted_clause 为原文摘录，保留条款编号及数据。\n"
    "- reasoning 明确针对选项（支持A/反对B/无法判断C），并说明逻辑。\n"
    "- 选项验证以文档原文为准，忽略选项括号内的辅助描述。\n"
    "- 每个选项独立输出一条evidence，其chunk_ids仅包含该选项相关段落，不跨选项。\n"
    "\n"
    "关键提醒：不要用题干问题去筛选选项！\n"
    "题干提供话题背景，正确选项是那些与题干陈述本身相关的选项。\n"
    "\n"
    "## 数值/计算敏感规则（重要）\n"
    "凡涉及数值比较（大于/小于/不低于/不超过/高于/低于/至少/至多/超过）、范围判断、求和、比例、金额量级（元/万元/亿元）、百分比换算时：\n"
    "- 严禁心算！必须在第一轮输出的 code 字段写 Python 代码由框架执行，再据执行结果回答。\n"
    "- code 要求：从文档原文逐个提取数值赋给变量（注释来源 chunk_id），统一单位（如全部换算成元或百分比小数），输出最终结果赋给变量 result（单个值）或 results（字典）。\n"
    "- 仅允许基础算术（+-*/**//%）、比较运算、abs/round/min/max/sum/len/sorted/range/int/float；禁用 import/open/eval/exec/网络/文件。\n"
    "- 单位换算务必显式（1亿=1e8，1万=1e4，1%=0.01），避免量级错误。\n"
    "- 若第一轮框架返回的 calc_result 与你的预期不符，以 calc_result 为准修正 correct_options。\n"
    "\n"
    "## 否定命题判断框架（重要）\n"
    "当选项声称【文档不包含XX】【未提及XX】【没有XX】【不存在XX】时，按以下步骤判断：\n"
    "1. 先看检索到的信息中，是否有直接证据证明XX存在 → 有则选项陈述错误 → 不选\n"
    "2. 若无直接证据，检查是否存在【NEGATIVE_VERIFIED】标记 → 有标记则视为已穷尽检索未找到，选项陈述正确 → 选\n"
    "3. 最终判断原则：\n"
    "   - 有正面证据反驳 → 选项错（高置信度）→ 不选\n"
    "   - 有NEGATIVE_VERIFIED标记或完整清单不含 → 选项对（中高置信度）→ 选\n"
)

ANSWER_CALC = (
    "你是一位保险精算与金融分析专家。根据所给段落，完成计算题。\n\n"
    "## 输出格式\n"
    + _answer_output_format() + "\n\n"
    + _COMMON_RULES + "\n"
    + "## 附加规则\n"
    "- correct_options 为正确选项列表，如[\"A\"]\n"
    "- 计算公式必须来自文档，不能自己造。\n"
    "- 多产品/情景需逐一计算后汇总对比。\n"
)

ANSWER_MCQ = (
    "你是一位金融文档审计专家。根据所给段落，选出唯一正确选项。\n\n"
    "## 输出格式\n"
    + _answer_output_format() + "\n\n"
    + _COMMON_RULES + "\n"
    + "## 附加规则\n"
    "- correct_options 为唯一正确选项，如[\"A\"]。\n"
)

ANSWER_MULTI = (
    "你是一位金融文档审计专家。根据所给段落，选出所有正确选项。\n\n"
    "## 输出格式\n"
    + _answer_output_format() + "\n\n"
    + _COMMON_RULES + "\n"
    + "## 附加规则\n"
    "- correct_options 为所有正确选项列表，不需要隔开，如[\"AC\"]\n"
)

ANSWER_TF = (
    "你是一位金融文档审计专家。根据所给段落，判断题干陈述是否全部成立。\n"
    "A=全部子命题成立，B=任一不成立。\n\n"
    "## 输出格式\n"
    + _answer_output_format() + "\n\n"
    + "## 附加规则\n"
    "- correct_options 为[\"A\"]或[\"B\"]。\n"
    "- 逐一验证每个子命题，全部成立则A，任一不成立则B。\n"
    "- 每个子命题独立输出一条evidence，其chunk_ids仅限该命题相关段落。"
)


def get_answer_system(question_data: dict) -> str:
    """根据题目类型路由到对应的 system prompt。"""
    qtype = question_data.get('type', '')
    af = question_data.get('answer_format', 'mcq')

    if '计算' in qtype:
        return ANSWER_CALC
    elif af == 'mcq':
        return ANSWER_MCQ
    elif af == 'multi':
        return ANSWER_MULTI
    elif af == 'tf':
        return ANSWER_TF
    return ANSWER_MCQ


def get_answer_instruction(question_data: dict) -> str:
    """根据题目类型生成用户 prompt 中的答题要求。"""
    qtype = question_data.get('type', '')
    af = question_data.get('answer_format', 'mcq')

    if '计算' in qtype:
        return (
            "这是一道**计算题**。请：\n"
            "1. 从题干中提取所有计算参数\n"
            "2. 从文档段落中找到对应的计算公式/条款\n"
            "3. 逐步计算每个产品/情景的结果\n"
            "4. 将计算结果与选项对比，选出匹配的选项\n"
        )
    elif af == 'mcq':
        return "这是一道**单选题**。请逐一分析每个选项，排除错误的，选出唯一正确的选项。\n"
    elif af == 'multi':
        return "这是一道**多选题**。请逐一判断每个选项是否正确，列出所有正确的选项（可能有多个）。\n"
    elif af == 'tf':
        return (
            "这是一道**判断题**。请逐一验证题干中的每个子命题是否都有文档证据支持。\n"
            "全部成立 → 答案A（True），任一不成立 → 答案B（False）。\n"
        )
    return ""


def _load_doc_registry() -> dict:
    """加载文档标题字典（新结构：按 category 分组，展开为 doc_id → info 的扁平映射）。"""
    import json
    from pathlib import Path
    reg_path = Path("data/doc_registry.json")
    if not reg_path.exists():
        return {}
    with open(reg_path, encoding="utf-8") as f:
        structured = json.load(f)
    flat = {}
    for cat, docs in structured.items():
        for doc_id, info in docs.items():
            flat[doc_id] = {**info, "category": cat}
    return flat


def build_unified_prompt(question_data: dict) -> str:
    """构建统一检索规划阶段的用户prompt——先文档理解，再检索规划。"""
    qid = question_data.get('qid', '')
    question = question_data.get('question', '')
    options = question_data.get('options', {})
    af = question_data.get('answer_format', '')
    doc_ids = question_data.get('doc_ids', [])
    qtype = question_data.get('type', '')

    prompt = f"题目ID: {qid}\n"
    prompt += f"答案格式: {af}\n"
    if qtype:
        prompt += f"题目类型: {qtype}\n"
    prompt += f"\n题干：\n{question}\n\n"

    # 文档列表——带完整标题，引导LLM做文档理解
    registry = _load_doc_registry()
    prompt += "## 可用文档列表（请先判断每个文档的相关性）\n\n"
    for i, did in enumerate(doc_ids, 1):
        title = registry.get(did, {}).get("title", "（未知文档）")
        category = registry.get(did, {}).get("category", "")
        cat_label = f"[{category}] " if category else ""
        prompt += f"  {i}. doc_id={did} {cat_label}{title}\n"
    prompt += "\n"

    prompt += "选项：\n"
    for key in sorted(options.keys()):
        prompt += f"{key}: {options[key]}\n"

    prompt += "\n## 任务\n\n"
    prompt += "1. 先阅读上方文档列表，判断每个文档与本题的相关性（high/medium/low）\n"
    prompt += "2. 然后设计检索计划，每个检索目标指定doc_ids列表\n"
    prompt += "3. 考虑是否需要第二轮检索补充\n\n"
    prompt += "请按输出格式返回完整JSON。"
    return prompt


# ============================================================
# Compress Step: LLM-based relevance filtering
# ============================================================

def build_compress_prompt(all_chunks: list[dict]) -> str:
    prompt = f"## 待压缩段落（共{len(all_chunks)}条）\n\n"
    for i, chunk in enumerate(all_chunks):
        cids = chunk.get('chunk_ids', ['?'])
        cids_str = ', '.join(cids) if isinstance(cids, list) else str(cids)
        prompt += f"[{i}] chunk_ids: {cids_str}\n"
        prompt += f"    description: {chunk.get('description', '?')}\n"
        prompt += f"    semantic_query: {chunk.get('semantic_query', '?')}\n"
        prompt += f"    text: {chunk.get('text', '')}\n\n"
    prompt += "\n请删除无关内容，保留原文措辞。compressed_text 以[产品名]开头。\n"
    return prompt


# ============================================================
# Answer Step: Build holistic prompt with full context
# ============================================================

def build_answer_prompt(question_data: dict, compressed: list[dict]) -> str:
    """构建Answer步骤的用户prompt，包含完整题干+全部选项+全部证据。"""
    question = question_data.get('question', '')
    options = question_data.get('options', {})
    doc_ids = question_data.get('doc_ids', [])

    prompt = f"## 题目\n\n{question}\n\n"

    # 文档映射（精简为一行）
    registry = _load_doc_registry()
    if doc_ids:
        prompt += "## 文档对照\n"
        doc_list = " | ".join(f"{did}: {registry.get(did, {}).get('title', '?')}" for did in doc_ids)
        prompt += f"{doc_list}\n\n"

    prompt += "## 选项\n\n"
    for key in sorted(options.keys()):
        prompt += f"{key}: {options[key]}\n"

    # Group compressed results by description
    from collections import defaultdict
    groups = defaultdict(list)
    for item in compressed:
        groups[item.get('description', '')].append(item)

    prompt += "\n## 检索到的信息\n\n"
    if not groups:
        prompt += "（未检索到相关信息）\n"
    else:
        for desc, items in groups.items():
            prompt += f"【{desc}】\n"
            for item in items:
                ct = item.get('compressed_text', '')
                cids = item.get('chunk_ids', [])
                cids_str = ', '.join(cids) if isinstance(cids, list) else str(cids)
                if ct:
                    prompt += f"  [{cids_str}] {ct}\n"
            prompt += "\n"

    # Type-specific answering instruction
    prompt += "\n## 答题要求\n\n"
    prompt += get_answer_instruction(question_data)

    # 数字/计算敏感 + 审题规则（替换原来的4条推理约束）
    prompt += "\n## 关键规则\n"
    prompt += "- 计算题：先列出文档中的公式（逐字引用），再代入数值，标注每个数值来源\n"
    prompt += "- 涉及比例/金额/日期：核对单位和量级（元/万元/亿元），确认是否已扣减免赔额\n"
    prompt += "- 不跨产品归因：A产品的条款不能用于支持B产品的结论\n"
    prompt += "- 选项声称'未提及'：找到→选项错，找不到→选项对\n"

    prompt += "\n请基于以上文档段落，给出你的判断。输出JSON。\n"

    return prompt


# ============================================================
# Answer Step (Round 1): Code generation for numeric questions
# ============================================================

def _codegen_output_format():
    return """{
  "need_calc": true,
  "code": "python代码字符串，无需计算时为空字符串",
  "extracted_values": [
    {"var": "变量名", "value": 数值, "from": "来源chunk_id", "note": "单位说明"}
  ],
  "expected": "本代码要算什么/比较什么，一句话"
}"""

def _codegen_common_rules():
    return """通用规则：
- 先判断本题是否需要数值计算/比较（need_calc），无需计算时 need_calc=false 且 code 为空字符串
- 涉及数值比较、范围判断、求和、比例、金额量级、百分比换算时 need_calc=true
- code 必须是可直接 exec 的纯 Python，禁用 import/open/eval/exec/网络/文件
- 仅允许：算术运算、比较运算、abs/round/min/max/sum/len/sorted/range/int/float/list/dict
- 从文档原文逐个提取数值赋给变量，注释来源 chunk_id，单位显式换算（1亿=1e8, 1万=1e4, 1%=0.01）
- 最终结果赋给变量 result（单个值）或 results（dict）
- 不要在 code 里写 print，框架会自动捕获 result/results 变量
- 选项比较题：results 用 {"选项字母": 计算值} 形式，便于框架对比
- 仅输出纯JSON，无markdown包裹，无额外文本"""

ANSWER_CALC_CODEGEN = (
    "你是一位保险精算与金融计算代码生成专家。这是计算题的第一轮：请生成 Python 代码由框架执行。\n\n"
    "## 输出格式\n"
    + _codegen_output_format() + "\n\n"
    + _codegen_common_rules() + "\n\n"
    "## 计算题特有要求\n"
    "- code 必须覆盖所有产品/情景的计算分支，最终汇总到 result 或 results\n"
    "- 公式来源在 extracted_values 的 note 中标注 chunk_id\n"
    "- 免赔额、赔付比例、累计限额等关键参数必须显式赋值，不能省略\n"
)

ANSWER_MCQ_CODEGEN = (
    "你是一位金融文档审计代码生成专家。这是单选/多选题的第一轮：判断选项是否涉及数值比较，若是则生成 Python 代码由框架执行。\n\n"
    "## 输出格式\n"
    + _codegen_output_format() + "\n\n"
    + _codegen_common_rules() + "\n\n"
    "## 选项题特有要求\n"
    "- 选项中含数值比较（大于/不超过/至少/介于等）时 need_calc=true\n"
    "- 选项只是定性陈述、不涉及数值比较时 need_calc=false\n"
    "- 多个选项都含数值比较时，在 results 中用 {\"A\": ..., \"B\": ...} 形式分别计算\n"
    "- 题干给出基准值（如某产品金额），选项声称与之比较时，把基准值和各选项值都算出来\n"
)

ANSWER_MULTI_CODEGEN = ANSWER_MCQ_CODEGEN  # 多选题与单选题代码生成规则一致


def get_answer_codegen_system(question_data: dict) -> str:
    """第一轮代码生成的 system prompt 路由。"""
    qtype = question_data.get('type', '')
    af = question_data.get('answer_format', 'mcq')

    if '计算' in qtype:
        return ANSWER_CALC_CODEGEN
    elif af == 'mcq':
        return ANSWER_MCQ_CODEGEN
    elif af == 'multi':
        return ANSWER_MULTI_CODEGEN
    return ANSWER_MCQ_CODEGEN


def build_answer_codegen_prompt(question_data: dict, compressed: list[dict]) -> str:
    """第一轮代码生成的 user prompt。复用 build_answer_prompt 的题干/选项/证据上下文。"""
    # 复用原 prompt 主体（题干、文档、选项、检索信息）
    prompt = build_answer_prompt(question_data, compressed)
    prompt += "\n## 本轮任务（第一轮·代码生成）\n"
    prompt += "请判断本题是否涉及数值计算/比较，并按输出格式返回 JSON。\n"
    prompt += "- 若需要计算：code 字段输出可执行的 Python 代码，最终结果赋给 result 或 results\n"
    prompt += "- 若无需计算（纯定性判断）：need_calc=false，code 为空字符串\n"
    return prompt


def build_answer_prompt_with_calc(question_data: dict, compressed: list[dict],
                                  calc_result: dict) -> str:
    """第二轮 user prompt：在原 prompt 基础上追加代码执行结果。"""
    prompt = build_answer_prompt(question_data, compressed)

    prompt += "\n## 代码执行结果（第一轮已由框架执行）\n\n"
    if calc_result.get('ok'):
        result_val = calc_result.get('result')
        results_val = calc_result.get('results')
        if results_val is not None:
            prompt += f"results = {results_val!r}\n"
        elif result_val is not None:
            prompt += f"result = {result_val!r}\n"
        else:
            prompt += "（代码执行成功但未找到 result/results 变量）\n"
        if calc_result.get('stdout'):
            prompt += f"stdout:\n{calc_result['stdout']}\n"
    else:
        prompt += f"执行失败：{calc_result.get('error', '未知错误')}\n"
        prompt += "请基于文档原文自行谨慎判断，并在 reasoning 中标注'代码执行失败，已手动核算'。\n"

    prompt += "\n## 本轮任务（第二轮·最终作答）\n"
    prompt += "基于上述代码执行结果给出最终答案。若执行结果与选项陈述矛盾，以执行结果为准。\n"
    prompt += "输出JSON（按原输出格式，含 correct_options 和 evidence_retrieval）。\n"
    return prompt
