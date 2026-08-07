# 05 · 问答流程（五阶段）

对每道题，Agent 按**五阶段**完成作答：`题目类型分析 → 锁定检索文档 → BM25 检索信息 → 信息压缩 → 答题`。代码上对应 `agent/` 下四个模块（阶段一、二都在分析模块内），编排入口是 `agent/qa_agent.py`。

## 5.0 阶段与代码映射

| 阶段 | 代码模块 | 职责 |
|------|----------|------|
| ① 分析题目类型 | `analyze_step.py` + `prompts.py` 路由 | 按题型选择检索与作答策略 |
| ② 锁定检索文档 | `analyze_step.py`（doc_analysis + retrieval_plan） | 文档相关性分级、检索目标拆解 |
| ③ BM25 检索信息 | `retrieve_step.py` | 按计划在目标文档内召回 Top-K |
| ④ 信息压缩 | `compress_step.py` | LLM 把原始片段压缩成证据 |
| ⑤ 答题 | `answer_step.py` | 计算题走代码执行，其余单轮作答 |

```text
题目 ──► ① 类型分析 ──► ② 锁定文档 ──► ③ BM25 检索 ──► ④ 证据压缩 ──► ⑤ 作答 ──► 答案+证据
```

## 5.1 阶段一：题目类型分析

第一步不是直接检索，而是**根据题型选择策略**。题目字段 `type`（单选/多选/判断/计算/抽取）与 `answer_format`（mcq/multi/tf）共同决定走哪套 Prompt：

| 题型 | 检索策略（`get_retrieval_system`） | 作答策略（`get_answer_system`） |
|------|----------------------------------|-------------------------------|
| 计算题（type 含"计算"） | `RETRIEVAL_CALC`（精算检索专家） | `ANSWER_CALC` + 代码生成 |
| 单选题（mcq） | `RETRIEVAL_MCQ`（逐选项判断题） | `ANSWER_MCQ` |
| 多选题（multi） | `RETRIEVAL_MULTI`（逐选项验证） | `ANSWER_MULTI` |
| 判断题（tf） | `RETRIEVAL_TF`（逐子命题验证） | `ANSWER_TF` |

**计算题优先级最高**：只要 type 含"计算"，一律先按计算策略走。类型分析的作用是把后续所有阶段约束到正确的策略上——选择题要"逐选项验证"，判断题要"逐子命题验证"，计算题要"先代码后结论"。

## 5.2 阶段二：锁定检索文档（检索规划）

`analyze_step.Analyzer.analyze()` 调用 LLM，输入由 `build_unified_prompt` 构造，包含：题目 ID、答案格式、题目类型、题干、**带标题的可用文档列表**（标题来自 `data/doc_registry.json`）、选项、任务说明。

LLM 输出一份 JSON 检索计划，核心是两个部分：

### ① 文档相关性分析（doc_analysis）

```json
{
  "doc_id": "insurance_1",
  "relevance": "high",
  "reason": "产品名直接对应"
}
```

`high` = 标题直接对应题干产品/机构/法规；`medium` = 主题相关但位置不确定；`low` = 明显无关可跳过。**第一轮只查 high + medium，第二轮才补 low**——这就是"锁定检索文档"。

### ② 检索计划（retrieval_plan）

```json
{
  "round": 1,
  "description": "验证A选项的身故给付比例",
  "semantic_query": "40岁身故保险金给付比例",
  "keywords": "身故保险金 给付比例 40岁",
  "answer_type": "条款",
  "doc_ids": ["insurance_1"]
}
```

每条检索目标的字段：

| 字段 | 作用 |
|------|------|
| `round` | 第几轮检索（第一轮核心，第二轮补漏） |
| `description` | 检索目的，同时是压缩/作答阶段的分组依据 |
| `semantic_query` | 自然语言查询，15~30 字，用于兜底召回 |
| `keywords` | BM25 关键词，空格分隔，5~8 个原文术语 |
| `answer_type` | 数值/比例/金额/公式/条款/是-否/日期/列表 |
| `doc_ids` | 目标文档 ID 列表（可跨文档合并） |

Prompt 中的核心约束（`_retrieval_common_rules`）：

- **每个选项独立成检索目标**，单选也查所有选项，防止"只查觉得对的那个"漏掉反例；
- **关键词禁用选项数值**：选项写"上限10亿元"，检索词只写"发行规模上限"，避免错误数值污染召回；
- **否定选项必须检索**：选项声称"不包含/未提及"时单独建目标，检索不到由压缩阶段打否定标记；
- **多轮策略**：`needs_second_round` + `second_round_reason` 决定是否补第二轮。

### 输出校验与兜底

LLM 输出先剥掉 markdown 围栏和多余文本，尝试直接解析 JSON；失败则提取第一个 `{...}` 块；再失败或字段不完整（缺 `semantic_query`、缺 `doc_ids/doc_id`）时，走 `_fallback` 兜底：整题作为查询，`keywords` 由 `jieba.cut_for_search` 自动切词，所有候选文档全查。

最后 `_flatten` 把计划展平为检索目标列表（按 round 排序，兼容旧的 `doc_id` 字符串格式）。

## 5.3 阶段三：BM25 检索信息

`retrieve_step.Retriever.retrieve()` 逐条执行检索目标：

```text
检索目标（keywords + doc_ids）
        │
        ▼
keywords 中的 "|" 备用同义词展开为空格 → bm25_query
        │
        ▼
多文档：逐文档 BM25 Top-3，chunk_id 去重合并
        │
        ├── 命中 ──► 返回
        └── 未命中 ──► 用 semantic_query 扩大召回 Top-5，补入前 2 条
```

细节：

- **Top-3 优先**：每个检索目标默认每文档召回 3 条（`BM25_TOP_K`，配置见 `config/config.yaml`）；
- **兜底补 2 条**：首轮全未命中时，改用 `semantic_query` 召回 5 条，按 `chunk_id` 去重补入最多 2 条（`BM25_FALLBACK_K`）；
- **逐文档检索 + 去重**：多文档目标逐个文档搜（支持前缀匹配 doc_id），再按 `chunk_id` 去重，保持返回顺序稳定；
- **容错**：单个目标检索异常只记 WARN，不影响其他目标。

底层检索器是增强 BM25（`retrieval/bm25_retriever.py`），三重打分：jieba 细粒度分词的 BM25 基础分 + 字符 Bigram 重叠分 + 精确子串加成，详见 [04-检索系统](04-检索系统.md)。

## 5.4 阶段四：信息压缩

`compress_step.Compressor.compress()` 把检索回来的原始 chunk 变成"只留证据"的精简文本：

1. **按目标分组**：以 `description + semantic_query` 为键合并同一检索目标的多个 chunk；
2. **LLM 一次压缩一组**：输入含 `description`、`semantic_query`、`chunk_ids`、原始段落（多个用 `---` 分隔）；
3. 输出严格 JSON 数组，每项 `{description, chunk_ids, semantic_query, compressed_text}`。

压缩系统提示（`COMPRESS_SYSTEM`）的核心原则：

- **原文优先**：保留原文措辞与数据，不做意译、不做总结，只删除明显无关段落；
- **三个不删**：不删数字和单位、不删限定词（除/不含/仅/非）、不删数值范围；
- `compressed_text` 以 `[产品名]` 开头，方便作答阶段对齐到具体产品。

### 否定验证标记

若某个检索目标**一条都没召回**（例如验证"选项声称文档未提及 XX"），压缩阶段自动生成：

```text
【NEGATIVE_VERIFIED】已在文档 [...] 中用关键词 [query] 检索，未找到相关内容。
```

这个标记是作答阶段判断"否定命题是否成立"的关键证据：有标记 → 视为已穷尽检索仍未找到 → 选项的否定陈述成立。

## 5.5 阶段五：答题

`answer_step.Answerer.answer()` 先判断是否走"两轮代码执行"流程：

```text
_needs_calc 预判：
  type 含"计算"            → 必走代码流程
  mcq/multi 且题干+选项+证据
  同时含数字和比较词         → 走代码流程
  其余（含 tf）             → 单轮作答
```

### 5.5.1 计算题：两轮"代码生成 → 受限执行 → 作答"

**第一轮（代码生成）**：LLM 基于压缩证据输出：

```json
{
  "need_calc": true,
  "code": "python代码字符串",
  "extracted_values": [
    {"var": "deductible", "value": 10000, "from": "ins_1_5", "note": "免赔额1万元"}
  ],
  "expected": "计算两个产品的赔付差额"
}
```

Prompt 约束（`_codegen_common_rules`）：从原文逐个提取数值赋给变量并注释来源 chunk_id；**单位显式换算**（1亿=1e8、1万=1e4、1%=0.01）；最终结果赋给 `result`（单值）或 `results`（字典，选项题用 `{"A": ..., "B": ...}`）；禁用 import/网络/文件。

**受限执行**（`_run_calc_code`）：

- 正则拦截危险关键字：`import`、`open`、`eval`、`exec`、`__import__`、`os.`、`sys.`、`subprocess`、`globals`、`locals` 等；
- 白名单内建函数：`abs/round/min/max/sum/len/sorted/range/int/float/str/bool/list/dict/tuple/set/enumerate/zip/map/filter/any/all/repr`；
- `print` 重定向到内存缓冲，不污染控制台；
- 线程执行 + 5 秒超时保护（Windows 兼容），超时或异常记录错误信息。

**第二轮（最终作答）**：把执行结果（`result` / `results` / `stdout`）追加进 prompt，让 LLM 基于**真实计算结果**输出答案；执行失败时要求 LLM"基于文档原文自行谨慎判断"，并在 reasoning 中标注"代码执行失败，已手动核算"。

### 5.5.2 选择题 / 判断题：单轮作答

LLM 直接基于压缩证据输出 JSON：

```json
{
  "correct_options": ["A"],
  "evidence_retrieval": [
    {
      "chunk_ids": ["ins_1_5", "ins_1_6"],
      "quoted_clause": "原文精确引用，保留条款编号和数据",
      "reasoning": "支持/反对选项X，逻辑说明"
    }
  ]
}
```

通用规则（`_COMMON_RULES`）里的几个关键点：

- **数值/计算敏感规则**：涉及数值比较、比例、金额量级时严禁心算，必须走第一轮代码；
- **否定命题判断框架**：选项声称"文档不含 XX"时，有正面证据反驳 → 选项错；有 `NEGATIVE_VERIFIED` 标记或完整清单不含 → 选项对；
- **逐选项证据**：每个选项独立一条 evidence，`chunk_ids` 不跨选项；
- **不跨产品归因**：A 产品的条款不能用于支持 B 产品的结论（写在用户 prompt 中）；
- **审题规则**：不要用题干去筛选选项，正确选项与题干陈述本身相关。

### 5.5.3 输出解析与格式化

`_parse_response` 按顺序：去 markdown 代码块围栏 → 去 `<thinking>` 标签 → 直接 JSON 解析 → 提取第一个 `{...}` 块 → 全部失败返回空答案并告警。

`_format_answer` 按题型格式化：

```python
mcq   → 取第一个选项字母
multi → 去重并按 A-D 排序后拼接（"ABD" 而非 "BAD"）
tf    → "A"=正确 / "B"=错误
```

## 5.6 Token 记账与中间产物

- `agent/llm_utils.py` 统一封装 DashScope 调用，返回 `(content, usage)`，usage 含 input/output tokens；
- `qa_agent.answer_single()` 汇总三步（分析/压缩/作答）token，写入结果的 `tokens` 字段；
- `self.trace` 保留全链路中间结果（检索计划、检索命中、压缩文本、代码生成原文、代码执行结果），供调试与复盘；
- 批量运行时（`scripts/run_agent.py`）每题答案、证据、日志单独落盘到 `output/evidence/`、`output/logs/`。

## 5.7 并发与稳定性

- `run_agent.py` 用线程池并发（默认 5 线程），**每个线程持有独立的 QAAgent**（BM25 索引在线程内加载一次），避免共享可变状态；
- 单题失败不中断整体，进度实时写入 `run_log.txt`；
- 每题产物独立落盘，便于逐题复盘与断点续跑。
