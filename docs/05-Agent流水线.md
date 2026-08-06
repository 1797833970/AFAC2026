# 05 · Agent 四步流水线

`agent/` 目录实现了核心的 QA Agent。对每一道题，流水线分四步执行，全部代码在 `agent/qa_agent.py` 中编排：

```text
题目 ──► Step1 分析（检索规划）──► Step2 检索（BM25）
             │                          │
             ▼                          ▼
        Step4 作答 ◄── Step3 证据压缩 ◄─┘
```

## 5.1 Step 1 · 分析：检索规划

`agent/analyze_step.py` —— 让 LLM 先"读懂题"，再决定查什么。

输入：题目文本 + 选项 + 候选文档标题。

输出一份 JSON 检索计划：

```json
{
  "doc_analysis": [
    {"doc_id": "insurance_1", "relevance": "high", "reason": "产品名直接对应"}
  ],
  "retrieval_plan": [
    {
      "round": 1,
      "description": "验证A选项的身故给付比例",
      "semantic_query": "40岁身故保险金给付比例",
      "keywords": "身故保险金 给付比例 40岁",
      "answer_type": "条款",
      "doc_ids": ["insurance_1"]
    }
  ],
  "needs_second_round": false
}
```

关键设计：

- **每个选项独立成检索目标**：单选题也要查所有选项，防止"只查觉得对的那个"漏掉反例。
- **关键词禁用选项数值**：选项说"上限10亿元"，检索词只写"发行规模上限"，避免错误数值污染召回。
- **否定选项必须验证**：选项声称"不包含/未提及"时，单独建检索目标；查不到会在 Step 3 打上否定标记。
- **多轮策略**：第一轮查 high+medium 文档，必要时第二轮补 low 文档。

LLM 输出不是合法 JSON 时，自动回退到"整题作为查询"的兜底计划。

## 5.2 Step 2 · 检索：BM25 召回

`agent/retrieve_step.py` —— 按计划执行检索。

- 每个检索目标在 `doc_ids` 范围内做 BM25，默认 **Top-3**；
- 首轮未命中时，用 `semantic_query` 扩大召回（Top-5）并补入前 2 条；
- 多文档目标逐文档检索后按 `chunk_id` 去重合并；
- 检索异常不中断，记警告后继续。

```text
检索目标（description + keywords + doc_ids）
        │
        ▼
BM25 Top-3 ── 命中? ── 是 ──► 结果
        │ 否
        ▼
扩大召回 Top-5，补入前 2 条 ──► 结果
```

## 5.3 Step 3 · 压缩：只留证据

`agent/compress_step.py` —— 检索回来的原始 chunk 往往很长，让 LLM 按检索目标压缩：

```text
输入：检索目标描述 + 查询 + 原始片段
输出：压缩后的"证据文本"（删除无关内容，保留与查询相关的句子和数字）
```

两个细节：

1. 同一检索目标的多个 chunk 合并后一次压缩，减少调用次数；
2. 检索为空的否定目标，自动生成 `【NEGATIVE_VERIFIED】已在文档中用关键词[...]检索，未找到相关内容`，供作答阶段判断"选项声称不存在的说法是否成立"。

## 5.4 Step 4 · 作答：综合判断

`agent/answer_step.py` —— 基于压缩证据输出最终答案。

### 计算题：两轮"写代码 → 执行 → 作答"

检测到题干含数值比较词（大于/不超过/同比/占比等）或题型为计算题时：

```text
第一轮：LLM 基于证据生成 Python 计算代码
   │
   ▼
沙箱执行（捕获 stdout/异常，限时）
   │
   ▼
第二轮：LLM 看到代码执行结果，输出最终答案
```

让"算数"发生在代码里而不是模型脑子里，显著降低计算错误。

### 选择题：单轮作答

LLM 直接输出 JSON：`{"correct_options": ["A"], "evidence_retrieval": [...]}`。解析器会：

1. 去掉 markdown 代码块围栏；
2. 去掉 `<thinking>` 标签；
3. 直接解析 JSON，失败则提取第一个 `{...}` 块；
4. 全部失败返回空答案并告警。

### 答案格式化

```python
mcq   → 取第一个选项字母
multi → 去重并按 A-D 排序后拼接（ABD 而不是 BAD）
tf    → "A"=正确 "B"=错误
```

## 5.5 Token 记账

每步 LLM 调用都会记录 `input_tokens / output_tokens`，`QAAgent` 汇总成每题的 `tokens` 字段，批量运行时累加为总 Token，用于评估 TokenScore。

## 5.6 并发与稳定性

- `scripts/run_agent.py` 用线程池并发（默认 5 线程），**每个线程独立加载一次 BM25 索引**，避免共享可变状态；
- 单题失败不中断整体，进度实时写入 `run_log.txt`；
- 每题产物（答案、证据、日志）单独落盘，方便事后逐题复盘。

## 5.7 Prompt 模板

所有 Prompt 集中在 `agent/prompts.py`，按题型分发（计算/单选/多选/判断），是调优的核心文件。改 Prompt 时建议：

1. 先改一套，跑 `--limit 5` 联调；
2. 对比 `output/evidence/` 看检索是否覆盖正确答案；
3. 再全量跑，比较准确率与 Token。
