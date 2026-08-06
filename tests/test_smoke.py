"""离线冒烟测试：不依赖 API Key、不访问网络、不触碰真实索引。

运行：python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. BM25 检索器（小语料，隔离索引路径）
# ---------------------------------------------------------------------------

def test_bm25_retriever(tmp_path):
    import retrieval.bm25_retriever as br

    # 重定向索引路径到临时目录，避免覆盖真实索引
    br.INDEX_DIR = str(tmp_path / "index")
    br.INDEX_PATH = str(tmp_path / "index" / "test.pkl")

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "insurance_1",
            "header_path": "第一章",
            "text": "身故保险金按照基本保险金额的百分之一百六十给付。",
        },
        {
            "chunk_id": "c2",
            "doc_id": "insurance_2",
            "header_path": "第二章",
            "text": "犹豫期内解除合同退还全部已交保险费。",
        },
        {
            "chunk_id": "c3",
            "doc_id": "financial_contracts_1",
            "header_path": "第三章",
            "text": "本期债券发行规模不超过人民币十亿元。",
        },
    ]
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    retriever = br.BM25Retriever()
    retriever.build(str(chunks_file), force=True)

    hits = retriever.search("身故保险金给付比例", top_k=2)
    assert hits, "应能检索到身故保险金相关内容"
    assert hits[0]["chunk_id"] == "c1"

    # doc_ids 过滤
    hits = retriever.search("发行规模", doc_ids={"financial_contracts_1"}, top_k=2)
    assert hits and hits[0]["doc_id"] == "financial_contracts_1"


# ---------------------------------------------------------------------------
# 2. 答案格式化
# ---------------------------------------------------------------------------

def test_answer_formatting():
    from agent.answer_step import Answerer

    answerer = Answerer()  # 仅构造，不调用 API
    assert answerer._format_answer(["B"], "mcq") == "B"
    assert answerer._format_answer(["ABD"], "multi") == "ABD"
    assert answerer._format_answer(["A", "D"], "multi") == "AD"
    assert answerer._format_answer(["A"], "tf") == "A"
    assert answerer._format_answer(["B"], "tf") == "B"
    assert answerer._format_answer([], "mcq") == ""


# ---------------------------------------------------------------------------
# 3. Markdown 分块引擎
# ---------------------------------------------------------------------------

def test_chunker_markdown():
    from data_processing.chunker_md import chunk_markdown

    sample = """# 平安智盈金生专属商业养老保险条款

## 第一章 保险责任

在本合同有效期内，本公司承担下列保险责任：

### 第一条 身故保险金

被保险人于本合同生效后身故的，本公司按照本合同约定给付身故保险金。

## 第二章 责任免除

因下列情形之一导致被保险人发生保险事故的，本公司不承担给付保险金的责任。
"""
    chunks = chunk_markdown("insurance_demo", sample)
    assert isinstance(chunks, list) and len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert chunk["text"].strip()


# ---------------------------------------------------------------------------
# 4. 提交 CSV 构建
# ---------------------------------------------------------------------------

def test_split_answer():
    from scripts.build_submission import split_answer

    assert split_answer("ACD", "多选题") == ["ACD", "", "", ""]
    assert split_answer("5.55；5.36；7.68；7.35", "计算题") == ["5.55", "5.36", "7.68", "7.35"]
    assert split_answer("14.41", "计算题") == ["14.41", "", "", ""]


def test_estimate_tokens():
    from scripts.build_submission import estimate_tokens

    assert estimate_tokens("") == 1
    assert estimate_tokens("中文十个字测试一下") > 0
    assert estimate_tokens("english text") > 0


def test_build_rows_and_csv(tmp_path):
    from scripts.build_submission import build_rows, write_csv

    questions = {
        "fc_b_001": {"qid": "fc_b_001", "type": "计算题", "question": "毛利率排序？", "options": {}},
        "fc_b_002": {"qid": "fc_b_002", "type": "多选题", "question": "哪些正确？", "options": {}},
    }
    answers = {
        "fc_b_001": {"answer": "5.55；5.36；7.68；7.35", "tokens": {"input": 1000, "output": 200}},
        "fc_b_002": {"answer": "ACD", "tokens": None},
    }
    rows, warnings = build_rows(questions, answers, {})
    assert len(rows) == 3  # 2 题 + summary
    assert rows[0]["answer_1"] == "5.55"
    assert rows[0]["total_tokens"] == 1200
    assert rows[1]["answer_1"] == "ACD"
    assert rows[-1]["qid"] == "summary"
    assert rows[-1]["total_tokens"] == rows[-1]["prompt_tokens"] + rows[-1]["completion_tokens"]

    out = tmp_path / "submit.csv"
    write_csv(rows, out)
    content = out.read_text(encoding="utf-8-sig")
    assert "qid,answer_1" in content
    assert "summary" in content


# ---------------------------------------------------------------------------
# 5. 官方题目数据 schema
# ---------------------------------------------------------------------------

def test_question_schema():
    from scripts.build_submission import load_questions
    from agent.settings import QUESTIONS_B_DIR

    questions = load_questions(str(PROJECT_ROOT / QUESTIONS_B_DIR))
    assert len(questions) >= 100  # B 榜 5 领域 × 20 题
    for qid, q in questions.items():
        assert qid.startswith(("fc_", "fin_", "ins_", "reg_", "res_"))
        assert "question" in q and q["question"].strip()
        assert "type" in q
