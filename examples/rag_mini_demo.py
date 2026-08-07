"""RAG 最小流程演示（零配置、零 API、零下载，30 秒跑完）。

用 3 条迷你"保险条款"演示 RAG 的核心三步：
  ① 分块：长文档切成可检索的小块
  ② 索引：给小块建关键词索引（BM25）
  ③ 检索：输入问题，找到最相关的块

这是理解本项目大管线的第一步——真实项目中：
  分块 → data_processing/chunker_md.py
  索引 → retrieval/bm25_retriever.py
  检索 → agent/retrieve_step.py（再交给 LLM 压缩、作答）

用法：
  python examples/rag_mini_demo.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---- 迷你语料：3 条保险条款（真实项目里是几百份 PDF 解析后的 Markdown）----
MINI_DOCS = {
    "insurance_a": """# 平安智盈金生专属商业养老保险条款

## 第一条 身故保险金

被保险人在本合同有效期内身故的，本公司按照下列约定给付身故保险金：
若被保险人在养老保险金开始领取日前身故，按被保险人身故时本合同保单账户价值给付。

## 第二条 犹豫期

自投保人签收本合同之日起十五日为犹豫期。犹豫期内投保人要求解除合同的，
本公司退还投保人已交的全部保险费。

## 第三条 责任免除

因投保人对被保险人的故意杀害、故意伤害导致被保险人发生保险事故的，
本公司不承担给付保险金的责任。
""",
    "insurance_b": """# 国寿增益宝终身寿险（万能型）条款

## 第一条 保险责任

在本合同保险期间内，本公司承担下列保险责任：被保险人于本合同生效之日起
一百八十日后因疾病身故的，本公司按本合同的基本保险金额给付身故保险金。

## 第二条 现金价值

本合同保单年度末的现金价值等于保单账户价值扣除退保费用后的余额。
""",
    "insurance_c": """# 平安e生保医疗保险条款

## 第一条 等待期

本合同生效之日起三十日为等待期。被保险人在等待期内发生的医疗费用，
本公司不承担给付保险金的责任，但因意外伤害导致的医疗费用不受等待期限制。

## 第二条 保险金额

本合同项下的年度保险金额以保险单载明为准，最高不超过三百万元。
""",
}


def main() -> int:
    print("=" * 64)
    print("RAG 最小流程演示：分块 → 索引 → 检索")
    print("=" * 64)

    # ---------- ① 分块 ----------
    from data_processing.chunker_md import chunk_markdown

    print("\n[① 分块] 把 3 篇文档按标题层级切成小块，每块带 header_path（位置信息）")
    all_chunks = []
    for doc_id, text in MINI_DOCS.items():
        chunks = chunk_markdown(doc_id, text)
        all_chunks.extend(chunks)
        for c in chunks:
            print(f"  [{c['chunk_id']}] hp={c['header_path']!r} tok={c['tokens']}")
            print(f"      {c['text'][:42].replace(chr(10), ' ')}...")

    # 分块结果落盘到临时目录（真实项目写入 data/chunks/all_chunks.json）
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_demo_"))
    chunks_file = tmp_dir / "all_chunks.json"
    chunks_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- ② 索引 ----------
    import retrieval.bm25_retriever as bm25_mod

    print("\n[② 索引] 用增强 BM25 给每个块建倒排索引（jieba 分词 + Bigram）")
    # 隔离索引路径，避免触碰真实项目索引 data/index/
    bm25_mod.INDEX_DIR = str(tmp_dir / "index")
    bm25_mod.INDEX_PATH = str(tmp_dir / "index" / "demo.pkl")
    retriever = bm25_mod.BM25Retriever()
    retriever.build(str(chunks_file), force=True)

    # ---------- ③ 检索 ----------
    print("\n[③ 检索] 输入问题，找到最相关的块")
    queries = ["身故保险金如何计算？", "犹豫期可以退保吗？", "等待期多久？"]
    for q in queries:
        hits = retriever.search(q, top_k=2)
        print(f"\n  问：{q}")
        for h in hits:
            snippet = h["text"][:60].replace("\n", " ")
            print(f"    → [{h['chunk_id']}] ({h['bm25_score']:.2f}) {snippet}...")

    # 反例：换个说法就查不到 → 引出"字面匹配的局限"
    print("\n[④ 反例] 换个说法试试：\"买完多久能退？\"")
    hits = retriever.search("买完多久能退？", top_k=2)
    if not hits:
        print("    → 0 条命中（原文写的是\"犹豫期/解除合同/退还\"，和\"买完/退\"没有词面重合）")
        print("      这就是纯关键词检索（BM25）的局限：字面不重叠就查不到。")
        print("      进阶方案靠两条路解决：")
        print("        1) 向量检索（意思相近也算匹配，见 retrieval/ 实验路径）")
        print("        2) LLM 先理解问题、拆成检索词（见 docs/05-Agent流水线.md）")

    print("\n" + "=" * 64)
    print("完成！真实项目中，检索到的块还会被 LLM 压缩成证据、再作答。")
    print("下一步：读 docs/00-RAG入门.md 理解概念，再按 docs/05-Agent流水线.md 看完整流程。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
