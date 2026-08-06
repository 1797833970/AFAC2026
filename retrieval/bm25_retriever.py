"""
BM25 Retriever: jieba 细粒度分词 + Bigram 短语匹配 + 精确子串加成

三重打分机制，解决中文金融术语精确匹配问题：
  1. BM25 基础分 — jieba.cut_for_search 细粒度分词 + IDF/TF
  2. Bigram 重叠分 — 字符级双字词组匹配，捕获"发行规模上限"这类连续短语
  3. 精确子串加成 — 查询词在原文中作为子串出现的 bonus

索引持久化到磁盘，首次构建后直接加载。

用法:
  retriever = BM25Retriever()
  retriever.build("data/chunks/all_chunks.json")   # 首次构建
  # 或
  retriever = BM25Retriever.load()                  # 从磁盘加载
  
  results = retriever.search("债券发行规模上限", doc_ids={"text01"}, top_k=5)
"""
import json
import math
import os
import pickle
import re
from collections import defaultdict

import jieba

INDEX_DIR = "data/index"
INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index.pkl")

# BM25 参数
K1 = 1.5
B = 0.75

# 加权系数
BIGRAM_WEIGHT = 0.3    # bigram 重叠分的权重
SUBSTR_WEIGHT = 0.25   # 精确子串加成的权重


def _char_bigrams(text: str) -> set[str]:
    """生成字符级双字词组（Chinese character bigrams）。
    例如 "发行规模" → {"发行", "行规", "规模"}
    纯数字/英文部分保留原词作为单独的 bigram。
    """
    bigrams = set()
    # 移除空格和标点后逐字生成 bigram
    cleaned = re.sub(r'\s+', '', text)
    for i in range(len(cleaned) - 1):
        bigrams.add(cleaned[i:i + 2])
    # 单字也加入，覆盖极短词
    for ch in cleaned:
        bigrams.add(ch)
    return bigrams


class BM25Retriever:
    """增强 BM25 检索器：jieba 细粒度分词 + Bigram + 子串匹配。"""

    def __init__(self):
        self.chunks: list[dict] = []
        self.doc_ids: list[str] = []              # chunk 索引 → doc_id
        self.tokenized_docs: list[list[str]] = []  # jieba 分词结果
        self.bigram_docs: list[set[str]] = []      # 每个 chunk 的字符 bigram 集合
        self.raw_texts: list[str] = []             # 原始文本（用于子串匹配）
        self.inverted: dict[str, list[tuple[int, int]]] = {}  # term → [(doc_idx, tf)]
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.doc_freq: dict[str, int] = {}
        self.N: int = 0

    # ── 构建 ──────────────────────────────────────────

    def build(self, chunks_path: str = "data/chunks/all_chunks.json",
              force: bool = False):
        """从 all_chunks.json 构建 BM25 索引。"""
        if not force and self._try_load():
            return

        print("[BM25] Loading chunks...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.N = len(self.chunks)
        print(f"  {self.N} chunks")

        # 分词 + bigram
        print("[BM25] Tokenizing (cut_for_search) + bigram...")
        self.doc_ids = []
        self.tokenized_docs = []
        self.bigram_docs = []
        self.raw_texts = []
        doc_lengths = []

        for c in self.chunks:
            text = c.get("text", "")
            self.raw_texts.append(text)
            self.doc_ids.append(c.get("doc_id", ""))

            # jieba.cut_for_search: 更细粒度分词，利于召回
            tokens = list(jieba.cut_for_search(text))
            self.tokenized_docs.append(tokens)
            doc_lengths.append(len(tokens))

            # 字符 bigram
            self.bigram_docs.append(_char_bigrams(text))

        self.avgdl = sum(doc_lengths) / self.N if self.N else 1.0

        # 统计 doc_freq + 构建倒排索引
        print("[BM25] Computing IDF + inverted index...")
        self.doc_freq = defaultdict(int)
        self.inverted = defaultdict(list)
        for idx, tokens in enumerate(self.tokenized_docs):
            seen = set(tokens)
            # doc_freq
            for t in seen:
                self.doc_freq[t] += 1
            # 倒排索引：term → [(doc_idx, tf)]
            tf_counts = defaultdict(int)
            for t in tokens:
                tf_counts[t] += 1
            for t, tf in tf_counts.items():
                self.inverted[t].append((idx, tf))

        # 计算 IDF
        self.idf = {}
        for term, df in self.doc_freq.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

        # 持久化
        self._save()
        print(f"[BM25] Built: {self.N} docs, avgdl={self.avgdl:.1f}, vocab={len(self.idf)}")

    # ── 搜索 ──────────────────────────────────────────

    def search(self, query: str, doc_ids: set[str] | None = None,
               top_k: int = 10) -> list[dict]:
        """增强搜索：倒排索引加速 + BM25 + bigram + 子串。"""
        if not self.tokenized_docs:
            return []

        query_tokens = list(jieba.cut_for_search(query))
        if not query_tokens:
            return []

        query_bigrams = _char_bigrams(query)

        # 倒排索引加速：只遍历包含至少一个查询词的文档
        candidate_idxs: set[int] = set()
        for term in query_tokens:
            if term in self.inverted:
                for idx, _tf in self.inverted[term]:
                    candidate_idxs.add(idx)

        if not candidate_idxs:
            return []

        # 过滤 doc_ids（支持前缀匹配：如 strict_v3_009 可匹配 strict_v3_009_xxx）
        if doc_ids:
            filtered = set()
            for i in candidate_idxs:
                did = self.doc_ids[i]
                if did in doc_ids:
                    filtered.add(i)
                    continue
                for prefix in doc_ids:
                    if did.startswith(prefix + "_") or did.startswith(prefix + "（") or did.startswith(prefix + "("):
                        filtered.add(i)
                        break
            candidate_idxs = filtered

        # 逐候选文档打分
        scores = []
        for idx in candidate_idxs:
            tokens = self.tokenized_docs[idx]
            doc_len = len(tokens)

            # 1. BM25（只计算包含在 inverted 中的 term）
            bm25 = 0.0
            for term in query_tokens:
                if term not in self.idf:
                    continue
                # 从倒排索引获取该文档中该 term 的 tf
                tf = 0
                for di, tfi in self.inverted.get(term, []):
                    if di == idx:
                        tf = tfi
                        break
                if tf == 0:
                    continue
                idf = self.idf[term]
                numerator = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * doc_len / self.avgdl)
                bm25 += idf * numerator / denominator

            if bm25 <= 0:
                continue

            # 2. Bigram 重叠加分
            bigram_overlap = len(query_bigrams & self.bigram_docs[idx])
            bigram_score = bigram_overlap * BIGRAM_WEIGHT

            # 3. 精确子串加成
            raw = self.raw_texts[idx]
            substr_hits = sum(1 for t in query_tokens if len(t) >= 2 and t in raw)
            substr_score = substr_hits * SUBSTR_WEIGHT

            total = bm25 + bigram_score + substr_score
            scores.append((idx, total))

        # 排序取 top_k
        scores.sort(key=lambda x: -x[1])
        results = []
        for idx, total in scores[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "doc_id": chunk.get("doc_id", ""),
                "header_path": chunk.get("header_path", ""),
                "text": chunk.get("text", ""),
                "bm25_score": round(total, 4),
            })
        return results

    # ── 持久化 ─────────────────────────────────────────

    def _save(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "doc_ids": self.doc_ids,
            "tokenized_docs": self.tokenized_docs,
            "bigram_docs": self.bigram_docs,
            "raw_texts": self.raw_texts,
            "inverted": dict(self.inverted),
            "avgdl": self.avgdl,
            "idf": dict(self.idf),
            "doc_freq": dict(self.doc_freq),
            "N": self.N,
        }
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25] Index saved to {INDEX_PATH}")

    def _try_load(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        print(f"[BM25] Loading index from {INDEX_PATH}...")
        with open(INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.doc_ids = data["doc_ids"]
        self.tokenized_docs = data["tokenized_docs"]
        self.bigram_docs = data.get("bigram_docs", [set() for _ in self.chunks])
        self.raw_texts = data.get("raw_texts", [c.get("text", "") for c in self.chunks])
        self.inverted = defaultdict(list, data.get("inverted", {}))
        self.avgdl = data["avgdl"]
        self.idf = data["idf"]
        self.doc_freq = data["doc_freq"]
        self.N = data["N"]
        print(f"  {self.N} docs, avgdl={self.avgdl:.1f}, vocab={len(self.idf)}")
        return True

    @staticmethod
    def ensure_built():
        """确保索引存在，不存在则构建。"""
        if os.path.exists(INDEX_PATH):
            return BM25Retriever.load()
        r = BM25Retriever()
        r.build()
        return r

    @classmethod
    def load(cls) -> "BM25Retriever":
        r = cls()
        r._try_load()
        return r
