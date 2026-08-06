"""
Hybrid Retriever: BM25(jieba) 关键词 + FAISS Dense 语义，双路独立。

BM25  — 自定义 jieba BM25，精确中文关键词匹配
语义 — FAISS IndexIVFFlat (Dense) → Reranker

两条路径各自独立返回结果，上层调用者负责合并去重。
"""
import time, json, os, pickle
from typing import Optional

from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.faiss_store import FAISSStore
from retrieval.bm25_retriever import BM25Retriever

CHUNKS_PATH = "data/chunks/all_chunks.json"
CHECKPOINT_PATH = "data/index/embedding_checkpoint.pkl"


class HybridRetriever:
    """BM25 + FAISS Dense 双路检索器。"""

    def __init__(self, bm25: Optional[BM25Retriever] = None):
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.faiss = FAISSStore()
        self.bm25 = bm25
        self.chunks: list[dict] = []
        self.chunk_id_map: dict[str, dict] = {}
        self._built = False

    # ── Build ──────────────────────────────────────────

    def build(self, chunks_path: str = CHUNKS_PATH):
        """构建全部索引。BM25 + Embedding + FAISS。"""
        t0 = time.time()

        print("[1/3] Loading chunks...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"  {len(self.chunks)} chunks")
        self._build_chunk_map()

        print("[2/3] Building BM25...")
        self.bm25 = BM25Retriever()
        self.bm25.build(chunks_path, force=True)

        print("[3/3] Generating embeddings + Building FAISS...")
        texts = [c["text"] for c in self.chunks]
        total = len(texts)

        # ── 断点续传 ──
        start_idx = 0
        embeddings = []
        if os.path.exists(CHECKPOINT_PATH):
            with open(CHECKPOINT_PATH, "rb") as f:
                ckpt = pickle.load(f)
            start_idx = ckpt.get("done", 0)
            embeddings = ckpt.get("embeddings", [])
            print(f"  Resuming from index {start_idx}/{total} (已有 {len(embeddings)} 条)")

        batch_size = self.embedder.batch_size
        for i in range(start_idx, total, batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = self.embedder.embed_batch(batch)
            embeddings.extend(batch_embs)

            # 每 50 批保存一次进度
            done = i + len(batch)
            if done % (batch_size * 50) < batch_size or done >= total:
                os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
                with open(CHECKPOINT_PATH, "wb") as f:
                    pickle.dump({"done": done, "embeddings": embeddings}, f)
                print(f"  checkpoint: {done}/{total}")

        # 完成，删除 checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

        # 检测空向量并重试
        failed = [i for i, e in enumerate(embeddings)
                  if len(e) != self.embedder.dim]
        if failed:
            print(f"  retrying {len(failed)} failed embeddings...")
            for idx in failed:
                retry = self.embedder.embed_batch([texts[idx]])
                if len(retry[0]) == self.embedder.dim:
                    embeddings[idx] = retry[0]
            for i, e in enumerate(embeddings):
                if len(e) != self.embedder.dim:
                    embeddings[i] = [0.0] * self.embedder.dim

        dense_vectors = embeddings
        chunk_ids = [c["chunk_id"] for c in self.chunks]
        doc_ids = [c.get("doc_id", "") for c in self.chunks]
        header_paths = [c.get("header_path", "") for c in self.chunks]

        self.faiss.build(
            dense_vectors=dense_vectors,
            chunk_ids=chunk_ids,
            texts=texts,
            doc_ids=doc_ids,
            header_paths=header_paths,
            force=True,
        )

        self._built = True
        print(f"Build done ({time.time()-t0:.0f}s)")

    def _build_chunk_map(self):
        self.chunk_id_map = {ch["chunk_id"]: ch for ch in self.chunks}

    # ── Load ────────────────────────────────────────────

    @classmethod
    def load(cls) -> "HybridRetriever":
        t0 = time.time()
        r = cls()

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            r.chunks = json.load(f)
        r._build_chunk_map()

        r.bm25 = BM25Retriever.ensure_built()
        r.faiss = FAISSStore.load()

        r._built = True
        print(f"Loaded: {len(r.chunks)} chunks ({time.time()-t0:.1f}s)")
        return r

    # ── Search ──────────────────────────────────────────

    def search_bm25(self, query: str, doc_ids: Optional[set[str]] = None,
                    top_k: int = 5) -> list[dict]:
        """BM25 关键词搜索。"""
        if not self.bm25:
            return []
        return self.bm25.search(query, doc_ids=doc_ids, top_k=top_k)

    def search_semantic(self, query: str, doc_ids: Optional[set[str]] = None,
                        top_k: int = 5, coarse_k: int = 50) -> list[dict]:
        """语义搜索 — FAISS Dense → Reranker。"""
        t0 = time.time()

        q_emb = self.embedder.embed_query(query)
        if not q_emb:
            return []

        # FAISS 粗排 — 如果指定了 doc_ids 需要更多候选避免过滤后不足
        effective_coarse = max(coarse_k, 100) if doc_ids else coarse_k
        dense_hits = self.faiss.search(q_emb, top_k=effective_coarse)

        # 过滤 doc_ids
        if doc_ids:
            dense_hits = [h for h in dense_hits if h.get("doc_id", "") in doc_ids]

        # 准备 rerank
        if not dense_hits:
            return []

        docs = [h["text"] for h in dense_hits]
        reranked = self.reranker.rerank(query, docs, top_k=top_k)

        if not reranked:
            reranked = [
                {"index": i, "score": h["score"], "text": h["text"]}
                for i, h in enumerate(dense_hits[:top_k])
            ]

        results = []
        for r in reranked:
            idx = r["index"]
            if idx >= len(dense_hits):
                continue
            h = dense_hits[idx]
            results.append({
                "chunk_id": h["chunk_id"],
                "doc_id": h.get("doc_id", ""),
                "header_path": h.get("header_path", ""),
                "text": r.get("text", h.get("text", "")),
                "rerank_score": round(r["score"], 4),
            })

        total_ms = (time.time() - t0) * 1000
        return results
