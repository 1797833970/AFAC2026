"""
FAISSStore: 基于 FAISS 的 Dense 向量存储与检索。

替代 Milvus Lite，无 gRPC 连接问题，无 Windows 文件锁 bug。
仅存储 dense_vector，sparse 向量不纳入（BM25 已覆盖关键词匹配）。

索引文件：
  data/index/faiss/
    index.faiss     — FAISS IndexFlatIP
    meta.pkl        — {chunk_ids, texts, doc_ids, header_paths}

用法:
  store = FAISSStore()
  store.build(dense_vectors, chunk_ids, texts, doc_ids, header_paths)
  # 或
  store = FAISSStore.load()
  results = store.search(query_vector, top_k=50)
"""
import os
import pickle

import numpy as np
import faiss

INDEX_DIR = "data/index/faiss"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")


class FAISSStore:
    """FAISS Dense 向量存储，内积(IP)相似度。"""

    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.doc_ids: list[str] = []
        self.header_paths: list[str] = []
        self._dense_dim: int = 0

    # ── 构建 ──────────────────────────────────────────

    def build(self, dense_vectors: list[list[float]],
              chunk_ids: list[str], texts: list[str],
              doc_ids: list[str] | None = None,
              header_paths: list[str] | None = None,
              force: bool = False):
        """构建 FAISS 索引并持久化。"""
        if not force and self._try_load():
            return

        self._dense_dim = len(dense_vectors[0])
        self.chunk_ids = list(chunk_ids)
        self.texts = list(texts)
        self.doc_ids = list(doc_ids) if doc_ids else [""] * len(chunk_ids)
        self.header_paths = list(header_paths) if header_paths else [""] * len(chunk_ids)

        vectors = np.array(dense_vectors, dtype=np.float32)
        N = len(vectors)

        self.index = faiss.IndexFlatIP(self._dense_dim)
        print(f"  [FAISS] Adding {N} vectors...")
        self.index.add(vectors)

        self._save()
        print(f"  [FAISS] Built: {self.index.ntotal} vectors, dim={self._dense_dim}")

    # ── 加载 ──────────────────────────────────────────

    def _try_load(self) -> bool:
        if not os.path.exists(INDEX_PATH):
            return False
        print(f"[FAISS] Loading index from {INDEX_DIR}...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.chunk_ids = meta["chunk_ids"]
        self.texts = meta["texts"]
        self.doc_ids = meta.get("doc_ids", [""] * len(self.chunk_ids))
        self.header_paths = meta.get("header_paths", [""] * len(self.chunk_ids))
        self._dense_dim = self.index.d
        print(f"  {self.index.ntotal} vectors, dim={self._dense_dim}")
        return True

    @classmethod
    def load(cls) -> "FAISSStore":
        s = cls()
        s._try_load()
        return s

    # ── 持久化 ─────────────────────────────────────────

    def _save(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        meta = {
            "chunk_ids": self.chunk_ids,
            "texts": self.texts,
            "doc_ids": self.doc_ids,
            "header_paths": self.header_paths,
        }
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    # ── 搜索 ──────────────────────────────────────────

    def search(self, query_vector: list[float], top_k: int = 50) -> list[dict]:
        """返回 top_k 个最相似的 chunk。"""
        if not self.index:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)  # L2 归一化，使 IP ≈ cosine similarity
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "doc_id": self.doc_ids[idx],
                "header_path": self.header_paths[idx],
                "text": self.texts[idx],
                "score": float(score),
            })
        return results
