"""
MilvusStore: Dense + Sparse 双路向量存储

单 collection 四字段：
  id              → VARCHAR(64) 主键
  text            → VARCHAR(4096) 原文
  dense_vector    → FLOAT_VECTOR(1024) + IVF_FLAT 索引  (text-embedding-v4 dense)
  sparse_embedding→ SPARSE_FLOAT_VECTOR + SPARSE_INVERTED_INDEX (text-embedding-v4 sparse)

BM25 由独立的 BM25Retriever 负责，不经过 Milvus。
连接策略：每次操作创建新 client，操作完立即释放。
"""
import os
from pathlib import Path

os.environ.setdefault('GRPC_KEEPALIVE_TIME_MS', '60000')
os.environ.setdefault('GRPC_KEEPALIVE_TIMEOUT_MS', '20000')
os.environ.setdefault('GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA', '0')

from pymilvus import (
    MilvusClient, DataType, FieldSchema, CollectionSchema,
)

DENSE_DIM = 1024
COLLECTION_NAME = "chunks"
DB_PATH = "data/index/milvus.db"


class MilvusStore:
    """Milvus Lite: 短连接模式，Dense + Sparse 双路。"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> MilvusClient:
        return MilvusClient(self.db_path)

    # ── Collection 管理 ──────────────────────────────────

    def create_collection(self, drop: bool = True):
        if drop:
            import shutil
            db_dir = Path(self.db_path)
            if db_dir.exists():
                try:
                    shutil.rmtree(db_dir)
                except Exception:
                    pass
            db_dir.parent.mkdir(parents=True, exist_ok=True)

        c = self._connect()
        try:
            if c.has_collection(COLLECTION_NAME):
                return

            schema = CollectionSchema(
                auto_id=False,
                fields=[
                    FieldSchema("id", DataType.VARCHAR, max_length=256, is_primary=True),
                    FieldSchema("text", DataType.VARCHAR, max_length=4096),
                    FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=DENSE_DIM),
                    FieldSchema("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR),
                ],
            )
            c.create_collection(collection_name=COLLECTION_NAME, schema=schema)
        finally:
            c.close()

    def create_index(self):
        self._clean_manifest_tmp()
        c = self._connect()
        try:
            index_params = c.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector", index_type="IVF_FLAT",
                metric_type="IP", params={"nlist": 128})
            index_params.add_index(
                field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP")

            for attempt in range(3):
                try:
                    self._clean_manifest_tmp()
                    c.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  create_index retry {attempt+1}: {e}")
                        import time; time.sleep(1)
                    else:
                        print(f"  create_index 警告: {e}")
        finally:
            c.close()

    def load_collection(self):
        c = self._connect()
        try:
            c.load_collection(COLLECTION_NAME)
        finally:
            c.close()

    def _clean_manifest_tmp(self):
        """删除残留 .tmp 及冲突的 manifest.json，避免 Windows rename 失败。"""
        db = Path(self.db_path)
        for tmp in list(db.glob("**/*.tmp")):
            try:
                target = tmp.with_suffix("")
                if target.exists():
                    target.unlink()
            except Exception:
                pass
            try:
                tmp.unlink()
            except Exception:
                pass

    # ── 数据写入 ──────────────────────────────────────────

    def insert(self, chunk_ids: list[str],
               dense_vectors: list[list[float]],
               sparse_embeddings: list[dict[int, float]],
               texts: list[str]):
        self._clean_manifest_tmp()
        c = self._connect()
        try:
            data = [
                {"id": cid, "text": txt, "dense_vector": dvec, "sparse_embedding": svec}
                for cid, dvec, svec, txt in zip(chunk_ids, dense_vectors, sparse_embeddings, texts)
            ]
            c.insert(COLLECTION_NAME, data)
        finally:
            c.close()

    # ── 搜索 ──────────────────────────────────────────

    def dense_search(self, query_vector: list[float], top_k: int = 50) -> list[dict]:
        c = self._connect()
        try:
            results = c.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector], anns_field="dense_vector",
                limit=top_k, output_fields=["id", "text"],
                search_params={"nprobe": 64})
            return self._parse_hits(results)
        finally:
            c.close()

    def sparse_search(self, query_sparse: dict[int, float], top_k: int = 50) -> list[dict]:
        c = self._connect()
        try:
            results = c.search(
                collection_name=COLLECTION_NAME,
                data=[query_sparse], anns_field="sparse_embedding",
                limit=top_k, output_fields=["id", "text"])
            return self._parse_hits(results)
        finally:
            c.close()

    # ── 内部 ──────────────────────────────────────────────

    def _parse_hits(self, results) -> list[dict]:
        if not results or not results[0]:
            return []
        hits = []
        for r in results[0]:
            entity = r.get("entity", {})
            hits.append({
                "chunk_id": r["id"],
                "text": entity.get("text", ""),
                "score": r["distance"],
            })
        return hits

    @property
    def count(self) -> int:
        c = self._connect()
        try:
            return c.get_collection_stats(COLLECTION_NAME)["row_count"]
        except Exception:
            return 0
        finally:
            c.close()
