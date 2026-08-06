"""
DashScope text-embedding-v4 wrapper.
Dense (1024-dim) embedding via output_type="dense".

text_type:
  "document" — 建库用，生成"正文"向量，信息全面，优化"被匹配"
  "query"    — 查询用，生成"标题"向量，方向性强，优化"提问/查找"
  instruct 仅 text_type="query" 时生效。
"""
import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import dashscope


# 检索场景指令（仅英文，仅 query 模式生效）
INSTRUCT = (
    "Given a Chinese financial question about contracts, reports, regulations, "
    "or insurance terms, retrieve the most relevant document sections that "
    "contain factual evidence to answer the question"
)

DENSE_DIM = 1024  # text-embedding-v4 with dimension=1024


class Embedder:
    """Qwen3-Embedding (text-embedding-v4) via DashScope."""

    def __init__(self, model: str = "text-embedding-v4", batch_size: int = 10,
                 api_key: Optional[str] = None):
        self.model = model
        self.batch_size = batch_size
        dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """嵌入一批文本，带重试。Returns: [[float, ...], ...]"""
        resp = None
        for attempt in range(3):
            try:
                resp = dashscope.TextEmbedding.call(
                    model=self.model,
                    input=texts,
                    dimension=DENSE_DIM,
                    output_type="dense",
                    text_type="document",
                )
                break
            except Exception as e:
                if attempt < 2:
                    print(f"\n  embedding retry {attempt+1}: {e}")
                    import time as _t; _t.sleep(2 * (attempt + 1))
                else:
                    print(f"\n  embedding failed after 3 retries: {e}")
                    resp = None

        if resp is None or resp.status_code != 200:
            code = getattr(resp, 'status_code', 'N/A') if resp else 'N/A'
            msg = getattr(resp, 'message', 'timeout') if resp else 'timeout'
            print(f"\n  embedding error ({code}): {msg}")
            return [[] for _ in texts]

        try:
            results = []
            for emb in resp.output.get("embeddings", []):
                dense = emb.get("embedding", [])
                results.append(list(dense) if dense else [])
        except Exception as e:
            print(f"\n  embedding parse error: {type(e).__name__}: {e}")
            return [[] for _ in texts]
        return results

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding。"""
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"  embedding {i+1}-{i+len(batch)}/{len(texts)}", end="\r")
            all_results.extend(self.embed_batch(batch))
            time.sleep(0.1)
        print()
        return all_results

    def embed_query(self, query: str) -> list[float]:
        """单条查询 embedding。text_type="query" + instruct，方向性优化。"""
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=query,
                dimension=DENSE_DIM,
                output_type="dense",
                text_type="query",
                instruct=INSTRUCT,
            )
        except Exception as e:
            print(f"  [embed_query] API exception: {type(e).__name__}: {e}")
            return []

        if resp.status_code != 200:
            print(f"  embed query error: code={resp.code} message={resp.message}")
            print(f"  (Check DASHSCOPE_API_KEY in .env, or network connectivity)")
            return []

        try:
            emb = resp.output.get("embeddings", [{}])[0]
            return list(emb.get("embedding", []))
        except Exception as e:
            print(f"  [embed_query] parse response error: {type(e).__name__}: {e}")
            return []

    @property
    def dim(self) -> int:
        return DENSE_DIM
