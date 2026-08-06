"""
DashScope qwen3-rerank wrapper.
Takes query + candidates, returns relevance scores.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import dashscope
from http import HTTPStatus


class Reranker:
    """Qwen3-Reranker via DashScope TextReRank API."""

    def __init__(self, model: str = "qwen3-rerank",
                 api_key: Optional[str] = None):
        self.model = model
        dashscope.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")

    def rerank(self, query: str, documents: list[str],
               top_k: int = 10) -> list[dict]:
        """对候选文档重排序。
        Args:
            query: 查询文本
            documents: 候选文档文本列表
            top_k: 返回前 N 个
        Returns:
            [{index: int, score: float, text: str}, ...] 按 score 降序
        """
        if not documents:
            return []

        try:
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception as e:
            print(f"  rerank API error: {e}")
            return []

        if resp.status_code != HTTPStatus.OK:
            print(f"  rerank error: {resp.code} {resp.message}")
            return []

        results = resp.output.get("results", [])
        return [
            {
                "index": r["index"],
                "score": r["relevance_score"],
                "text": documents[r["index"]],
            }
            for r in sorted(results, key=lambda x: -x["relevance_score"])
        ]
