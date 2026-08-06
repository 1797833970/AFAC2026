"""Step 2: Retrieve — BM25 关键词检索，top 3，未命中则补 2 条。"""

from retrieval.bm25_retriever import BM25Retriever
from .settings import BM25_TOP_K, BM25_FALLBACK_K

class Retriever:

    def __init__(self, retriever=None):
        self.retriever = retriever or BM25Retriever.load()

    def retrieve(self, analyses: list[dict], default_doc_ids: list) -> list[dict]:
        """[{description, semantic_query, keywords, doc_ids}, ...] → [{description, chunks: [...]}, ...]"""
        results = []
        for item in analyses:
            desc = item.get('description', '?')
            semantic_query = item.get('semantic_query', '')
            keywords = item.get('keywords', '')
            doc_ids = item.get('doc_ids', [])

            if not semantic_query:
                results.append({'description': desc, 'semantic_query': semantic_query, 'doc_ids': [], 'chunks': []})
                continue

            # 保持原始顺序的doc_ids列表
            target_docs_list = doc_ids if doc_ids else list(default_doc_ids)
            target_docs_set = set(target_docs_list)

            # keywords 中 | 是备用同义词，展开为空格分隔供 BM25 使用
            bm25_query = keywords.replace('|', ' ') if keywords else semantic_query

            try:
                # ── BM25 top 3 ──
                hits = self._bm25_search(bm25_query, target_docs_list, target_docs_set, BM25_TOP_K)

                # ── 未命中则用 semantic_query 补 2 条 ──
                if not hits:
                    fallback = self._bm25_search(semantic_query, target_docs_list, target_docs_set, BM25_FALLBACK_K)
                    # 补入最多 2 条
                    seen = set()
                    for h in hits:
                        seen.add(h.get('chunk_id', ''))
                    for h in fallback:
                        if h.get('chunk_id', '') not in seen:
                            hits.append(h)
                            seen.add(h.get('chunk_id', ''))
                            if len(hits) >= 2:
                                break

            except Exception as e:
                print(f'  [WARN] Retrieval for "{desc[:30]}" failed: {e}')
                hits = []

            results.append({'description': desc, 'semantic_query': semantic_query, 'doc_ids': target_docs_list, 'chunks': hits})

        return results

    def _bm25_search(self, query: str, target_docs_list: list, target_docs_set: set, top_k: int) -> list[dict]:
        """BM25 检索，多文档逐文档检索后合并去重。"""
        if len(target_docs_list) > 1:
            all_hits = []
            for doc_id in target_docs_list:
                for h in self.retriever.search(query=query, doc_ids={doc_id}, top_k=top_k):
                    h['source'] = 'bm25'
                    h['score'] = h.get('bm25_score', 0)
                    all_hits.append(h)
        else:
            all_hits = []
            for h in self.retriever.search(query=query, doc_ids=target_docs_set, top_k=top_k):
                h['source'] = 'bm25'
                h['score'] = h.get('bm25_score', 0)
                all_hits.append(h)

        # chunk_id 去重
        merged = {}
        for h in all_hits:
            cid = h.get('chunk_id', '')
            if cid and cid not in merged:
                merged[cid] = h
        return list(merged.values())
