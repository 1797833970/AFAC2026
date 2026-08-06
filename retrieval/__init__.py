"""
双路融合检索系统

Pipeline:
  Query → BM25(本地) + FAISS(Dense) → 合并去重 → qwen3-rerank → final top-N

双路召回：
  BM25   → jieba 分词 + BM25 关键词匹配
  Dense  → text-embedding-v4 (1024-dim)，FAISS IndexIVFFlat ANN
"""