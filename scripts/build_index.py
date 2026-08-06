"""构建 BM25 检索索引（Agent 实际使用的检索路径）。

用法：
  python scripts/build_index.py build           # 从 data/chunks 构建索引
  python scripts/build_index.py build --force   # 强制重建（忽略已有索引）
  python scripts/build_index.py search "查询词"  # 用已有索引做检索测试
  python scripts/build_index.py info            # 查看索引统计
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agent.settings import BM25_INDEX_DIR, BM25_INDEX_FILE, CHUNKS_FILE  # noqa: E402
from retrieval.bm25_retriever import BM25Retriever  # noqa: E402


def _index_path() -> Path:
    return Path(BM25_INDEX_DIR) / BM25_INDEX_FILE


def cmd_build(force: bool) -> int:
    if not Path(CHUNKS_FILE).exists():
        print(f"[ERROR] 未找到分块文件：{CHUNKS_FILE}")
        print("        请先完成数据处理（见 docs/03-数据准备与分块.md）")
        return 1

    retriever = BM25Retriever()
    retriever.build(CHUNKS_FILE, force=force)
    print(f"[DONE] 索引已保存：{_index_path()}")
    return 0


def cmd_search(query: str, top_k: int) -> int:
    if not _index_path().exists():
        print(f"[ERROR] 索引不存在：{_index_path()}，请先 build")
        return 1
    retriever = BM25Retriever.load()
    hits = retriever.search(query, top_k=top_k)
    print(f"查询：{query}")
    print(f"命中 {len(hits)} 条：\n")
    for i, hit in enumerate(hits, 1):
        text = hit.get("text", "")[:120].replace("\n", " ")
        print(f"[{i}] {hit.get('doc_id')} | {hit.get('chunk_id')} | score={hit.get('bm25_score', 0):.4f}")
        print(f"    {text}\n")
    return 0


def cmd_info() -> int:
    if not _index_path().exists():
        print(f"[ERROR] 索引不存在：{_index_path()}，请先 build")
        return 1
    retriever = BM25Retriever.load()
    print(f"索引文件：{_index_path()}")
    print(f"分块总数：{retriever.N}")
    print(f"词汇表大小：{len(retriever.idf)}")
    print(f"平均文档长度：{retriever.avgdl:.1f}")

    chunks_root = Path("data/chunks")
    if chunks_root.is_dir():
        print("\n按领域统计（data/chunks 子目录）：")
        for domain_dir in sorted(chunks_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            count = sum(1 for f in domain_dir.glob("*.json"))
            print(f"  {domain_dir.name}: {count} 个文件")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BM25 索引构建与检索")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="构建索引")
    p_build.add_argument("--force", action="store_true", help="强制重建")

    p_search = sub.add_parser("search", help="检索测试")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("--top-k", type=int, default=5, help="返回条数（默认 5）")

    sub.add_parser("info", help="索引统计")

    args = parser.parse_args()
    if args.cmd == "build":
        return cmd_build(args.force)
    if args.cmd == "search":
        return cmd_search(args.query, args.top_k)
    if args.cmd == "info":
        return cmd_info()
    return 1


if __name__ == "__main__":
    sys.exit(main())
