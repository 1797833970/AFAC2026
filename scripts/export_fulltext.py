"""把分块数据按领域导出为纯文本（便于人工通读/排查）。

用法：
  python scripts/export_fulltext.py [--chunks data/chunks] [--out full_text]
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def export_domain(chunks_dir: Path, domain: str, out_path: Path) -> int:
    domain_dir = chunks_dir / domain
    if not domain_dir.is_dir():
        return 0

    lines = []
    for fname in sorted(domain_dir.glob("*.json")):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except json.JSONDecodeError:
            print(f"  [WARN] 跳过无法解析的文件：{fname.name}")
            continue
        for chunk in chunks:
            doc_id = chunk.get("doc_id", fname.stem)
            text = chunk.get("text", "")
            lines.append(f"[DOC:{doc_id}] {text}")

    if not lines:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"  {domain}: {len(lines)} chunks -> {out_path}")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="导出全文")
    parser.add_argument("--chunks", default="data/chunks", help="分块目录")
    parser.add_argument("--out", default="full_text", help="输出目录")
    parser.add_argument("--domains", default="", help="逗号分隔的领域列表，默认全部")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks)
    out_dir = Path(args.out)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or [
        "financial_reports",
        "financial_contracts",
        "insurance",
        "regulatory",
        "research",
    ]

    total = 0
    for domain in domains:
        total += export_domain(chunks_dir, domain, out_dir / f"{domain}.txt")
    print(f"\n完成：共导出 {total} 个 chunk 到 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
