"""
Chunk 二次切分：将过长 chunk（>350 tokens）按语义边界拆分为 200-400 tokens 的子块。

切分规则（优先级递减）：
  1. 段落边界（\n\n）
  2. 句号边界（。！？）
  3. 逗号/分号边界（，；）
  4. 硬截断

每个子块约 300 tokens，相邻子块间保留约 40 tokens 的重叠上下文。

用法：
  python scripts/split_chunks.py              # 预览模式（不改文件）
  python scripts/split_chunks.py --apply       # 执行切分并覆盖文件
"""
import json
import os
import re
import sys
from pathlib import Path

# 配置
MAX_TOKENS = 350        # 超过此值的 chunk 进行拆分
TARGET_CHARS = 400      # 目标子块字符数（≈300 tokens）
OVERLAP_CHARS = 70      # 相邻子块重叠字符数（≈40 tokens）
CHARS_PER_TOKEN = 1.8   # 中文字符/Token 估算比

# 切分边界正则（优先级递减）
SPLITTERS = [
    (r'\n\s*\n', 'paragraph'),     # 段落
    (r'[。！？]', 'sentence'),      # 句号
    (r'[；]', 'semicolon'),        # 分号
    (r'[，]', 'comma'),            # 逗号
]

CHUNKS_DIR = Path("data/chunks")
ALL_CHUNKS_PATH = CHUNKS_DIR / "all_chunks.json"


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。"""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def find_split_point(text: str, target: int, min_chars: int = 150) -> int:
    """在 text 中找最佳切分点，尽量接近 target 但不超过 target + target*0.5。"""
    upper = min(target + target // 2, len(text))

    for pattern, _name in SPLITTERS:
        # 在 target 附近找分割点
        best = -1
        for m in re.finditer(pattern, text):
            pos = m.end()
            if min_chars <= pos <= upper:
                best = pos  # 尽可能靠近 target
            elif pos > upper:
                break
        if best > 0:
            return best

    # 硬截断
    return target


def should_not_split(text: str) -> bool:
    """检测不应拆分的文本模式：表格、短文本。"""
    # 表格（含多个 |）
    if text.count('|') > 5:
        return True
    # 极短文本（< 100 字符）不再拆分
    if len(text) < 100:
        return True
    return False


def split_text(text: str) -> list[str]:
    """将 text 按语义边界拆分为多个子块。返回子块文本列表。"""
    parts = []
    remaining = text

    while estimate_tokens(remaining) > MAX_TOKENS:
        if should_not_split(remaining):
            break

        split_at = find_split_point(remaining, TARGET_CHARS)

        # 取当前段
        current = remaining[:split_at].strip()
        parts.append(current)

        # 下一段：从重叠点开始
        overlap_start = max(0, split_at - OVERLAP_CHARS)
        # 回溯到句子边界
        for pattern, _name in SPLITTERS:
            sub = remaining[overlap_start:split_at]
            matches = list(re.finditer(pattern, sub))
            if matches:
                overlap_start += matches[-1].end()
                break

        remaining = remaining[overlap_start:].strip()

    if remaining.strip():
        parts.append(remaining.strip())

    return parts if parts else [text]


def split_chunks(input_path: str, dry_run: bool = True) -> list[dict]:
    """读取 chunks，拆分过长 chunk，返回新 chunk 列表。"""
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    stats = {"total": len(chunks), "split": 0, "kept": 0, "before": 0, "after": 0}
    new_chunks = []

    for c in chunks:
        tokens = c.get("tokens", estimate_tokens(c.get("text", "")))
        text = c.get("text", "")
        chunk_id = c.get("chunk_id", "?")
        header = c.get("header_path", "")
        doc_id = c.get("doc_id", "")

        if tokens <= MAX_TOKENS:
            c["tokens"] = tokens
            new_chunks.append(c)
            stats["kept"] += 1
            continue

        # 需要拆分
        sub_parts = split_text(text)
        if len(sub_parts) <= 1:
            c["tokens"] = tokens
            new_chunks.append(c)
            stats["kept"] += 1
            continue

        stats["split"] += 1
        for i, part_text in enumerate(sub_parts, 1):
            part_tokens = estimate_tokens(part_text)
            sub_header = f"{header} (part {i}/{len(sub_parts)})" if header else f"(part {i}/{len(sub_parts)})"
            new_chunks.append({
                "doc_id": doc_id,
                "header_path": sub_header,
                "text": part_text,
                "tokens": part_tokens,
                "chunk_id": f"{chunk_id}_s{i}",
            })

        if dry_run:
            print(f"  [SPLIT] {chunk_id}: {tokens}t → {len(sub_parts)} parts "
                  f"({', '.join(str(estimate_tokens(p))+'t' for p in sub_parts)})")

    stats["before"] = len(chunks)
    stats["after"] = len(new_chunks)

    print(f"\n  统计: {stats['before']} → {stats['after']} chunks "
          f"(拆分 {stats['split']}, 保留 {stats['kept']})")

    return new_chunks


def process_all_domains(dry_run: bool = True):
    """处理所有领域的 chunk 文件 + all_chunks.json。"""
    # 1. 处理各领域文件
    for domain_dir in sorted(CHUNKS_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        for chunk_file in sorted(domain_dir.glob("*.json")):
            print(f"\n[{'DRY RUN' if dry_run else 'APPLY'}] {chunk_file}")
            new_chunks = split_chunks(str(chunk_file), dry_run=dry_run)
            if not dry_run:
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(new_chunks, f, ensure_ascii=False, indent=2)

    # 2. 重新聚合 all_chunks.json
    if not dry_run:
        print(f"\n[APPLY] Rebuilding {ALL_CHUNKS_PATH}...")
        all_chunks = []
        for domain_dir in sorted(CHUNKS_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            for chunk_file in sorted(domain_dir.glob("*.json")):
                with open(chunk_file, "r", encoding="utf-8") as f:
                    all_chunks.extend(json.load(f))
        with open(ALL_CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"  {ALL_CHUNKS_PATH}: {len(all_chunks)} chunks")
    else:
        print(f"\n[DRY RUN] 预览 all_chunks.json...")
        all_chunks = []
        for domain_dir in sorted(CHUNKS_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            for chunk_file in sorted(domain_dir.glob("*.json")):
                with open(chunk_file, "r", encoding="utf-8") as f:
                    all_chunks.extend(json.load(f))
        split_result = split_chunks(str(ALL_CHUNKS_PATH), dry_run=True)
        print(f"  预计: {len(all_chunks)} → {len(split_result)} chunks")


if __name__ == "__main__":
    dry_run = "--apply" not in sys.argv
    if dry_run:
        print("预览模式（加 --apply 执行实际切分）\n")
    process_all_domains(dry_run=dry_run)

    if dry_run:
        print("\n确认无误后执行: python scripts/split_chunks.py --apply")
    else:
        print("\n切分完成！请重新构建索引: python scripts/build_index.py build")
