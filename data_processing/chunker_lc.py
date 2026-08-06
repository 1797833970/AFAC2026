"""
Markdown 分块（LangChain 版）
=============================
1. 用 chunk_md 的 rank 规则把平级 ## 改写成多级 #
2. 交给 MarkdownHeaderTextSplitter 分块
3. 二次切分 + token 控制

安装: pip install langchain-text-splitters

Moved from chunk_lc.py into data_processing package.
Duplicated functions (tok, _split_long, _number) removed — imported from _utils.
"""

import re, json
from pathlib import Path

from data_processing._utils import (
    tok, _match, _number, _split_long, _WEAK_RE,
    _table_aware_split, _split_table_rows,
)

# ---- config ----
MD_DIR   = "data/processed_md"
OUT_DIR  = "data/chunks_lc"
TARGET, MIN_T, MAX_T = 512, 80, 1024


# ============================================================
#  Post-chunk processing: secondary split + token control
# ============================================================

def _post_chunk(doc_id, hp, text):
    """Split a coarse chunk: table-aware paragraph split → sentence split → token pack"""
    # Table-aware split: tables stay in text as markdown, only images extracted
    paras, imgs = _table_aware_split(text)

    if not paras:
        return []

    flat = []
    for p in paras:
        pt = tok(p)
        if pt > MAX_T:
            # Is it a markdown table?
            if p.startswith("|"):
                flat.extend(_split_table_rows(p, MAX_T))
            else:
                flat.extend(_split_long(p, MAX_T))
        else:
            flat.append(p)

    chunks = []
    buf, bt = [], 0
    for p in flat:
        pt = tok(p)
        if bt + pt <= MAX_T:
            buf.append(p)
            bt += pt
        else:
            if buf:
                c = {"doc_id": doc_id, "header_path": hp,
                     "text": "\n\n".join(buf), "tokens": bt}
                if imgs:
                    c["images"] = list(imgs)
                chunks.append(c)
            buf, bt = [p], pt
    if buf:
        c = {"doc_id": doc_id, "header_path": hp,
             "text": "\n\n".join(buf), "tokens": bt}
        if imgs:
            c["images"] = list(imgs)
        chunks.append(c)

    # merge short chunks within section
    merged = []
    for c in chunks:
        if c["tokens"] < MIN_T and merged:
            comb = tok(merged[-1]["text"] + "\n\n" + c["text"])
            if comb <= MAX_T:
                merged[-1]["text"] += "\n\n" + c["text"]
                merged[-1]["tokens"] = comb
                continue
        merged.append(c)
    return merged


# ============================================================
#  Heading depth normalization: ## → #～###### according to rank stack
# ============================================================

def _normalize(text):
    """
    Scan all markdown heading lines, use chunk_md's rank stack to determine
    true depth, replace original # count with stack depth.
    """
    lines = text.split("\n")
    out = []
    stack = []          # [(rank, is_weak)]

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            out.append(line)
            continue

        content = m.group(2).strip()
        if not content:
            out.append(line)
            continue

        # weak heading detection (strip spaces)
        no_space = content.replace(" ", "").replace("　", "")
        is_weak = bool(_WEAK_RE.search(no_space))
        rank, _ = _match(content)

        # ---- stack update (same logic as chunk_md) ----
        if rank is None:                         # PLAIN / weak
            if is_weak:
                stack.append((None, True))
            elif stack and stack[-1][0] is None and not stack[-1][1]:
                # consecutive non-weak PLAIN merged (placeholder, depth unchanged)
                pass
            else:
                stack.append((None, False))
        else:                                    # numbered heading
            while stack and stack[-1][0] is None:    # pop PLAIN
                stack.pop()
            while stack and stack[-1][0] is not None and stack[-1][0] >= rank:
                stack.pop()
            stack.append((rank, False))

        depth = len(stack)
        out.append(f"{'#' * depth} {content}")

    return "\n".join(out)


# ============================================================
#  Chunk
# ============================================================

def chunk_markdown(doc_id, full_text):
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    # 1. Normalize: flat ## → true multi-level headings
    normalized = _normalize(full_text)

    # 2. LangChain split by heading level
    headers = [(f"{'#' * i} ", f"H{i}") for i in range(1, 7)]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    )
    docs = splitter.split_text(normalized)

    # 3. Build header_path + secondary split
    out = []
    for doc in docs:
        meta = doc.metadata
        parts = []
        for i in range(1, 7):
            v = meta.get(f"H{i}", "").strip()
            if v:
                parts.append(v)
        hp = " > ".join(parts)

        sub = _post_chunk(doc_id, hp, doc.page_content)
        out.extend(sub)

    if not out:
        out = _post_chunk(doc_id, "", full_text)

    return _number(out, doc_id)


# ============================================================
#  Main entry point
# ============================================================

def run():
    od = Path(OUT_DIR)
    od.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    md_root = Path(MD_DIR)

    for fp in sorted(Path(MD_DIR).glob("**/*.md")):
        rel = fp.relative_to(md_root)
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fp.read_text(encoding="gbk")
        cs = chunk_markdown(fp.stem, text)
        out_sub = od / str(rel.parent)
        out_sub.mkdir(parents=True, exist_ok=True)
        json.dump(cs, (out_sub / f"{fp.stem}.json").open("w", encoding="utf-8"),
                  ensure_ascii=False)
        avg = sum(c["tokens"] for c in cs) // max(len(cs), 1)
        print(f"  {rel!s:50s} {len(cs):3d} chunks  (avg {avg:4d} tok)")
        all_chunks.extend(cs)

    json.dump(all_chunks, (od / "all_chunks.json").open("w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"Done: {len(all_chunks)} chunks")


if __name__ == "__main__":
    run()
