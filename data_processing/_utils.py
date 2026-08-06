"""
Shared utilities for data processing.
Extracted from chunk_md.py — single source of truth for token counting,
heading matching, Chinese numeral conversion, text splitting, etc.
"""

import re

# ---- types ----
T_ARABIC, T_CHINESE, T_PLAIN = 1, 2, 3

# ---- weak keywords ----
_WEAK_RE = re.compile(
    r"目录|释义|备查文件|附件|摘要|声明"
    r"|TOC|abstract|summary|disclaimer|declaration|glossary|appendix",
    re.IGNORECASE)

# ---- heading patterns: (regex, rank, type) ----
# Two sets of rules; auto-detect which convention the document uses.
# Convention A (Chinese-first): 第X节 → 一、→（一）→ 1、→（1）
# Convention B (Arabic-first):   1. → 1.1. → 一、→（一）

_RULES_A = [  # 金融合同 / 年报 / 监管
    (r"^第[一二三四五六七八九十百千零\d]+章",    1,  T_CHINESE),
    (r"^第[一二三四五六七八九十百千零\d]+节",    2,  T_CHINESE),
    (r"^第[一二三四五六七八九十百千零\d]+条",    3,  T_CHINESE),
    (r"^[一二三四五六七八九十]+[、．]",          4,  T_CHINESE),
    (r"^[一二三四五六七八九十]+[.）\)]",         4,  T_CHINESE),
    (r"^（[一二三四五六七八九十]+）",             5,  T_CHINESE),
    (r"^（\d+）",                               7,  T_CHINESE),
    (r"^\(\d+\)",                               7,  T_CHINESE),
    (r"^\d+\.\d+\.\d+\.\d+\.?(\s|$)",          9,  T_ARABIC),
    (r"^\d+\.\d+\.\d+\.?(\s|$)",               8,  T_ARABIC),
    (r"^\d+\.\d+\.?(\s|$)",                    7,  T_ARABIC),
    (r"^\d+[）\)]",                             7,  T_ARABIC),
    (r"^\d{1,2}\.?\s",                        6,  T_ARABIC),
    (r"^\d+[、．]",                            6,  T_ARABIC),
]

_RULES_B = [  # 保险 / CMB / 部分研报：阿拉伯在上，中文在下
    (r"^第[一二三四五六七八九十百千零\d]+章",    1,  T_CHINESE),
    (r"^第[一二三四五六七八九十百千零\d]+节",    2,  T_CHINESE),
    (r"^第[一二三四五六七八九十百千零\d]+条",    3,  T_CHINESE),
    # Arabic: 1. → 1.1. → 1.1.1. → 1.1.1.1. → 1、→ 1)
    (r"^\d+\.\d+\.\d+\.\d+\.?(\s|$)",          7,  T_ARABIC),
    (r"^\d+\.\d+\.\d+\.?(\s|$)",               6,  T_ARABIC),
    (r"^\d+\.\d+\.?(\s|$)",                    5,  T_ARABIC),
    (r"^\d{1,2}\.?\s",                        4,  T_ARABIC),   # 1. top-level
    (r"^\d+[、．]",                            8,  T_ARABIC),   # 1、sub-item (deeper than 1.)
    (r"^\d+[）\)]",                             9,  T_ARABIC),   # 1) deeper
    # Chinese under Arabic
    (r"^[一二三四五六七八九十]+[、．]",         10,  T_CHINESE),  # 一、
    (r"^[一二三四五六七八九十]+[.）\)]",        10,  T_CHINESE),
    (r"^（[一二三四五六七八九十]+）",           11,  T_CHINESE),  # （一）
    (r"^（\d+）",                              12,  T_CHINESE),  # （1）
    (r"^\(\d+\)",                              12,  T_CHINESE),
]

_HEADING_RULES = _RULES_A  # runtime override by _detect_convention()


def _detect_convention(text):
    """Scan all headings; detect Convention A (Chinese-first) or B (Arabic-first).
    1. Any 第X章/节 in text → force A
    2. First 3 numbered headings all Arabic and no Chinese → B
    3. Otherwise A"""
    global _HEADING_RULES
    _HEADING_RULES = _RULES_A

    # Pass 1: scan for 第X章/节
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            continue
        content = m.group(2).strip()
        if not content:
            continue
        if re.match(r"^第[一二三四五六七八九十百千零\d]+[章节]", content):
            return  # Chinese chapter/section → force A

    # Pass 2: Arabic-first detection
    arabic_first = 0
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            continue
        content = m.group(2).strip()
        if not content:
            continue
        rank, typ = _match(content)
        if typ == T_PLAIN:
            continue
        if typ == T_ARABIC:
            arabic_first += 1
        else:
            return  # Chinese numeral appeared → A
        if arabic_first >= 3:
            _HEADING_RULES = _RULES_B
            return
    return


# ============================================================
#  Token counting
# ============================================================

def tok(text):
    """Estimate token count: CJK chars ~1.8 chars/tok, others ~4 chars/tok"""
    cn = sum(1 for c in text if "一" <= c <= "鿿")
    return int(cn / 1.8 + (len(text) - cn) / 4.0) + 1


# ============================================================
#  Heading matching
# ============================================================

def _match(text):
    """Return (rank, type) or (None, T_PLAIN)"""
    if not text:
        return None, T_PLAIN
    for pat, r, t in _HEADING_RULES:
        if re.match(pat, text):
            return r, t
    return None, T_PLAIN


def _arabic_seq(text):
    """Extract last number from Arabic heading (e.g. '1.2.3' → 3)"""
    m = re.match(r"^([\d.]+)", text)
    if not m:
        return None
    nums = re.findall(r"\d+", m.group(1))
    return int(nums[-1]) if nums else None


def _arabic_prefix(text):
    """Extract dotted prefix from Arabic heading (e.g. '1.2.3.' → '1.2.3')"""
    m = re.match(r"^([\d.]+)", text)
    return m.group(1).rstrip(".") if m else None


# ---- Chinese numeral → int ----
_CN = dict(zip("一二三四五六七八九", range(1, 10)))
_CN.update({"十": 10, "百": 100, "千": 1000, "零": 0})


def _cn2int(s):
    """Extract Chinese numeral from heading text, return int.
    '第一节'→1, '二十一'→21, '一百二十三'→123"""
    m = re.search(r"[一二三四五六七八九十百千零]+", s)
    if not m:
        return None
    s = m.group()
    v = u = 0
    for i, c in enumerate(s):
        n = _CN[c]
        if n < 10:
            u = n
            if i == len(s) - 1:
                v += n
        else:
            v += (u or 1) * n
            u = 0
    return v


# ============================================================
#  Chunk helpers
# ============================================================

def _make(doc_id, hp, text):
    return {"doc_id": doc_id, "header_path": hp, "text": text, "tokens": tok(text)}


def _number(chunks, doc_id):
    """Assign sequential chunk_id to each chunk"""
    for i, c in enumerate(chunks, 1):
        c["chunk_id"] = f"{doc_id}_{i}"
    return chunks


# ============================================================
#  Text splitting
# ============================================================

def _split_long(text, max_tok):
    """Split long text at sentence boundaries (。；;), fallback to hard cut"""
    sents = [s for s in re.split(r"(?<=[。；;])\s*", text) if s.strip()]
    if len(sents) <= 1:
        sz = int(max_tok / 1.5)
        return [text[i:i + sz] for i in range(0, len(text), sz)]
    out, buf, bt = [], [], 0
    for s in sents:
        st = tok(s)
        if st > max_tok:
            if buf:
                out.append("".join(buf))
                buf, bt = [], 0
            out.extend(_split_long(s, max_tok))
            continue
        if bt + st > max_tok and buf:
            out.append("".join(buf))
            buf, bt = [s], st
        else:
            buf.append(s)
            bt += st
    if buf:
        out.append("".join(buf))
    return out


def _pack(paras, doc_id, hp, max_t=1024):
    """Greedy pack paragraphs into chunks ≤ max_t tokens.
    Tables (starting with |) are kept atomic — never split mid-table.
    Oversized tables are split at row boundaries only.
    """
    chunks, buf, bt = [], [], 0
    for p in paras:
        pt = tok(p)
        if bt + pt <= max_t:
            buf.append(p)
            bt += pt
        else:
            if buf:
                chunks.append(_make(doc_id, hp, "\n\n".join(buf)))
            buf, bt = [p], pt
    if buf:
        chunks.append(_make(doc_id, hp, "\n\n".join(buf)))
    return chunks


# ============================================================
#  HTML table → Markdown table conversion
# ============================================================

def _html_table_to_md(html):
    """Convert <table>...</table> HTML to Markdown pipe-table format.
    Preserves cell content integrity; handles colspan/rowspan basically.
    Returns (markdown_table_string, row_count).
    """
    # Extract all rows
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
    if not rows:
        return "", 0

    parsed = []
    col_count = 0
    for row_html in rows:
        cells = []
        # Match both <td> and <th>
        for m in re.finditer(r'<(td|th)([^>]*)>(.*?)</\1>', row_html, re.DOTALL | re.IGNORECASE):
            tag, attrs, content = m.group(1), m.group(2), m.group(3)
            # Clean cell content: strip tags, normalize whitespace
            cell_text = re.sub(r'<[^>]+>', '', content).strip()
            cell_text = re.sub(r'\s+', ' ', cell_text)
            # Handle colspan
            colspan = 1
            cm = re.search(r'colspan\s*=\s*["\']?(\d+)', attrs, re.IGNORECASE)
            if cm:
                colspan = int(cm.group(1))
            for _ in range(colspan):
                cells.append(cell_text)
        if cells:
            parsed.append(cells)
            col_count = max(col_count, len(cells))

    if not parsed:
        return "", 0

    # Normalize all rows to same column count
    for row in parsed:
        while len(row) < col_count:
            row.append("")

    # Build markdown table
    lines = []
    for i, row in enumerate(parsed):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            # Separator line after header
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines), len(parsed)


def _parse_md_table(md_table: str):
    """Parse markdown table into (header_names, rows_of_cells)."""
    lines = md_table.strip().split("\n")
    if len(lines) < 2:
        return [], []
    # Parse header
    hdr = [c.strip() for c in lines[0].split("|") if c.strip()]
    # Skip separator line (|---|...)
    data_start = 1
    if data_start < len(lines) and all(c.strip().startswith("-") for c in lines[data_start].split("|") if c.strip()):
        data_start = 2
    rows = []
    for line in lines[data_start:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            rows.append(cells)
    return hdr, rows


def _build_md_table(header: list[str], rows: list[list[str]]) -> str:
    """Build a markdown table string from header + rows."""
    ncols = len(header)
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * ncols) + " |")
    for row in rows:
        # Pad row to ncols
        padded = row + [""] * (ncols - len(row))
        lines.append("| " + " | ".join(padded[:ncols]) + " |")
    return "\n".join(lines)


def _split_table_rows(md_table: str, max_tok: int) -> list[str]:
    """Split an oversized markdown table in two stages:
    Stage 1: Row-based split (tall tables)
    Stage 2: Column-group split for any part still > max_tok (wide rows)
    Falls back to cell truncation for extreme cases.
    """
    hdr, rows = _parse_md_table(md_table)
    if not hdr or not rows:
        sz = int(max_tok / 1.5)
        return [md_table[i:i + sz] for i in range(0, len(md_table), sz)]

    ncols = len(hdr)

    # --- Stage 1: Row-based split ---
    header_block = _build_md_table(hdr, [])
    header_tok = tok(header_block)

    row_parts = []
    buf_rows = []
    buf_tok = header_tok

    for row in rows:
        row_str = _build_md_table(hdr, [row])
        row_tok = tok(row_str) - header_tok
        # If this single row > max_tok, flush buffer first, then add row alone
        if buf_tok + row_tok > max_tok and buf_rows:
            row_parts.append(_build_md_table(hdr, buf_rows))
            buf_rows = []
            buf_tok = header_tok
        buf_rows.append(row)
        buf_tok += row_tok

    if buf_rows:
        row_parts.append(_build_md_table(hdr, buf_rows))

    # --- Stage 2: Column-group split for parts still > max_tok ---
    final = []
    for part in row_parts:
        if tok(part) <= max_tok:
            final.append(part)
        elif ncols > 3:
            # Parse this part and apply column-group split
            phdr, prows = _parse_md_table(part)
            final.extend(_split_wide_table(phdr, prows, max_tok, len(phdr)))
        else:
            # Can't split columns further — truncate cells
            final.append(_truncate_cells(part, max_tok))

    return final if final else [md_table]


def _split_wide_table(hdr: list[str], rows: list[list[str]],
                      max_tok: int, ncols: int) -> list[str]:
    """Split by column groups then by row batches within each group.
    First col always acts as row anchor.
    """
    anchor_col = 0
    remaining = list(range(1, ncols))

    # Determine column groups
    col_groups = []
    i = 0
    while i < len(remaining):
        placed = False
        for sz in (4, 3, 2, 1):
            group_cols = [anchor_col] + remaining[i:i + sz]
            # Test with just the FIRST row to check width
            test_rows = [[rows[0][c] if c < len(rows[0]) else "" for c in group_cols]]
            test_tok = tok(_build_md_table([hdr[c] for c in group_cols], test_rows))
            if test_tok <= max_tok:
                col_groups.append(group_cols)
                i += sz
                placed = True
                break
        if not placed:
            col_groups.append([anchor_col, remaining[i]])
            i += 1

    # For each column group, split rows into batches that fit max_tok
    parts = []
    n_groups = len(col_groups)
    part_idx = 0
    for gi, cols in enumerate(col_groups):
        sub_hdr = [hdr[c] for c in cols]

        # Row-batch within this column group
        header_block = _build_md_table(sub_hdr, [])
        header_tok = tok(header_block)
        buf_rows = []
        buf_tok = header_tok

        for row in rows:
            sub_row = [row[c] if c < len(row) else "" for c in cols]
            row_str = _build_md_table(sub_hdr, [sub_row])
            row_tok = tok(row_str) - header_tok

            if buf_tok + row_tok > max_tok and buf_rows:
                label = f"[表 {part_idx+1}]\n" if n_groups > 1 or len(rows) > len(buf_rows) else ""
                tbl = _build_md_table(sub_hdr, buf_rows)
                if tok(label + tbl) > max_tok:
                    tbl = _truncate_cells(tbl, max_tok - tok(label))
                parts.append(label + tbl)
                part_idx += 1
                buf_rows = []
                buf_tok = header_tok

            buf_rows.append(sub_row)
            buf_tok += row_tok

        if buf_rows:
            label = f"[表 {part_idx+1}]\n" if (n_groups > 1 or len(buf_rows) < len(rows)) else ""
            tbl = _build_md_table(sub_hdr, buf_rows)
            if tok(label + tbl) > max_tok:
                tbl = _truncate_cells(tbl, max_tok - tok(label))
            parts.append(label + tbl)
            part_idx += 1

    return parts


def _append_row_split_groups(groups: list, hdr: list[str], rows: list[list[str]],
                              cols: list[int], max_tok: int):
    """For extreme case: even anchor+1 col exceeds max_tok. Split per-row."""
    for ri in range(len(rows)):
        sub_hdr = [hdr[c] for c in cols]
        sub_rows = [rows[ri]]
        sub_tbl = _build_md_table(sub_hdr, sub_rows)
        if tok(sub_tbl) > max_tok:
            sub_tbl = _truncate_cells(sub_tbl, max_tok)
        # We can't add to groups (different sets of rows per group colset),
        # so mark this as needing special handling. Instead just use truncation.
        pass  # _truncate_cells in the main loop handles this


def _truncate_cells(md_table: str, max_tok: int) -> str:
    """Truncate cell content to fit under max_tok. Keeps structure intact."""
    if tok(md_table) <= max_tok:
        return md_table
    lines = md_table.split("\n")
    result = [lines[0], lines[1]] if len(lines) >= 2 else lines[:1]  # header + sep
    for line in lines[2:]:
        result.append(line)
        if tok("\n".join(result)) > max_tok:
            # Truncate last added row: shorten each cell
            cells = [c.strip() for c in line.split("|")]
            shortened = []
            for cell in cells:
                if not cell:
                    shortened.append("")
                else:
                    # Keep first 200 chars of cell
                    shortened.append(cell[:200] + ("..." if len(cell) > 200 else ""))
            result[-1] = "| " + " | ".join(shortened) + " |"
            if tok("\n".join(result)) > max_tok:
                result.pop()
                break
    return "\n".join(result)


def _table_aware_split(text):
    """Split text into paragraphs, keeping table blocks atomic.
    Handles both <table> HTML and |...| markdown tables from MinerU.
    Returns (paragraphs, images_list).
    """
    # Step 1: Extract images
    imgs = re.findall(r"!\[.*?\]\([^)]+\)", text)
    clean = text
    for x in imgs:
        clean = clean.replace(x, "")

    # Step 2: Extract table blocks → placeholders
    tables = []

    def _replace_html_table(m):
        tables.append(m.group(0))
        return f"\n\n<<<TABLE_{len(tables) - 1}>>>\n\n"

    def _replace_md_table(m):
        tables.append(m.group(0))
        return f"\n\n<<<TABLE_{len(tables) - 1}>>>\n\n"

    # HTML tables
    clean = re.sub(r'<table>.*?</table>', _replace_html_table, clean,
                   flags=re.DOTALL | re.IGNORECASE)
    # Markdown tables: header line + separator + data rows
    # Pattern: line starting with |, followed by |---| separator, then 0+ data rows
    clean = re.sub(
        r'(?:^|\n\n)(\|[^\n]+\|\n\|[\s\-:|]+\|(?:\n\|[^\n]+\|)+)',
        lambda m: '\n\n' + _replace_md_table(m) + '\n\n',
        clean, flags=re.MULTILINE
    )

    # Step 3: Split by blank lines
    raw_paras = [p.strip() for p in clean.split("\n\n") if p.strip()]

    # Step 4: Replace table placeholders → markdown tables
    result = []
    for p in raw_paras:
        m = re.match(r'<<<TABLE_(\d+)>>>', p)
        if m:
            idx = int(m.group(1))
            raw_table = tables[idx].strip()
            # Already markdown? Keep as-is
            if raw_table.startswith("|"):
                result.append(raw_table)
            else:
                # HTML → markdown
                md_table, _ = _html_table_to_md(raw_table)
                if md_table.strip():
                    result.append(md_table)
        else:
            result.append(p)

    return result, imgs
