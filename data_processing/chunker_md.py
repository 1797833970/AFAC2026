"""
Markdown / TXT 分块引擎
=======================
规则见 chunks_rules.md

Moved from chunk_md.py into data_processing package.
All shared utilities imported from data_processing._utils.
"""

import re, json
from pathlib import Path

from data_processing._utils import (
    tok, _match, _number, _split_long, _pack, _make,
    _detect_convention, _cn2int, _arabic_seq, _arabic_prefix,
    _table_aware_split, _split_table_rows,
    _WEAK_RE, T_ARABIC, T_CHINESE, T_PLAIN,
)

# ---- config ----
MD_DIR   = "data/processed_md"
TXT_DIR  = "data/raw/raw/regulatory/txt"
OUT_DIR  = "data/chunks"
TARGET, MIN_T, MAX_T = 350, 100, 400


# ============================================================
#  Markdown chunking
# ============================================================

def chunk_markdown(doc_id, full_text):
    _detect_convention(full_text)        # auto-select A/B rules
    lines = full_text.split("\n")

    # ===== pass 1: section list =====
    # sec: {rank, title, lines, type, seq, prefix, is_weak}
    secs = []
    cur = {"rank": None, "title": "", "lines": [], "type": T_PLAIN,
           "seq": None, "prefix": None, "is_weak": False}

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if not m:
            s = line.strip()
            cur["lines"].append(s if s else "")
            continue

        content = m.group(2).strip()
        if not content:                          # empty heading → treat as body
            cur["lines"].append(line.strip())
            continue

        # save previous section
        if cur["title"]:
            secs.append(cur)

        # classify new section
        is_weak = bool(_WEAK_RE.search(content.replace(" ", "").replace("　", "")))
        rank, typ = _match(content)

        cur = {"title": content, "lines": [], "is_weak": is_weak, "type": typ}

        if typ == T_ARABIC:
            cur["rank"] = rank
            cur["seq"] = _arabic_seq(content)
            cur["prefix"] = _arabic_prefix(content)
        elif typ == T_CHINESE:
            cur["rank"] = rank
            cur["seq"] = _cn2int(content)
            cur["prefix"] = None
        else:
            cur["rank"] = None
            cur["seq"] = None
            cur["prefix"] = None

    if cur["title"]:                             # last section
        secs.append(cur)

    # ===== pass 2: stack + pack =====
    stack = []          # [(title, rank, is_weak)]
    results = []        # [{chunks, tok, rank, type, seq, first_disp, last_disp, is_weak, hp}]

    for sec in secs:
        rank, title = sec["rank"], sec["title"]
        is_weak = sec["is_weak"]

        # ---- update stack ----
        if rank is None:                         # PLAIN / weak
            if is_weak:
                stack.append((title, None, True))      # weak: separate, no merge
            elif stack and stack[-1][1] is None and not stack[-1][2]:
                stack[-1] = (stack[-1][0] + " + " + title, None, False)  # non-weak PLAIN merge
            else:
                stack.append((title, None, False))
        else:                                    # numbered heading
            while stack and stack[-1][1] is None:    # pop PLAIN (incl weak)
                stack.pop()
            while stack and stack[-1][1] is not None and stack[-1][1] >= rank:
                stack.pop()
            stack.append((title, rank, False))

        # ---- hp ----
        if is_weak:
            hp = title                             # weak: direct title
        else:
            n_plain = sum(1 for _, r, _ in stack if r is None)
            if n_plain == len(stack):
                if len(stack) == 1 and " + " not in stack[0][0]:
                    hp = stack[0][0]               # single non-weak PLAIN → own title
                else:
                    hp = ""                        # multiple PLAIN → ""
            else:
                hp = " > ".join(t for t, r, _ in stack if r is not None)  # numbered titles form hp

        slines = [title] + sec["lines"]             # title always in text
        if len(slines) == 1 and not slines[0]:
            continue

        raw = "\n".join(slines)
        raw = re.sub(r'\n{3,}', '\n\n', raw)         # compress excessive blank lines

        # ---- table-aware split: tables stay in text as markdown, only images extracted ----
        paras, imgs = _table_aware_split(raw)
        extra = {}
        if imgs:
            extra["images"] = imgs

        # only title left, no content → skip
        if len(paras) == 1 and paras[0] == title and not extra:
            continue
        if not paras:
            if extra:
                paras = [""]
            else:
                continue

        flat = []
        for p in paras:
            pt = tok(p)
            if pt > MAX_T:
                # Is it a markdown table? (starts with |)
                if p.startswith("|"):
                    flat.extend(_split_table_rows(p, MAX_T))
                else:
                    flat.extend(_split_long(p, MAX_T))
            else:
                flat.append(p)
        if not flat and extra:
            flat = [""]

        chunks = _pack(flat, doc_id, hp, MAX_T)

        # section-internal merge (short chunks)
        merged = []
        for c in chunks:
            if c["tokens"] < MIN_T and merged:
                comb = tok(merged[-1]["text"] + "\n\n" + c["text"])
                if comb <= MAX_T:
                    merged[-1]["text"] += "\n\n" + c["text"]
                    merged[-1]["tokens"] = comb
                    continue
            merged.append(c)

        for c in merged:
            if imgs:
                c["images"] = list(imgs)  # each chunk gets its own copy
        if is_weak:
            for c in merged:
                c["_weak"] = True

        # display range: ARABIC uses prefix, CHINESE uses full title
        disp = sec["prefix"] if sec["type"] == T_ARABIC else (
               title if sec["type"] == T_CHINESE else None)

        results.append({
            "chunks": merged,
            "tok": sum(c["tokens"] for c in merged),
            "rank": rank,
            "type": sec["type"],
            "seq": sec["seq"],
            "first_disp": disp,
            "last_disp": disp,
            "is_weak": is_weak,
            "hp": hp,
        })

    # ================================================================
    #  Same-rank merge (iterative to convergence)
    #  Adjacent, same rank, same parent, consecutive seq → merge;
    #  full coverage → promote hp to parent
    # ================================================================

    def _parent(hp):
        parts = hp.split(" > ")
        return " > ".join(parts[:-1]) if len(parts) > 1 else ""

    # init labels / last_seq
    for r in results:
        r["labels"] = [r["first_disp"]] if r["first_disp"] else []
        r["last_seq"] = r["seq"]

    # iterate until no adjacent pair can be merged
    changed = True
    while changed:
        changed = False
        nxt_list = []
        i = 0
        while i < len(results):
            a = results[i]

            # skip weak headings
            if a["is_weak"]:
                nxt_list.append(a)
                i += 1
                continue

            # try merge with next
            if i + 1 < len(results):
                b = results[i + 1]
                # Condition 1: same rank, same parent, consecutive seq → peer merge
                same_rank = (not b["is_weak"] and b["rank"] == a["rank"]
                      and a["rank"] is not None and b["rank"] is not None
                      and _parent(a["hp"]) == _parent(b["hp"])
                      and a["last_seq"] is not None and b["seq"] is not None
                      and b["seq"] == a["last_seq"] + 1)
                # Condition 2: same hp (PLAIN under numbered heading) → direct merge
                same_hp = (a["hp"] == b["hp"] and a["hp"] != "" and not b["is_weak"])
                if (same_rank or same_hp) and a["tok"] + b["tok"] <= MAX_T:

                    # ---- merge a ← b (text + images) ----
                    txt = "\n\n".join(c["text"] for c in a["chunks"])
                    txt += "\n\n" + "\n\n".join(c["text"] for c in b["chunks"])
                    a["chunks"] = a["chunks"][:1]
                    a["chunks"][0]["text"] = txt
                    a["chunks"][0]["tokens"] = tok(txt)
                    for cc in b["chunks"]:
                        if "images" in cc:
                            a["chunks"][0].setdefault("images", []).extend(cc["images"])
                    a["tok"] = tok(txt)

                    if same_rank:
                        # Peer merge: update labels / hp / full-coverage
                        a["labels"].extend(b["labels"])
                        a["last_seq"] = b["last_seq"]
                        a["last_disp"] = b["last_disp"]
                        parts = a["hp"].split(" > ")
                        parts[-1] = " + ".join(a["labels"])
                        a["hp"] = " > ".join(parts)

                        rest = nxt_list + [a] + results[i + 2:]
                        p = _parent(a["hp"])
                        other = any(
                            r["rank"] == a["rank"] and _parent(r["hp"]) == p
                            for r in rest if r is not a and not r["is_weak"]
                        )
                        if not other and p:
                            a["hp"] = p
                            last_seg = p.split(" > ")[-1]
                            prank, ptype = _match(last_seg)
                            a["rank"] = prank
                            if ptype != T_PLAIN:
                                a["type"] = ptype
                            a["first_disp"] = last_seg
                            a["last_disp"]  = last_seg
                            a["labels"] = [last_seg] if ptype != T_PLAIN else []
                            if ptype == T_ARABIC:
                                a["seq"] = _arabic_seq(last_seg)
                            elif ptype == T_CHINESE:
                                a["seq"] = _cn2int(last_seg)
                            else:
                                a["seq"] = None
                            a["last_seq"] = a["seq"]
                    # same_hp: hp/rank/labels unchanged, pure text absorption

                    a["chunks"][0]["header_path"] = a["hp"]  # sync to chunk
                    nxt_list.append(a)
                    i += 2
                    changed = True
                    continue

            nxt_list.append(a)
            i += 1

        results = nxt_list

    # ---- PLAIN merge ----
    merged = []
    for sec in results:
        if (merged and sec["type"] == T_PLAIN and merged[-1]["type"] == T_PLAIN
              and not sec["is_weak"] and not merged[-1]["is_weak"]
              and merged[-1]["tok"] + sec["tok"] <= MAX_T):
            need_prepend = (merged[-1]["chunks"][0]["header_path"] != "")
            first_hp = merged[-1]["hp"]
            # absorb
            txt = "\n\n".join(c["text"] for c in merged[-1]["chunks"])
            txt += "\n\n" + "\n\n".join(c["text"] for c in sec["chunks"])
            merged[-1]["chunks"] = merged[-1]["chunks"][:1]
            merged[-1]["chunks"][0]["text"] = txt
            merged[-1]["chunks"][0]["tokens"] = tok(txt)
            merged[-1]["tok"] = tok(txt)
            if need_prepend and first_hp:
                t = merged[-1]["chunks"][0]
                t["text"] = first_hp + "\n\n" + t["text"]
                t["tokens"] = tok(t["text"])
            merged[-1]["chunks"][0]["header_path"] = ""
            merged[-1]["hp"] = ""
            continue
        merged.append(sec)

    # flatten + clean + remove empty chunks
    out = []
    for sec in merged:
        for c in sec["chunks"]:
            c.pop("_weak", None)
            # Skip empty chunks but NEVER those with tables or images
            if _is_empty_chunk(c):
                continue
            out.append(c)
    return _number(out, doc_id)


def _is_empty_chunk(c: dict) -> bool:
    """Check if chunk is effectively empty/meaningless.
    NEVER remove chunks containing tables or images.
    """
    import re as _re
    # Safety: never remove chunks with tables or images
    if "images" in c:
        return False
    text = c.get("text", "").strip()
    if "|" in text and "---" in text:
        return False

    # Remove boilerplate + pipe separators
    cleaned = text
    for m in ["无。", "无．", "不适用"]:
        cleaned = cleaned.replace(m, "")
    cleaned = _re.sub(r'[□]\s*适用', '', cleaned)
    cleaned = _re.sub(r'[□]\s*不适用', '', cleaned)
    cleaned = cleaned.replace("|", "").strip()

    # Empty after cleaning -> remove
    if len(cleaned) < 10:
        return True
    # TOC-style: >=2 page number references like ".. 309"
    if len(_re.findall(r'\.\.\s*\d+', cleaned)) >= 2:
        return True
    return False



# ============================================================
#  TXT chunking (by 第X条 articles)
# ============================================================

_ARTICLE_RE = re.compile(r"^第[一-鿿\d]+条[\s　]*")

def chunk_txt_articles(doc_id, full_text):
    lines = full_text.split("\n")
    arts, ct, cb = [], "", []
    for line in lines:
        s = line.strip()
        if _ARTICLE_RE.match(s):
            if ct or cb:
                arts.append((ct, cb))
            ct = s
            cb = []
        else:
            cb.append(s if s else "")
    if ct or cb:
        arts.append((ct, cb))

    chunks = []
    for title, body in arts:
        text = title + "\n" + "\n".join(body)
        hp = title
        if tok(text) <= MAX_T:
            chunks.append(_make(doc_id, hp, text))
            continue
        paras = [p.strip() for p in "\n".join(body).split("\n\n") if p.strip()]
        if len(paras) <= 1:
            paras = [p.strip() for p in "\n".join(body).split("\n") if p.strip()]
        flat = []
        for p in paras:
            if tok(p) > MAX_T:
                flat.extend(_split_long(p, MAX_T))
            else:
                flat.append(p)
        chunks.extend(_pack(flat, doc_id, hp, MAX_T))
    return _number(chunks, doc_id)


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

    for fp in sorted(Path(TXT_DIR).glob("*.txt")):
        try:
            text = fp.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = fp.read_text(encoding="gbk")
        cs = chunk_txt_articles(fp.stem, text)
        out_sub = od / "regulatory/txt"
        out_sub.mkdir(parents=True, exist_ok=True)
        json.dump(cs, (out_sub / f"{fp.stem}.json").open("w", encoding="utf-8"),
                  ensure_ascii=False)
        avg = sum(c["tokens"] for c in cs) // max(len(cs), 1)
        print(f"  regulatory/txt/{fp.stem}.txt   {len(cs):3d} chunks  (avg {avg:4d} tok)")
        all_chunks.extend(cs)

    json.dump(all_chunks, (od / "all_chunks.json").open("w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"Done: {len(all_chunks)} chunks")


if __name__ == "__main__":
    run()
