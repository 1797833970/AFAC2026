"""
PDF/TXT/HTML → 按标题层级构建树形结构，同时输出平坦段落。

Merged from process_docs.py + src/document_processor.py into data_processing package.

Two parsing strategies:
  - Rich (process_docs): word grouping, font size, header/footer filtering (better for PDF)
  - Simple (document_processor): line-by-line heading detection (better for TXT/HTML)

Usage:
  python -m data_processing.document_parser          # process all
  python -m data_processing.document_parser check    # validate doc mapping
  python -m data_processing.document_parser qa       # show question-answer mapping
"""

import re, json
from pathlib import Path
from collections import Counter
import pdfplumber


# ===== Domain constants (from src/document_processor.py) =====
DOMAINS = ["insurance", "regulatory", "financial_contracts", "financial_reports", "research"]
_DOMAIN_CN = {"insurance": "保险条款", "regulatory": "监管法规",
              "financial_contracts": "金融合同", "financial_reports": "财务报表",
              "research": "行业研报"}


# ===== Heading patterns (from process_docs.py — rich) =====
PATTERNS = [
    (1, r'^\s*#?\s*(\d+)\.\s+([^\d].{3,})$'),           # 1. 标题
    (2, r'^\s*#?\s*(\d+\.\d+)\.\s*(.{3,})$'),          # 1.1
    (3, r'^\s*#?\s*(\d+\.\d+\.\d+)\.\s*(.{3,})$'),   # 1.1.1
    (4, r'^\s*#?\s*(\d+\.\d+\.\d+\.\d+)\.\s*(.{3,})$'), # 1.1.1.1
    (1, r'^\s*第[一二三四五六七八九十百零\d]+章[、\s\.]+(.+)$'),
    (2, r'^\s*第[一二三四五六七八九十百零\d]+条[、\s\.]+(.+)$'),
    (3, r'^\s*第[一二三四五六七八九十\d]+类[：:]\s*(.{2,})$'),
    (1, r'^\s*[一二三四五六七八九十]+[、]\s*(.{2,})$'),
    (3, r'^\s*[（(][一二三四五六七八九十\d]+[）)]\s*(.{2,})$'),
]

# Heading patterns (from src/document_processor.py — simple)
HEADER_PATTERNS_SIMPLE = [
    (re.compile(r"^第[一二三四五六七八九十百千\d]+[章节条]\s+"), 1),
    (re.compile(r"^第\s*\d+\s*条\s+"), 1),
    (re.compile(r"^\(\s*[一二三四五六七八九十百千\d]+\s*\)\s+"), 2),
    (re.compile(r"^(\d+\.)+\d+\s+[^\d]"), 3),
    (re.compile(r"^\d+\.\d+\s+[^\d\s]"), 3),
    (re.compile(r"^[A-Z]\.[\s]\s*[A-Z一-鿿]"), 4),
]

# List patterns (not headings)
LIST_PATTERNS = [
    re.compile(r"^\s*（\d+）\s*"),
    re.compile(r"^\s*[\(（]\d+[\)）]\s*"),
    re.compile(r"^\s*[a-zA-Z][\.．]\s*"),
    re.compile(r"^\s*[①-⑩]\s*"),
    re.compile(r"^\s*\d+[、,，]\s*"),
]

# Page-level header/footer patterns to filter out
HEADER_FILTER_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*\d{1,4}\s*/\s*\d{1,4}\s*$"),
    re.compile(r"请扫描以查询验证条款"),
    re.compile(r"请务必阅读正文之后的免责条款"),
]


# ===== Section class (from process_docs.py — enhanced) =====
class Section:
    def __init__(self, level=0, title="", content=""):
        self.level = level
        self.title = title
        self.content = content
        self.children = []

    def to_dict(self):
        return {"level": self.level, "title": self.title,
                "content": self.content,
                "children": [c.to_dict() for c in self.children]}

    def to_flat(self, parent_path=""):
        path = f"{parent_path} > {self.title}" if parent_path and self.title else (self.title or parent_path)
        flat = [{"header_path": path, "title": self.title,
                 "level": self.level, "content": self.content}]
        for c in self.children:
            flat.extend(c.to_flat(path))
        return flat


# ===== Rich heading detection (from process_docs.py) =====
def _match_heading(text, body_size=10.5, font_size=10.5):
    text = text.strip()
    if not text or len(text) > 200:
        return False, 0, ""
    for p in LIST_PATTERNS:
        if p.match(text):
            return False, 0, ""
    for level, pattern in PATTERNS:
        m = re.match(pattern, text)
        if not m:
            continue
        title = m.groups()[-1].strip()
        if len(title) < 3:
            continue
        if font_size == body_size or (font_size / body_size >= {1: 1.15, 2: 1.08, 3: 1.05, 4: 1.02}.get(level, 1.02)):
            return True, level, title
    return False, 0, ""


# ===== Simple heading detection (from src/document_processor.py) =====
def is_header(line, patterns=None):
    if patterns is None:
        patterns = HEADER_PATTERNS_SIMPLE
    for pat, lv in patterns:
        if pat.match(line):
            return True, lv
    return False, 0


# ===== PDF parsing =====
def _group_words(words):
    """Group raw pdfplumber word dicts into (text, avg_font_size) lines."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"] / 3), w["x0"]))
    lines, cur, cur_top = [], [], round(words[0]["top"] / 3)
    for w in words:
        t = round(w["top"] / 3)
        if t == cur_top:
            cur.append(w)
        else:
            if cur:
                cur.sort(key=lambda x: x["x0"])
                sizes = [x.get("size", 0) for x in cur if x.get("size")]
                lines.append(("".join(x["text"] for x in cur),
                             sum(sizes) / len(sizes) if sizes else 10.5))
            cur, cur_top = [w], t
    if cur:
        cur.sort(key=lambda x: x["x0"])
        sizes = [x.get("size", 0) for x in cur if x.get("size")]
        lines.append(("".join(x["text"] for x in cur),
                     sum(sizes) / len(sizes) if sizes else 10.5))
    return lines


# ===== Tree building (from process_docs.py) =====
def _build_tree(lines_data, body_size=10.5):
    root = Section(level=0, title="文档")
    stack = [root]
    for text, fs in lines_data:
        h, lv, title = _match_heading(text, body_size, fs)
        if h:
            sec = Section(level=lv, title=title)
            while stack and stack[-1].level >= lv:
                stack.pop()
            if stack:
                stack[-1].children.append(sec)
            else:
                root.children.append(sec)
            stack.append(sec)
        else:
            stack[-1].content += text + "\n"
    return root


def parse_pdf(path):
    """Rich PDF parsing: word grouping, font size, header/footer filtering."""
    all_pages = []
    counter = Counter()
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            ws = p.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
            all_pages.append(ws)
            seen = set()
            for w in ws:
                t = w["text"].strip()
                if 0 < len(t) < 40:
                    seen.add(t)
            for t in seen:
                counter[t] += 1
    total = len(all_pages)
    hf_texts = {t for t, c in counter.items()
                if c >= total * 0.7 and not re.search(r"[，。；！？一-龥]{5,}", t)}

    all_lines = []
    with pdfplumber.open(path) as pdf:
        for pi, p in enumerate(pdf.pages):
            ws = all_pages[pi]
            if not ws:
                continue
            sizes = [w.get("size", 0) for w in ws if w.get("size")]
            bs = Counter(sizes).most_common(1)[0][0] if sizes else 10.5
            for text, fs in _group_words(ws):
                if text.strip() in hf_texts:
                    continue
                for hfp in HEADER_FILTER_PATTERNS:
                    if hfp.search(text):
                        break
                else:
                    all_lines.append((text, fs))

    root = _build_tree(all_lines, bs)
    tree = [root.to_dict()]
    flat = root.to_flat()
    return tree, flat


def parse_txt(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f.read().split("\n") if l.strip()]
    root = _build_tree([(l, 10.5) for l in lines])
    return [root.to_dict()], root.to_flat()


def parse_html(path):
    text = open(path, "r", encoding="utf-8").read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"&[a-z]+;", "", text)
    root = _build_tree([(l.strip(), 10.5) for l in text.split("\n") if l.strip()])
    return [root.to_dict()], root.to_flat()


# ===== Simple section parsing (from src/document_processor.py) =====
def parse_sections(lines, patterns=None):
    """Simple line-by-line heading detection and section building."""
    secs = []
    cur = Section()
    for line in lines:
        h, lv = is_header(line, patterns)
        if h:
            if cur.content or cur.title:
                secs.append(cur)
            cur = Section()
            cur.title = line
            cur.level = lv
        else:
            cur.content += line + "\n"
    if cur.content or cur.title:
        secs.append(cur)
    if len(secs) >= 2 and not secs[0].title:
        secs[1].content = secs[0].content + secs[1].content
        secs.pop(0)
    return secs


# ===== File processing =====
def process_file(path, doc_id):
    """Process a single file: auto-detect format, return tree + flat (rich mode)."""
    p = Path(path)
    s = p.suffix.lower()
    tree, flat = {"txt": parse_txt, "html": parse_html}.get(s, parse_pdf)(path)
    return {"doc_id": doc_id, "file": str(p),
            "total_sections": len([x for x in flat if x["title"]]),
            "section_tree": tree, "flat_sections": flat}


def process_one(filepath, doc_id, domain):
    """Process a single file with domain info (simple mode, from document_processor)."""
    path = Path(filepath)
    suf = path.suffix.lower()
    lines = {"txt": parse_txt, "html": parse_html}.get(suf, parse_pdf)(filepath)
    # Use the flat output from rich parser
    _, flat = {"txt": parse_txt, "html": parse_html}.get(suf, parse_pdf)(filepath)
    return {"doc_id": doc_id, "domain": domain, "domain_cn": _DOMAIN_CN.get(domain, ""),
            "file": filepath, "total_lines": len(lines),
            "total_sections": len([x for x in flat if x["title"]]),
            "engine": suf, "sections": [{"header": s["title"], "level": s["level"],
                                         "text": s["content"],
                                         "line_count": s["content"].count("\n"),
                                         "char_count": len(s["content"])}
                                        for s in flat]}


# ===== File mapping (unified) =====
def build_mapping(raw_dir="data/raw/raw", domains=None):
    """Build doc_id -> filepath mapping.
    If domains is None, walks all subdirectories.
    If domains is a list, only walks those domain subdirectories.
    """
    mapping = {}
    rd = Path(raw_dir)
    if domains:
        dirs = [(rd / d, True) for d in domains if (rd / d).exists()]
    else:
        dirs = [(rd / d.name, True) for d in rd.iterdir() if d.is_dir()]

    for dpath, _ in dirs:
        for f in sorted(dpath.glob("*")):
            if f.suffix.lower() in (".pdf", ".txt", ".html"):
                mapping[f.stem] = str(f)
        for sub in ["txt", "html", "attachments"]:
            sp = dpath / sub
            if sp.is_dir():
                for f in sorted(sp.glob("*")):
                    if f.suffix.lower() in (".pdf", ".txt", ".html"):
                        mapping[f.stem] = str(f)
    return mapping


def validate_mapping(raw_dir="data/raw/raw"):
    """Verify that all doc_ids in questions JSON have corresponding files."""
    mapping = build_mapping(raw_dir)
    for qf in sorted(Path("data/raw/questions").glob("*/*_questions.json")):
        qs = json.loads(qf.read_text(encoding="utf-8"))
        docs = {d for q in qs for d in q.get("doc_ids", [])}
        missing = [d for d in docs if d not in mapping]
        print(f"  {qf.stem.replace('_questions',''):20s} {len(docs)-len(missing)}/{len(docs)}"
              + (f"  MISSING: {missing}" if missing else ""))


def show_qa_mapping(qdir="data/raw/questions"):
    """Show question-answer document coverage."""
    for qf in sorted(Path(qdir).glob("*_questions.json")):
        domain = qf.stem.replace("_questions", "")
        with open(qf, encoding="utf-8") as f:
            qs = json.load(f)
        all_docs = set()
        for q in qs:
            for d in q.get("doc_ids", []):
                all_docs.add(d)
        mapping = build_mapping()
        missing = [d for d in all_docs if d not in mapping]
        print(f"  {domain:20s} {len(qs):2d} q -> {len(all_docs):2d} docs  missing={missing or 'NONE'}")


# ===== Pipeline (unified) =====
def process_all(raw_dir="data/raw/raw", out_dir="data/processed", mode="rich"):
    """Process all documents into JSON.

    mode: "rich" (process_docs style) or "simple" (document_processor style)
    """
    rd, od = Path(raw_dir), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    mapping = build_mapping(str(rd))
    print(f"文档总数: {len(mapping)}")

    for doc_id, fp in mapping.items():
        domain = Path(fp).parent.name
        if domain in ("txt", "html", "attachments"):
            domain = "regulatory"
        try:
            if mode == "simple":
                r = process_one(fp, doc_id, domain)
            else:
                r = process_file(fp, doc_id)
            (od / f"{doc_id}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {Path(fp).parent.name:20s} {doc_id:35s} {r.get('total_sections', '?' )} sections")
        except Exception as e:
            print(f"  ERROR {doc_id}: {e}")
    print(f"\n完成，输出目录: {od}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "check":
            validate_mapping()
        elif cmd == "qa":
            show_qa_mapping()
        else:
            print(f"未知命令: {cmd}")
    else:
        process_all()
