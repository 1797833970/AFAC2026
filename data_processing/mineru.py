"""
MinerU API 批量处理：PDF / TXT / HTML → Markdown
================================================
流程：
  1. 收集原始文件
  2. 扫描已有输出，跳过已处理的
  3. 拆分 >200 页的 PDF
  4. POST + PUT 上传到 MinerU
  5. 轮询获取解析结果 → Markdown
  6. 合并拆分文件的各部分

注意：此脚本仅处理 .pdf / .txt / .html 三种格式。
如有 .doc/.docx/.ppt/.xlsx 等 Office 文件，请先转换为 PDF 再使用本脚本。

使用前请设置环境变量:
  export MINERU_TOKEN="your_token"    (Linux/Mac)
  set MINERU_TOKEN="your_token"       (Windows)

用法：
  python -m data_processing.mineru check           # 验证文档映射
  python -m data_processing.mineru                  # 处理全部文档（自动跳过已有结果）
  python -m data_processing.mineru --dry-run        # 扫描模式：只看不传
  python -m data_processing.mineru --limit 3        # 只处理前3个（测试用）
  python -m data_processing.mineru --force          # 强制重新处理所有文件

Moved from mineru_process.py into data_processing package.
"""

import os, sys, json, time
from pathlib import Path

# Auto-load .env file
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# ---- MinerU 配置 ----
TOKEN = os.environ.get("MINERU_TOKEN", "")
API_BASE = "https://mineru.net/api/v4"
BATCH_SIZE = 50            # max 50 files per batch
BATCH_INTERVAL = 65        # interval between batches (sec), rate limit 50 files/min
DAILY_LIMIT = 5000         # daily upload limit
HTML_DAILY_LIMIT = 100     # HTML daily limit
POLL_INTERVAL = 30         # poll interval (sec)
MAX_POLL_TIME = 1800       # max wait per batch (sec)

import requests
from pypdf import PdfReader, PdfWriter


# ============================================================
#  Utility functions
# ============================================================

def fix_proxy():
    """Bypass broken system proxy for critical domains."""
    domains = {
        "mineru.net", "cdn-mineru.openxlab.org.cn",
        "dashscope.aliyuncs.com",
        "localhost", "127.0.0.1", "::1",
    }
    cur = os.environ.get("NO_PROXY", "")
    cur_set = set(d.strip() for d in cur.split(",") if d.strip())
    missing = domains - cur_set
    if missing:
        os.environ["NO_PROXY"] = cur + ("," if cur else "") + ",".join(sorted(missing))


def _split_pdf(filepath, doc_id, max_pages=200, temp_dir="data/temp"):
    """Split oversized PDF into chunks ≤ max_pages, return [(chunk_path, chunk_doc_id), ...]."""
    reader = PdfReader(filepath)
    total = len(reader.pages)
    if total <= max_pages:
        return [(filepath, doc_id)]

    os.makedirs(temp_dir, exist_ok=True)
    chunks = []
    for start in range(0, total, max_pages):
        part = (start // max_pages) + 1
        cid = f"{doc_id}_p{part}"
        cpath = os.path.join(temp_dir, f"{cid}.pdf")

        writer = PdfWriter()
        for i in range(start, min(start + max_pages, total)):
            writer.add_page(reader.pages[i])
        with open(cpath, "wb") as f:
            writer.write(f)
        chunks.append((cpath, cid))

    print(f"  {doc_id} ({total}pg) -> {len(chunks)} chunks")
    return chunks


def _merge_chunks(base_id, output_dir, part_count, subdir=""):
    """Merge chunked markdown files, delete intermediates."""
    od = Path(output_dir) / subdir if subdir else Path(output_dir)
    chunk_files = [od / f"{base_id}_p{i}.md" for i in range(1, part_count + 1)]
    missing = [str(f) for f in chunk_files if not f.exists()]
    if missing:
        print(f"  [{base_id}] merge skip: missing {len(missing)}/{part_count}")
        return False
    merged = ""
    for i, cf in enumerate(chunk_files, 1):
        merged += f"\n<!-- Part {i} -->\n\n" + cf.read_text(encoding="utf-8")
        cf.unlink()
    (od / f"{base_id}.md").write_text(merged, encoding="utf-8")
    print(f"  [{base_id}] merged {part_count} parts -> {subdir}/")
    return True


def _cleanup_temp(doc_id, temp_dir="data/temp"):
    """Clean up temporary split PDFs."""
    for f in Path(temp_dir).glob(f"{doc_id}_p*.pdf"):
        try:
            f.unlink()
        except OSError:
            pass


# ============================================================
#  File collection & validation
# ============================================================

def _walk_raw_dir(raw_dir="data/raw/raw"):
    """Walk raw directory, yield (doc_id, filepath, ext, subdir).

    subdir mirrors the directory hierarchy under raw_dir:
      raw_dir/financial_contracts/text01.pdf       ->  subdir = "financial_contracts"
      raw_dir/regulatory/attachments/xxx.pdf       ->  subdir = "regulatory/attachments"
      raw_dir/regulatory/txt/xxx.txt               ->  subdir = "regulatory/txt"
    """
    EXT = {".pdf", ".html"}
    rd = Path(raw_dir)
    for root in rd.iterdir():
        if not root.is_dir():
            continue
        for sub in ["", "txt", "html", "attachments"]:
            sp = root / sub if sub else root
            if not sp.is_dir():
                continue
            for f in sorted(sp.glob("*")):
                if f.suffix.lower() in EXT:
                    rel = f.parent.relative_to(rd)
                    subdir = str(rel) if str(rel) != "." else ""
                    yield (f.stem, str(f), f.suffix.lower(), subdir)


def collect_files(raw_dir="data/raw/raw"):
    """Collect all raw files (no splitting)."""
    return list(_walk_raw_dir(raw_dir))


def build_mapping(raw_dir="data/raw/raw"):
    """doc_id -> filepath mapping (for validation)."""
    return {doc_id: fp for doc_id, fp, *_ in _walk_raw_dir(raw_dir)}


def validate_mapping(raw_dir="data/raw/raw"):
    """Verify that all doc_ids in questions JSON have corresponding files."""
    mapping = build_mapping(raw_dir)
    for qf in sorted(Path("data/raw/questions").glob("*/*_questions.json")):
        qs = json.loads(qf.read_text(encoding="utf-8"))
        docs = {d for q in qs for d in q.get("doc_ids", [])}
        missing = [d for d in docs if d not in mapping]
        print(f"  {qf.stem.replace('_questions',''):20s} {len(docs)-len(missing)}/{len(docs)}"
              + (f"  MISSING: {missing}" if missing else ""))


# ============================================================
#  MinerU API interaction
# ============================================================

def upload_batch(batch_files, model_version="vlm"):
    """Upload a batch of files to MinerU, return batch_id or None."""
    if not TOKEN:
        print("  错误: 未设置 MINERU_TOKEN")
        return None

    url = f"{API_BASE}/file-urls/batch"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

    payload = {
        "files": [{"name": os.path.basename(fp), "data_id": did}
                  for did, fp, *_ in batch_files],
        "model_version": model_version,
    }

    print(f"  申请上传链接: {len(batch_files)} 个文件...")
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"  请求失败: HTTP {resp.status_code}")
        return None

    result = resp.json()
    if result.get("code") != 0:
        print(f"  接口返回错误: {result.get('msg')}")
        return None

    batch_id = result["data"]["batch_id"]
    file_urls = result["data"]["file_urls"]
    print(f"  batch_id: {batch_id}")

    success = 0
    for i, (did, fp, *_) in enumerate(batch_files):
        url_i = file_urls[i] if i < len(file_urls) else None
        if not url_i:
            print(f"    [{i+1}] {did}: 没有上传链接")
            continue
        try:
            res = requests.put(url_i, data=open(fp, "rb").read())
            if res.status_code == 200:
                success += 1
                print(f"    [{i+1}/{len(batch_files)}] {did}: OK")
            else:
                print(f"    [{i+1}/{len(batch_files)}] {did}: HTTP {res.status_code}")
        except Exception as e:
            print(f"    [{i+1}/{len(batch_files)}] {did}: {e}")

    print(f"  上传: {success}/{len(batch_files)}")
    return batch_id if success > 0 else None


def poll_results(batch_id, upload_files, output_dir="data/processed_md"):
    """Poll parsing progress, auto-download markdown on completion."""
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # file_name -> (doc_id, subdir)
    name_map = {os.path.basename(fp): (did, sd) for did, fp, *_, sd in upload_files}

    print(f"  等待解析 (batch: {batch_id})...")
    api_url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    start = time.time()
    done = {}
    failed = {}
    progress = {}

    while time.time() - start < MAX_POLL_TIME:
        try:
            resp = requests.get(api_url, headers={"Authorization": f"Bearer {TOKEN}"})
        except Exception as e:
            print(f"  请求异常: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}")
            time.sleep(POLL_INTERVAL)
            continue

        body = resp.json()
        if body.get("code") != 0:
            print(f"  接口异常: {body.get('msg')}")
            time.sleep(POLL_INTERVAL)
            continue

        results = body.get("data", {}).get("extract_result", [])
        if not results:
            time.sleep(POLL_INTERVAL)
            continue

        for item in results:
            fname = item.get("file_name", "")
            did, sd = name_map.get(fname, (fname, ""))
            state = item.get("state", "unknown")

            if did in done or did in failed:
                continue

            if state == "done":
                zip_url = item.get("full_zip_url", "")
                if zip_url:
                    print(f"\n    [{did}] done, downloading...")
                    try:
                        zr = requests.get(zip_url,
                            headers={"Authorization": f"Bearer {TOKEN}"} if zip_url.startswith("http") else {})
                        if zr.status_code == 200:
                            import zipfile, io
                            with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
                                md_files = sorted(n for n in zf.namelist() if n.endswith(".md"))
                                if md_files:
                                    content = "\n".join(zf.read(m).decode("utf-8") for m in md_files)
                                    tdir = od / sd if sd else od
                                    tdir.mkdir(parents=True, exist_ok=True)
                                    (tdir / f"{did}.md").write_text(content, encoding="utf-8")
                                    done[did] = zip_url
                                    print(f"    [{did}] -> {sd}/  ({len(content):,} chars)")
                                else:
                                    failed[did] = "no md in zip"
                                    print(f"    [{did}] no .md in zip")
                        else:
                            print(f"    [{did}] download HTTP {zr.status_code}")
                    except Exception as e:
                        print(f"    [{did}] download error: {e}")
                        failed[did] = str(e)
                else:
                    done[did] = ""

            elif state == "failed":
                print(f"    [{did}] FAIL: {item.get('err_msg', '')}")
                failed[did] = item.get("err_msg", "")

            elif state == "running":
                pg = item.get("extract_progress", {})
                cur, tot = pg.get("extracted_pages", 0), pg.get("total_pages", 0)
                if (cur, tot) != progress.get(did):
                    pct = int(cur / tot * 100) if tot else 0
                    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                    print(f"    [{did}] {bar} {pct}% ({cur}/{tot} pg)", end="\r")
                    progress[did] = (cur, tot)

            elif state in ("waiting-file", "pending", "converting"):
                labels = {"waiting-file": "waiting", "pending": "queued", "converting": "converting"}
                print(f"    [{did}] {labels.get(state, state)}...", end="\r")

        if len(done) + len(failed) == len(upload_files):
            print(f"\n  batch done: {len(done)}/{len(upload_files)}")
            break

        time.sleep(POLL_INTERVAL)

    elapsed = int(time.time() - start)
    print(f"  batch finish: {len(done)} ok, {len(failed)} fail, {elapsed}s")
    return done, failed


# ============================================================
#  Main pipeline
# ============================================================

def process_all(raw_dir="data/raw/raw", out_dir="data/processed_md",
                limit=None, force=False, dry_run=False):
    """Process all documents: collect → skip completed → split large PDFs → upload → poll → merge."""
    if not TOKEN and not dry_run:
        print("错误: 未设置 MINERU_TOKEN")
        print("  export MINERU_TOKEN=xxx  或  --token xxx")
        return 0

    fix_proxy()

    # ===== Phase 1: Collect raw files =====
    raw = list(_walk_raw_dir(raw_dir))
    print(f"原始文件: {len(raw)}")
    if limit:
        raw = raw[:limit]
        print(f"限制处理: {limit}")

    # ===== Phase 2: Skip completed =====
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    skipped = 0

    if not force:
        pending = []
        for did, fp, ext, sd in raw:
            tdir = od / sd if sd else od
            if (tdir / f"{did}.md").exists():
                skipped += 1
                continue
            pending.append((did, fp, ext, sd))
        raw = pending
        if skipped:
            print(f"跳过已处理: {skipped} 个  (--force 强制重处理)")
    else:
        print("--force: 不跳过任何文件")

    if not raw:
        print("全部已完成")
        return 0

    # ===== Phase 3: Split large PDFs (only for files still needing processing) =====
    upload_list = []        # (chunk_doc_id, filepath, ext, subdir)
    chunk_of = {}           # base_id -> [chunk_doc_id, ...]
    chunk_sd = {}           # base_id -> subdir

    for did, fp, ext, sd in raw:
        if ext == ".pdf":
            parts = _split_pdf(fp, did)
            for pfp, pid in parts:
                upload_list.append((pid, pfp, ".pdf", sd))
                if pid != did:          # was actually split
                    chunk_of.setdefault(did, []).append(pid)
                    chunk_sd[did] = sd
        else:
            upload_list.append((did, fp, ext, sd))

    n_chunks = sum(len(v) for v in chunk_of.values())
    print(f"待上传: {len(upload_list)}" + (f"  (含 {n_chunks} 个分块)" if n_chunks else ""))

    # ===== Phase 4: Dry-run report =====
    if dry_run:
        by_dir = {}
        for did, fp, ext, sd in raw:
            by_dir.setdefault(sd or "(root)", []).append(did)
        print(f"\n  === Dry-run ===")
        print(f"  已跳过: {skipped}")
        print(f"  待处理: {len(raw)} 个文档 -> {len(upload_list)} 次上传")
        for d in sorted(by_dir):
            print(f"    {d}/  ({len(by_dir[d])})")
        print(f"\n  去掉 --dry-run 开始处理")
        return 0

    # ===== Phase 5: Batch upload + poll =====
    total_ok = 0
    for i in range(0, len(upload_list), BATCH_SIZE):
        batch = upload_list[i:i + BATCH_SIZE]
        n = i // BATCH_SIZE + 1
        total_n = (len(upload_list) - 1) // BATCH_SIZE + 1
        print(f"\n批次 {n}/{total_n}")

        vlm_files = [x for x in batch if x[2] != ".html"]
        html_files = [x for x in batch if x[2] == ".html"]

        for model_files, model_ver in [(vlm_files, "vlm"), (html_files, "MinerU-HTML")]:
            if not model_files:
                continue
            bid = upload_batch(model_files, model_version=model_ver)
            if bid:
                done, _ = poll_results(bid, model_files, out_dir)
                total_ok += len(done)

        if i + BATCH_SIZE < len(upload_list):
            print(f"频控等待 {BATCH_INTERVAL}s (50文件/分钟)...")
            time.sleep(BATCH_INTERVAL)

    # ===== Phase 6: Merge chunks =====
    merged = 0
    for base_id, cids in chunk_of.items():
        sd = chunk_sd.get(base_id, "")
        if _merge_chunks(base_id, out_dir, len(cids), sd):
            merged += 1
        _cleanup_temp(base_id)

    if merged:
        print(f"\n合并: {merged} 个文档")
    print(f"\n完成: {total_ok}/{len(upload_list)} 上传成功")
    return total_ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        validate_mapping()
    else:
        limit = None
        force = "--force" in sys.argv
        dry_run = "--dry-run" in sys.argv
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                limit = int(sys.argv[idx + 1])
        if "--token" in sys.argv:
            idx = sys.argv.index("--token")
            if idx + 1 < len(sys.argv):
                TOKEN = sys.argv[idx + 1]
        process_all(limit=limit, force=force, dry_run=dry_run)
