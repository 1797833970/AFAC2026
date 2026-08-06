"""批量运行四步 Agent 流水线，输出答案、证据与日志。

用法：
  # B 榜全部题目（100 题，5 线程）
  python scripts/run_agent.py --split B

  # 只跑某个领域、前 3 题（联调用）
  python scripts/run_agent.py --split B --domain insurance --limit 3

  # 使用自定义题目目录（A 榜或新数据集，目录内为 json/jsonl）
  python scripts/run_agent.py --questions /path/to/questions --limit 5

输出：
  output/answers.json        全部答案 + token 统计
  output/evidence/<qid>.json 每题检索证据
  output/logs/<qid>.log      每题运行日志
  output/run_log.txt         总体进度
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agent.qa_agent import QAAgent  # noqa: E402
from agent.settings import ensure_output_dirs, QUESTIONS_B_DIR  # noqa: E402


# 领域 → 题目文件名（B 榜官方布局）
DOMAIN_FILES = {
    "financial_contracts": "financial_contracts_b_question.json",
    "financial_reports": "financial_reports_b_questions.jsonl",
    "insurance": "insurance_b_questions.json",
    "regulatory": "regulatory_b_questions.jsonl",
    "research": "research_b_question.jsonl",
}

_thread_local = threading.local()


def get_agent() -> QAAgent:
    """每个线程持有独立的 Agent 实例（BM25 索引在线程内加载一次）。"""
    if not hasattr(_thread_local, "agent"):
        _thread_local.agent = QAAgent()
    return _thread_local.agent


def load_questions(questions_dir: str, domain: str = "") -> list[dict]:
    """加载题目。默认读取 B 榜目录；也可指向任意含 json/jsonl 的目录。"""
    qdir = Path(questions_dir)
    if not qdir.exists():
        raise FileNotFoundError(f"题目目录不存在：{qdir}")

    all_q: list[dict] = []

    def parse_file(path: Path, tag: str) -> None:
        with open(path, "r", encoding="utf-8-sig") as f:
            if path.suffix == ".jsonl":
                items = [json.loads(line) for line in f if line.strip()]
            else:
                items = json.load(f)
        for q in items:
            q["_domain"] = tag
            all_q.append(q)

    if domain:
        fname = DOMAIN_FILES.get(domain)
        if not fname:
            raise ValueError(f"未知领域：{domain}，可选 {list(DOMAIN_FILES)}")
        path = qdir / fname
        if path.exists():
            parse_file(path, domain)
    else:
        for tag, fname in DOMAIN_FILES.items():
            path = qdir / fname
            if path.exists():
                parse_file(path, tag)
        # 兼容任意 json/jsonl（自定义目录）
        if not all_q:
            for path in sorted(qdir.iterdir()):
                if path.suffix in (".json", ".jsonl"):
                    parse_file(path, path.stem)

    print(f"共加载 {len(all_q)} 道题（目录：{qdir}）")
    return all_q


def answer_one(question: dict) -> dict:
    """单题作答（线程内调用）。"""
    agent = get_agent()
    t0 = time.time()
    result = agent.answer_single(question, verbose=False)
    result["_elapsed"] = round(time.time() - t0, 1)
    return result


def save_artifacts(result: dict, evidence_dir: Path, logs_dir: Path) -> None:
    qid = result.get("qid", "unknown")
    evidence = result.get("evidence_retrieval", [])
    (evidence_dir / f"{qid}.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_lines = [
        f"qid: {qid}",
        f"answer: {result.get('answer', '')}",
        f"tokens: {result.get('tokens', {})}",
        f"elapsed: {result.get('_elapsed', 0)}s",
        f"question: {result.get('question', '')[:200]}",
    ]
    (logs_dir / f"{qid}.log").write_text("\n".join(log_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="批量运行 AFAC2026 Agent")
    parser.add_argument("--split", choices=["A", "B"], default="B",
                        help="数据集划分标记（默认 B；A 榜题目需用 --questions 指定目录）")
    parser.add_argument("--questions", default="", help="自定义题目目录（优先于 --split）")
    parser.add_argument("--domain", default="", help="只跑指定领域")
    parser.add_argument("--limit", type=int, default=0, help="最多跑 N 题（0=全部）")
    parser.add_argument("--workers", type=int, default=5, help="并发线程数")
    parser.add_argument("--outdir", default="output", help="输出目录")
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = Path(args.outdir)
    evidence_dir = out_dir / "evidence"
    logs_dir = out_dir / "logs"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # B 榜使用官方题目目录；--questions 可覆盖（如 A 榜或自定义数据）
    questions_dir = args.questions or str(PROJECT_ROOT / QUESTIONS_B_DIR)
    questions = load_questions(questions_dir, args.domain)
    if args.limit > 0:
        questions = questions[: args.limit]

    total_input = total_output = 0
    results = []
    t_start = time.time()
    log_handle = open(out_dir / "run_log.txt", "a", encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(answer_one, q): q for q in questions}
            done = 0
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [FAIL] {q.get('qid')}: {exc}")
                    log_handle.write(f"{q.get('qid')}: ERROR {exc}\n")
                    continue

                qid = result.get("qid", "?")
                answer = result.get("answer", "")
                tokens = result.get("tokens", {})
                total_input += tokens.get("input", 0)
                total_output += tokens.get("output", 0)
                save_artifacts(result, evidence_dir, logs_dir)
                results.append(result)
                done += 1
                line = (
                    f"[{done}/{len(questions)}] {qid}: {answer or '(empty)'} "
                    f"tok={tokens.get('input', 0)}/{tokens.get('output', 0)} "
                    f"elapsed={result.get('_elapsed', 0)}s"
                )
                print(line)
                log_handle.write(line + "\n")
                log_handle.flush()
    finally:
        log_handle.close()

    elapsed = time.time() - t_start
    summary = {
        "total_questions": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_question": round(elapsed / max(len(results), 1), 1),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }
    print("\n== 汇总 ==")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    (out_dir / "answers.json").write_text(
        json.dumps({"summary": summary, "answers": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存：{out_dir / 'answers.json'}")
    print(f"生成提交文件：python scripts/build_submission.py --answers {out_dir / 'answers.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
