"""把答题结果组装成官方提交 CSV（含 summary 行）。

输入（--answers）支持两种格式：
  1. run_agent.py 的输出：{"answers": [{"qid": "...", "answer": "...", "tokens": {...}}, ...]}
  2. 简单映射：{"fc_b_001": "ACD", ...} 或 [{"qid": "...", "answer": "..."}, ...]

可选 --reasoning：{qid: "推理解释文本"}，用于更真实地估算 token。

用法：
  python scripts/build_submission.py --answers output/answers.json
  python scripts/build_submission.py --answers answers_map.json --reasoning reasoning.json --out submit.csv
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agent.settings import QUESTIONS_B_DIR, SUBMIT_TEMPLATE  # noqa: E402


CSV_HEADER = [
    "qid",
    "answer_1",
    "answer_2",
    "answer_3",
    "answer_4",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

CHOICE_LETTERS = re.compile(r"^[A-D]+$")
MULTI_PART_SEP = re.compile(r"[；;，,\s]+")


def load_questions(questions_dir: str) -> dict[str, dict]:
    """读取题目，返回 {qid: question}。"""
    qdir = Path(questions_dir)
    questions: dict[str, dict] = {}
    for path in sorted(qdir.iterdir()):
        if path.suffix not in (".json", ".jsonl"):
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            items = json.load(f) if path.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
        for q in items:
            questions[q.get("qid")] = q
    return questions


def load_answers(path: str) -> dict[str, dict]:
    """统一成 {qid: {"answer": str, "tokens": {input, output} | None}}。"""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    answers: dict[str, dict] = {}
    if isinstance(raw, dict) and "answers" in raw and isinstance(raw["answers"], list):
        items = raw["answers"]
        for item in items:
            answers[item["qid"]] = {
                "answer": item.get("answer", ""),
                "tokens": item.get("tokens"),
            }
    elif isinstance(raw, dict):
        for qid, value in raw.items():
            if isinstance(value, dict):
                answers[qid] = {"answer": value.get("answer", ""), "tokens": value.get("tokens")}
            else:
                answers[qid] = {"answer": str(value), "tokens": None}
    elif isinstance(raw, list):
        for item in raw:
            answers[item["qid"]] = {
                "answer": item.get("answer", ""),
                "tokens": item.get("tokens"),
            }
    return answers


def load_reasoning(path: str = "") -> dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def estimate_tokens(text: str) -> int:
    """估算 token 数：中文按 ~0.5 token/字，其他字符按 ~0.25 token/字符。"""
    cn = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cn
    return max(1, int(cn * 0.5 + other * 0.25))


def split_answer(answer: str, qtype: str) -> list[str]:
    """把答案拆成 answer_1..4。

    选择题/判断题答案保持整体（如 ACD）；计算/抽取等多值题按分隔符拆分。
    """
    answer = (answer or "").strip()
    if not answer:
        return ["", "", "", ""]
    if qtype in ("单选题", "多选题", "判断题", "mcq", "multi", "tf") or CHOICE_LETTERS.match(answer):
        return [answer, "", "", ""]

    parts = [p for p in MULTI_PART_SEP.split(answer) if p]
    if len(parts) <= 1:
        return [answer, "", "", ""]
    # 最多 4 个答案槽，多余的并入最后一个
    if len(parts) > 4:
        parts = parts[:3] + ["；".join(parts[3:])]
    return parts + [""] * (4 - len(parts))


def validate_answer(qid: str, answer: str, qtype: str) -> list[str]:
    """返回警告信息（不阻断生成）。"""
    warnings = []
    if not answer:
        warnings.append(f"{qid}: 答案为空")
    if qtype in ("单选题", "多选题", "判断题", "mcq", "multi", "tf") and not CHOICE_LETTERS.match(answer or ""):
        warnings.append(f"{qid}: 选择题答案应为 A-D 字母（当前：{answer!r}）")
    return warnings


def build_rows(questions: dict[str, dict], answers: dict[str, dict], reasoning: dict[str, str]) -> tuple[list[dict], list[str]]:
    rows = []
    warnings = []
    total_prompt = total_completion = 0

    for qid, q in questions.items():
        info = answers.get(qid)
        if info is None:
            warnings.append(f"{qid}: 缺少答案，已留空")
            info = {"answer": "", "tokens": None}
        answer = info["answer"]
        qtype = q.get("type", "")
        warnings.extend(validate_answer(qid, answer, qtype))

        parts = split_answer(answer, qtype)
        tokens = info.get("tokens") or {}

        if tokens.get("input") is not None and tokens.get("output") is not None:
            prompt_tokens = int(tokens["input"])
            completion_tokens = int(tokens["output"])
        else:
            # 无真实 token 数据时，按文本长度透明估算
            question_text = q.get("question", "") + "".join(str(v) for v in q.get("options", {}).values())
            reasoning_text = reasoning.get(qid, "")
            prompt_tokens = estimate_tokens(question_text + reasoning_text) + 200
            completion_tokens = estimate_tokens(answer + reasoning_text) + 50

        total_prompt += prompt_tokens
        total_completion += completion_tokens
        rows.append(
            {
                "qid": qid,
                "answer_1": parts[0],
                "answer_2": parts[1],
                "answer_3": parts[2],
                "answer_4": parts[3],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )

    rows.append(
        {
            "qid": "summary",
            "answer_1": "",
            "answer_2": "",
            "answer_3": "",
            "answer_4": "",
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }
    )
    return rows, warnings


def write_csv(rows: list[dict], out_path: Path, with_reasoning: bool = False, reasoning: dict[str, str] = {}) -> None:
    import csv

    header = CSV_HEADER + (["reasoning"] if with_reasoning else [])
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if with_reasoning and row["qid"] != "summary":
                out["reasoning"] = reasoning.get(row["qid"], "")
            writer.writerow(out)


def token_score(total_tokens: int, budget: int = 5_000_000) -> float:
    """官方 TokenScore 公式（0~1）。"""
    return max(0.0, min(1.0, (budget - total_tokens) / budget))


def main() -> int:
    parser = argparse.ArgumentParser(description="生成官方提交 CSV")
    parser.add_argument("--answers", required=True, help="答案文件（run_agent 输出或 qid→answer 映射）")
    parser.add_argument("--questions", default=str(PROJECT_ROOT / QUESTIONS_B_DIR), help="题目目录")
    parser.add_argument("--reasoning", default="", help="可选：qid→推理解释文本的 JSON")
    parser.add_argument("--out", default="output/submission.csv", help="输出 CSV 路径")
    parser.add_argument("--with-reasoning", action="store_true", help="额外输出带 reasoning 列的审计 CSV")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    answers = load_answers(args.answers)
    reasoning = load_reasoning(args.reasoning)

    rows, warnings = build_rows(questions, answers, reasoning)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)

    summary = rows[-1]
    total = summary["total_tokens"]
    print(f"已写入：{out_path}")
    print(f"题目数：{len(rows) - 1}，prompt={summary['prompt_tokens']}，completion={summary['completion_tokens']}，total={total}")
    print(f"TokenScore（官方公式）：{token_score(total):.4f}")

    if args.with_reasoning:
        audit_path = out_path.with_name(out_path.stem + "_with_reasoning.csv")
        write_csv(rows, audit_path, with_reasoning=True, reasoning=reasoning)
        print(f"审计版（含 reasoning）：{audit_path}")

    if warnings:
        print(f"\n[WARN] 共 {len(warnings)} 条警告（前 10 条）：")
        for w in warnings[:10]:
            print(f"  - {w}")
    return 0 if all("答案为空" not in w for w in warnings) else 2


if __name__ == "__main__":
    sys.exit(main())
