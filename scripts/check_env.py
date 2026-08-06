"""环境自检：确认依赖、密钥、数据文件是否就绪。

用法：
  python scripts/check_env.py            # 离线检查（推荐先跑这个）
  python scripts/check_env.py --api      # 额外做一次 DashScope API 连通性测试
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from agent.settings import (  # noqa: E402
    CHUNKS_FILE,
    DOC_REGISTRY,
    MODEL,
    QUESTIONS_B_DIR,
    SUBMIT_TEMPLATE,
)


REQUIRED_PACKAGES = [
    "dashscope",
    "jieba",
    "dotenv",
    "yaml",
    "requests",
]


def check_packages() -> bool:
    print("== 依赖检查 ==")
    ok = True
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            ok = False
    return ok


def check_env_file() -> bool:
    print("\n== 环境变量 ==")
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("  [WARN] 未找到 .env（可执行 cp .env.example .env 后填写）")
        return False
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key and not key.startswith("sk-"):
        print("  [WARN] DASHSCOPE_API_KEY 格式可疑")
    if key:
        print(f"  [OK] DASHSCOPE_API_KEY 已设置（{key[:6]}...）")
    else:
        print("  [MISSING] DASHSCOPE_API_KEY 未设置")
        return False
    return True


def check_data() -> bool:
    print("\n== 数据文件 ==")
    ok = True
    for label, path in [
        ("题目目录(B榜)", QUESTIONS_B_DIR),
        ("提交模板", SUBMIT_TEMPLATE),
        ("文档注册表", DOC_REGISTRY),
        ("分块文件", CHUNKS_FILE),
    ]:
        exists = Path(path).exists()
        print(f"  [{'OK' if exists else 'MISSING'}] {label}: {path}")
        ok = ok and exists
    return ok


def check_api() -> bool:
    print("\n== API 连通性 ==")
    try:
        import dashscope
        from http import HTTPStatus

        resp = dashscope.MultiModalConversation.call(
            model=MODEL,
            messages=[
                {"role": "user", "content": [{"text": "请只回复：ok"}]},
            ],
            max_tokens=8,
        )
        if resp.status_code == HTTPStatus.OK:
            print(f"  [OK] 模型 {MODEL} 连通正常")
            return True
        print(f"  [FAIL] status={resp.status_code} code={resp.code} msg={resp.message}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="AFAC2026 环境自检")
    parser.add_argument("--api", action="store_true", help="额外执行 API 连通性测试")
    args = parser.parse_args()

    results = [
        check_packages(),
        check_env_file(),
        check_data(),
    ]
    if args.api:
        results.append(check_api())

    ok = all(results)
    print(f"\n{'全部就绪' if ok else '存在缺失项，请按 docs/02-快速开始.md 修复'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
