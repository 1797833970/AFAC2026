"""项目配置加载：集中读取 config/config.yaml 与环境变量。"""

import os
from pathlib import Path

from dotenv import load_dotenv
import yaml

# 项目根目录 = 本文件上一级的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> dict:
    """读取 config.yaml；文件不存在时返回空字典（全部走默认值）。"""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG = load_config()


def _get(*keys: str, default=None):
    """按嵌套路径取配置，例如 _get("model", "name")。"""
    node = _CONFIG
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


# ---- 常用配置（带环境变量覆盖与默认值） ----

MODEL = os.environ.get("AFAC_MODEL", _get("model", "name", default="qwen3.7-plus"))
TEMPERATURE = float(os.environ.get("AFAC_TEMPERATURE", _get("model", "temperature", default=0.1)))
MAX_TOKENS = int(os.environ.get("AFAC_MAX_TOKENS", _get("model", "max_tokens", default=4096)))

CHUNKS_FILE = str(PROJECT_ROOT / _get("data", "chunks_file", default="data/chunks/all_chunks.json"))
DOC_REGISTRY = str(PROJECT_ROOT / _get("data", "doc_registry", default="data/doc_registry.json"))
QUESTIONS_B_DIR = str(PROJECT_ROOT / _get("data", "questions_b_dir", default="data/upload_b/question_b"))
SUBMIT_TEMPLATE = str(PROJECT_ROOT / _get("data", "submit_template", default="data/upload_b/submit.csv"))

BM25_INDEX_DIR = str(PROJECT_ROOT / _get("retrieval", "bm25", "index_dir", default="data/index"))
BM25_INDEX_FILE = _get("retrieval", "bm25", "index_file", default="bm25_index.pkl")
BM25_TOP_K = int(_get("retrieval", "bm25", "top_k", default=3))
BM25_FALLBACK_K = int(_get("retrieval", "bm25", "fallback_k", default=5))

BATCH_SIZE = int(os.environ.get("AFAC_BATCH_SIZE", _get("agent", "batch_size", default=5)))
SAVE_EVIDENCE = bool(_get("agent", "save_evidence", default=True))

OUTPUT_DIR = str(PROJECT_ROOT / _get("output", "dir", default="output"))
EVIDENCE_DIR = str(PROJECT_ROOT / _get("output", "evidence_dir", default="output/evidence"))
LOGS_DIR = str(PROJECT_ROOT / _get("output", "logs_dir", default="output/logs"))

TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", 5_000_000))


def ensure_output_dirs() -> None:
    """创建输出目录（幂等）。"""
    for path in (OUTPUT_DIR, EVIDENCE_DIR, LOGS_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
